# app/services/matching_service.py
import numpy as np
import re
import logging
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import torch
import json
import os
from datetime import datetime
from flask import current_app

logger = logging.getLogger(__name__)

class MatchingService:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.llm_service = current_app.llm_service if current_app else None
        self.skill_model = SentenceTransformer('all-mpnet-base-v2')
        self.match_cache = {}
        self.cache_file_path = os.path.join(current_app.config['EMBEDDINGS_STORAGE_PATH'], 'match_cache.json') if current_app else "match_cache.json"
        
        try:
            self.context_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            logger.info("Loaded context model for MatchingService")
        except Exception as e:
            logger.warning(f"Could not load context model: {e}")
            self.context_model = None
        
        self._load_cache()

    def _load_cache(self):
        """Load match cache from file if exists"""
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, 'r') as f:
                    self.match_cache = json.load(f)
                logger.info(f"Loaded match cache with {len(self.match_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load match cache: {e}")
                self.match_cache = {}

    def _save_cache(self):
        """Save current match cache to file"""
        try:
            with open(self.cache_file_path, 'w') as f:
                json.dump(self.match_cache, f)
            logger.info(f"Saved match cache with {len(self.match_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save match cache: {e}")

    def _get_cache_key(self, resume_id: str, job_id: str) -> str:
        """Generate consistent cache key"""
        return f"{resume_id}||{job_id}"

    def _get_cached_match(self, resume_id: str, job_id: str) -> Optional[Dict]:
        """Retrieve cached match if available and recent"""
        cache_key = self._get_cache_key(resume_id, job_id)
        cached = self.match_cache.get(cache_key)
        
        if cached:
            cache_time = datetime.fromisoformat(cached['timestamp'])
            if (datetime.now() - cache_time).days < 7:
                return cached['data']
        return None

    def _cache_match(self, resume_id: str, job_id: str, result: Dict):
        """Cache match results"""
        cache_key = self._get_cache_key(resume_id, job_id)
        self.match_cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'data': result
        }
        self._save_cache()

    def match_resume_to_job(self, resume_id: str, job_id: str) -> Tuple[Optional[Dict], Optional[str]]:
        # Check cache first
        cached_result = self._get_cached_match(resume_id, job_id)
        if cached_result:
            logger.info(f"Using cached result for {resume_id} and {job_id}")
            return cached_result, None

        if not self.embedding_service:
            return None, "Embedding service not initialized"
        if not hasattr(self, 'llm_service') or not self.llm_service:
            return None, "LLM service not initialized"

        try:
            res = self.embedding_service.get_resume_data(resume_id)
            job = self.embedding_service.get_job_data(job_id)
            
            if res is None:
                return None, f"Resume '{resume_id}' not found"
            if job is None:
                return None, f"Job '{job_id}' not found"

            r_ent = res.get('refined_entities', {})
            j_ent = job.get('refined_entities', {})

            # Prepare data for matching
            resume_data = {
                'skills': [s['name'] if isinstance(s, dict) and 'name' in s else str(s) 
                          for s in r_ent.get('skills', [])],
                'experience': r_ent.get('experience', []),
                'education': r_ent.get('education', [])
            }
            
            job_data = {
                'skills': [s['name'] if isinstance(s, dict) and 'name' in s else str(s) 
                          for s in j_ent.get('skills', [])],
                'experience': j_ent.get('experience', []),
                'education': j_ent.get('education', [])
            }

            # Get scores from LLM
            llm_scores, token_usage = self.llm_service.calculate_match_score(resume_data, job_data)
            
            if 'error' in llm_scores:
                return None, llm_scores['error']

            result = {
                'scores': {
                    'skills': llm_scores.get('skills_match', 0),
                    'experience': llm_scores.get('experience_match', 0),
                    'education': llm_scores.get('education_match', 0),
                    'overall': llm_scores.get('overall_score', 0),
                    'domain_alignment': self._calculate_domain_alignment(
                        job_data['skills'], 
                        resume_data['skills']
                    )
                },
                'match_analysis': llm_scores.get('match_analysis', ''),
                'candidate_info': {
                    'name': r_ent.get('name', 'N/A'),
                    'email': r_ent.get('email', 'N/A'),
                    'phone': r_ent.get('phone', 'N/A'),
                    'skills': resume_data['skills'],
                    'experience': r_ent.get('experience', []),
                    'education': r_ent.get('education', []),
                    'certifications': r_ent.get('certifications', []),
                    'projects': r_ent.get('projects', []),
                    'summary': r_ent.get('summary_objective', '')
                },
                'job_info': {
                    'skills': job_data['skills'],
                    'experience': j_ent.get('experience', []),
                    'education': j_ent.get('education', []),
                    'summary': j_ent.get('summary_objective', ''),
                    'title': j_ent.get('job_title', ''),
                    'company': j_ent.get('company', ''),
                    'location': j_ent.get('location', '')
                },
                'ids': {
                    'resume_id': resume_id,
                    'job_id': job_id
                },
                'metadata': {
                    'llm_token_usage': token_usage,
                    'timestamp': datetime.now().isoformat()
                }
            }

            self._cache_match(resume_id, job_id, result)
            return result, None

        except Exception as e:
            logger.error(f"Error matching resume {resume_id} to job {job_id}: {str(e)}")
            return None, f"Internal server error: {str(e)}"

    def _calculate_domain_alignment(self, job_skills: List[str], resume_skills: List[str]) -> float:
        if not job_skills or not resume_skills:
            return 0.0
        
        skill_categories = {
            "AI/ML": ["machine learning", "deep learning", "neural networks", "ai", "artificial intelligence",
                     "tensorflow", "pytorch", "keras", "scikit-learn", "llm", "nlp", "computer vision", "cv"],
            "Data": ["data science", "data analysis", "data engineering", "pandas", "numpy", "spark", "hadoop",
                    "etl", "data pipeline", "data visualization", "tableau", "powerbi"],
            "Cloud": ["aws", "azure", "gcp", "google cloud", "amazon web services", "cloud computing",
                     "kubernetes", "docker", "terraform", "devops", "ci/cd"],
            "Programming": ["python", "java", "javascript", "c++", "c#", "sql", "r", "scala", "go", "typescript"],
            "Web": ["html", "css", "react", "angular", "vue", "node.js", "django", "flask", "spring", "express"],
            "Database": ["sql", "nosql", "mysql", "postgresql", "mongodb", "cassandra", "redis", "oracle", "dynamodb"]
        }
        
        category_coverage = {}
        job_skills_lower = [s.lower() for s in job_skills]
        resume_skills_lower = [s.lower() for s in resume_skills]
        
        for category, keywords in skill_categories.items():
            job_count = sum(1 for kw in keywords if any(kw in js for js in job_skills_lower))
            resume_count = sum(1 for kw in keywords if any(kw in rs for rs in resume_skills_lower))
            
            if job_count > 0:
                category_coverage[category] = min(resume_count / job_count, 1.0)
        
        if not category_coverage:
            return 0.0
        
        return sum(category_coverage.values()) / len(category_coverage)

    def find_top_matches_for_job(self, job_id: str, top_n: int = 5) -> Tuple[List[Dict], Optional[str]]:
        if not self.embedding_service:
            return [], "Embedding service not initialized"

        job = self.embedding_service.get_job_data(job_id)
        if job is None:
            return [], f"Job '{job_id}' not found."

        try:
            cached_results = []
            uncached_resume_ids = []
            
            for rid in self.embedding_service.resume_embeddings.keys():
                cache_key = self._get_cache_key(rid, job_id)
                cached = self.match_cache.get(cache_key)
                if cached:
                    cached_results.append({
                        'resume_id': rid,
                        'candidate_name': cached['data']['candidate_info'].get('name', 'N/A'),
                        'overall_score': cached['data']['scores'].get('overall', 0),
                        'skills_score': cached['data']['scores'].get('skills', 0),
                        'experience_score': cached['data']['scores'].get('experience', 0),
                        'education_score': cached['data']['scores'].get('education', 0),
                        'domain_alignment': cached['data']['scores'].get('domain_alignment', 0),
                        'from_cache': True
                    })
                else:
                    uncached_resume_ids.append(rid)

            new_results = []
            for rid in uncached_resume_ids:
                details, err = self.match_resume_to_job(rid, job_id)
                if not err:
                    new_results.append({
                        'resume_id': rid,
                        'candidate_name': details['candidate_info']['name'],
                        'overall_score': details['scores']['overall'],
                        'skills_score': details['scores']['skills'],
                        'experience_score': details['scores']['experience'],
                        'education_score': details['scores']['education'],
                        'domain_alignment': details['scores']['domain_alignment'],
                        'from_cache': False
                    })

            all_results = cached_results + new_results
            all_results.sort(key=lambda x: x['overall_score'], reverse=True)
            
            return all_results[:top_n], None

        except Exception as e:
            logger.error(f"Error in find_top_matches_for_job: {str(e)}")
            return [], f"Internal server error: {str(e)}"

    def precompute_matches_for_job(self, job_id: str) -> Tuple[bool, str]:
        """Precompute matches for a job to warm up the cache"""
        if not self.embedding_service:
            return False, "Embedding service not initialized"

        job = self.embedding_service.get_job_data(job_id)
        if job is None:
            return False, f"Job '{job_id}' not found"
        
        try:
            resume_ids = list(self.embedding_service.resume_embeddings.keys())
            logger.info(f"Precomputing matches for job {job_id} against {len(resume_ids)} resumes")
            
            for rid in resume_ids:
                self.match_resume_to_job(rid, job_id)
                
            return True, f"Precomputed matches for {len(resume_ids)} resumes"
        
        except Exception as e:
            logger.error(f"Error precomputing matches for job {job_id}: {str(e)}")
            return False, f"Error precomputing matches: {str(e)}"