# app/services/matching_service.py
import json
import logging
import os
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
from flask import current_app
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MatchingService:
    """
    Orchestrates overall matching using:
      - EmbeddingService for skill similarity
      - LLMService for weighted final scores
    Includes a simple file-backed cache so repeat matches are fast.
    """

    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.llm_service = current_app.llm_service if current_app else None

        # cached embeddings for extra semantic context if needed
        try:
            self.context_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            logger.info("Loaded context model for MatchingService")
        except Exception as e:
            logger.warning(f"Could not load context model: {e}")
            self.context_model = None

        self.match_cache: Dict[str, dict] = {}
        base_path = (
            current_app.config.get("EMBEDDINGS_STORAGE_PATH")
            if current_app
            else "."
        )
        self.cache_file_path = os.path.join(base_path, "match_cache.json")

        self._load_cache()

    # ---- Cache helpers ------------------------------------------------------

    def _get_cache_key(self, resume_id: str, job_id: str) -> str:
        return f"{resume_id}||{job_id}"

    def _load_cache(self) -> None:
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, "r") as f:
                    self.match_cache = json.load(f)
                logger.info(f"Loaded match cache with {len(self.match_cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load match cache: {e}")
                self.match_cache = {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_file_path, "w") as f:
                json.dump(self.match_cache, f)
            logger.info(f"Saved match cache with {len(self.match_cache)} entries")
        except Exception as e:
            logger.warning(f"Failed to save match cache: {e}")

    def _get_cached_match(self, resume_id: str, job_id: str) -> Optional[Dict]:
        key = self._get_cache_key(resume_id, job_id)
        cached = self.match_cache.get(key)
        if cached:
            try:
                cache_time = datetime.fromisoformat(cached["timestamp"])
                if (datetime.now() - cache_time).days < 7:
                    return cached["data"]
            except Exception:
                return None
        return None

    def _cache_match(self, resume_id: str, job_id: str, result: Dict) -> None:
        key = self._get_cache_key(resume_id, job_id)
        self.match_cache[key] = {
            "timestamp": datetime.now().isoformat(),
            "data": result,
        }
        self._save_cache()

    # ---- Matching core ------------------------------------------------------

    def match_resume_to_job(self, resume_id: str, job_id: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Returns a full matching payload used by the /jobs/match-specific endpoint.
        """
        cached = self._get_cached_match(resume_id, job_id)
        if cached:
            logger.info(f"Using cached result for {resume_id} and {job_id}")
            return cached, None

        if not self.embedding_service:
            return None, "Embedding service not initialized"
        if not (hasattr(self, "llm_service") and self.llm_service):
            return None, "LLM service not initialized"

        try:
            rdata = self.embedding_service.get_resume_data(resume_id)
            jdata = self.embedding_service.get_job_data(job_id)
            if rdata is None:
                return None, f"Resume '{resume_id}' not found"
            if jdata is None:
                return None, f"Job '{job_id}' not found"

            r_ent = rdata.get("refined_entities", {})
            j_ent = jdata.get("refined_entities", {})

            # Skill similarity
            skill_sim = self.embedding_service.get_skill_similarity_matrix(resume_id, job_id)

            resume_payload = {
                "matched_skills": r_ent.get("matched_skills", []),
                "skills": r_ent.get("matched_skills", r_ent.get("skills", [])),
                "experience": r_ent.get("experience", []),
                "education": r_ent.get("education", []),
                "skill_categories": r_ent.get("skill_categories", {}),
            }
            job_payload = {
                "matched_skills": j_ent.get("matched_skills", []),
                "skills": j_ent.get("matched_skills", j_ent.get("skills", [])),
                "experience": j_ent.get("experience", []),
                "education": j_ent.get("education", []),
                "skill_categories": j_ent.get("skill_categories", {}),
            }

            llm_scores, token_usage = self.llm_service.calculate_match_score(
                resume_payload, job_payload, skill_sim
            )
            if "error" in llm_scores:
                return None, llm_scores["error"]

            result = {
                "scores": {
                    "skills": llm_scores.get("scores", {}).get("skills", 0.0),
                    "experience": llm_scores.get("scores", {}).get("experience", 0.0),
                    "education": llm_scores.get("scores", {}).get("education", 0.0),
                    "overall": llm_scores.get("scores", {}).get("overall", 0.0),
                },
                "skill_analysis": {
                    "total_job_skills": skill_sim.get("total_job_skills", 0),
                    "total_resume_skills": skill_sim.get("total_resume_skills", 0),
                    "matched_skills_count": skill_sim.get("matched_count", 0),
                    "skill_match_percentage": skill_sim.get("match_percentage", 0),
                    "detailed_matches": skill_sim.get("matched_skills", []),
                },
                "matched_skills": llm_scores.get("matched_skills", []),
                "match_analysis": llm_scores.get("match_analysis", ""),
                "candidate_info": {
                    "name": r_ent.get("name", "N/A"),
                    "email": r_ent.get("email", "N/A"),
                    "phone": r_ent.get("phone", "N/A"),
                    "matched_skills": resume_payload.get("matched_skills", []),
                    "all_extracted_skills": r_ent.get("skills", []),
                    "skill_categories": r_ent.get("skill_categories", {}),
                    "experience": r_ent.get("experience", []),
                    "education": r_ent.get("education", []),
                    "certifications": r_ent.get("certifications", []),
                    "projects": r_ent.get("projects", []),
                    "summary": r_ent.get("summary_objective", ""),
                },
                "job_info": {
                    "required_skills": job_payload.get("matched_skills", []),
                    "all_extracted_skills": j_ent.get("skills", []),
                    "skill_categories": j_ent.get("skill_categories", {}),
                    "experience": j_ent.get("experience", []),
                    "education": j_ent.get("education", []),
                    "summary": j_ent.get("summary_objective", ""),
                    "title": j_ent.get("job_title", ""),
                    "company": j_ent.get("company", ""),
                    "location": j_ent.get("location", ""),
                },
                "ids": {"resume_id": resume_id, "job_id": job_id},
                "metadata": {
                    "llm_token_usage": token_usage,
                    "timestamp": datetime.now().isoformat(),
                    "matching_version": "2.0_skill_library_enhanced",
                },
            }

            self._cache_match(resume_id, job_id, result)
            return result, None
        except Exception as e:
            logger.exception(f"Error matching resume {resume_id} to job {job_id}: {e}")
            return None, f"Internal server error: {str(e)}"

    # ---- Batch helpers used by list/match screens --------------------------

    def find_top_matches_for_job(self, job_id: str, top_n: int = 5) -> Tuple[List[dict], Optional[str]]:
        """
        For a job, rank all resumes by overall score.
        Returns a list of {resume_id, candidate_name, overall_score, details?}
        """
        job = self.embedding_service.get_job_data(job_id)
        if job is None:
            return [], f"Job '{job_id}' not found"

        results = []
        for resume_id in list(self.embedding_service.resume_embeddings.keys()):
            match_data, err = self.match_resume_to_job(resume_id, job_id)
            if err or not match_data:
                continue
            overall = float(match_data.get("scores", {}).get("overall", 0.0))
            candidate_name = match_data.get("candidate_info", {}).get("name", "Unknown")
            results.append(
                {
                    "resume_id": resume_id,
                    "candidate_name": candidate_name,
                    "overall_score": round(overall, 3),
                    "match_details": match_data,  # optional, useful if you want full detail
                }
            )

        results.sort(key=lambda x: x["overall_score"], reverse=True)
        return results[: max(1, top_n)], None

    def find_top_matches_for_resume(self, resume_id: str, top_n: int = 5) -> Tuple[List[dict], Optional[str]]:
        """
        For a resume, rank all jobs by overall score.
        Returns a list of {job_id, overall_score, details?}
        """
        resume = self.embedding_service.get_resume_data(resume_id)
        if resume is None:
            return [], f"Resume '{resume_id}' not found"

        results = []
        for job_id in list(self.embedding_service.job_embeddings.keys()):
            match_data, err = self.match_resume_to_job(resume_id, job_id)
            if err or not match_data:
                continue
            overall = float(match_data.get("scores", {}).get("overall", 0.0))
            results.append(
                {
                    "job_id": job_id,
                    "overall_score": round(overall, 3),
                    "match_details": match_data,
                }
            )

        results.sort(key=lambda x: x["overall_score"], reverse=True)
        return results[: max(1, top_n)], None
