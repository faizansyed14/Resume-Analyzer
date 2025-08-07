# app/services/llm_service.py
import os
import json
from openai import OpenAI
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class LLMService:
    def __init__(self, api_key: str = None, model_name: str = 'gpt-4o-mini'):
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError('OPENAI_API_KEY environment variable not set or invalid.')
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.skill_model = SentenceTransformer('all-mpnet-base-v2')
        print(f'OpenAI LLMService initialized with model: {self.model_name}')

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> tuple[str, dict | None]:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
            {'role': 'system', 'content': 'Respond only with a valid JSON object.'}
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                response_format={'type': 'json_object'}
            )
            content = response.choices[0].message.content
            usage = response.usage.model_dump() if response.usage else None
            return content, usage
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return json.dumps({"error": f"LLM call failed: {e}"}), None

    def calculate_match_score(self, resume_data: Dict, job_data: Dict) -> Tuple[Dict, Optional[Dict]]:
        system_prompt = """You are an expert recruiter evaluating how well a candidate matches a job description.
        Analyze the following and return a JSON response with:
        1. scores:
        - skills_match (weight: 80%)
        - experience_match (weight: 15%)
        - education_match (weight: 5%)
        - overall_score (weighted average using the above weights)
        2. matched_skills: list of skills that actually match
        3. match_analysis: brief explanation

        Format:
        {
            "scores": {
                "skills": 0.85,
                "experience": 0.75,
                "education": 0.90,
                "overall": 0.82,
                "domain_alignment": 0.88
            },
            "matched_skills": ["Python", "AWS"],
            "skill_match_details": [
                {
                    "job_skill": "Python",
                    "confidence": 0.95,
                    "match_type": "exact"
                }
            ],
            "match_analysis": "The candidate has strong core skills match but lacks some domain experience"
        }"""

        
        user_prompt = f"""Candidate Resume:
        Skills: {', '.join(resume_data.get('skills', []))}
        Experience: {str(resume_data.get('experience', []))}
        Education: {str(resume_data.get('education', []))}
        
        Job Description:
        Required Skills: {', '.join(job_data.get('skills', []))}
        Required Experience: {str(job_data.get('experience', []))}
        Required Education: {str(job_data.get('education', []))}
        
        Provide detailed matching analysis as specified in the system prompt."""
        
        llm_response_str, token_usage = self._call_llm(system_prompt, user_prompt)
        try:
            data = json.loads(llm_response_str)
            
            # Ensure all required fields are present
            if not all(k in data.get('scores', {}) for k in ['skills', 'experience', 'education', 'overall']):
                raise ValueError("Missing required score fields")
                
            return {
                "scores": {
                    "skills": data["scores"].get("skills", 0),
                    "experience": data["scores"].get("experience", 0),
                    "education": data["scores"].get("education", 0),
                    "overall": data["scores"].get("overall", 0),
                    "domain_alignment": data["scores"].get("domain_alignment", 0)
                },
                "matched_skills": data.get("matched_skills", []),
                "skill_match_details": data.get("skill_match_details", []),
                "match_analysis": data.get("match_analysis", "No analysis provided"),
                "token_usage": token_usage
            }, token_usage
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing LLM response: {str(e)}")
            return {
                "scores": {
                    "skills": 0,
                    "experience": 0,
                    "education": 0,
                    "overall": 0,
                    "domain_alignment": 0
                },
                "matched_skills": [],
                "skill_match_details": [],
                "match_analysis": f"Scoring failed: {str(e)}",
                "token_usage": token_usage
            }, token_usage
    def generate_interview_questions(self, context: dict) -> Tuple[Dict, Dict]:
        """Generate technical and behavioral interview questions"""
        system_prompt = """You are an experienced technical interviewer. Generate specific interview questions based on:
        - Candidate skills: {candidate_skills}
        - Job requirements: {job_skills}
        Return JSON with:
        {
            "technical_questions": ["question1", "question2"],
            "behavioral_questions": ["question1", "question2"]
        }"""
        
        user_prompt = f"""Candidate Profile:
        Skills: {', '.join(context.get('candidate_skills', []))}
        Experience: {', '.join(context.get('candidate_experience', [])) if isinstance(context.get('candidate_experience', []), list) else str(context.get('candidate_experience', []))}
        
        Job Requirements:
        Skills: {', '.join(context.get('job_skills', []))}
        Experience: {', '.join(context.get('job_experience', [])) if isinstance(context.get('job_experience', []), list) else str(context.get('job_experience', []))}
        
        Generate 5 technical and 3 behavioral questions as a JSON object with arrays."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Parse and validate response
            questions = json.loads(response.choices[0].message.content)
            
            # Ensure questions are strings
            if 'technical_questions' in questions:
                questions['technical_questions'] = [str(q) for q in questions['technical_questions']]
            if 'behavioral_questions' in questions:
                questions['behavioral_questions'] = [str(q) for q in questions['behavioral_questions']]
            
            return questions, response.usage.model_dump()
            
        except Exception as e:
            return {"error": str(e)}, None