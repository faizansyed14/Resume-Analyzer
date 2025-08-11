# app/services/embedding_service.py
import os
import pickle
import uuid
from typing import Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .skill_library import SkillLibrary


class EmbeddingService:
    """
    Generates, stores and loads text embeddings.
    - Individual embeddings per matched skill
    - Combined embeddings for experience and education
    Pickled to disk for persistence.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.skill_library = SkillLibrary()
        print(f"Sentence Transformer model '{model_name}' loaded successfully.")
        self.resume_embeddings: Dict[str, dict] = {}
        self.job_embeddings: Dict[str, dict] = {}
        self.embeddings_storage_path: Optional[str] = None
        print(f"EmbeddingService initialized with model: {model_name}")

    # ---- Storage ------------------------------------------------------------

    def set_storage_path(self, path: str) -> None:
        self.embeddings_storage_path = path
        os.makedirs(self.embeddings_storage_path, exist_ok=True)

    def _resume_pickle_path(self) -> str:
        return os.path.join(self.embeddings_storage_path, "resume_embeddings.pkl")

    def _job_pickle_path(self) -> str:
        return os.path.join(self.embeddings_storage_path, "job_embeddings.pkl")

    def load_embeddings(self) -> None:
        """Load stored embeddings to memory."""
        if not self.embeddings_storage_path:
            print("Embeddings storage path not set. Cannot load embeddings.")
            return

        rpath = self._resume_pickle_path()
        jpath = self._job_pickle_path()

        if os.path.exists(rpath):
            try:
                with open(rpath, "rb") as f:
                    self.resume_embeddings = pickle.load(f)
                print(f"Loaded {len(self.resume_embeddings)} resume embeddings from {rpath}")
            except Exception as e:
                print(f"Error loading resume embeddings from {rpath}: {e}")
                self.resume_embeddings = {}
        else:
            print(f"No resume embeddings file found at {rpath}")

        if os.path.exists(jpath):
            try:
                with open(jpath, "rb") as f:
                    self.job_embeddings = pickle.load(f)
                print(f"Loaded {len(self.job_embeddings)} job embeddings from {jpath}")
            except Exception as e:
                print(f"Error loading job embeddings from {jpath}: {e}")
                self.job_embeddings = {}
        else:
            print(f"No job embeddings file found at {jpath}")

    def save_embeddings(self) -> None:
        """Persist in-memory embeddings to disk."""
        if not self.embeddings_storage_path:
            print("Embeddings storage path not set. Cannot save embeddings.")
            return

        rpath = self._resume_pickle_path()
        jpath = self._job_pickle_path()

        try:
            with open(rpath, "wb") as f:
                pickle.dump(self.resume_embeddings, f)
            print(f"Saved {len(self.resume_embeddings)} resume embeddings to {rpath}")
        except Exception as e:
            print(f"Error saving resume embeddings to {rpath}: {e}")

        try:
            with open(jpath, "wb") as f:
                pickle.dump(self.job_embeddings, f)
            print(f"Saved {len(self.job_embeddings)} job embeddings to {jpath}")
        except Exception as e:
            print(f"Error saving job embeddings to {jpath}: {e}")

    # ---- Embedding helpers --------------------------------------------------

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text:
            return np.array([])
        return self.model.encode(text, convert_to_tensor=False)

    def _extract_and_match_skills(self, extracted_skills: list) -> list:
        """Map extracted skill-like strings to canonical skill names."""
        if not extracted_skills:
            return []

        skill_strings: list[str] = []
        for skill in extracted_skills:
            if isinstance(skill, dict):
                skill_name = skill.get("name", skill.get("skill", ""))
            else:
                skill_name = str(skill)
            if skill_name and skill_name.strip():
                skill_strings.append(skill_name.strip())

        matched = self.skill_library.find_matching_skills(skill_strings)
        print(f"Matched {len(matched)} skills from {len(skill_strings)} extracted skills")
        return matched

    def _create_individual_skill_embeddings(self, matched_skills: list) -> dict:
        """Create individual embeddings per canonical skill."""
        out: dict[str, np.ndarray] = {}
        for skill in matched_skills:
            text = f"Technical skill: {skill}"
            out[skill] = self.generate_embedding(text)
        return out

    def _create_experience_embedding(self, experience_entries: list) -> np.ndarray:
        if not experience_entries:
            return np.array([])

        texts = []
        for exp in experience_entries:
            if isinstance(exp, dict):
                parts = []
                if exp.get("title"):
                    parts.append(f"Title: {exp['title']}")
                if exp.get("company"):
                    parts.append(f"Company: {exp['company']}")
                if exp.get("description"):
                    parts.append(f"Description: {exp['description']}")
                if exp.get("duration") or exp.get("start_date") or exp.get("end_date"):
                    when = exp.get("duration") or f"{exp.get('start_date','?')} - {exp.get('end_date','?')}"
                    parts.append(f"Duration: {when}")
                if parts:
                    texts.append(". ".join(parts))
            else:
                texts.append(str(exp))

        if not texts:
            return np.array([])

        combined = "Professional Experience: " + " | ".join(texts)
        return self.generate_embedding(combined)

    def _create_education_embedding(self, education_entries: list) -> np.ndarray:
        if not education_entries:
            return np.array([])

        texts = []
        for edu in education_entries:
            if isinstance(edu, dict):
                parts = []
                if edu.get("degree"):
                    parts.append(f"Degree: {edu['degree']}")
                if edu.get("field"):
                    parts.append(f"Field: {edu['field']}")
                if edu.get("institution"):
                    parts.append(f"Institution: {edu['institution']}")
                if edu.get("dates"):
                    parts.append(f"Year: {edu['dates']}")
                if parts:
                    texts.append(". ".join(parts))
            else:
                texts.append(str(edu))

        if not texts:
            return np.array([])

        combined = "Educational Background: " + " | ".join(texts)
        return self.generate_embedding(combined)

    # ---- Public API: store & fetch -----------------------------------------

    def store_resume_embedding(
        self, resume_id: str, original_text: str, refined_entities: dict
    ) -> dict:
        """Create embeddings & persist structured resume info."""
        print(f"Processing resume {resume_id} for embeddings...")
        extracted_skills = refined_entities.get("skills", [])
        matched_skills = self._extract_and_match_skills(extracted_skills)

        skill_embeddings = self._create_individual_skill_embeddings(matched_skills)
        exp_emb = self._create_experience_embedding(
            refined_entities.get("experience", [])
        )
        edu_emb = self._create_education_embedding(
            refined_entities.get("education", [])
        )

        embeddings = {
            "individual_skills": skill_embeddings,
            "experience": exp_emb,
            "education": edu_emb,
            "matched_skills_list": matched_skills,
        }

        refined_entities_updated = dict(refined_entities)
        refined_entities_updated["matched_skills"] = matched_skills
        refined_entities_updated["skill_categories"] = self.skill_library.categorize_skills(
            matched_skills
        )

        self.resume_embeddings[resume_id] = {
            "embeddings": embeddings,
            "refined_entities": refined_entities_updated,
            "original_text": original_text,
        }

        self.save_embeddings()
        print(
            f"Stored resume embeddings: {len(matched_skills)} skills, "
            f"experience: {len(exp_emb) if isinstance(exp_emb, np.ndarray) else 0}, "
            f"education: {len(edu_emb) if isinstance(edu_emb, np.ndarray) else 0}"
        )
        return embeddings

    def store_job_embedding(
        self, job_id: str, original_text: str, refined_entities: dict
    ) -> dict:
        """Create embeddings & persist structured JD info."""
        print(f"Processing job {job_id} for embeddings...")
        extracted_skills = refined_entities.get("skills", [])
        matched_skills = self._extract_and_match_skills(extracted_skills)

        skill_embeddings = self._create_individual_skill_embeddings(matched_skills)
        exp_emb = self._create_experience_embedding(
            refined_entities.get("experience", [])
        )
        edu_emb = self._create_education_embedding(
            refined_entities.get("education", [])
        )

        embeddings = {
            "individual_skills": skill_embeddings,
            "experience": exp_emb,
            "education": edu_emb,
            "matched_skills_list": matched_skills,
        }

        refined_entities_updated = dict(refined_entities)
        refined_entities_updated["matched_skills"] = matched_skills
        refined_entities_updated["skill_categories"] = self.skill_library.categorize_skills(
            matched_skills
        )

        self.job_embeddings[job_id] = {
            "embeddings": embeddings,
            "refined_entities": refined_entities_updated,
            "original_text": original_text,
        }

        self.save_embeddings()
        print(
            f"Stored job embeddings: {len(matched_skills)} required skills, "
            f"experience: {len(exp_emb) if isinstance(exp_emb, np.ndarray) else 0}, "
            f"education: {len(edu_emb) if isinstance(edu_emb, np.ndarray) else 0}"
        )
        return embeddings

    def get_resume_data(self, resume_id: str) -> dict | None:
        return self.resume_embeddings.get(resume_id)

    def get_job_data(self, job_id: str) -> dict | None:
        return self.job_embeddings.get(job_id)

    def store_job_text_input(self, text_content: str, refined_entities: dict) -> tuple[str, dict]:
        job_id = str(uuid.uuid4())
        embeddings = self.store_job_embedding(job_id, text_content, refined_entities)
        print(f"Stored new job description with ID: {job_id}")
        return job_id, embeddings

    # ---- Skill similarity matrix -------------------------------------------

    def get_skill_similarity_matrix(self, resume_id: str, job_id: str) -> dict:
        """
        Calculate similarity matrix between resume skills and job skills.
        Returns detailed skill matching information.
        """
        rdata = self.get_resume_data(resume_id)
        jdata = self.get_job_data(job_id)
        if not rdata or not jdata:
            return {}

        rskills: dict = rdata["embeddings"]["individual_skills"]
        jskills: dict = jdata["embeddings"]["individual_skills"]
        if not rskills or not jskills:
            return {}

        similarity_matrix = {}
        matched_skills = []

        for job_skill, job_emb in jskills.items():
            best_match = None
            best_sim = 0.0

            for resume_skill, resume_emb in rskills.items():
                if len(job_emb) > 0 and len(resume_emb) > 0:
                    sim = float(
                        np.dot(job_emb, resume_emb)
                        / (np.linalg.norm(job_emb) * np.linalg.norm(resume_emb))
                    )
                    if sim > best_sim:
                        best_sim = sim
                        best_match = resume_skill

            similarity_matrix[job_skill] = {
                "best_match": best_match,
                "similarity_score": float(best_sim),
                "is_exact_match": (job_skill.lower() == best_match.lower()) if best_match else False,
            }

            if best_sim > 0.7 or (
                best_match and job_skill.lower() == best_match.lower()
            ):
                matched_skills.append(
                    {
                        "job_skill": job_skill,
                        "resume_skill": best_match,
                        "similarity": float(best_sim),
                        "match_type": "exact"
                        if best_match
                        and job_skill.lower() == best_match.lower()
                        else "semantic",
                    }
                )

        return {
            "similarity_matrix": similarity_matrix,
            "matched_skills": matched_skills,
            "total_job_skills": len(jskills),
            "total_resume_skills": len(rskills),
            "matched_count": len(matched_skills),
            "match_percentage": (len(matched_skills) / len(jskills)) * 100 if jskills else 0.0,
        }
