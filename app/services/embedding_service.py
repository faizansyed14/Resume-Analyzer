# app/services/embedding_service.py

import os
import pickle
import uuid
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    """
    Service for generating, storing, and loading text embeddings.
    Uses a higher-capacity MPNet model by default for improved semantic understanding.
    Generates separate embeddings for skills, experience, and education.
    """
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        print(f"Sentence Transformer model '{model_name}' loaded successfully.")
        self.resume_embeddings = {}
        self.job_embeddings = {}
        self.embeddings_storage_path = None
        print(f"EmbeddingService initialized with model: {model_name}")

    def set_storage_path(self, path):
        self.embeddings_storage_path = path
        os.makedirs(self.embeddings_storage_path, exist_ok=True)

    def load_embeddings(self):
        """Loads embeddings from pickle files into memory."""
        if not self.embeddings_storage_path:
            print("Embeddings storage path not set. Cannot load embeddings.")
            return

        resume_path = os.path.join(self.embeddings_storage_path, 'resume_embeddings.pkl')
        job_path = os.path.join(self.embeddings_storage_path, 'job_embeddings.pkl')

        if os.path.exists(resume_path):
            try:
                with open(resume_path, 'rb') as f:
                    self.resume_embeddings = pickle.load(f)
                print(f"Loaded {len(self.resume_embeddings)} resume embeddings from {resume_path}")
            except Exception as e:
                print(f"Error loading resume embeddings from {resume_path}: {e}")
                self.resume_embeddings = {}
        else:
            print(f"No resume embeddings file found at {resume_path}")

        if os.path.exists(job_path):
            try:
                with open(job_path, 'rb') as f:
                    self.job_embeddings = pickle.load(f)
                print(f"Loaded {len(self.job_embeddings)} job embeddings from {job_path}")
            except Exception as e:
                print(f"Error loading job embeddings from {job_path}: {e}")
                self.job_embeddings = {}
        else:
            print(f"No job embeddings file found at {job_path}")

    def save_embeddings(self):
        """Saves current in-memory embeddings to pickle files."""
        if not self.embeddings_storage_path:
            print("Embeddings storage path not set. Cannot save embeddings.")
            return

        resume_path = os.path.join(self.embeddings_storage_path, 'resume_embeddings.pkl')
        job_path = os.path.join(self.embeddings_storage_path, 'job_embeddings.pkl')

        try:
            with open(resume_path, 'wb') as f:
                pickle.dump(self.resume_embeddings, f)
            print(f"Saved {len(self.resume_embeddings)} resume embeddings to {resume_path}")
        except Exception as e:
            print(f"Error saving resume embeddings to {resume_path}: {e}")

        try:
            with open(job_path, 'wb') as f:
                pickle.dump(self.job_embeddings, f)
            print(f"Saved {len(self.job_embeddings)} job embeddings to {job_path}")
        except Exception as e:
            print(f"Error saving job embeddings to {job_path}: {e}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generates a vector embedding for a given text."""
        if not text:
            return np.array([])
        return self.model.encode(text, convert_to_tensor=False)

    def store_resume_embedding(self, resume_id: str, original_text: str, refined_entities: dict) -> dict:
        """
        Stores resume data including separate embeddings for skills, experience, and education.
        Returns dictionary of embeddings.
        """
        embeddings = {}
        
        # Skills embedding
        skills_text = "Skills: " + ", ".join(
            s['name'] if isinstance(s, dict) and 'name' in s else str(s)
            for s in refined_entities.get('skills', [])
        )
        embeddings['skills'] = self.generate_embedding(skills_text)
        
        # Experience embedding
        experience_text = "Experience: " + ". ".join(
            e['title'] if isinstance(e, dict) and 'title' in e else
            e['description'] if isinstance(e, dict) and 'description' in e else str(e)
            for e in refined_entities.get('experience', [])
        )
        embeddings['experience'] = self.generate_embedding(experience_text)
        
        # Education embedding (safe stringify)
        education_entries = refined_entities.get('education', [])
        edu_strs = []
        for edu in education_entries:
            if isinstance(edu, str):
                edu_strs.append(edu)
            elif isinstance(edu, dict):
                if 'name' in edu and isinstance(edu['name'], str):
                    edu_strs.append(edu['name'])
                elif 'degree' in edu and isinstance(edu['degree'], str):
                    edu_strs.append(edu['degree'])
                else:
                    edu_strs.append(" ".join(str(v) for v in edu.values()))
            else:
                edu_strs.append(str(edu))
        education_text = "Education: " + ", ".join(edu_strs)
        embeddings['education'] = self.generate_embedding(education_text)

        # Store all embeddings
        self.resume_embeddings[resume_id] = {
            'embeddings': embeddings,
            'refined_entities': refined_entities,
            'original_text': original_text
        }
        self.save_embeddings()
        return embeddings

    def store_job_embedding(self, job_id: str, original_text: str, refined_entities: dict) -> dict:
        """
        Stores job description data including separate embeddings for skills, experience, and education.
        Returns dictionary of embeddings.
        """
        embeddings = {}
        
        # Skills embedding
        skills_text = "Required Skills: " + ", ".join(
            s['name'] if isinstance(s, dict) and 'name' in s else str(s)
            for s in refined_entities.get('skills', [])
        )
        embeddings['skills'] = self.generate_embedding(skills_text)
        
        # Experience embedding
        experience_text = "Required Experience: " + ". ".join(
            e['title'] if isinstance(e, dict) and 'title' in e else
            e['description'] if isinstance(e, dict) and 'description' in e else str(e)
            for e in refined_entities.get('experience', [])
        )
        embeddings['experience'] = self.generate_embedding(experience_text)
        
        # Education embedding (safe stringify)
        education_entries = refined_entities.get('education', [])
        edu_strs = []
        for edu in education_entries:
            if isinstance(edu, str):
                edu_strs.append(edu)
            elif isinstance(edu, dict):
                if 'name' in edu and isinstance(edu['name'], str):
                    edu_strs.append(edu['name'])
                elif 'degree' in edu and isinstance(edu['degree'], str):
                    edu_strs.append(edu['degree'])
                else:
                    edu_strs.append(" ".join(str(v) for v in edu.values()))
            else:
                edu_strs.append(str(edu))
        education_text = "Required Education: " + ", ".join(edu_strs)
        embeddings['education'] = self.generate_embedding(education_text)

        # Store all embeddings
        self.job_embeddings[job_id] = {
            'embeddings': embeddings,
            'refined_entities': refined_entities,
            'original_text': original_text
        }
        self.save_embeddings()
        return embeddings

    def get_resume_data(self, resume_id: str) -> dict | None:
        """Returns the full data for a resume."""
        return self.resume_embeddings.get(resume_id)

    def get_job_data(self, job_id: str) -> dict | None:
        """Returns the full data for a job description."""
        return self.job_embeddings.get(job_id)

    def store_job_text_input(self, text_content: str, refined_entities: dict) -> tuple[str, dict]:
        """
        Handles job description input, generates a unique ID,
        and stores the embeddings along with refined entities.
        """
        job_id = str(uuid.uuid4())
        embeddings = self.store_job_embedding(job_id, text_content, refined_entities)
        print(f"Stored new job description with ID: {job_id}")
        return job_id, embeddings
