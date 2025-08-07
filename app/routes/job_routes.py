import os
import uuid
from flask import Blueprint, request, jsonify, current_app, flash
from werkzeug.utils import secure_filename
from flask_login import login_required
from ..services.parsing_service import ParsingService
from .. import embedding_service_instance as embedding_service
from ..services.matching_service import MatchingService
from flask_login import login_required

job_bp = Blueprint('job_bp', __name__)

parsing_service = ParsingService()
matching_service = MatchingService(embedding_service)

@job_bp.route('/upload', methods=['POST'])
@login_required
def upload_job_description():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Process file
        unique_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower()
        save_path = os.path.join(current_app.config['JOB_STORAGE_PATH'], f"{unique_id}{ext}")
        
        file.save(save_path)
        
        # Use services from current_app
        text_content, entities, token_usage = current_app.parsing_service.parse_job_pdf(save_path)
        embeddings_dict = current_app.embedding_service.store_job_embedding(unique_id, text_content, entities)
        
        return jsonify({
            'message': 'Job description uploaded and processed successfully',
            'job_id': unique_id,
            'extracted_entities': entities,
            'llm_token_usage': token_usage,
            'embedding_shapes': {
                'skills': list(embeddings_dict['skills'].shape) if 'skills' in embeddings_dict else [],
                'experience': list(embeddings_dict['experience'].shape) if 'experience' in embeddings_dict else [],
                'education': list(embeddings_dict['education'].shape) if 'education' in embeddings_dict else []
            }
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Error processing job description: {e}")
        return jsonify({'error': str(e)}), 500

@job_bp.route('/upload_text', methods=['POST'])
# @login_required
def upload_job_text():
    job_text = request.json.get('job_description_text')
    if not job_text:
        return jsonify({"error": "No job description text provided"}), 400

    try:
        job_text_content, refined_job_entities, llm_token_usage = parsing_service.parse_job_text(job_text)
        job_id, embeddings_dict = embedding_service.store_job_text_input(
            text_content=job_text_content,
            refined_entities=refined_job_entities
        )

        return jsonify({
            "message": "Job Description text processed successfully",
            "job_id": job_id,
            "extracted_entities": refined_job_entities,
            "llm_token_usage": llm_token_usage,
            "embedding_shapes": {
                'skills': list(embeddings_dict['skills'].shape) if 'skills' in embeddings_dict else [],
                'experience': list(embeddings_dict['experience'].shape) if 'experience' in embeddings_dict else [],
                'education': list(embeddings_dict['education'].shape) if 'education' in embeddings_dict else []
            }
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error processing job description text: {e}")
        return jsonify({"error": str(e)}), 500

@job_bp.route('/<job_id>', methods=['GET'])
# @login_required
def get_job_data(job_id: str):
    job_data = embedding_service.get_job_data(job_id)
    if job_data:
        # Check if any embedding exists in the dictionary
        has_embedding = any(
            emb.size > 0 
            for emb in job_data.get('embeddings', {}).values()
        )
        
        return jsonify({
            'job_id': job_id,
            'entities': job_data.get('refined_entities', {}),
            'embedding_exists': has_embedding
        }), 200
    return jsonify({'error': f'Job Description with ID "{job_id}" not found.'}), 404

@job_bp.route('/list', methods=['GET'])
# @login_required
def list_jobs():
    jobs_list = []
    if hasattr(embedding_service, 'job_embeddings') and isinstance(embedding_service.job_embeddings, dict):
        for job_id, data in embedding_service.job_embeddings.items():
            # Check if any embedding exists
            has_embedding = any(
                emb.size > 0 
                for emb in data.get('embeddings', {}).values()
            )
            
            refined_entities = data.get('refined_entities', {})
            skills_required = refined_entities.get('skills', [])
            jobs_list.append({
                'job_id': job_id,
                'has_embedding': has_embedding,
                'num_skills_required': len(skills_required)
            })
    return jsonify(jobs_list), 200

@job_bp.route('/match-resumes/<job_id>', methods=['GET'])
@login_required
def match_resumes_for_job(job_id):
    try:
        top_n = request.args.get('top_n', default=5, type=int)
        matches, error = current_app.matching_service.find_top_matches_for_job(job_id, top_n)
        
        if error:
            current_app.logger.error(f"Error finding top matches: {error}")
            return jsonify({'error': error}), 404 if "not found" in error.lower() else 500
        
        return jsonify({
            'job_id': job_id,
            'top_resume_matches': matches
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Unexpected error in match_resumes_for_job: {str(e)}")
        return jsonify({'error': str(e)}), 500
@job_bp.route('/match-specific/<job_id>/<resume_id>')
def get_specific_match(job_id, resume_id):
    try:
        # Get data from embedding service
        resume = current_app.embedding_service.get_resume_data(resume_id)
        job = current_app.embedding_service.get_job_data(job_id)
        
        if not resume or not job:
            return jsonify({"error": "Resume or job not found"}), 404

        # Prepare data for LLM
        resume_data = {
            "skills": resume.get('refined_entities', {}).get('skills', []),
            "experience": resume.get('refined_entities', {}).get('experience', []),
            "education": resume.get('refined_entities', {}).get('education', [])
        }
        
        job_data = {
            "skills": job.get('refined_entities', {}).get('skills', []),
            "experience": job.get('refined_entities', {}).get('experience', []),
            "education": job.get('refined_entities', {}).get('education', [])
        }
        
        # Get match from LLM
        match_result, token_usage = current_app.llm_service.calculate_match_score(
            resume_data, 
            job_data
        )
        
        return jsonify({
            "match_details": {
                **match_result,
                "candidate_info": {
                    "name": resume.get('refined_entities', {}).get('name'),
                    "email": resume.get('refined_entities', {}).get('email'),
                    "skills": resume_data["skills"],
                    "experience": resume_data["experience"],
                    "education": resume_data["education"]
                },
                "job_info": {
                    "job_title": job.get('refined_entities', {}).get('job_title'),
                    "company": job.get('refined_entities', {}).get('company'),
                    "skills": job_data["skills"],
                    "experience": job_data["experience"],
                    "education": job_data["education"]
                },
                "metadata": {
                    "resume_id": resume_id,
                    "job_id": job_id,
                    "llm_token_usage": token_usage
                }
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Error in get_specific_match: {str(e)}")
        return jsonify({"error": str(e)}), 500