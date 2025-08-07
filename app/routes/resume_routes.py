import os
import uuid
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from ..services.parsing_service import ParsingService
from .. import embedding_service_instance as embedding_service
from .. import llm_service_instance as llm_service
from ..services.matching_service import MatchingService

from flask_login import login_required

resume_bp = Blueprint('resume_bp', __name__)

parsing_service = ParsingService()
matching_service = MatchingService(embedding_service)

@resume_bp.route('/upload', methods=['POST'])
@login_required
def upload_resumes_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        files = request.files.getlist('file')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        results = []
        for file in files:
            # Process each file
            unique_id = str(uuid.uuid4())
            ext = os.path.splitext(file.filename)[1].lower()
            save_path = os.path.join(current_app.config['RESUME_STORAGE_PATH'], f"{unique_id}{ext}")
            
            file.save(save_path)
            
            # Use services from current_app
            text_content, entities, token_usage = current_app.parsing_service.parse_resume_pdf(save_path)
            embeddings_dict = current_app.embedding_service.store_resume_embedding(unique_id, text_content, entities)
            
            results.append({
                'filename': file.filename,
                'resume_id': unique_id,
                'extracted_entities': entities,
                'embedding_shapes': {
                    'skills': list(embeddings_dict['skills'].shape) if 'skills' in embeddings_dict else [],
                    'experience': list(embeddings_dict['experience'].shape) if 'experience' in embeddings_dict else [],
                    'education': list(embeddings_dict['education'].shape) if 'education' in embeddings_dict else []
                },
                'llm_token_usage': token_usage
            })
        
        return jsonify({'results': results}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@resume_bp.route('/<resume_id>', methods=['GET'])
def get_resume_data(resume_id: str):
    resume_data = embedding_service.get_resume_data(resume_id)
    if resume_data:
        # Check if any embedding exists
        has_embedding = any(
            emb.size > 0 
            for emb in resume_data.get('embeddings', {}).values()
        )
        
        return jsonify({
            'resume_id': resume_id,
            'entities': resume_data.get('refined_entities', {}),
            'embedding_exists': has_embedding
        }), 200
    return jsonify({'error': f'Resume with ID "{resume_id}" not found.'}), 404

@resume_bp.route('/list', methods=['GET'])
def list_resumes():
    resumes_list = []
    for resume_id, data in embedding_service.resume_embeddings.items():
        # Check if any embedding exists
        has_embedding = any(
            emb.size > 0 
            for emb in data.get('embeddings', {}).values()
        )
        
        resumes_list.append({
            'resume_id': resume_id,
            'has_embedding': has_embedding,
            'num_skills': len(data.get('refined_entities', {}).get('skills', [])),
            'name': data.get('refined_entities', {}).get('name', 'N/A')
        })
    return jsonify(resumes_list), 200

@resume_bp.route('/match-jobs/<resume_id>', methods=['GET'])
def match_jobs_for_resume(resume_id: str):
    top_n = request.args.get('top_n', 5, type=int)
    matches, error = matching_service.find_top_matches_for_resume(resume_id, top_n)
    if error:
        return jsonify({'error': error}), 404

    return jsonify({
        'resume_id': resume_id,
        'top_job_matches': matches
    }), 200

@resume_bp.route('/<resume_id>/generate-interview-questions/<job_id>')
def generate_interview_questions(resume_id, job_id):
    # Get resume and job data
    resume_data = current_app.embedding_service.get_resume_data(resume_id)
    job_data = current_app.embedding_service.get_job_data(job_id)
    
    if not resume_data or not job_data:
        return jsonify({"error": "Resume or job not found"}), 404
    
    # Prepare context with proper string conversion
    def safe_stringify(data):
        if isinstance(data, list):
            return ', '.join(str(item) for item in data)
        return str(data)
    
    context = {
        'candidate_skills': resume_data.get('refined_entities', {}).get('skills', []),
        'candidate_experience': safe_stringify(resume_data.get('refined_entities', {}).get('experience', [])),
        'job_skills': job_data.get('refined_entities', {}).get('skills', []),
        'job_experience': safe_stringify(job_data.get('refined_entities', {}).get('experience', [])),
        'job_title': job_data.get('refined_entities', {}).get('job_title', 'the role')
    }
    
    # Generate questions
    questions, usage = current_app.llm_service.generate_interview_questions(context)
    
    if 'error' in questions:
        return jsonify({"error": questions['error']}), 500
    
    return jsonify({
        "interview_questions": {
            "technical_questions": questions.get("technical_questions", []),
            "behavioral_questions": questions.get("behavioral_questions", [])
        },
        "token_usage": usage
    })