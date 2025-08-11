# app/routes/job_routes.py
import os
import uuid
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from flask_login import login_required

from ..services.jd_parser import JDParser

job_bp = Blueprint("job_bp", __name__)
jd_parser = JDParser()


@job_bp.route("/upload", methods=["POST"])
@login_required
def upload_job_description():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        unique_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower()
        os.makedirs(current_app.config["JOB_STORAGE_PATH"], exist_ok=True)
        save_path = os.path.join(current_app.config["JOB_STORAGE_PATH"], f"{unique_id}{ext}")
        file.save(save_path)

        text_content, entities, token_usage = current_app.jd_parser.parse_job_pdf(save_path)

        embeddings = current_app.embedding_service.store_job_embedding(unique_id, text_content, entities)
        job_data = current_app.embedding_service.get_job_data(unique_id)
        refined_entities = job_data.get("refined_entities", {}) if job_data else entities

        return jsonify(
            {
                "message": "Job description uploaded and processed successfully",
                "job_id": unique_id,
                "extracted_entities": refined_entities,
                "llm_token_usage": token_usage,
                "embedding_shapes": {
                    "individual_skills_count": len(embeddings.get("individual_skills", {})),
                    "experience": list(embeddings["experience"].shape)
                    if isinstance(embeddings.get("experience"), np.ndarray) and embeddings["experience"].size > 0
                    else [],
                    "education": list(embeddings["education"].shape)
                    if isinstance(embeddings.get("education"), np.ndarray) and embeddings["education"].size > 0
                    else [],
                },
            }
        ), 200

    except Exception as e:
        current_app.logger.error(f"Error processing job description: {e}")
        return jsonify({"error": str(e)}), 500


@job_bp.route("/upload_text", methods=["POST"])
@login_required
def upload_job_text():
    data = request.get_json(silent=True) or {}
    job_text = data.get("job_description_text")
    if not job_text:
        return jsonify({"error": "No job description text provided"}), 400

    try:
        job_text_content, refined_job_entities, llm_token_usage = current_app.jd_parser.parse_job_text(job_text)
        job_id, embeddings = current_app.embedding_service.store_job_text_input(
            text_content=job_text_content, refined_entities=refined_job_entities
        )
        job_data = current_app.embedding_service.get_job_data(job_id)
        refined_entities = job_data.get("refined_entities", {}) if job_data else refined_job_entities

        return jsonify(
            {
                "message": "Job Description text processed successfully",
                "job_id": job_id,
                "extracted_entities": refined_entities,
                "llm_token_usage": llm_token_usage,
                "embedding_shapes": {
                    "individual_skills_count": len(embeddings.get("individual_skills", {})),
                    "experience": list(embeddings["experience"].shape)
                    if isinstance(embeddings.get("experience"), np.ndarray) and embeddings["experience"].size > 0
                    else [],
                    "education": list(embeddings["education"].shape)
                    if isinstance(embeddings.get("education"), np.ndarray) and embeddings["education"].size > 0
                    else [],
                },
            }
        ), 200
    except Exception as e:
        current_app.logger.error(f"Error processing job description text: {e}")
        return jsonify({"error": str(e)}), 500


@job_bp.route("/<job_id>", methods=["GET"])
@login_required
def get_job_data(job_id: str):
    job_data = current_app.embedding_service.get_job_data(job_id)
    if job_data:
        has_embedding = any(
            isinstance(emb, np.ndarray) and emb.size > 0
            for emb in job_data.get("embeddings", {}).values()
        )
        return jsonify(
            {
                "job_id": job_id,
                "entities": job_data.get("refined_entities", {}),
                "embedding_exists": has_embedding,
            }
        ), 200
    return jsonify({"error": f'Job Description with ID "{job_id}" not found.'}), 404


@job_bp.route("/list", methods=["GET"])
@login_required
def list_jobs():
    jobs_list = []
    store = current_app.embedding_service.job_embeddings
    if isinstance(store, dict):
        for job_id, data in store.items():
            has_embedding = any(
                isinstance(emb, np.ndarray) and emb.size > 0
                for emb in data.get("embeddings", {}).values()
            )
            ents = data.get("refined_entities", {})
            skills_required = ents.get("matched_skills", [])
            jobs_list.append(
                {
                    "job_id": job_id,
                    "has_embedding": has_embedding,
                    "num_skills_required": len(skills_required),
                }
            )
    return jsonify(jobs_list), 200


@job_bp.route("/match-resumes/<job_id>", methods=["GET"])
@login_required
def match_resumes_for_job(job_id: str):
    try:
        top_n = request.args.get("top_n", default=5, type=int)
        matches, error = current_app.matching_service.find_top_matches_for_job(job_id, top_n)
        if error:
            current_app.logger.error(f"Error finding top matches: {error}")
            return jsonify({"error": error}), 404 if "not found" in error.lower() else 500

        # return minimal info for your UI's "Top Candidates" view
        minimal = [
            {
                "resume_id": m["resume_id"],
                "candidate_name": m.get("candidate_name", "Unknown"),
                "overall_score": m.get("overall_score", 0.0),
            }
            for m in matches
        ]
        return jsonify({"job_id": job_id, "top_resume_matches": minimal}), 200

    except Exception as e:
        current_app.logger.error(f"Unexpected error in match_resumes_for_job: {str(e)}")
        return jsonify({"error": str(e)}), 500


@job_bp.route("/match-specific/<job_id>/<resume_id>", methods=["GET"])
@login_required
def get_specific_match(job_id: str, resume_id: str):
    try:
        match_data, error = current_app.matching_service.match_resume_to_job(resume_id, job_id)
        if error:
            current_app.logger.error(f"Match error: {error}")
            return jsonify({"error": error}), 404 if "not found" in error.lower() else 500

        return jsonify({"job_id": job_id, "resume_id": resume_id, "match_details": match_data}), 200
    except Exception as e:
        current_app.logger.error(f"Error in get_specific_match: {str(e)}")
        return jsonify({"error": str(e)}), 500
