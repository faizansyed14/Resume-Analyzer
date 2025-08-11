# app/routes/resume_routes.py
import os
import json
import uuid
from typing import Generator

import numpy as np
from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
from flask_login import login_required

# Use the alias from your file: ResumeParser = EnhancedResumeParser
from ..services.resume_parser import ResumeParser

resume_bp = Blueprint("resume_bp", __name__)
resume_parser = ResumeParser()


def _ndjson_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


@resume_bp.route("/upload", methods=["POST"])
@login_required
def upload_resumes_single():
    """
    Simple single-file endpoint (kept for compatibility).
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        unique_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1].lower()
        save_path = os.path.join(current_app.config["RESUME_STORAGE_PATH"], f"{unique_id}{ext}")
        os.makedirs(current_app.config["RESUME_STORAGE_PATH"], exist_ok=True)
        file.save(save_path)

        # Use the new parser sync API to get rich structure
        parsed = current_app.resume_parser.process_single_resume_sync(save_path)
        raw_text = parsed.get("raw_text", "")
        entities = parsed.get("entities", {})

        # Store embeddings
        embeddings = current_app.embedding_service.store_resume_embedding(unique_id, raw_text, entities)

        # Pull refined entities (includes matched_skills)
        resume_data = current_app.embedding_service.get_resume_data(unique_id)
        refined_entities = (resume_data or {}).get("refined_entities", entities)

        return jsonify(
            {
                "filename": file.filename,
                "resume_id": unique_id,
                "entities": refined_entities,
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
        current_app.logger.exception(f"Upload single resume failed: {e}")
        return jsonify({"error": str(e)}), 500


@resume_bp.route("/batch-upload", methods=["POST"])
@login_required
def upload_resumes_batch_stream():
    """
    NDJSON streaming endpoint to match the front-end expectations.
    Emits lines with {"type": "...", ...} events: progress | result | error | complete
    """
    try:
        files = request.files.getlist("files")
        if not files:
            # Some clients use "file" for multiple selection as well
            files = request.files.getlist("file")
        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        processing_mode = request.form.get("processing_mode", "threaded")

        os.makedirs(current_app.config["RESUME_STORAGE_PATH"], exist_ok=True)

        @stream_with_context
        def generate() -> Generator[str, None, None]:
            total = len(files)
            successful = 0
            failed = 0

            for i, file in enumerate(files, start=1):
                try:
                    unique_id = str(uuid.uuid4())
                    ext = os.path.splitext(file.filename)[1].lower()
                    save_path = os.path.join(
                        current_app.config["RESUME_STORAGE_PATH"], f"{unique_id}{ext}"
                    )
                    file.save(save_path)

                    # progress tick
                    yield _ndjson_line(
                        {
                            "type": "progress",
                            "current": i - 1,
                            "total": total,
                            "current_file": file.filename,
                        }
                    )

                    # parse
                    parsed = current_app.resume_parser.process_single_resume_sync(save_path)
                    raw_text = parsed.get("raw_text", "")
                    entities = parsed.get("entities", {})
                    processing_time = parsed.get("processing_time")

                    # store + refine
                    embeddings = current_app.embedding_service.store_resume_embedding(unique_id, raw_text, entities)
                    resume_data = current_app.embedding_service.get_resume_data(unique_id)
                    refined_entities = (resume_data or {}).get("refined_entities", entities)

                    # result payload shaped for your UI
                    resume_payload = {
                        "filename": file.filename,
                        "resume_id": unique_id,
                        "entities": refined_entities,
                        "processing_time": processing_time,
                    }
                    successful += 1

                    yield _ndjson_line({"type": "result", "data": resume_payload})

                except Exception as ex:
                    failed += 1
                    current_app.logger.exception(f"Batch parse error for {file.filename}: {ex}")
                    yield _ndjson_line(
                        {
                            "type": "error",
                            "filename": getattr(file, "filename", "unknown"),
                            "error": str(ex),
                        }
                    )

                # progress after each file
                yield _ndjson_line(
                    {"type": "progress", "current": i, "total": total, "current_file": file.filename}
                )

            # completion
            yield _ndjson_line(
                {
                    "type": "complete",
                    "successful": successful,
                    "failed": failed,
                    "total": total,
                }
            )

        return Response(generate(), mimetype="application/x-ndjson")

    except Exception as e:
        current_app.logger.exception(f"Batch upload failed: {e}")
        return jsonify({"error": str(e)}), 500


@resume_bp.route("/<resume_id>", methods=["GET"])
@login_required
def get_resume_data(resume_id: str):
    data = current_app.embedding_service.get_resume_data(resume_id)
    if not data:
        return jsonify({"error": f'Resume with ID "{resume_id}" not found.'}), 404

    has_embedding = any(
        isinstance(emb, np.ndarray) and emb.size > 0
        for emb in data.get("embeddings", {}).values()
    )

    return jsonify(
        {
            "resume_id": resume_id,
            "entities": data.get("refined_entities", {}),
            "embedding_exists": has_embedding,
        }
    ), 200


@resume_bp.route("/list", methods=["GET"])
@login_required
def list_resumes():
    items = []
    for rid, data in current_app.embedding_service.resume_embeddings.items():
        has_embedding = any(
            isinstance(emb, np.ndarray) and emb.size > 0
            for emb in data.get("embeddings", {}).values()
        )
        ents = data.get("refined_entities", {})
        matched_skills = ents.get("matched_skills", [])
        items.append(
            {
                "resume_id": rid,
                "has_embedding": has_embedding,
                "num_skills": len(matched_skills),
                "name": ents.get("name", "N/A"),
            }
        )
    return jsonify(items), 200


@resume_bp.route("/match-jobs/<resume_id>", methods=["GET"])
@login_required
def match_jobs_for_resume(resume_id: str):
    top_n = request.args.get("top_n", 5, type=int)
    matches, error = current_app.matching_service.find_top_matches_for_resume(resume_id, top_n)
    if error:
        return jsonify({"error": error}), 404
    return jsonify({"resume_id": resume_id, "top_job_matches": matches}), 200


@resume_bp.route("/<resume_id>/generate-interview-questions/<job_id>", methods=["GET"])
@login_required
def generate_interview_questions(resume_id: str, job_id: str):
    # Pull stored entities
    resume_data = current_app.embedding_service.get_resume_data(resume_id)
    job_data = current_app.embedding_service.get_job_data(job_id)
    if not resume_data or not job_data:
        return jsonify({"error": "Resume or job not found"}), 404

    r_ent = resume_data.get("refined_entities", {})
    j_ent = job_data.get("refined_entities", {})

    context = {
        "candidate_skills": r_ent.get("matched_skills", []),
        "candidate_experience": r_ent.get("experience", []),
        "candidate_projects": r_ent.get("projects", []),
        "matched_skills": r_ent.get("matched_skills", []),
        "job_skills": j_ent.get("matched_skills", []),
        "job_experience": j_ent.get("experience", []),
        "job_title": j_ent.get("job_title", "the role"),
        "company": j_ent.get("company", ""),
    }

    questions, usage = current_app.llm_service.generate_interview_questions(context)
    if "error" in questions:
        return jsonify({"error": questions["error"]}), 500

    return jsonify({"interview_questions": questions, "token_usage": usage}), 200
