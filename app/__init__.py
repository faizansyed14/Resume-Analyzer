# app/__init__.py
import os

import boto3
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user

from config import DevelopmentConfig, ProductionConfig
from .models.local_user import LocalUser  # Your lightweight dev user model

# Globals for optional import-access
login_manager = LoginManager()
llm_service_instance = None
embedding_service_instance = None
resume_parser_instance = None
jd_parser_instance = None


def create_app(config_name: str = "development"):
    """
    Flask application factory.
    """
    app = Flask(__name__, static_folder="static", static_url_path="")
    # Config
    if config_name.lower() == "development":
        app.config.from_object(DevelopmentConfig)
        setup_local_auth(app)
    else:
        app.config.from_object(ProductionConfig)
        setup_aws_auth(app)

    # Login
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # Initialize services on app
    initialize_services(app)

    # Blueprints
    register_blueprints(app)

    # Stub out Chrome DevTools probe to prevent 404 spam
    @app.route("/.well-known/appspecific/com.chrome.devtools.json")
    def chrome_devtools_stub():
        return jsonify({}), 200

    # Provide auth status without redirect (used by front-end checks)
    @app.route("/auth/check")
    def auth_check():
        return jsonify(authenticated=current_user.is_authenticated), 200

    # Home route renders your index.html (protected via login in templates)
    @app.route("/")
    def index():
        if not current_user.is_authenticated:
            return redirect(url_for("auth.login"))
        return render_template("index.html")

    return app


# ---- Auth setup ------------------------------------------------------------

def setup_local_auth(app: Flask) -> None:
    """
    Setup for local development with in-memory user store.
    """
    if not hasattr(LocalUser, "_users"):
        LocalUser._users = {}
    # Ensure default admin exists
    LocalUser.ensure_user_exists("admin@alpha.ae", "Alphadata.123")

    @login_manager.user_loader
    def load_user(user_id):
        return LocalUser.get(user_id)


def setup_aws_auth(app: Flask) -> None:
    """
    Setup for production with AWS Cognito / DynamoDB (skeleton).
    """
    app.dynamodb = boto3.resource(
        "dynamodb",
        region_name=app.config["AWS_REGION"],
        aws_access_key_id=app.config["AWS_ACCESS_KEY"],
        aws_secret_access_key=app.config["AWS_SECRET_KEY"],
    )
    app.cognito = boto3.client(
        "cognito-idp",
        region_name=app.config["AWS_REGION"],
        aws_access_key_id=app.config["AWS_ACCESS_KEY"],
        aws_secret_access_key=app.config["AWS_SECRET_KEY"],
    )

    # Ensure DynamoDB user table exists
    from .models.dynamodb_user import ensure_user_table_exists, DynamoDBUser

    ensure_user_table_exists(app.dynamodb)

    @login_manager.user_loader
    def load_user(user_id):
        return DynamoDBUser.get(user_id, app.dynamodb)


# ---- Services init ---------------------------------------------------------

def initialize_services(app: Flask) -> None:
    """
    Initialize embedding, LLM, parsing, and matching services; attach to app.
    """
    from .services.embedding_service import EmbeddingService
    from .services.llm_service import LLMService
    from .services.resume_parser import ResumeParser
    from .services.jd_parser import JDParser
    from .services.matching_service import MatchingService

    global llm_service_instance, embedding_service_instance, resume_parser_instance, jd_parser_instance

    # Embeddings
    embedding_service = EmbeddingService(model_name=app.config["EMBEDDING_MODEL_NAME"])
    embedding_service.set_storage_path(app.config["EMBEDDINGS_STORAGE_PATH"])
    embedding_service.load_embeddings()

    # LLM
    llm_service = LLMService(
        api_key=app.config.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
        model_name=app.config["OPENAI_LLM_MODEL"],
    )

    # Parsers
    resume_parser = ResumeParser()
    jd_parser = JDParser()

    # Matching
    matching_service = MatchingService(embedding_service)
    # Ensure matching service has LLM access (in case no current_app at import time)
    matching_service.llm_service = llm_service

    # Attach to app
    app.embedding_service = embedding_service
    app.llm_service = llm_service
    app.resume_parser = resume_parser
    app.jd_parser = jd_parser
    app.matching_service = matching_service

    # Export as module globals too (some routes import these)
    embedding_service_instance = embedding_service
    llm_service_instance = llm_service
    resume_parser_instance = resume_parser
    jd_parser_instance = jd_parser


def register_blueprints(app: Flask) -> None:
    """
    Register Flask blueprints.
    """
    from .routes.auth_routes import auth_bp
    from .routes.resume_routes import resume_bp
    from .routes.job_routes import job_bp

    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(resume_bp, url_prefix="/resumes")
    app.register_blueprint(job_bp, url_prefix="/jobs")
