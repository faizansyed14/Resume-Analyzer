import os
import boto3
from flask import Flask, jsonify, request, redirect, url_for, render_template
from flask_login import LoginManager, current_user
from werkzeug.security import generate_password_hash
from config import DevelopmentConfig, ProductionConfig

# Models
from .models.local_user import LocalUser

# Initialize the login manager and global service placeholders
login_manager = LoginManager()
llm_service_instance = None
embedding_service_instance = None

def create_app(config_name='development'):
    """Application factory"""
    app = Flask(__name__, static_folder="static", static_url_path="")

    # Configuration
    if config_name.lower() == 'development':
        app.config.from_object(DevelopmentConfig)
        setup_local_auth(app)
    else:
        app.config.from_object(ProductionConfig)
        setup_aws_auth(app)

    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # Initialize services
    initialize_services(app)

    # Register Blueprints
    register_blueprints(app)

    # Stub out Chrome DevTools probe to prevent 404 spam
    @app.route('/.well-known/appspecific/com.chrome.devtools.json')
    def chrome_devtools_stub():
        return jsonify({}), 200

    # Provide auth status without redirect
    @app.route('/auth/check')
    def auth_check():
        return jsonify(authenticated=current_user.is_authenticated), 200

    # Index route
    @app.route('/')
    def index():
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        return render_template('index.html')

    return app


def setup_local_auth(app):
    """Setup for local development with in-memory user store"""
    if not hasattr(LocalUser, '_users'):
        LocalUser._users = {}

    # Ensure default admin exists
    LocalUser.ensure_user_exists('admin@alpha.ae', 'Alphadata.123')

    @login_manager.user_loader
    def load_user(user_id):
        return LocalUser.get(user_id)


def setup_aws_auth(app):
    """Setup for production with AWS Cognito/DynamoDB"""
    # Initialize AWS resources
    app.dynamodb = boto3.resource(
        'dynamodb',
        region_name=app.config['AWS_REGION'],
        aws_access_key_id=app.config['AWS_ACCESS_KEY'],
        aws_secret_access_key=app.config['AWS_SECRET_KEY']
    )
    app.cognito = boto3.client(
        'cognito-idp',
        region_name=app.config['AWS_REGION'],
        aws_access_key_id=app.config['AWS_ACCESS_KEY'],
        aws_secret_access_key=app.config['AWS_SECRET_KEY']
    )

    # Ensure DynamoDB user table exists
    from .models.dynamodb_user import ensure_user_table_exists, DynamoDBUser
    ensure_user_table_exists(app.dynamodb)

    @login_manager.user_loader
    def load_user(user_id):
        return DynamoDBUser.get(user_id, app.dynamodb)


def initialize_services(app):
    """Initialize embedding, LLM, parsing, and matching services"""
    from .services.embedding_service import EmbeddingService
    from .services.llm_service import LLMService
    from .services.parsing_service import ParsingService
    from .services.matching_service import MatchingService

    global embedding_service_instance, llm_service_instance

    # Embedding service
    embedding_service = EmbeddingService(
        model_name=app.config['EMBEDDING_MODEL_NAME']
    )
    embedding_service.set_storage_path(app.config['EMBEDDINGS_STORAGE_PATH'])
    embedding_service.load_embeddings()

    # LLM service
    llm_service = LLMService(
        api_key=app.config['OPENAI_API_KEY'],
        model_name=app.config['OPENAI_LLM_MODEL']
    )

    # Parsing & matching
    parsing_service = ParsingService()
    matching_service = MatchingService(embedding_service)
    matching_service.llm_service = llm_service

    # Attach to app
    app.embedding_service = embedding_service
    app.llm_service = llm_service
    app.parsing_service = parsing_service
    app.matching_service = matching_service

    # Update globals for import in routes
    embedding_service_instance = embedding_service
    llm_service_instance = llm_service


def register_blueprints(app):
    """Register Flask blueprints"""
    from .routes.auth_routes import auth_bp
    from .routes.resume_routes import resume_bp
    from .routes.job_routes import job_bp

    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(resume_bp, url_prefix='/resumes')
    app.register_blueprint(job_bp, url_prefix='/jobs')
