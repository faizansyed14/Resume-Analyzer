import os

class Config:
    # Base configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-mpnet-base-v2')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_LLM_MODEL = os.getenv('OPENAI_LLM_MODEL', 'gpt-4-mini')

    
    # File storage
    RESUME_STORAGE_PATH = os.getenv('RESUME_STORAGE_PATH', 'data/resumes')
    JOB_STORAGE_PATH = os.getenv('JOB_STORAGE_PATH', 'data/job_descriptions')
    EMBEDDINGS_STORAGE_PATH = os.getenv('EMBEDDINGS_STORAGE_PATH', 'data/embeddings')
    
    # Session
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour

class DevelopmentConfig(Config):  # Note: Capital 'D' and 'C'
    DEBUG = True
    ENV = 'development'
    
    # Local storage paths
    RESUME_STORAGE_PATH = 'data/resumes'
    JOB_STORAGE_PATH = 'data/job_descriptions'
    EMBEDDINGS_STORAGE_PATH = 'data/embeddings'

class ProductionConfig(Config):  # Note: Capital 'P' and 'C'
    DEBUG = False
    ENV = 'production'
    
    # AWS Configuration
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    
    # Cognito Configuration
    COGNITO_USER_POOL_ID = os.getenv('COGNITO_USER_POOL_ID')
    COGNITO_CLIENT_ID = os.getenv('COGNITO_CLIENT_ID')
    
    # S3 Configuration
    S3_BUCKET = os.getenv('S3_BUCKET', 'alpha-data-resumes')
    
    # DynamoDB Configuration
    DYNAMODB_TABLE_PREFIX = os.getenv('DYNAMODB_TABLE_PREFIX', 'AlphaData_')
    
    # Update storage paths for S3
    RESUME_STORAGE_PATH = f's3://{S3_BUCKET}/resumes/'
    JOB_STORAGE_PATH = f's3://{S3_BUCKET}/jobs/'
    EMBEDDINGS_STORAGE_PATH = f's3://{S3_BUCKET}/embeddings/'