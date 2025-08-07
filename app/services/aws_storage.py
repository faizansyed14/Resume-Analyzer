import boto3
from werkzeug.utils import secure_filename
import os
from io import BytesIO
from typing import Union

class S3Storage:
    def __init__(self, bucket_name, region='us-east-1'):
        self.s3 = boto3.resource('s3', region_name=region)
        self.bucket = self.s3.Bucket(bucket_name)
    
    def save_file(self, file_stream, file_path: str) -> str:
        """Save file to S3 and return its URL"""
        self.bucket.upload_fileobj(
            file_stream,
            file_path,
            ExtraArgs={'ACL': 'private'}
        )
        return f"s3://{self.bucket.name}/{file_path}"
    
    def get_file(self, file_path: str) -> BytesIO:
        """Get file from S3 as BytesIO object"""
        buffer = BytesIO()
        self.bucket.download_fileobj(file_path, buffer)
        buffer.seek(0)
        return buffer
    
    def delete_file(self, file_path: str) -> bool:
        """Delete file from S3"""
        try:
            self.bucket.Object(file_path).delete()
            return True
        except Exception:
            return False
    
    def generate_presigned_url(self, file_path: str, expiration=3600) -> str:
        """Generate a presigned URL to share an S3 object"""
        return self.s3.meta.client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket.name, 'Key': file_path},
            ExpiresIn=expiration
        )

class StorageService:
    def __init__(self, app=None):
        self.app = app
        self.storage = None
        
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        if app.config['ENV'] == 'production':
            self.storage = S3Storage(app.config['S3_BUCKET'], app.config['AWS_REGION'])
        else:
            from .local_storage import LocalStorage
            self.storage = LocalStorage(
                base_path=app.config['RESUME_STORAGE_PATH']
            )
    
    def save_resume(self, file_stream, filename: str) -> str:
        """Save resume file and return its path"""
        safe_name = secure_filename(filename)
        path = f"resumes/{safe_name}"
        return self.storage.save_file(file_stream, path)
    
    def save_job_description(self, file_stream, filename: str) -> str:
        """Save job description file and return its path"""
        safe_name = secure_filename(filename)
        path = f"jobs/{safe_name}"
        return self.storage.save_file(file_stream, path)