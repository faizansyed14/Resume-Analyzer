from app import create_app
from app.models.local_user import LocalUser
import os

app = create_app('development')

# Create test user and ensure directories exist
with app.app_context():
    # Create test user
    LocalUser.ensure_user_exists('admin@alpha.ae', 'Alphadata.123')
    
    # Ensure directories exist
    os.makedirs(app.config['RESUME_STORAGE_PATH'], exist_ok=True)
    os.makedirs(app.config['JOB_STORAGE_PATH'], exist_ok=True)
    os.makedirs(app.config['EMBEDDINGS_STORAGE_PATH'], exist_ok=True)

if __name__ == '__main__':
    app.run(debug=True)