"""
Setup script to initialize the Ocean ML platform
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.init_db import init_db
from backend.config.settings import get_settings


def setup_platform():
    """Initialize platform setup"""
    print("=" * 60)
    print("Ocean ML Platform Setup")
    print("=" * 60)
    
    # Check settings
    print("\n1. Loading configuration...")
    settings = get_settings()
    print(f"   Database: {settings.database_url}")
    print(f"   API: {settings.api_host}:{settings.api_port}")
    print(f"   MLflow: {settings.mlflow_tracking_uri}")
    
    # Initialize database
    print("\n2. Initializing database...")
    try:
        init_db()
        print("   ✓ Database tables created")
    except Exception as e:
        print(f"   ✗ Database initialization failed: {e}")
        return False
    
    # Create data directories
    print("\n3. Creating data directories...")
    directories = [
        settings.data_dir,
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.models_dir
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ {directory}")
    
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the API server:")
    print("   python -m uvicorn backend.api.main:app --reload")
    print("\n2. Or use Docker:")
    print("   docker-compose up -d")
    print("\n3. Run demo workflow:")
    print("   python examples/demo_workflow.py")
    print("\n4. Access API docs:")
    print("   http://localhost:8000/docs")
    
    return True


if __name__ == "__main__":
    success = setup_platform()
    sys.exit(0 if success else 1)
