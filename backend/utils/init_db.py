"""
Database initialization script
"""
from sqlalchemy import create_engine
from backend.config.settings import get_settings
from backend.datastore.models import Base as DatastoreBase
from backend.models.registry import Base as ModelsBase


def init_db():
    """Initialize database tables"""
    settings = get_settings()
    engine = create_engine(settings.database_url)
    
    # Create all tables
    DatastoreBase.metadata.create_all(bind=engine)
    ModelsBase.metadata.create_all(bind=engine)
    
    print("Database tables created successfully!")


if __name__ == "__main__":
    init_db()
