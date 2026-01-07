"""
Test suite for datastore module
"""
import pytest
from datetime import datetime
from backend.datastore.models import DataStore, DataEntryCreate, DataType, DataSource
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db_session():
    """Create in-memory database for testing"""
    from backend.datastore.models import Base
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_data_entry(db_session):
    """Test creating a data entry"""
    datastore = DataStore(db_session)
    
    entry_data = DataEntryCreate(
        data_type=DataType.IMAGE,
        source=DataSource.IOT_CAMERA,
        location_lat=37.7749,
        location_lon=-122.4194,
        tags=["ocean", "monitoring"]
    )
    
    entry = datastore.create_entry(entry_data)
    
    assert entry.id is not None
    assert entry.data_type == "image"
    assert entry.source == "iot_camera"
    assert len(entry.tags) == 2


def test_list_entries(db_session):
    """Test listing data entries"""
    datastore = DataStore(db_session)
    
    # Create multiple entries
    for i in range(5):
        entry_data = DataEntryCreate(
            data_type=DataType.IMAGE,
            source=DataSource.IOT_CAMERA,
            tags=[f"tag{i}"]
        )
        datastore.create_entry(entry_data)
    
    entries = datastore.list_entries()
    assert len(entries) == 5


def test_update_tags(db_session):
    """Test updating entry tags"""
    datastore = DataStore(db_session)
    
    entry_data = DataEntryCreate(
        data_type=DataType.IMAGE,
        source=DataSource.IOT_CAMERA,
        tags=["original"]
    )
    
    entry = datastore.create_entry(entry_data)
    updated = datastore.update_tags(entry.id, ["new", "tags"])
    
    assert len(updated.tags) == 2
    assert "new" in updated.tags
