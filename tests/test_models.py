"""
Test suite for model registry
"""
import pytest
from backend.models.registry import ModelRegistry, MLModelCreate, ModelType, ModelFramework, ModelStatus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db_session():
    """Create in-memory database for testing"""
    from backend.models.registry import Base
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_create_model(db_session):
    """Test creating a model"""
    registry = ModelRegistry(db_session)
    
    model_data = MLModelCreate(
        name="Test Model",
        model_type=ModelType.OCEAN_PLASTIC_DETECTION,
        framework=ModelFramework.PYTORCH,
        version="1.0.0",
        hyperparameters={"batch_size": 32}
    )
    
    model = registry.create_model(model_data)
    
    assert model.id is not None
    assert model.name == "Test Model"
    assert model.model_type == "ocean_plastic_detection"
    assert model.status == "created"


def test_list_models(db_session):
    """Test listing models"""
    registry = ModelRegistry(db_session)
    
    # Create multiple models
    for i in range(3):
        model_data = MLModelCreate(
            name=f"Model {i}",
            model_type=ModelType.OCEAN_PLASTIC_DETECTION,
            framework=ModelFramework.PYTORCH,
            version=f"1.0.{i}"
        )
        registry.create_model(model_data)
    
    models = registry.list_models()
    assert len(models) == 3


def test_update_model_status(db_session):
    """Test updating model status"""
    registry = ModelRegistry(db_session)
    
    model_data = MLModelCreate(
        name="Test Model",
        model_type=ModelType.OCEAN_PLASTIC_DETECTION,
        framework=ModelFramework.PYTORCH,
        version="1.0.0"
    )
    
    model = registry.create_model(model_data)
    updated = registry.update_status(model.id, ModelStatus.TRAINING)
    
    assert updated.status == "training"


def test_update_metrics(db_session):
    """Test updating model metrics"""
    registry = ModelRegistry(db_session)
    
    model_data = MLModelCreate(
        name="Test Model",
        model_type=ModelType.OCEAN_PLASTIC_DETECTION,
        framework=ModelFramework.PYTORCH,
        version="1.0.0"
    )
    
    model = registry.create_model(model_data)
    updated = registry.update_metrics(
        model.id,
        accuracy=0.95,
        precision=0.93,
        recall=0.94,
        f1_score=0.935
    )
    
    assert updated.accuracy == 0.95
    assert updated.precision == 0.93
