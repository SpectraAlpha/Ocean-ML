"""
ML Model Management System
"""
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from sqlalchemy import Column, String, DateTime, JSON, Float, Integer, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field


Base = declarative_base()


class ModelType(str, Enum):
    """Type of ML model"""
    OCEAN_PLASTIC_DETECTION = "ocean_plastic_detection"
    HARMFUL_ALGAE_DETECTION = "harmful_algae_detection"
    MARINE_DEBRIS_CLASSIFICATION = "marine_debris_classification"
    WATER_QUALITY_ASSESSMENT = "water_quality_assessment"
    COASTAL_POLLUTION_MONITORING = "coastal_pollution_monitoring"


class ModelStatus(str, Enum):
    """Model training/deployment status"""
    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"


class ModelFramework(str, Enum):
    """ML framework used"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    CUSTOM = "custom"


class MLModel(Base):
    """Database model for ML models"""
    __tablename__ = "ml_models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    framework = Column(String, nullable=False)
    version = Column(String, nullable=False)
    status = Column(String, default=ModelStatus.CREATED.value)
    description = Column(Text, nullable=True)
    
    # Model artifacts
    model_path = Column(String, nullable=True)
    s3_model_key = Column(String, nullable=True)
    checkpoint_path = Column(String, nullable=True)
    
    # Training configuration
    hyperparameters = Column(JSON, default={})
    training_data_ids = Column(JSON, default=[])
    
    # Metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    metrics = Column(JSON, default={})
    
    # Metadata
    training_started_at = Column(DateTime, nullable=True)
    training_completed_at = Column(DateTime, nullable=True)
    deployed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # MLflow integration
    mlflow_run_id = Column(String, nullable=True)
    wandb_run_id = Column(String, nullable=True)


class TrainingJob(Base):
    """Database model for training jobs"""
    __tablename__ = "training_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False)
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    current_epoch = Column(Integer, default=0)
    total_epochs = Column(Integer, nullable=False)
    logs = Column(Text, default="")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MLModelCreate(BaseModel):
    """Schema for creating ML models"""
    name: str
    model_type: ModelType
    framework: ModelFramework
    version: str = "1.0.0"
    description: Optional[str] = None
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class MLModelResponse(BaseModel):
    """Schema for ML model responses"""
    id: str
    name: str
    model_type: str
    framework: str
    version: str
    status: str
    description: Optional[str]
    model_path: Optional[str]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ModelRegistry:
    """Model registry for managing ML models"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_model(self, model_data: MLModelCreate) -> MLModel:
        """Register a new model"""
        model = MLModel(
            name=model_data.name,
            model_type=model_data.model_type.value,
            framework=model_data.framework.value,
            version=model_data.version,
            description=model_data.description,
            hyperparameters=model_data.hyperparameters
        )
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model
    
    def get_model(self, model_id: str) -> Optional[MLModel]:
        """Get model by ID"""
        return self.db.query(MLModel).filter(MLModel.id == model_id).first()
    
    def list_models(
        self,
        skip: int = 0,
        limit: int = 100,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None
    ) -> List[MLModel]:
        """List models with filters"""
        query = self.db.query(MLModel)
        
        if model_type:
            query = query.filter(MLModel.model_type == model_type.value)
        if status:
            query = query.filter(MLModel.status == status.value)
        
        return query.order_by(MLModel.created_at.desc()).offset(skip).limit(limit).all()
    
    def update_status(self, model_id: str, status: ModelStatus) -> Optional[MLModel]:
        """Update model status"""
        model = self.get_model(model_id)
        if model:
            model.status = status.value
            self.db.commit()
            self.db.refresh(model)
        return model
    
    def update_metrics(
        self,
        model_id: str,
        accuracy: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1_score: Optional[float] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[MLModel]:
        """Update model metrics"""
        model = self.get_model(model_id)
        if model:
            if accuracy is not None:
                model.accuracy = accuracy
            if precision is not None:
                model.precision = precision
            if recall is not None:
                model.recall = recall
            if f1_score is not None:
                model.f1_score = f1_score
            if metrics is not None:
                model.metrics = metrics
            self.db.commit()
            self.db.refresh(model)
        return model
    
    def get_latest_version(self, model_type: ModelType) -> Optional[MLModel]:
        """Get latest version of a model type"""
        return (
            self.db.query(MLModel)
            .filter(MLModel.model_type == model_type.value)
            .order_by(MLModel.created_at.desc())
            .first()
        )
    
    def create_training_job(self, model_id: str, total_epochs: int) -> TrainingJob:
        """Create a new training job"""
        job = TrainingJob(
            model_id=model_id,
            total_epochs=total_epochs
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job
