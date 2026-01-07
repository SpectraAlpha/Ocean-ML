"""
FastAPI application for Ocean ML platform
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional
import logging
from datetime import datetime

from backend.config.settings import get_settings
from backend.datastore.models import (
    Base as DatastoreBase,
    DataStore,
    DataEntryCreate,
    DataEntryResponse,
    DataType,
    DataSource
)
from backend.models.registry import (
    Base as ModelsBase,
    ModelRegistry,
    MLModelCreate,
    MLModelResponse,
    ModelType,
    ModelStatus
)
from backend.datastore.llm_enrichment import LLMTagEnricher

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Ocean ML Platform API",
    description="AI Workflow Management Platform for Ocean Waste and Harmful Algae Detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
settings = get_settings()
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
DatastoreBase.metadata.create_all(bind=engine)
ModelsBase.metadata.create_all(bind=engine)

# Initialize LLM enricher
llm_enricher = LLMTagEnricher(
    api_key=settings.openai_api_key if settings.enable_llm_tagging else None,
    provider="openai"
)


def get_db():
    """Dependency for database sessions"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Health check
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Ocean ML Platform API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# Data ingestion endpoints
@app.post("/api/data/ingest", response_model=DataEntryResponse)
async def ingest_data(
    data: DataEntryCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Ingest IoT data stream"""
    try:
        datastore = DataStore(db)
        entry = datastore.create_entry(data)
        
        # Schedule LLM enrichment if enabled and it's an image
        if llm_enricher.enabled and data.data_type == DataType.IMAGE and data.file_path:
            background_tasks.add_task(enrich_entry_with_llm, entry.id, data.file_path, db)
        
        return entry
    except Exception as e:
        logger.error(f"Error ingesting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def enrich_entry_with_llm(entry_id: str, file_path: str, db: Session):
    """Background task to enrich entry with LLM tags"""
    try:
        tags, description = llm_enricher.enrich_image(file_path)
        datastore = DataStore(db)
        datastore.add_llm_enrichment(entry_id, tags, description)
        logger.info(f"Enriched entry {entry_id} with LLM tags")
    except Exception as e:
        logger.error(f"Error enriching entry {entry_id}: {e}")


@app.get("/api/data/entries", response_model=List[DataEntryResponse])
async def list_data_entries(
    skip: int = 0,
    limit: int = 100,
    data_type: Optional[DataType] = None,
    source: Optional[DataSource] = None,
    processed: Optional[bool] = None,
    db: Session = Depends(get_db)
):
    """List data entries with filters"""
    try:
        datastore = DataStore(db)
        entries = datastore.list_entries(
            skip=skip,
            limit=limit,
            data_type=data_type,
            source=source,
            processed=processed
        )
        return entries
    except Exception as e:
        logger.error(f"Error listing entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/entries/{entry_id}", response_model=DataEntryResponse)
async def get_data_entry(entry_id: str, db: Session = Depends(get_db)):
    """Get specific data entry"""
    try:
        datastore = DataStore(db)
        entry = datastore.get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Entry not found")
        return entry
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints
@app.post("/api/models/register", response_model=MLModelResponse)
async def register_model(
    model_data: MLModelCreate,
    db: Session = Depends(get_db)
):
    """Register a new ML model"""
    try:
        registry = ModelRegistry(db)
        model = registry.create_model(model_data)
        return model
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models", response_model=List[MLModelResponse])
async def list_models(
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[ModelType] = None,
    status: Optional[ModelStatus] = None,
    db: Session = Depends(get_db)
):
    """List registered models"""
    try:
        registry = ModelRegistry(db)
        models = registry.list_models(
            skip=skip,
            limit=limit,
            model_type=model_type,
            status=status
        )
        return models
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_id}", response_model=MLModelResponse)
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get specific model"""
    try:
        registry = ModelRegistry(db)
        model = registry.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/{model_id}/train")
async def trigger_training(
    model_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Trigger model training pipeline"""
    try:
        registry = ModelRegistry(db)
        model = registry.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Update status to training
        registry.update_status(model_id, ModelStatus.TRAINING)
        
        # Create training job
        job = registry.create_training_job(model_id, total_epochs=50)
        
        # Schedule training in background
        background_tasks.add_task(run_training_pipeline, model_id, db)
        
        return {
            "message": "Training pipeline triggered",
            "model_id": model_id,
            "job_id": job.id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training_pipeline(model_id: str, db: Session):
    """Background task to run training pipeline"""
    try:
        registry = ModelRegistry(db)
        model = registry.get_model(model_id)
        
        # Simulate training (in production, would run actual training)
        logger.info(f"Starting training for model {model_id}")
        
        # Update metrics after training
        registry.update_metrics(
            model_id,
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85
        )
        
        # Update status
        registry.update_status(model_id, ModelStatus.TRAINED)
        
        logger.info(f"Training completed for model {model_id}")
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        registry = ModelRegistry(db)
        registry.update_status(model_id, ModelStatus.FAILED)


@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get platform statistics"""
    try:
        datastore = DataStore(db)
        registry = ModelRegistry(db)
        
        total_entries = len(datastore.list_entries(limit=10000))
        total_models = len(registry.list_models(limit=10000))
        
        return {
            "total_data_entries": total_entries,
            "total_models": total_models,
            "active_models": len(registry.list_models(status=ModelStatus.DEPLOYED, limit=10000)),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
