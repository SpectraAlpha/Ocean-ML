# Ocean ML Platform Architecture

## System Overview

The Ocean ML Platform is designed as a microservices-based architecture for managing IoT data streams and ML model lifecycle for ocean monitoring applications.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  (Web Dashboard, Mobile Apps, IoT Devices, API Clients)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway / Load Balancer            │
│                         (FastAPI)                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
│   Datastore    │ │    Model     │ │    Training     │
│    Service     │ │   Registry   │ │   Orchestrator  │
└────────┬───────┘ └──────┬───────┘ └────────┬────────┘
         │                │                   │
         └────────────────┼───────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
│   PostgreSQL   │ │    Redis     │ │     MLflow      │
│   (Metadata)   │ │   (Cache)    │ │   (Tracking)    │
└────────────────┘ └──────────────┘ └─────────────────┘
         │
         ▼
┌────────────────┐
│   S3 / Object  │
│    Storage     │
│  (Images/Data) │
└────────────────┘
```

## Component Details

### 1. API Layer (`backend/api/`)

**Technology**: FastAPI
**Responsibilities**:
- RESTful API endpoints
- Request validation (Pydantic)
- Authentication & authorization
- CORS handling
- Background task scheduling

**Key Files**:
- `main.py`: API application & endpoints
- Swagger/OpenAPI documentation at `/docs`

### 2. Datastore Module (`backend/datastore/`)

**Technology**: SQLAlchemy + PostgreSQL
**Responsibilities**:
- IoT data ingestion
- Metadata storage & indexing
- Tag management
- LLM-based enrichment
- Query & filtering

**Key Components**:
- `models.py`: Database models & schemas
- `llm_enrichment.py`: LLM integration for image analysis

**Database Schema**:
```sql
data_entries (
  id UUID PRIMARY KEY,
  data_type VARCHAR,
  source VARCHAR,
  timestamp TIMESTAMP,
  location_lat FLOAT,
  location_lon FLOAT,
  file_path VARCHAR,
  s3_key VARCHAR,
  metadata JSONB,
  tags JSONB,
  llm_tags JSONB,
  llm_description TEXT,
  processed BOOLEAN,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)
```

### 3. Model Registry (`backend/models/`)

**Technology**: SQLAlchemy + PostgreSQL + MLflow
**Responsibilities**:
- Model versioning
- Status tracking
- Metrics storage
- Model artifacts management
- Training job management

**Key Components**:
- `registry.py`: Model registry logic
- Individual model implementations (5 models)

**Database Schema**:
```sql
ml_models (
  id UUID PRIMARY KEY,
  name VARCHAR,
  model_type VARCHAR,
  framework VARCHAR,
  version VARCHAR,
  status VARCHAR,
  description TEXT,
  model_path VARCHAR,
  s3_model_key VARCHAR,
  hyperparameters JSONB,
  training_data_ids JSONB,
  accuracy FLOAT,
  precision FLOAT,
  recall FLOAT,
  f1_score FLOAT,
  metrics JSONB,
  mlflow_run_id VARCHAR,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)

training_jobs (
  id UUID PRIMARY KEY,
  model_id UUID REFERENCES ml_models(id),
  status VARCHAR,
  progress FLOAT,
  current_epoch INT,
  total_epochs INT,
  logs TEXT,
  error_message TEXT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
)
```

### 4. Training Pipeline (`backend/training/`)

**Technology**: PyTorch + asyncio
**Responsibilities**:
- Training orchestration
- Concurrent job management
- Progress tracking
- Model checkpointing
- Metrics logging

**Key Components**:
- `pipeline.py`: Base training pipeline & orchestrator
- Model-specific pipelines in `backend/models/`

**Training Flow**:
```
1. Submit Training Job
   ↓
2. Allocate Resources
   ↓
3. Load Data
   ↓
4. Build Model
   ↓
5. Training Loop
   ├─ Forward pass
   ├─ Backward pass
   ├─ Update weights
   └─ Log metrics
   ↓
6. Validation
   ↓
7. Save Checkpoints
   ↓
8. Update Registry
   ↓
9. Complete Job
```

### 5. ML Models

Five specialized models for ocean monitoring:

#### Model 1: Ocean Plastic Detector
- **Architecture**: ResNet50
- **Task**: Multi-class classification
- **Classes**: 3 (no waste, light, severe)
- **Input**: 224x224 RGB images

#### Model 2: Harmful Algae Detector
- **Architecture**: EfficientNet-B0
- **Task**: Multi-class classification
- **Classes**: 4 (no bloom, low, medium, high)
- **Input**: 224x224 RGB images

#### Model 3: Marine Debris Classifier
- **Architecture**: VGG16-BN
- **Task**: Multi-class classification
- **Classes**: 6 (bottles, bags, nets, metal, glass, other)
- **Input**: 224x224 RGB images

#### Model 4: Water Quality Assessor
- **Architecture**: DenseNet121
- **Task**: Regression
- **Outputs**: 5 quality metrics (0-1 scale)
- **Input**: 224x224 RGB images

#### Model 5: Coastal Pollution Monitor
- **Architecture**: MobileNetV3-Large
- **Task**: Multi-class classification
- **Classes**: 5 (clean to critical)
- **Input**: 224x224 RGB images

### 6. Configuration (`backend/config/`)

**Technology**: Pydantic Settings
**Responsibilities**:
- Environment variable management
- Settings validation
- Configuration caching

### 7. Data Storage

#### PostgreSQL
- Metadata storage
- Model registry
- Training jobs
- User data

#### Redis
- Caching layer
- Session storage
- Task queue (future)

#### S3 / Object Storage
- Raw images/videos
- Processed datasets
- Model artifacts
- Training checkpoints

#### MLflow
- Experiment tracking
- Model versioning
- Metrics logging
- Artifact storage

## Data Flow

### Ingestion Flow
```
IoT Device → API → Validation → Database → [Optional: LLM Enrichment] → Storage
```

### Training Flow
```
Data Entry → Auto-import → Dataset Creation → Training Pipeline → Model Registry → Deployment
```

### Inference Flow (Future)
```
New Data → API → Model Service → Inference → Results → Database → Notification
```

## Scalability Considerations

### Horizontal Scaling
- Stateless API servers (behind load balancer)
- Multiple training workers
- Distributed storage (S3)
- Database read replicas

### Vertical Scaling
- GPU-enabled training nodes
- Larger database instances
- Increased cache memory

### Performance Optimization
- Database indexing (timestamp, model_type, status)
- Redis caching for frequent queries
- Batch processing for data ingestion
- Async operations for I/O-bound tasks

## Security Architecture

### Current Implementation
- Input validation (Pydantic)
- SQL injection prevention (SQLAlchemy ORM)
- CORS configuration

### Future Enhancements
- API key authentication
- JWT tokens
- Role-based access control (RBAC)
- Encryption at rest
- Encryption in transit (TLS)
- Audit logging
- Rate limiting

## Monitoring & Observability

### Metrics Collection
- API request metrics (Prometheus)
- Training metrics (MLflow)
- Database metrics
- Resource utilization

### Logging
- Structured logging (JSON)
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Centralized log aggregation (future)

### Alerting (Future)
- Training failures
- API errors
- Resource exhaustion
- Model drift

## Deployment

### Docker Compose (Development)
- All services in single host
- Persistent volumes
- Network isolation

### Kubernetes (Production - Future)
- Pod auto-scaling
- Service mesh
- Persistent volume claims
- ConfigMaps & Secrets

## Technology Stack Summary

| Layer | Technology |
|-------|-----------|
| API | FastAPI, Uvicorn |
| ML Framework | PyTorch, TensorFlow |
| Database | PostgreSQL 15 |
| Cache | Redis 7 |
| ML Tracking | MLflow 2.8 |
| Storage | S3-compatible |
| ORM | SQLAlchemy 2.0 |
| Validation | Pydantic 2.4 |
| Containerization | Docker, Docker Compose |
| Language | Python 3.10+ |

## Future Enhancements

1. **Real-time Inference Service**
2. **Web Dashboard (Frontend)**
3. **Kubernetes Deployment**
4. **CI/CD Pipeline**
5. **Model A/B Testing**
6. **Federated Learning**
7. **Edge Deployment**
8. **Mobile SDK**
