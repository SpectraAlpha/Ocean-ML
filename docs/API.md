# Ocean ML Platform API Documentation

## Overview

The Ocean ML Platform provides a RESTful API for managing IoT data streams, ML models, and training pipelines focused on ocean waste detection and harmful algae monitoring.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API uses open access for development. Production deployments should implement:
- API key authentication
- OAuth2/JWT tokens
- Rate limiting

## Endpoints

### Health & Status

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-03T12:00:00.000000"
}
```

#### GET /api/stats
Get platform statistics.

**Response:**
```json
{
  "total_data_entries": 150,
  "total_models": 5,
  "active_models": 3,
  "timestamp": "2024-01-03T12:00:00.000000"
}
```

---

### Data Ingestion

#### POST /api/data/ingest
Ingest IoT data stream with metadata.

**Request Body:**
```json
{
  "data_type": "image",
  "source": "iot_camera",
  "timestamp": "2024-01-03T12:00:00Z",
  "location_lat": 37.7749,
  "location_lon": -122.4194,
  "file_path": "/data/raw/image_001.jpg",
  "s3_key": "raw/2024/01/03/image_001.jpg",
  "metadata": {
    "camera_id": "CAM-001",
    "weather": "sunny",
    "sea_state": "calm"
  },
  "tags": ["ocean", "surface", "monitoring"]
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "data_type": "image",
  "source": "iot_camera",
  "timestamp": "2024-01-03T12:00:00Z",
  "location_lat": 37.7749,
  "location_lon": -122.4194,
  "file_path": "/data/raw/image_001.jpg",
  "s3_key": "raw/2024/01/03/image_001.jpg",
  "metadata": {...},
  "tags": ["ocean", "surface", "monitoring"],
  "llm_tags": [],
  "llm_description": null,
  "processed": false,
  "created_at": "2024-01-03T12:00:00Z",
  "updated_at": "2024-01-03T12:00:00Z"
}
```

#### GET /api/data/entries
List data entries with optional filters.

**Query Parameters:**
- `skip` (int): Number of entries to skip (default: 0)
- `limit` (int): Maximum entries to return (default: 100)
- `data_type` (string): Filter by data type (image, sensor, video, metadata)
- `source` (string): Filter by source (iot_camera, satellite, drone, buoy_sensor, vessel_sensor)
- `processed` (bool): Filter by processing status

**Response:**
```json
[
  {
    "id": "...",
    "data_type": "image",
    "source": "iot_camera",
    ...
  }
]
```

#### GET /api/data/entries/{entry_id}
Get a specific data entry by ID.

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "data_type": "image",
  ...
}
```

---

### Model Management

#### POST /api/models/register
Register a new ML model.

**Request Body:**
```json
{
  "name": "Ocean Plastic Detector v1.0",
  "model_type": "ocean_plastic_detection",
  "framework": "pytorch",
  "version": "1.0.0",
  "description": "ResNet50-based model for detecting ocean plastic waste",
  "hyperparameters": {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "optimizer": "adam",
    "num_classes": 3
  }
}
```

**Response:**
```json
{
  "id": "650e8400-e29b-41d4-a716-446655440000",
  "name": "Ocean Plastic Detector v1.0",
  "model_type": "ocean_plastic_detection",
  "framework": "pytorch",
  "version": "1.0.0",
  "status": "created",
  "description": "ResNet50-based model for detecting ocean plastic waste",
  "model_path": null,
  "accuracy": null,
  "precision": null,
  "recall": null,
  "f1_score": null,
  "metrics": {},
  "created_at": "2024-01-03T12:00:00Z",
  "updated_at": "2024-01-03T12:00:00Z"
}
```

#### GET /api/models
List all registered models.

**Query Parameters:**
- `skip` (int): Number of models to skip
- `limit` (int): Maximum models to return
- `model_type` (string): Filter by model type
- `status` (string): Filter by status (created, training, trained, deployed, failed)

**Response:**
```json
[
  {
    "id": "...",
    "name": "Ocean Plastic Detector v1.0",
    "model_type": "ocean_plastic_detection",
    "status": "trained",
    ...
  }
]
```

#### GET /api/models/{model_id}
Get a specific model by ID.

**Response:**
```json
{
  "id": "650e8400-e29b-41d4-a716-446655440000",
  "name": "Ocean Plastic Detector v1.0",
  ...
}
```

#### POST /api/models/{model_id}/train
Trigger training pipeline for a model.

**Response:**
```json
{
  "message": "Training pipeline triggered",
  "model_id": "650e8400-e29b-41d4-a716-446655440000",
  "job_id": "750e8400-e29b-41d4-a716-446655440000"
}
```

---

## Data Types

### DataType Enum
- `image`: Image data
- `sensor`: Sensor readings
- `video`: Video streams
- `metadata`: Metadata only

### DataSource Enum
- `iot_camera`: IoT camera devices
- `satellite`: Satellite imagery
- `drone`: Drone footage
- `buoy_sensor`: Ocean buoy sensors
- `vessel_sensor`: Vessel-mounted sensors

### ModelType Enum
- `ocean_plastic_detection`: Ocean plastic waste detection
- `harmful_algae_detection`: Harmful algae bloom detection
- `marine_debris_classification`: Marine debris classification
- `water_quality_assessment`: Water quality assessment
- `coastal_pollution_monitoring`: Coastal pollution monitoring

### ModelFramework Enum
- `pytorch`: PyTorch
- `tensorflow`: TensorFlow
- `sklearn`: Scikit-learn
- `custom`: Custom framework

### ModelStatus Enum
- `created`: Model registered but not trained
- `training`: Currently training
- `trained`: Training completed
- `evaluating`: Under evaluation
- `deployed`: Deployed for inference
- `failed`: Training/deployment failed
- `archived`: Archived model

---

## Error Responses

All endpoints may return error responses:

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error message"
}
```

---

## Rate Limiting

Currently not implemented. Production should include:
- Per-IP rate limits
- Per-API-key rate limits
- Burst allowances

---

## Webhooks

Future feature: Register webhooks for:
- Training completion
- Model deployment
- Data processing completion
- Alert triggers

---

## Interactive Documentation

Access interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
