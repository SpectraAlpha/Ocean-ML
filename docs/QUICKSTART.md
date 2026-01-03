# Quick Start Guide - Ocean ML Platform

## Prerequisites

Before starting, ensure you have:
- Python 3.10 or higher
- Docker and Docker Compose
- 4GB+ RAM available
- 10GB+ disk space

## Option 1: Docker Compose (Recommended)

### 1. Clone and Setup

```bash
git clone https://github.com/SpectraAlpha/Ocean-ML.git
cd Ocean-ML

# Copy environment file
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file (optional, defaults work for development):

```bash
# Database
DATABASE_URL=postgresql://oceanml:oceanml_pass@postgres:5432/oceanml

# LLM Integration (optional)
OPENAI_API_KEY=your_key_here
ENABLE_LLM_TAGGING=false
```

### 3. Start All Services

```bash
# Start all services in background
docker-compose up -d

# View logs
docker-compose logs -f api

# Check service status
docker-compose ps
```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Access interactive API docs
open http://localhost:8000/docs
```

### 5. Run Demo Workflow

```bash
# Install requests library if not in container
pip install requests

# Run demo
python examples/demo_workflow.py
```

### 6. Stop Services

```bash
docker-compose down

# Remove volumes (clean slate)
docker-compose down -v
```

---

## Option 2: Local Development

### 1. Setup Python Environment

```bash
# Clone repository
git clone https://github.com/SpectraAlpha/Ocean-ML.git
cd Ocean-ML

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup PostgreSQL

```bash
# Install PostgreSQL 15+
# Create database
createdb oceanml

# Or use Docker for just the database
docker run -d \
  --name oceanml-postgres \
  -e POSTGRES_USER=oceanml \
  -e POSTGRES_PASSWORD=oceanml_pass \
  -e POSTGRES_DB=oceanml \
  -p 5432:5432 \
  postgres:15
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your database connection
```

### 4. Initialize Database

```bash
python setup.py
```

### 5. Start API Server

```bash
python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Test API

```bash
# In another terminal
curl http://localhost:8000/health

# Run demo workflow
python examples/demo_workflow.py
```

---

## Using the Platform

### Ingest IoT Data

```python
import requests

data = {
    "data_type": "image",
    "source": "iot_camera",
    "location_lat": 37.7749,
    "location_lon": -122.4194,
    "file_path": "/data/raw/ocean_image.jpg",
    "tags": ["ocean", "monitoring"]
}

response = requests.post("http://localhost:8000/api/data/ingest", json=data)
print(response.json())
```

### Register a Model

```python
model_data = {
    "name": "Ocean Plastic Detector v1.0",
    "model_type": "ocean_plastic_detection",
    "framework": "pytorch",
    "version": "1.0.0",
    "hyperparameters": {
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 0.001
    }
}

response = requests.post("http://localhost:8000/api/models/register", json=model_data)
model_id = response.json()["id"]
```

### Trigger Training

```python
response = requests.post(f"http://localhost:8000/api/models/{model_id}/train")
print(response.json())
```

### List Models

```python
response = requests.get("http://localhost:8000/api/models")
models = response.json()
for model in models:
    print(f"{model['name']}: {model['status']}")
```

---

## MLflow Tracking

Access MLflow UI to track experiments:

```bash
# MLflow is available at
open http://localhost:5000
```

View:
- Experiment runs
- Model metrics
- Training parameters
- Model artifacts

---

## API Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=backend tests/

# Run specific test file
pytest tests/test_datastore.py
```

---

## Troubleshooting

### Port Already in Use

```bash
# Check what's using port 8000
lsof -i :8000

# Change port in .env
API_PORT=8001
```

### Database Connection Error

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

### Module Import Errors

```bash
# Ensure you're in the project root
cd /path/to/Ocean-ML

# Reinstall dependencies
pip install -r requirements.txt
```

### Docker Issues

```bash
# Rebuild containers
docker-compose build --no-cache

# Remove all containers and volumes
docker-compose down -v
docker system prune -a

# Start fresh
docker-compose up -d
```

---

## Development Workflow

### 1. Make Code Changes

Edit files in `backend/` directory

### 2. API Auto-Reloads

When running with `--reload`, API automatically restarts on file changes

### 3. Test Changes

```bash
# Run specific tests
pytest tests/test_models.py -v

# Run with print statements
pytest tests/test_models.py -s
```

### 4. Check Logs

```bash
# API logs
docker-compose logs -f api

# All logs
docker-compose logs -f
```

---

## Next Steps

1. **Read the Documentation**
   - [API Reference](docs/API.md)
   - [Architecture](docs/ARCHITECTURE.md)

2. **Explore Examples**
   - Run `examples/demo_workflow.py`
   - Check example API calls in docs

3. **Add Your Data**
   - Place images in `data/raw/`
   - Use ingestion API to register

4. **Train Models**
   - Register models via API
   - Trigger training pipelines
   - Monitor in MLflow

5. **Extend the Platform**
   - Add new model types
   - Customize training pipelines
   - Integrate with your IoT devices

---

## Support

- **Documentation**: See `docs/` directory
- **Issues**: GitHub Issues
- **Examples**: See `examples/` directory

---

## What's Included

✅ 5 Pre-configured ML Models:
- Ocean Plastic Waste Detection
- Harmful Algae Bloom Detection
- Marine Debris Classification
- Water Quality Assessment
- Coastal Pollution Monitoring

✅ Complete Backend:
- REST API with 15+ endpoints
- PostgreSQL database
- Redis caching
- MLflow tracking
- Docker deployment

✅ Documentation:
- API reference
- Architecture guide
- This quick start guide
- Example workflows

Happy Ocean Monitoring! 🌊
