# Ocean-ML: AI Workflow Management Platform

An advanced AI workflow management platform focused on infrastructure and ML model lifecycle management for ocean waste detection and harmful algae bloom monitoring.

## 🌊 Overview

Ocean-ML is a comprehensive platform that combines IoT data ingestion, ML model management, and automated training pipelines to address critical ocean health challenges:

- **Ocean Plastic Waste Detection**: Identify and classify plastic debris in marine environments
- **Harmful Algae Bloom (HAB) Detection**: Monitor and detect harmful algae blooms
- **Marine Debris Classification**: Categorize different types of marine debris
- **Water Quality Assessment**: Evaluate water quality from visual indicators
- **Coastal Pollution Monitoring**: Track and monitor coastal pollution levels

## 🏗️ Architecture

### Backend Components

1. **Datastore Module** (`backend/datastore/`)
   - IoT image and sensor data ingestion
   - Metadata tagging system
   - Optional LLM-based image tag enrichment (OpenAI GPT-4 Vision, Anthropic Claude)
   - PostgreSQL-backed persistent storage

2. **ML Model Registry** (`backend/models/`)
   - Version-controlled model management
   - Model status tracking (created, training, trained, deployed, failed)
   - MLflow integration for experiment tracking
   - Support for PyTorch, TensorFlow, and scikit-learn

3. **Training Pipeline** (`backend/training/`)
   - Automated training orchestration
   - Multi-model concurrent training support
   - Progress tracking and logging
   - Continuous improvement workflow

4. **REST API** (`backend/api/`)
   - FastAPI-based RESTful endpoints
   - Data ingestion endpoints
   - Model management and deployment
   - Training pipeline triggers
   - Real-time monitoring

## 🤖 ML Models

### 1. Ocean Plastic Waste Detector
- **Architecture**: ResNet50-based CNN
- **Purpose**: Detect and classify plastic waste in ocean images
- **Classes**: No waste, light pollution, severe pollution

### 2. Harmful Algae Bloom Detector
- **Architecture**: EfficientNet-B0
- **Purpose**: Identify harmful algae blooms
- **Classes**: No bloom, low intensity, medium intensity, high intensity

### 3. Marine Debris Classifier
- **Architecture**: VGG16-BN
- **Purpose**: Multi-class classification of marine debris types
- **Classes**: Plastic bottles, bags, fishing nets, metal, glass, other

### 4. Water Quality Assessor
- **Architecture**: DenseNet121
- **Purpose**: Regression model for water quality scoring
- **Outputs**: Turbidity, color, clarity, pollution level, overall quality

### 5. Coastal Pollution Monitor
- **Architecture**: MobileNetV3-Large
- **Purpose**: Efficient real-time coastal pollution detection
- **Classes**: Clean, light pollution, moderate pollution, heavy pollution, critical

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- PostgreSQL 15+ (or use Docker)
- Redis (or use Docker)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SpectraAlpha/Ocean-ML.git
cd Ocean-ML
```

2. **Create environment file**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize database**
```bash
python -m backend.utils.init_db
```

### Running with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Running Locally

```bash
# Start the API server
python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

### Data Ingestion

- `POST /api/data/ingest` - Ingest IoT data stream
- `GET /api/data/entries` - List data entries with filters
- `GET /api/data/entries/{entry_id}` - Get specific entry

### Model Management

- `POST /api/models/register` - Register new ML model
- `GET /api/models` - List all models
- `GET /api/models/{model_id}` - Get model details
- `POST /api/models/{model_id}/train` - Trigger training pipeline

### Monitoring

- `GET /health` - Health check
- `GET /api/stats` - Platform statistics

## 🔧 Configuration

Key configuration options in `.env`:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/oceanml

# Storage
S3_BUCKET_NAME=ocean-ml-data
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# LLM Integration (Optional)
OPENAI_API_KEY=your_openai_key
ENABLE_LLM_TAGGING=true

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Training
MAX_TRAINING_JOBS=5
DEFAULT_BATCH_SIZE=32
DEFAULT_EPOCHS=50
```

## 📊 Data Flow

1. **Data Ingestion**: IoT devices/sensors send image/data streams
2. **Metadata Tagging**: Automatic and manual tagging with location, timestamp
3. **LLM Enrichment** (Optional): AI-powered image analysis and description
4. **Auto-Import**: Data automatically imported into training pipeline
5. **Model Training**: Scheduled or triggered training on new data
6. **Version Tracking**: Model versions tracked in registry
7. **Deployment**: Trained models deployed for inference
8. **Continuous Improvement**: Feedback loop for model refinement

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_datastore.py

# Run with coverage
pytest --cov=backend tests/
```

## 📦 Project Structure

```
Ocean-ML/
├── backend/
│   ├── api/              # FastAPI endpoints
│   ├── config/           # Configuration management
│   ├── datastore/        # Data ingestion & storage
│   ├── models/           # ML model implementations
│   ├── training/         # Training pipelines
│   └── utils/            # Utility functions
├── data/
│   ├── raw/             # Raw IoT data
│   ├── processed/       # Processed datasets
│   └── models/          # Saved model artifacts
├── tests/               # Test suite
├── docs/                # Documentation
├── docker-compose.yml   # Docker services
├── Dockerfile          # API container
└── requirements.txt    # Python dependencies
```

## 🔐 Security

- API key authentication for LLM services
- Database credentials managed via environment variables
- CORS middleware for API security
- Input validation using Pydantic models

## 🌟 Features

✅ IoT data stream ingestion  
✅ Metadata tagging system  
✅ LLM-based image enrichment (GPT-4 Vision, Claude)  
✅ Model versioning and registry  
✅ Automated training pipelines  
✅ MLflow experiment tracking  
✅ RESTful API with FastAPI  
✅ Docker containerization  
✅ PostgreSQL & Redis integration  
✅ 5 specialized ML models  
✅ Continuous improvement workflow  

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

This platform addresses critical ocean health challenges including:
- Ocean plastic pollution monitoring
- Harmful algae bloom detection
- Marine ecosystem protection
- Water quality assessment
- Coastal environmental monitoring

## 📞 Support

For issues and questions:
- GitHub Issues: [Ocean-ML Issues](https://github.com/SpectraAlpha/Ocean-ML/issues)
- Email: support@oceanml.org

---

**Status**: Backend ~50% Complete | 5 ML Models Implemented | Active Development

Built with ❤️ for ocean conservation