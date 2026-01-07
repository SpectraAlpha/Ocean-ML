"""
Example usage of Ocean ML Platform
"""
import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"


def check_health():
    """Check API health"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.json()


def ingest_sample_data():
    """Ingest sample IoT data"""
    data = {
        "data_type": "image",
        "source": "iot_camera",
        "timestamp": datetime.utcnow().isoformat(),
        "location_lat": 37.7749,
        "location_lon": -122.4194,
        "file_path": "/data/raw/sample_ocean_image.jpg",
        "metadata": {
            "camera_id": "CAM-001",
            "weather": "sunny",
            "sea_state": "calm"
        },
        "tags": ["ocean", "surface", "monitoring"]
    }
    
    response = requests.post(f"{BASE_URL}/api/data/ingest", json=data)
    print("\nData Ingestion Response:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def list_data_entries():
    """List all data entries"""
    response = requests.get(f"{BASE_URL}/api/data/entries?limit=10")
    print("\nData Entries:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def register_model():
    """Register a new ML model"""
    model_data = {
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
    
    response = requests.post(f"{BASE_URL}/api/models/register", json=model_data)
    print("\nModel Registration Response:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def list_models():
    """List all registered models"""
    response = requests.get(f"{BASE_URL}/api/models")
    print("\nRegistered Models:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def trigger_training(model_id):
    """Trigger training pipeline for a model"""
    response = requests.post(f"{BASE_URL}/api/models/{model_id}/train")
    print(f"\nTraining Trigger Response for {model_id}:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def get_platform_stats():
    """Get platform statistics"""
    response = requests.get(f"{BASE_URL}/api/stats")
    print("\nPlatform Statistics:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def demo_workflow():
    """Run a complete demo workflow"""
    print("=" * 60)
    print("Ocean ML Platform Demo Workflow")
    print("=" * 60)
    
    # 1. Check health
    print("\n1. Checking API health...")
    check_health()
    
    # 2. Ingest data
    print("\n2. Ingesting sample IoT data...")
    entry = ingest_sample_data()
    
    # 3. List entries
    print("\n3. Listing data entries...")
    list_data_entries()
    
    # 4. Register models
    print("\n4. Registering ML models...")
    model1 = register_model()
    
    # Register more models
    models_to_register = [
        {
            "name": "Harmful Algae Detector v1.0",
            "model_type": "harmful_algae_detection",
            "framework": "pytorch",
            "version": "1.0.0",
            "hyperparameters": {"batch_size": 32, "epochs": 50, "num_classes": 4}
        },
        {
            "name": "Marine Debris Classifier v1.0",
            "model_type": "marine_debris_classification",
            "framework": "pytorch",
            "version": "1.0.0",
            "hyperparameters": {"batch_size": 32, "epochs": 50, "num_classes": 6}
        },
        {
            "name": "Water Quality Assessor v1.0",
            "model_type": "water_quality_assessment",
            "framework": "pytorch",
            "version": "1.0.0",
            "hyperparameters": {"batch_size": 32, "epochs": 50, "output_dim": 5}
        },
        {
            "name": "Coastal Pollution Monitor v1.0",
            "model_type": "coastal_pollution_monitoring",
            "framework": "pytorch",
            "version": "1.0.0",
            "hyperparameters": {"batch_size": 32, "epochs": 50, "num_classes": 5}
        }
    ]
    
    for model_data in models_to_register:
        response = requests.post(f"{BASE_URL}/api/models/register", json=model_data)
        print(f"\nRegistered: {response.json()['name']}")
    
    # 5. List all models
    print("\n5. Listing all registered models...")
    models = list_models()
    
    # 6. Trigger training (for demo, we'll skip actual training)
    # print("\n6. Triggering training pipeline...")
    # if models:
    #     trigger_training(models[0]['id'])
    
    # 7. Get stats
    print("\n7. Getting platform statistics...")
    get_platform_stats()
    
    print("\n" + "=" * 60)
    print("Demo workflow completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        demo_workflow()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server.")
        print("Make sure the server is running:")
        print("  - With Docker: docker-compose up -d")
        print("  - Standalone: python -m uvicorn backend.api.main:app --reload")
    except Exception as e:
        print(f"Error: {e}")
