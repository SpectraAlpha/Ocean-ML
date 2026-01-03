# Contributing to Ocean-ML Platform

Thank you for your interest in contributing to the Ocean-ML Platform! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Screenshots if applicable

### Suggesting Features

Feature suggestions are welcome! Please create an issue with:
- Clear description of the feature
- Use case and benefits
- Potential implementation approach
- Any related examples

### Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/SpectraAlpha/Ocean-ML.git
   cd Ocean-ML
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make Your Changes**
   - Follow the code style guidelines below
   - Add tests for new features
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run tests
   pytest
   
   # Check code style
   flake8 backend/
   black --check backend/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

## Development Setup

### Prerequisites
- Python 3.10+
- Docker and Docker Compose
- PostgreSQL 15+
- Git

### Setup
```bash
# Clone repository
git clone https://github.com/SpectraAlpha/Ocean-ML.git
cd Ocean-ML

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 mypy

# Setup database
python setup.py

# Run tests
pytest
```

## Code Style Guidelines

### Python Style
- Follow PEP 8
- Use Black for formatting: `black backend/`
- Use type hints where appropriate
- Maximum line length: 100 characters

### Documentation
- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update API docs when adding endpoints
- Keep README.md up to date

### Example Docstring
```python
def train_model(model_id: str, epochs: int) -> Dict[str, Any]:
    """
    Train a registered ML model.
    
    Args:
        model_id: Unique identifier of the model
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing training results and metrics
        
    Raises:
        ValueError: If model_id is invalid
        RuntimeError: If training fails
    """
    pass
```

### Commit Messages
- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep first line under 72 characters
- Add detailed description if needed

Good examples:
```
Add water quality assessment model
Fix database connection pooling issue
Update API documentation for new endpoints
```

## Project Structure

```
Ocean-ML/
├── backend/
│   ├── api/          # API endpoints
│   ├── config/       # Configuration
│   ├── datastore/    # Data management
│   ├── models/       # ML models
│   ├── training/     # Training pipelines
│   └── utils/        # Utilities
├── tests/            # Test files
├── docs/             # Documentation
└── examples/         # Example scripts
```

## Adding New Features

### Adding a New ML Model

1. Create model file in `backend/models/`:
   ```python
   # backend/models/new_model.py
   from ..training.pipeline import TrainingPipeline
   import torch.nn as nn
   
   class NewModel(nn.Module):
       def __init__(self):
           super().__init__()
           # Define layers
   
   class NewModelPipeline(TrainingPipeline):
       def build_model(self):
           return NewModel()
       
       def prepare_data(self):
           # Implement data loading
           pass
   ```

2. Add model type to `backend/models/registry.py`:
   ```python
   class ModelType(str, Enum):
       # ... existing types
       NEW_MODEL_TYPE = "new_model_type"
   ```

3. Add tests in `tests/test_new_model.py`

4. Update documentation

### Adding a New API Endpoint

1. Add endpoint to `backend/api/main.py`:
   ```python
   @app.get("/api/new-endpoint")
   async def new_endpoint():
       """Endpoint description"""
       return {"result": "data"}
   ```

2. Add tests in `tests/test_api.py`

3. Update `docs/API.md`

### Adding a New Data Source

1. Add to `backend/datastore/models.py`:
   ```python
   class DataSource(str, Enum):
       # ... existing sources
       NEW_SOURCE = "new_source"
   ```

2. Update ingestion logic if needed

3. Add tests

## Testing Guidelines

### Writing Tests
- Write tests for all new features
- Aim for >80% code coverage
- Use fixtures for common setup
- Test both success and error cases

### Test Structure
```python
def test_feature_success():
    """Test successful operation"""
    # Arrange
    # Act
    # Assert
    
def test_feature_failure():
    """Test error handling"""
    # Arrange
    # Act
    # Assert raises exception
```

### Running Tests
```bash
# All tests
pytest

# Specific file
pytest tests/test_models.py

# With coverage
pytest --cov=backend tests/

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Documentation

### API Documentation
- Update `docs/API.md` for API changes
- Include request/response examples
- Document all parameters and errors

### Architecture Documentation
- Update `docs/ARCHITECTURE.md` for structural changes
- Include diagrams when helpful
- Explain design decisions

### Code Documentation
- Add docstrings to all public functions/classes
- Include examples in docstrings when helpful
- Keep comments up to date

## Review Process

### Pull Request Checklist
- [ ] Code follows style guidelines
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] Ready for review

### Review Criteria
- Code quality and style
- Test coverage
- Documentation completeness
- Performance impact
- Security considerations
- Backward compatibility

## Getting Help

- **Questions**: Create a GitHub issue with "Question:" prefix
- **Discussion**: Use GitHub Discussions
- **Bugs**: Create a GitHub issue with bug template
- **Features**: Create a GitHub issue with feature template

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project README

Thank you for contributing to Ocean-ML! 🌊
