#!/usr/bin/env python3
"""
Validation script for Ocean ML Platform setup
"""
import sys
import subprocess
import importlib.util
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.10+)")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking required packages...")
    
    required = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "pydantic",
        "torch",
        "torchvision",
        "requests",
        "redis",
        "boto3"
    ]
    
    missing = []
    for package in required:
        spec = importlib.util.find_spec(package)
        if spec is None:
            print(f"  ✗ {package} not installed")
            missing.append(package)
        else:
            print(f"  ✓ {package} installed")
    
    if missing:
        print(f"\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    return True


def check_docker():
    """Check if Docker is installed"""
    print("\nChecking Docker...")
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✓ {result.stdout.strip()}")
            return True
        else:
            print("  ✗ Docker not found")
            return False
    except FileNotFoundError:
        print("  ✗ Docker not installed")
        return False


def check_docker_compose():
    """Check if Docker Compose is installed"""
    print("\nChecking Docker Compose...")
    try:
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✓ {result.stdout.strip()}")
            return True
        else:
            print("  ✗ Docker Compose not found")
            return False
    except FileNotFoundError:
        print("  ✗ Docker Compose not installed")
        return False


def check_project_structure():
    """Check if project structure is correct"""
    print("\nChecking project structure...")
    
    required_dirs = [
        "backend",
        "backend/api",
        "backend/config",
        "backend/datastore",
        "backend/models",
        "backend/training",
        "data",
        "tests",
        "docs",
        "examples"
    ]
    
    required_files = [
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile",
        ".env.example",
        "README.md"
    ]
    
    all_good = True
    
    for directory in required_dirs:
        path = Path(directory)
        if path.exists() and path.is_dir():
            print(f"  ✓ {directory}/")
        else:
            print(f"  ✗ {directory}/ missing")
            all_good = False
    
    for file in required_files:
        path = Path(file)
        if path.exists() and path.is_file():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} missing")
            all_good = False
    
    return all_good


def check_env_file():
    """Check if .env file exists"""
    print("\nChecking environment configuration...")
    env_path = Path(".env")
    example_path = Path(".env.example")
    
    if env_path.exists():
        print("  ✓ .env file exists")
        return True
    elif example_path.exists():
        print("  ⚠ .env file not found (using defaults)")
        print("  → Copy .env.example to .env and configure")
        return True
    else:
        print("  ✗ No .env configuration found")
        return False


def check_api_syntax():
    """Check if main API file has syntax errors"""
    print("\nChecking API syntax...")
    try:
        import py_compile
        py_compile.compile("backend/api/main.py", doraise=True)
        print("  ✓ API syntax valid")
        return True
    except Exception as e:
        print(f"  ✗ API syntax error: {e}")
        return False


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("Ocean ML Platform - Setup Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Docker", check_docker),
        ("Docker Compose", check_docker_compose),
        ("Project Structure", check_project_structure),
        ("Environment Config", check_env_file),
        ("API Syntax", check_api_syntax)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"  ✗ Error checking {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. docker-compose up -d          # Start services")
        print("2. python examples/demo_workflow.py  # Run demo")
        print("3. open http://localhost:8000/docs   # View API docs")
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
