#!/bin/bash

# Development utility script for Face Verification API
# This script provides common development tasks

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup development environment
setup_dev() {
    print_info "Setting up development environment..."
    
    # Check Python version
    if ! command_exists python; then
        print_error "Python is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $python_version"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    print_info "Installing dependencies..."
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    
    # Copy environment file if it doesn't exist
    if [ ! -f ".env" ]; then
        print_info "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please update .env file with your configuration"
    fi
    
    # Install pre-commit hooks
    print_info "Installing pre-commit hooks..."
    pre-commit install
    
    print_success "Development environment setup complete!"
    print_info "Don't forget to update your .env file with appropriate values"
}

# Function to run code quality checks
check_quality() {
    print_info "Running code quality checks..."
    
    # Format check
    print_info "Checking code formatting with black..."
    black --check --diff .
    
    # Linting
    print_info "Running flake8 linting..."
    flake8 .
    
    # Type checking
    print_info "Running mypy type checking..."
    mypy domain/ application/ infrastructure/ presentation/ --ignore-missing-imports
    
    # Security scan
    print_info "Running security scan with bandit..."
    bandit -r . -ll
    
    # Dependency check
    print_info "Checking dependencies for security issues..."
    safety check
    
    print_success "All quality checks passed!"
}

# Function to format code
format_code() {
    print_info "Formatting code with black..."
    black .
    
    print_info "Sorting imports with isort..."
    isort .
    
    print_success "Code formatting complete!"
}

# Function to run tests
run_tests() {
    local test_type=${1:-"all"}
    
    case $test_type in
        "unit")
            print_info "Running unit tests..."
            pytest tests/unit/ -v
            ;;
        "integration")
            print_info "Running integration tests..."
            pytest tests/integration/ -v
            ;;
        "performance")
            print_info "Running performance tests..."
            locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 60s --host http://localhost:8000
            ;;
        "coverage")
            print_info "Running tests with coverage..."
            pytest tests/ --cov=domain --cov=application --cov=infrastructure --cov=presentation --cov-report=html --cov-report=term
            ;;
        "all"|*)
            print_info "Running all tests..."
            pytest tests/ -v --cov=domain --cov=application --cov=infrastructure --cov=presentation --cov-report=html
            ;;
    esac
    
    print_success "Tests completed!"
}

# Function to start development server
start_dev() {
    print_info "Starting development server..."
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
}

# Function to build Docker image
build_docker() {
    print_info "Building Docker image..."
    docker build -t face-verification-api:latest .
    print_success "Docker image built successfully!"
}

# Function to start Docker services
start_docker() {
    print_info "Starting Docker services..."
    docker-compose up -d
    print_success "Docker services started!"
    print_info "API available at: http://localhost:8000"
    print_info "API docs available at: http://localhost:8000/docs"
}

# Function to stop Docker services
stop_docker() {
    print_info "Stopping Docker services..."
    docker-compose down
    print_success "Docker services stopped!"
}

# Function to view Docker logs
logs_docker() {
    docker-compose logs -f face-verification-api
}

# Function to run database migrations
migrate_db() {
    print_info "Running database migrations..."
    # Add your migration commands here
    print_success "Database migrations complete!"
}

# Function to generate documentation
generate_docs() {
    print_info "Generating documentation..."
    
    # Generate API documentation
    python -c "
import json
from main import app
from fastapi.openapi.utils import get_openapi

openapi_schema = get_openapi(
    title=app.title,
    version=app.version,
    description=app.description,
    routes=app.routes,
)

with open('docs/openapi.json', 'w') as f:
    json.dump(openapi_schema, f, indent=2)
"
    
    print_success "Documentation generated!"
}

# Function to clean up temporary files
cleanup() {
    print_info "Cleaning up temporary files..."
    
    # Remove Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove test artifacts
    rm -rf .pytest_cache/ htmlcov/ .coverage coverage.xml junit/ 2>/dev/null || true
    
    # Remove temporary files
    rm -rf temp/ logs/*.log 2>/dev/null || true
    
    print_success "Cleanup complete!"
}

# Function to show help
show_help() {
    echo "Face Verification API Development Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  setup          Setup development environment"
    echo "  check          Run code quality checks"
    echo "  format         Format code with black and isort"
    echo "  test [type]    Run tests (unit|integration|performance|coverage|all)"
    echo "  dev            Start development server"
    echo "  docker-build   Build Docker image"
    echo "  docker-start   Start Docker services"
    echo "  docker-stop    Stop Docker services"
    echo "  docker-logs    View Docker logs"
    echo "  migrate        Run database migrations"
    echo "  docs           Generate documentation"
    echo "  cleanup        Clean up temporary files"
    echo "  help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup              # Setup development environment"
    echo "  $0 test unit          # Run unit tests only"
    echo "  $0 test coverage      # Run tests with coverage"
    echo "  $0 docker-start       # Start all services with Docker"
}

# Main script logic
case "${1:-help}" in
    "setup")
        setup_dev
        ;;
    "check")
        check_quality
        ;;
    "format")
        format_code
        ;;
    "test")
        run_tests "$2"
        ;;
    "dev")
        start_dev
        ;;
    "docker-build")
        build_docker
        ;;
    "docker-start")
        start_docker
        ;;
    "docker-stop")
        stop_docker
        ;;
    "docker-logs")
        logs_docker
        ;;
    "migrate")
        migrate_db
        ;;
    "docs")
        generate_docs
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_help
        ;;
esac
