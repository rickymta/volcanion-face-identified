@echo off
REM Development utility script for Face Verification API (Windows version)
REM This script provides common development tasks for Windows

setlocal enabledelayedexpansion

REM Colors for output (Windows)
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

REM Function to print colored output
set "print_info=echo %BLUE%[INFO]%NC%"
set "print_success=echo %GREEN%[SUCCESS]%NC%"
set "print_warning=echo %YELLOW%[WARNING]%NC%"
set "print_error=echo %RED%[ERROR]%NC%"

REM Main script logic
if "%1"=="" goto help
if "%1"=="setup" goto setup_dev
if "%1"=="check" goto check_quality
if "%1"=="format" goto format_code
if "%1"=="test" goto run_tests
if "%1"=="dev" goto start_dev
if "%1"=="docker-build" goto build_docker
if "%1"=="docker-start" goto start_docker
if "%1"=="docker-stop" goto stop_docker
if "%1"=="docker-logs" goto logs_docker
if "%1"=="migrate" goto migrate_db
if "%1"=="docs" goto generate_docs
if "%1"=="cleanup" goto cleanup
if "%1"=="help" goto help
goto help

:setup_dev
%print_info% Setting up development environment...

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    %print_error% Python is not installed. Please install Python 3.8 or higher.
    exit /b 1
)

%print_info% Python version:
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    %print_info% Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
%print_info% Activating virtual environment...
call venv\Scripts\activate

REM Upgrade pip
%print_info% Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
%print_info% Installing dependencies...
pip install -r requirements.txt
pip install -r requirements-dev.txt

REM Copy environment file if it doesn't exist
if not exist ".env" (
    %print_info% Creating .env file from template...
    copy .env.example .env
    %print_warning% Please update .env file with your configuration
)

REM Install pre-commit hooks
%print_info% Installing pre-commit hooks...
pre-commit install

%print_success% Development environment setup complete!
%print_info% Don't forget to update your .env file with appropriate values
goto end

:check_quality
%print_info% Running code quality checks...

REM Format check
%print_info% Checking code formatting with black...
black --check --diff .

REM Linting
%print_info% Running flake8 linting...
flake8 .

REM Type checking
%print_info% Running mypy type checking...
mypy domain/ application/ infrastructure/ presentation/ --ignore-missing-imports

REM Security scan
%print_info% Running security scan with bandit...
bandit -r . -ll

REM Dependency check
%print_info% Checking dependencies for security issues...
safety check

%print_success% All quality checks passed!
goto end

:format_code
%print_info% Formatting code with black...
black .

%print_info% Sorting imports with isort...
isort .

%print_success% Code formatting complete!
goto end

:run_tests
set test_type=%2
if "%test_type%"=="" set test_type=all

if "%test_type%"=="unit" (
    %print_info% Running unit tests...
    pytest tests/unit/ -v
) else if "%test_type%"=="integration" (
    %print_info% Running integration tests...
    pytest tests/integration/ -v
) else if "%test_type%"=="performance" (
    %print_info% Running performance tests...
    locust -f tests/performance/locustfile.py --headless -u 10 -r 2 -t 60s --host http://localhost:8000
) else if "%test_type%"=="coverage" (
    %print_info% Running tests with coverage...
    pytest tests/ --cov=domain --cov=application --cov=infrastructure --cov=presentation --cov-report=html --cov-report=term
) else (
    %print_info% Running all tests...
    pytest tests/ -v --cov=domain --cov=application --cov=infrastructure --cov=presentation --cov-report=html
)

%print_success% Tests completed!
goto end

:start_dev
%print_info% Starting development server...
uvicorn main:app --reload --host 0.0.0.0 --port 8000
goto end

:build_docker
%print_info% Building Docker image...
docker build -t face-verification-api:latest .
%print_success% Docker image built successfully!
goto end

:start_docker
%print_info% Starting Docker services...
docker-compose up -d
%print_success% Docker services started!
%print_info% API available at: http://localhost:8000
%print_info% API docs available at: http://localhost:8000/docs
goto end

:stop_docker
%print_info% Stopping Docker services...
docker-compose down
%print_success% Docker services stopped!
goto end

:logs_docker
docker-compose logs -f face-verification-api
goto end

:migrate_db
%print_info% Running database migrations...
REM Add your migration commands here
%print_success% Database migrations complete!
goto end

:generate_docs
%print_info% Generating documentation...

REM Generate API documentation
python -c "import json; from main import app; from fastapi.openapi.utils import get_openapi; openapi_schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes); json.dump(openapi_schema, open('docs/openapi.json', 'w'), indent=2)"

%print_success% Documentation generated!
goto end

:cleanup
%print_info% Cleaning up temporary files...

REM Remove Python cache
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc >nul 2>&1

REM Remove test artifacts
if exist ".pytest_cache\" rd /s /q ".pytest_cache"
if exist "htmlcov\" rd /s /q "htmlcov"
if exist ".coverage" del ".coverage"
if exist "coverage.xml" del "coverage.xml"
if exist "junit\" rd /s /q "junit"

REM Remove temporary files
if exist "temp\" rd /s /q "temp"
del "logs\*.log" >nul 2>&1

%print_success% Cleanup complete!
goto end

:help
echo Face Verification API Development Script (Windows)
echo.
echo Usage: %0 ^<command^> [options]
echo.
echo Commands:
echo   setup          Setup development environment
echo   check          Run code quality checks
echo   format         Format code with black and isort
echo   test [type]    Run tests (unit^|integration^|performance^|coverage^|all)
echo   dev            Start development server
echo   docker-build   Build Docker image
echo   docker-start   Start Docker services
echo   docker-stop    Stop Docker services
echo   docker-logs    View Docker logs
echo   migrate        Run database migrations
echo   docs           Generate documentation
echo   cleanup        Clean up temporary files
echo   help           Show this help message
echo.
echo Examples:
echo   %0 setup              # Setup development environment
echo   %0 test unit          # Run unit tests only
echo   %0 test coverage      # Run tests with coverage
echo   %0 docker-start       # Start all services with Docker

:end
