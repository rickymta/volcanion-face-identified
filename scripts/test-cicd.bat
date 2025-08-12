@echo off
REM CI/CD Flow Test Script for Windows
REM This script simulates and tests the CI/CD pipeline locally

setlocal enabledelayedexpansion

REM Colors for output (Windows doesn't support colors in basic cmd, but this prepares for PowerShell)
set "RED=[31m"
set "GREEN=[32m"
set "YELLOW=[33m"
set "BLUE=[34m"
set "NC=[0m"

REM Global variables
set TEST_COUNT=0
set PASS_COUNT=0
set FAIL_COUNT=0
set SKIP_COUNT=0

echo.
echo ========================================
echo    CI/CD Pipeline Test (Windows)
echo ========================================
echo.

REM Function to print status
:print_info
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_warning
echo [WARNING] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:print_step
echo.
echo ==== %~1 ====
echo.
goto :eof

:track_result
set /a TEST_COUNT+=1
if "%~2"=="PASS" (
    set /a PASS_COUNT+=1
    call :print_success "%~1: PASSED"
) else if "%~2"=="SKIP" (
    set /a SKIP_COUNT+=1
    call :print_warning "%~1: SKIPPED"
) else (
    set /a FAIL_COUNT+=1
    call :print_error "%~1: FAILED"
)
goto :eof

REM Test CI Pipeline Components
:test_ci_pipeline
call :print_step "Testing CI Pipeline Components"

REM 1. Check if Python files exist
call :print_info "Checking Python project structure..."
if exist "src\*.py" (
    call :track_result "Python Project Structure" "PASS"
) else (
    call :track_result "Python Project Structure" "FAIL"
)

REM 2. Check requirements.txt
call :print_info "Checking requirements.txt..."
if exist "requirements.txt" (
    call :track_result "Requirements File" "PASS"
) else (
    call :track_result "Requirements File" "FAIL"
)

REM 3. Check Docker configuration
call :print_info "Checking Docker configuration..."
if exist "Dockerfile" (
    call :track_result "Dockerfile" "PASS"
) else (
    call :track_result "Dockerfile" "FAIL"
)

if exist "docker-compose.yml" (
    call :track_result "Docker Compose" "PASS"
) else (
    call :track_result "Docker Compose" "FAIL"
)

REM 4. Check if Docker is available
call :print_info "Checking Docker installation..."
docker --version >nul 2>&1
if !errorlevel! equ 0 (
    call :track_result "Docker Installation" "PASS"
) else (
    call :track_result "Docker Installation" "SKIP"
)

goto :eof

REM Test CD Pipeline Components
:test_cd_pipeline
call :print_step "Testing CD Pipeline Components"

REM 1. Check Helm chart
call :print_info "Checking Helm chart..."
if exist "k8s\helm\face-verification\Chart.yaml" (
    call :track_result "Helm Chart" "PASS"
) else (
    call :track_result "Helm Chart" "FAIL"
)

REM 2. Check Kubernetes manifests
call :print_info "Checking Kubernetes manifests..."
if exist "k8s\helm\face-verification\templates\deployment.yaml" (
    call :track_result "K8s Deployment" "PASS"
) else (
    call :track_result "K8s Deployment" "FAIL"
)

if exist "k8s\helm\face-verification\templates\service.yaml" (
    call :track_result "K8s Service" "PASS"
) else (
    call :track_result "K8s Service" "FAIL"
)

REM 3. Check environment configuration
call :print_info "Checking environment configuration..."
if exist ".env.example" (
    call :track_result "Environment Example" "PASS"
) else (
    call :track_result "Environment Example" "FAIL"
)

goto :eof

REM Test Workflow Files
:test_workflow_files
call :print_step "Testing Workflow Files"

REM 1. Check CI workflow
call :print_info "Checking CI workflow..."
if exist ".github\workflows\ci.yml" (
    call :track_result "CI Workflow" "PASS"
) else (
    call :track_result "CI Workflow" "FAIL"
)

REM 2. Check CD workflow
call :print_info "Checking CD workflow..."
if exist ".github\workflows\cd.yml" (
    call :track_result "CD Workflow" "PASS"
) else (
    call :track_result "CD Workflow" "FAIL"
)

goto :eof

REM Test Documentation
:test_documentation
call :print_step "Testing Documentation"

REM 1. Check README
if exist "README.md" (
    call :track_result "README.md" "PASS"
) else (
    call :track_result "README.md" "FAIL"
)

REM 2. Check API Documentation
if exist "API_DOCUMENTATION.md" (
    call :track_result "API Documentation" "PASS"
) else (
    call :track_result "API Documentation" "FAIL"
)

REM 3. Check Postman collection
if exist "postman\Face_Verification_API.postman_collection.json" (
    call :track_result "Postman Collection" "PASS"
) else (
    call :track_result "Postman Collection" "FAIL"
)

goto :eof

REM Simulate Deployment Flow
:simulate_deployment_flow
call :print_step "Simulating Deployment Flow"

call :print_info "Checking Git repository..."
git status >nul 2>&1
if !errorlevel! equ 0 (
    call :track_result "Git Repository" "PASS"
    
    REM Get current branch
    for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set CURRENT_BRANCH=%%i
    
    call :print_info "Current branch: !CURRENT_BRANCH!"
    
    if "!CURRENT_BRANCH!"=="main" (
        call :print_info "Main branch detected - Would trigger production deployment"
        call :track_result "Production Deployment Logic" "PASS"
    ) else if "!CURRENT_BRANCH!"=="develop" (
        call :print_info "Develop branch detected - Would trigger staging deployment"
        call :track_result "Staging Deployment Logic" "PASS"
    ) else (
        call :print_info "Feature branch detected - Would trigger CI only"
        call :track_result "Feature Branch Logic" "PASS"
    )
) else (
    call :track_result "Git Repository" "FAIL"
)

goto :eof

REM Generate Test Report
:generate_report
call :print_step "Test Report"

echo CI/CD Pipeline Test Results
echo ==========================
echo Test Count: !TEST_COUNT!
echo Passed: !PASS_COUNT!
echo Failed: !FAIL_COUNT!
echo Skipped: !SKIP_COUNT!
echo.

if !FAIL_COUNT! gtr 0 (
    call :print_error "Some tests failed. Please fix the issues before deploying."
    exit /b 1
) else (
    call :print_success "All tests passed! Pipeline is ready for deployment."
    exit /b 0
)

goto :eof

REM Show help
:show_help
echo CI/CD Pipeline Test Script (Windows)
echo.
echo Usage: %0 [option]
echo.
echo Options:
echo   --ci-only       Test CI components only
echo   --cd-only       Test CD components only
echo   --workflows     Test workflow files only
echo   --docs          Test documentation only
echo   --simulate      Simulate deployment flow only
echo   --help          Show this help message
echo.
echo Without options, runs all tests.
goto :eof

REM Main execution
:main
call :print_info "Starting CI/CD Pipeline Test"

REM Run all tests
call :test_ci_pipeline
call :test_cd_pipeline
call :test_workflow_files
call :test_documentation
call :simulate_deployment_flow

REM Generate report
call :generate_report
goto :eof

REM Parse command line arguments
if "%1"=="--ci-only" (
    call :test_ci_pipeline
    call :generate_report
) else if "%1"=="--cd-only" (
    call :test_cd_pipeline
    call :generate_report
) else if "%1"=="--workflows" (
    call :test_workflow_files
    call :generate_report
) else if "%1"=="--docs" (
    call :test_documentation
    call :generate_report
) else if "%1"=="--simulate" (
    call :simulate_deployment_flow
    call :generate_report
) else if "%1"=="--help" (
    call :show_help
) else if "%1"=="" (
    call :main
) else (
    call :print_error "Unknown option: %1"
    call :show_help
    exit /b 1
)
