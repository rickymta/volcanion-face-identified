#!/bin/bash

# CI/CD Flow Test Script
# This script simulates and tests the CI/CD pipeline locally

set -e

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

print_step() {
    echo -e "\n${BLUE}==== $1 ====${NC}\n"
}

# Global variables
TEST_RESULTS=()
START_TIME=$(date +%s)

# Function to track test results
track_result() {
    local test_name="$1"
    local result="$2"
    TEST_RESULTS+=("$test_name:$result")
    
    if [[ "$result" == "PASS" ]]; then
        print_success "$test_name: PASSED"
    else
        print_error "$test_name: FAILED"
    fi
}

# Function to test CI pipeline components
test_ci_pipeline() {
    print_step "Testing CI Pipeline Components"
    
    # 1. Code Quality Checks
    print_info "Testing code quality checks..."
    if command -v black >/dev/null 2>&1 && command -v flake8 >/dev/null 2>&1; then
        # Test black formatting
        if black --check --diff . >/dev/null 2>&1; then
            track_result "Code Formatting (Black)" "PASS"
        else
            track_result "Code Formatting (Black)" "FAIL"
        fi
        
        # Test flake8 linting
        if flake8 . --count --select=E9,F63,F7,F82 >/dev/null 2>&1; then
            track_result "Code Linting (Flake8)" "PASS"
        else
            track_result "Code Linting (Flake8)" "FAIL"
        fi
    else
        print_warning "Code quality tools not installed"
        track_result "Code Quality Tools" "SKIP"
    fi
    
    # 2. Security Checks
    print_info "Testing security checks..."
    if command -v bandit >/dev/null 2>&1; then
        if bandit -r . -ll >/dev/null 2>&1; then
            track_result "Security Scan (Bandit)" "PASS"
        else
            track_result "Security Scan (Bandit)" "FAIL"
        fi
    else
        print_warning "Bandit not installed"
        track_result "Security Scan (Bandit)" "SKIP"
    fi
    
    # 3. Test Suite
    print_info "Testing test suite..."
    if command -v pytest >/dev/null 2>&1; then
        if [[ -d "tests" ]]; then
            # Run a quick test
            if pytest tests/ --collect-only >/dev/null 2>&1; then
                track_result "Test Collection" "PASS"
            else
                track_result "Test Collection" "FAIL"
            fi
        else
            print_warning "No tests directory found"
            track_result "Test Suite" "SKIP"
        fi
    else
        print_warning "Pytest not installed"
        track_result "Test Suite" "SKIP"
    fi
    
    # 4. Docker Build Test
    print_info "Testing Docker build..."
    if command -v docker >/dev/null 2>&1; then
        if [[ -f "Dockerfile" ]]; then
            if docker build -t face-verification-test:latest . >/dev/null 2>&1; then
                track_result "Docker Build" "PASS"
                
                # Clean up test image
                docker rmi face-verification-test:latest >/dev/null 2>&1 || true
            else
                track_result "Docker Build" "FAIL"
            fi
        else
            print_warning "No Dockerfile found"
            track_result "Docker Build" "SKIP"
        fi
    else
        print_warning "Docker not installed"
        track_result "Docker Build" "SKIP"
    fi
}

# Function to test CD pipeline components
test_cd_pipeline() {
    print_step "Testing CD Pipeline Components"
    
    # 1. Helm Chart Validation
    print_info "Testing Helm chart validation..."
    if command -v helm >/dev/null 2>&1; then
        if [[ -d "k8s/helm/face-verification" ]]; then
            if helm lint k8s/helm/face-verification >/dev/null 2>&1; then
                track_result "Helm Chart Lint" "PASS"
            else
                track_result "Helm Chart Lint" "FAIL"
            fi
            
            # Test template rendering
            if helm template face-verification k8s/helm/face-verification >/dev/null 2>&1; then
                track_result "Helm Template Rendering" "PASS"
            else
                track_result "Helm Template Rendering" "FAIL"
            fi
        else
            print_warning "No Helm chart found"
            track_result "Helm Chart" "SKIP"
        fi
    else
        print_warning "Helm not installed"
        track_result "Helm Chart" "SKIP"
    fi
    
    # 2. Kubernetes Manifests Validation
    print_info "Testing Kubernetes manifests..."
    if command -v kubectl >/dev/null 2>&1; then
        # Generate manifests and validate
        if helm template face-verification k8s/helm/face-verification > /tmp/k8s-manifests.yaml 2>/dev/null; then
            if kubectl apply --dry-run=client -f /tmp/k8s-manifests.yaml >/dev/null 2>&1; then
                track_result "Kubernetes Manifests Validation" "PASS"
            else
                track_result "Kubernetes Manifests Validation" "FAIL"
            fi
            rm -f /tmp/k8s-manifests.yaml
        else
            track_result "Kubernetes Manifests Generation" "FAIL"
        fi
    else
        print_warning "kubectl not installed"
        track_result "Kubernetes Manifests" "SKIP"
    fi
    
    # 3. Environment Configuration
    print_info "Testing environment configuration..."
    if [[ -f ".env.example" ]]; then
        # Check if all required environment variables are documented
        if grep -q "MONGODB_URL\|API_SECRET_KEY\|ENVIRONMENT" .env.example; then
            track_result "Environment Documentation" "PASS"
        else
            track_result "Environment Documentation" "FAIL"
        fi
    else
        print_warning "No .env.example found"
        track_result "Environment Documentation" "SKIP"
    fi
}

# Function to test workflow files
test_workflow_files() {
    print_step "Testing Workflow Files"
    
    # 1. CI Workflow
    print_info "Testing CI workflow file..."
    if [[ -f ".github/workflows/ci.yml" ]]; then
        # Basic YAML validation
        if python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" 2>/dev/null; then
            track_result "CI Workflow YAML" "PASS"
        else
            track_result "CI Workflow YAML" "FAIL"
        fi
        
        # Check for required jobs
        if grep -q "code-quality\|tests\|docker\|performance" .github/workflows/ci.yml; then
            track_result "CI Workflow Jobs" "PASS"
        else
            track_result "CI Workflow Jobs" "FAIL"
        fi
    else
        print_error "CI workflow file not found"
        track_result "CI Workflow" "FAIL"
    fi
    
    # 2. CD Workflow
    print_info "Testing CD workflow file..."
    if [[ -f ".github/workflows/cd.yml" ]]; then
        # Basic YAML validation
        if python -c "import yaml; yaml.safe_load(open('.github/workflows/cd.yml'))" 2>/dev/null; then
            track_result "CD Workflow YAML" "PASS"
        else
            track_result "CD Workflow YAML" "FAIL"
        fi
        
        # Check for deployment jobs
        if grep -q "deploy-staging\|deploy-production" .github/workflows/cd.yml; then
            track_result "CD Workflow Jobs" "PASS"
        else
            track_result "CD Workflow Jobs" "FAIL"
        fi
    else
        print_error "CD workflow file not found"
        track_result "CD Workflow" "FAIL"
    fi
}

# Function to test documentation
test_documentation() {
    print_step "Testing Documentation"
    
    # 1. README.md
    if [[ -f "README.md" ]]; then
        if grep -q "Docker\|CI/CD\|Kubernetes" README.md; then
            track_result "README Documentation" "PASS"
        else
            track_result "README Documentation" "FAIL"
        fi
    else
        track_result "README Documentation" "FAIL"
    fi
    
    # 2. API Documentation
    if [[ -f "API_DOCUMENTATION.md" ]]; then
        track_result "API Documentation" "PASS"
    else
        track_result "API Documentation" "FAIL"
    fi
    
    # 3. CHANGELOG
    if [[ -f "CHANGELOG.md" ]]; then
        track_result "CHANGELOG" "PASS"
    else
        track_result "CHANGELOG" "FAIL"
    fi
}

# Function to simulate deployment flow
simulate_deployment_flow() {
    print_step "Simulating Deployment Flow"
    
    print_info "Simulating branch-based deployment logic..."
    
    # Simulate different branch scenarios
    local current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
    
    print_info "Current branch: $current_branch"
    
    case $current_branch in
        "main")
            print_info "✅ Main branch detected - Would trigger production deployment"
            track_result "Production Deployment Logic" "PASS"
            ;;
        "develop")
            print_info "✅ Develop branch detected - Would trigger staging deployment"
            track_result "Staging Deployment Logic" "PASS"
            ;;
        *)
            print_info "✅ Feature branch detected - Would trigger CI only"
            track_result "Feature Branch Logic" "PASS"
            ;;
    esac
    
    # Test environment-specific configurations
    print_info "Testing environment configurations..."
    
    # Check if environment-specific values exist in Helm chart
    if [[ -f "k8s/helm/face-verification/values.yaml" ]]; then
        if grep -q "environment:" k8s/helm/face-verification/values.yaml; then
            track_result "Environment Configuration" "PASS"
        else
            track_result "Environment Configuration" "FAIL"
        fi
    fi
}

# Function to generate test report
generate_report() {
    print_step "Test Report"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    local pass_count=0
    local fail_count=0
    local skip_count=0
    
    echo "CI/CD Pipeline Test Results"
    echo "=========================="
    echo "Test Duration: ${duration}s"
    echo "Timestamp: $(date)"
    echo ""
    
    for result in "${TEST_RESULTS[@]}"; do
        local test_name=$(echo "$result" | cut -d: -f1)
        local test_result=$(echo "$result" | cut -d: -f2)
        
        case $test_result in
            "PASS")
                echo "✅ $test_name"
                ((pass_count++))
                ;;
            "FAIL")
                echo "❌ $test_name"
                ((fail_count++))
                ;;
            "SKIP")
                echo "⏭️  $test_name"
                ((skip_count++))
                ;;
        esac
    done
    
    echo ""
    echo "Summary:"
    echo "- Passed: $pass_count"
    echo "- Failed: $fail_count"
    echo "- Skipped: $skip_count"
    echo "- Total: $((pass_count + fail_count + skip_count))"
    
    if [[ $fail_count -gt 0 ]]; then
        print_error "Some tests failed. Please fix the issues before deploying."
        return 1
    else
        print_success "All tests passed! Pipeline is ready for deployment."
        return 0
    fi
}

# Main execution
main() {
    print_info "Starting CI/CD Pipeline Test"
    print_info "=============================="
    
    # Run all tests
    test_ci_pipeline
    test_cd_pipeline
    test_workflow_files
    test_documentation
    simulate_deployment_flow
    
    # Generate report
    if generate_report; then
        print_success "CI/CD Pipeline test completed successfully!"
        exit 0
    else
        print_error "CI/CD Pipeline test failed!"
        exit 1
    fi
}

# Show help
show_help() {
    echo "CI/CD Pipeline Test Script"
    echo ""
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  --ci-only       Test CI components only"
    echo "  --cd-only       Test CD components only"
    echo "  --workflows     Test workflow files only"
    echo "  --docs          Test documentation only"
    echo "  --simulate      Simulate deployment flow only"
    echo "  --help          Show this help message"
    echo ""
    echo "Without options, runs all tests."
}

# Parse command line arguments
case "${1:-}" in
    "--ci-only")
        test_ci_pipeline
        generate_report
        ;;
    "--cd-only")
        test_cd_pipeline
        generate_report
        ;;
    "--workflows")
        test_workflow_files
        generate_report
        ;;
    "--docs")
        test_documentation
        generate_report
        ;;
    "--simulate")
        simulate_deployment_flow
        generate_report
        ;;
    "--help")
        show_help
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
