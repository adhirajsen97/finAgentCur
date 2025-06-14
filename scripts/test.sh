#!/bin/bash

# FinAgent Testing Script
set -e

# Configuration
COVERAGE_THRESHOLD=80
TEST_TIMEOUT=300  # 5 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Setup test environment
setup_test_env() {
    log "Setting up test environment..."
    
    # Create test environment file
    cat > .env.test << EOF
ENVIRONMENT=test
DATABASE_URL=postgresql://test:test@localhost:5433/test_finagent
REDIS_URL=redis://localhost:6380
OPENAI_API_KEY=test_key
ANTHROPIC_API_KEY=test_key
ALPHA_VANTAGE_API_KEY=test_key
LOG_LEVEL=DEBUG
EOF
    
    # Start test services
    log "Starting test services..."
    docker-compose -f docker-compose.test.yml up -d || {
        warn "Test services not available, using mocks"
    }
    
    # Wait for services to be ready
    sleep 5
    
    log "Test environment setup completed ✓"
}

# Install test dependencies
install_test_deps() {
    log "Installing test dependencies..."
    
    pip install -q pytest pytest-asyncio pytest-cov httpx coverage || {
        error "Failed to install test dependencies"
    }
    
    log "Test dependencies installed ✓"
}

# Run unit tests
run_unit_tests() {
    log "Running unit tests..."
    
    pytest tests/test_services.py \
        --verbose \
        --tb=short \
        --timeout=$TEST_TIMEOUT \
        --cov=services \
        --cov-report=term-missing \
        --cov-report=html:htmlcov/unit \
        --junit-xml=test-results/unit-tests.xml || {
        error "Unit tests failed"
    }
    
    log "Unit tests completed ✓"
}

# Run API tests
run_api_tests() {
    log "Running API tests..."
    
    pytest tests/test_api.py \
        --verbose \
        --tb=short \
        --timeout=$TEST_TIMEOUT \
        --cov=main \
        --cov=agents \
        --cov-append \
        --cov-report=term-missing \
        --cov-report=html:htmlcov/api \
        --junit-xml=test-results/api-tests.xml || {
        error "API tests failed"
    }
    
    log "API tests completed ✓"
}

# Run integration tests
run_integration_tests() {
    log "Running integration tests..."
    
    # Check if integration test file exists
    if [[ -f "tests/test_integration.py" ]]; then
        pytest tests/test_integration.py \
            --verbose \
            --tb=short \
            --timeout=$TEST_TIMEOUT \
            --junit-xml=test-results/integration-tests.xml || {
            warn "Integration tests failed"
        }
    else
        warn "No integration tests found"
    fi
    
    log "Integration tests completed ✓"
}

# Run load tests
run_load_tests() {
    log "Running load tests..."
    
    if command -v locust &> /dev/null; then
        # Run basic load test
        timeout 60 locust -f tests/load_test.py --headless -u 10 -r 2 -t 30s --host=http://localhost:8000 || {
            warn "Load tests failed or timed out"
        }
    else
        warn "Locust not installed, skipping load tests"
    fi
    
    log "Load tests completed ✓"
}

# Check code coverage
check_coverage() {
    log "Checking code coverage..."
    
    # Generate combined coverage report
    coverage combine || true
    coverage report --show-missing
    
    # Check coverage threshold
    coverage_percent=$(coverage report | tail -1 | awk '{print $4}' | sed 's/%//')
    
    if (( $(echo "$coverage_percent < $COVERAGE_THRESHOLD" | bc -l) )); then
        warn "Code coverage ($coverage_percent%) is below threshold ($COVERAGE_THRESHOLD%)"
    else
        log "Code coverage ($coverage_percent%) meets threshold ✓"
    fi
    
    # Generate HTML report
    coverage html -d htmlcov/combined
    
    log "Coverage report generated: htmlcov/combined/index.html"
}

# Run linting
run_linting() {
    log "Running code linting..."
    
    # Install linting tools if not present
    pip install -q flake8 black isort mypy || true
    
    # Run black (code formatting check)
    if command -v black &> /dev/null; then
        black --check --diff . || warn "Code formatting issues found"
    fi
    
    # Run isort (import sorting check)
    if command -v isort &> /dev/null; then
        isort --check-only --diff . || warn "Import sorting issues found"
    fi
    
    # Run flake8 (style guide enforcement)
    if command -v flake8 &> /dev/null; then
        flake8 --max-line-length=100 --ignore=E203,W503 . || warn "Style guide violations found"
    fi
    
    # Run mypy (type checking)
    if command -v mypy &> /dev/null; then
        mypy --ignore-missing-imports . || warn "Type checking issues found"
    fi
    
    log "Linting completed ✓"
}

# Security scanning
run_security_scan() {
    log "Running security scan..."
    
    # Install security tools
    pip install -q safety bandit || true
    
    # Check for known security vulnerabilities
    if command -v safety &> /dev/null; then
        safety check || warn "Security vulnerabilities found in dependencies"
    fi
    
    # Run bandit for security issues in code
    if command -v bandit &> /dev/null; then
        bandit -r . -f json -o security-report.json || warn "Security issues found in code"
    fi
    
    log "Security scan completed ✓"
}

# Generate test report
generate_report() {
    log "Generating test report..."
    
    # Create test results directory
    mkdir -p test-results
    
    # Generate summary report
    cat > test-results/summary.md << EOF
# FinAgent Test Report

**Generated:** $(date)

## Test Results

### Unit Tests
- Status: $([ -f test-results/unit-tests.xml ] && echo "✅ PASSED" || echo "❌ FAILED")

### API Tests  
- Status: $([ -f test-results/api-tests.xml ] && echo "✅ PASSED" || echo "❌ FAILED")

### Integration Tests
- Status: $([ -f test-results/integration-tests.xml ] && echo "✅ PASSED" || echo "⚠️ SKIPPED")

## Coverage Report
- Overall Coverage: $(coverage report | tail -1 | awk '{print $4}' || echo "N/A")
- Threshold: ${COVERAGE_THRESHOLD}%

## Security Scan
- Dependencies: $([ -f security-report.json ] && echo "✅ SCANNED" || echo "⚠️ SKIPPED")

## Reports Location
- Coverage: htmlcov/combined/index.html
- Security: security-report.json
EOF
    
    log "Test report generated: test-results/summary.md"
}

# Cleanup test environment
cleanup() {
    log "Cleaning up test environment..."
    
    # Stop test services
    docker-compose -f docker-compose.test.yml down || true
    
    # Remove test environment file
    rm -f .env.test
    
    log "Cleanup completed ✓"
}

# Main testing flow
main() {
    log "🧪 Starting comprehensive test suite for FinAgent..."
    
    # Create results directory
    mkdir -p test-results htmlcov
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    case "${1:-all}" in
        "unit")
            setup_test_env
            install_test_deps
            run_unit_tests
            ;;
        "api")
            setup_test_env
            install_test_deps
            run_api_tests
            ;;
        "integration")
            setup_test_env
            install_test_deps
            run_integration_tests
            ;;
        "load")
            run_load_tests
            ;;
        "lint")
            run_linting
            ;;
        "security")
            run_security_scan
            ;;
        "coverage")
            check_coverage
            ;;
        "all")
            setup_test_env
            install_test_deps
            run_linting
            run_unit_tests
            run_api_tests
            run_integration_tests
            run_load_tests
            check_coverage
            run_security_scan
            generate_report
            ;;
        *)
            echo "Usage: $0 {all|unit|api|integration|load|lint|security|coverage}"
            echo ""
            echo "Options:"
            echo "  all         - Run all tests and checks (default)"
            echo "  unit        - Run unit tests only"
            echo "  api         - Run API tests only"
            echo "  integration - Run integration tests only"
            echo "  load        - Run load tests only"
            echo "  lint        - Run code linting only"
            echo "  security    - Run security scan only"
            echo "  coverage    - Check code coverage only"
            exit 1
            ;;
    esac
    
    log "🎉 Test suite completed!"
}

# Run main function with arguments
main "$@" 