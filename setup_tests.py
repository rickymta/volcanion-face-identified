"""
Comprehensive Test Runner và Configuration
"""

# pytest.ini configuration content
PYTEST_INI = """
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=domain
    --cov=application
    --cov=infrastructure
    --cov=presentation
    --cov-report=term-missing
    --cov-report=html:htmlcov
    --cov-report=xml:coverage.xml
    --cov-fail-under=85
markers =
    unit: Unit tests
    integration: Integration tests
    api: API tests
    ml: Machine Learning tests
    monitoring: Monitoring tests
    slow: Slow running tests
    smoke: Smoke tests for basic functionality
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
"""

# conftest.py content
CONFTEST_PY = '''
"""
Pytest configuration and fixtures
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime
import numpy as np
from PIL import Image
import io

# Test configuration
pytest_plugins = ["pytest_asyncio"]

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_image():
    """Create sample image for testing"""
    # Create a simple RGB image
    image = Image.new('RGB', (640, 480), color='red')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

@pytest.fixture
def sample_document_image():
    """Create sample document image"""
    # Create a document-like image with text area
    image = Image.new('RGB', (800, 600), color='white')
    
    # Add some colored rectangles to simulate text/content
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Simulate text blocks
    draw.rectangle([50, 50, 750, 100], fill='lightgray')  # Header
    draw.rectangle([50, 150, 350, 200], fill='lightblue')  # ID number
    draw.rectangle([50, 250, 550, 300], fill='lightgreen')  # Name
    draw.rectangle([50, 350, 400, 400], fill='lightyellow')  # DOB
    
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

@pytest.fixture
def sample_face_image():
    """Create sample face image"""
    # Create an image with a simple face-like pattern
    image = Image.new('RGB', (300, 400), color='peachpuff')
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Simple face outline
    draw.ellipse([50, 50, 250, 300], fill='peachpuff', outline='black', width=2)
    
    # Eyes
    draw.ellipse([80, 120, 120, 160], fill='white', outline='black')
    draw.ellipse([180, 120, 220, 160], fill='white', outline='black')
    draw.ellipse([90, 130, 110, 150], fill='black')  # Left pupil
    draw.ellipse([190, 130, 210, 150], fill='black')  # Right pupil
    
    # Nose
    draw.polygon([(150, 180), (140, 220), (160, 220)], fill='rosybrown')
    
    # Mouth
    draw.arc([120, 240, 180, 280], 0, 180, fill='black', width=3)
    
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

@pytest.fixture
def mock_cv2():
    """Mock OpenCV for testing"""
    with patch('cv2.imread') as mock_imread, \\
         patch('cv2.cvtColor') as mock_cvtcolor, \\
         patch('cv2.resize') as mock_resize:
        
        # Mock image as numpy array
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        mock_cvtcolor.return_value = mock_image
        mock_resize.return_value = mock_image
        
        yield {
            'imread': mock_imread,
            'cvtColor': mock_cvtcolor,
            'resize': mock_resize,
            'mock_image': mock_image
        }

@pytest.fixture
def mock_pil():
    """Mock PIL for testing"""
    with patch('PIL.Image.open') as mock_open:
        mock_image = Mock()
        mock_image.size = (640, 480)
        mock_image.mode = 'RGB'
        mock_image.format = 'JPEG'
        mock_open.return_value = mock_image
        
        yield {
            'open': mock_open,
            'mock_image': mock_image
        }

@pytest.fixture
def mock_mongodb():
    """Mock MongoDB for testing"""
    mock_collection = Mock()
    mock_db = Mock()
    mock_client = Mock()
    
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection
    
    # Mock common operations
    mock_collection.insert_one.return_value = Mock(inserted_id="test_id")
    mock_collection.find_one.return_value = {"_id": "test_id", "data": "test"}
    mock_collection.find.return_value = [{"_id": "test_id", "data": "test"}]
    mock_collection.update_one.return_value = Mock(modified_count=1)
    mock_collection.delete_one.return_value = Mock(deleted_count=1)
    
    with patch('pymongo.MongoClient', return_value=mock_client):
        yield {
            'client': mock_client,
            'db': mock_db,
            'collection': mock_collection
        }

@pytest.fixture
def mock_ml_models():
    """Mock ML models for testing"""
    mocks = {}
    
    # Mock face detection
    with patch('domain.services.face_detection_service.FaceDetectionService') as mock_face_service:
        mock_face_service.return_value.detect_faces.return_value = {
            "faces": [{"x": 10, "y": 10, "width": 100, "height": 100, "confidence": 0.95}],
            "face_count": 1
        }
        mocks['face_detection'] = mock_face_service
    
    # Mock OCR
    with patch('domain.services.ocr_service.OCRService') as mock_ocr_service:
        mock_ocr_service.return_value.extract_text.return_value = {
            "text": "Sample extracted text",
            "confidence": 0.9,
            "fields": {"id_number": "123456789"}
        }
        mocks['ocr'] = mock_ocr_service
    
    yield mocks

@pytest.fixture
def api_client():
    """FastAPI test client"""
    from fastapi.testclient import TestClient
    from main import app
    
    return TestClient(app)

@pytest.fixture
def sample_api_payload():
    """Sample API payload for testing"""
    return {
        "document_type": "CCCD",
        "confidence_threshold": 0.8,
        "enable_quality_check": True,
        "enable_tamper_detection": True
    }

# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def create_test_image(width=640, height=480, color='red'):
        """Create test image"""
        image = Image.new('RGB', (width, height), color=color)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    @staticmethod
    def create_test_file_upload(content, filename="test.jpg", content_type="image/jpeg"):
        """Create test file upload"""
        from fastapi import UploadFile
        from io import BytesIO
        
        file_obj = BytesIO(content)
        return UploadFile(
            filename=filename,
            file=file_obj,
            content_type=content_type
        )
    
    @staticmethod
    def assert_api_response(response, expected_status=200, expected_keys=None):
        """Assert API response format"""
        assert response.status_code == expected_status
        
        if expected_keys:
            data = response.json()
            for key in expected_keys:
                assert key in data
    
    @staticmethod
    def assert_ml_result(result, required_fields=None):
        """Assert ML result format"""
        assert isinstance(result, dict)
        assert "success" in result
        
        if required_fields:
            for field in required_fields:
                assert field in result

# Performance testing helpers
@pytest.fixture
def performance_timer():
    """Timer for performance testing"""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            
        def start(self):
            self.start_time = time.time()
            
        def stop(self):
            self.end_time = time.time()
            
        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
    
    return Timer()

# Database test helpers
@pytest.fixture
def clean_database():
    """Clean test database"""
    # This would clean test database in real implementation
    yield
    # Cleanup after test

# Async test helpers
@pytest.mark.asyncio
async def async_test_helper():
    """Helper for async testing"""
    pass

# Custom markers for test organization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.api = pytest.mark.api
pytest.mark.ml = pytest.mark.ml
pytest.mark.monitoring = pytest.mark.monitoring
pytest.mark.slow = pytest.mark.slow
pytest.mark.smoke = pytest.mark.smoke
'''

# Test runner script
TEST_RUNNER_PY = '''
#!/usr/bin/env python3
"""
Comprehensive Test Runner for Volcanion Face Identification System
"""
import subprocess
import sys
import os
import argparse
import time
from pathlib import Path

class TestRunner:
    """Test runner with various options"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        
    def run_all_tests(self, coverage=True, html_report=True):
        """Run all tests with coverage"""
        cmd = ["python", "-m", "pytest"]
        
        if coverage:
            cmd.extend([
                "--cov=domain",
                "--cov=application", 
                "--cov=infrastructure",
                "--cov=presentation",
                "--cov-report=term-missing"
            ])
            
            if html_report:
                cmd.append("--cov-report=html:htmlcov")
                
        cmd.extend(["-v", "--tb=short"])
        
        print("🧪 Running all tests...")
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        end_time = time.time()
        
        print(f"\\n⏱️ Test execution time: {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            if html_report:
                print(f"📊 Coverage report: {self.project_root}/htmlcov/index.html")
        else:
            print("❌ Some tests failed!")
            
        return result.returncode == 0
        
    def run_unit_tests(self):
        """Run only unit tests"""
        cmd = ["python", "-m", "pytest", "-m", "unit", "-v"]
        
        print("🧪 Running unit tests...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
        
    def run_integration_tests(self):
        """Run only integration tests"""
        cmd = ["python", "-m", "pytest", "-m", "integration", "-v"]
        
        print("🧪 Running integration tests...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
        
    def run_api_tests(self):
        """Run API tests"""
        cmd = ["python", "-m", "pytest", "-m", "api", "-v"]
        
        print("🧪 Running API tests...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
        
    def run_ml_tests(self):
        """Run ML tests"""
        cmd = ["python", "-m", "pytest", "-m", "ml", "-v"]
        
        print("🧪 Running ML tests...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
        
    def run_monitoring_tests(self):
        """Run monitoring tests"""
        cmd = ["python", "-m", "pytest", "-m", "monitoring", "-v"]
        
        print("🧪 Running monitoring tests...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
        
    def run_smoke_tests(self):
        """Run smoke tests"""
        cmd = ["python", "-m", "pytest", "-m", "smoke", "-v", "--tb=line"]
        
        print("🧪 Running smoke tests...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
        
    def run_specific_test(self, test_path):
        """Run specific test file or function"""
        cmd = ["python", "-m", "pytest", test_path, "-v"]
        
        print(f"🧪 Running specific test: {test_path}")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
        
    def run_parallel_tests(self, num_workers=4):
        """Run tests in parallel"""
        try:
            import pytest_xdist
            cmd = ["python", "-m", "pytest", f"-n{num_workers}", "-v"]
            
            print(f"🧪 Running tests in parallel ({num_workers} workers)...")
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode == 0
            
        except ImportError:
            print("❌ pytest-xdist not installed. Install with: pip install pytest-xdist")
            return False
            
    def check_coverage(self, min_coverage=85):
        """Check if coverage meets minimum requirement"""
        cmd = ["python", "-m", "pytest", "--cov=.", f"--cov-fail-under={min_coverage}", "--cov-report=term"]
        
        print(f"📊 Checking coverage (minimum: {min_coverage}%)...")
        result = subprocess.run(cmd, cwd=self.project_root)
        return result.returncode == 0
        
    def generate_test_report(self):
        """Generate comprehensive test report"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_{timestamp}.html"
        
        cmd = [
            "python", "-m", "pytest",
            f"--html={report_file}",
            "--self-contained-html",
            "--cov=.",
            "--cov-report=html:htmlcov",
            "-v"
        ]
        
        print(f"📋 Generating test report: {report_file}")
        result = subprocess.run(cmd, cwd=self.project_root)
        
        if result.returncode == 0:
            print(f"✅ Test report generated: {self.project_root}/{report_file}")
        
        return result.returncode == 0

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Volcanion Face ID Test Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only") 
    parser.add_argument("--api", action="store_true", help="Run API tests only")
    parser.add_argument("--ml", action="store_true", help="Run ML tests only")
    parser.add_argument("--monitoring", action="store_true", help="Run monitoring tests only")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--coverage", action="store_true", help="Check coverage")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    parser.add_argument("--parallel", type=int, metavar="N", help="Run tests in parallel with N workers")
    parser.add_argument("--specific", type=str, metavar="PATH", help="Run specific test file/function")
    parser.add_argument("--min-coverage", type=int, default=85, help="Minimum coverage percentage")
    
    args = parser.parse_args()
    runner = TestRunner()
    
    success = True
    
    if args.all or not any([args.unit, args.integration, args.api, args.ml, args.monitoring, args.smoke, args.coverage, args.specific]):
        success &= runner.run_all_tests(coverage=True, html_report=args.report)
    
    if args.unit:
        success &= runner.run_unit_tests()
        
    if args.integration:
        success &= runner.run_integration_tests()
        
    if args.api:
        success &= runner.run_api_tests()
        
    if args.ml:
        success &= runner.run_ml_tests()
        
    if args.monitoring:
        success &= runner.run_monitoring_tests()
        
    if args.smoke:
        success &= runner.run_smoke_tests()
        
    if args.coverage:
        success &= runner.check_coverage(args.min_coverage)
        
    if args.parallel:
        success &= runner.run_parallel_tests(args.parallel)
        
    if args.specific:
        success &= runner.run_specific_test(args.specific)
        
    if args.report:
        runner.generate_test_report()
    
    if success:
        print("\\n🎉 All selected tests completed successfully!")
        sys.exit(0)
    else:
        print("\\n💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

def create_test_files():
    """Create test configuration files"""
    files_to_create = [
        ("pytest.ini", PYTEST_INI),
        ("conftest.py", CONFTEST_PY), 
        ("run_tests.py", TEST_RUNNER_PY)
    ]
    
    for filename, content in files_to_create:
        print(f"Creating {filename}...")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {filename} created")

if __name__ == "__main__":
    create_test_files()
    print("\\n🎉 Test configuration files created successfully!")
    print("\\nUsage:")
    print("  python run_tests.py --all          # Run all tests")
    print("  python run_tests.py --unit         # Run unit tests only")
    print("  python run_tests.py --api          # Run API tests only") 
    print("  python run_tests.py --coverage     # Check coverage")
    print("  python run_tests.py --report       # Generate HTML report")
    print("  python run_tests.py --parallel 4   # Run tests in parallel")
