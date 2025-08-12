from locust import HttpUser, task, between
import json
import os
import random
from io import BytesIO


class FaceVerificationAPIUser(HttpUser):
    """
    Performance test user for Face Verification API
    Simulates realistic user behavior with various endpoints
    """
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Setup method called once per user"""
        # Test if health endpoint is working
        self.client.get("/health")
        
        # Store test data
        self.test_images = self._create_test_images()
        
    def _create_test_images(self):
        """Create mock image data for testing"""
        # Create small test images (1x1 pixel)
        test_images = {}
        
        # Simple 1x1 pixel image data
        pixel_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x9d\xcc\xc2\xc4\x00\x00\x00\x00IEND\xaeB`\x82'
        
        test_images['document'] = ('document.png', pixel_data, 'image/png')
        test_images['face'] = ('face.png', pixel_data, 'image/png')
        test_images['selfie'] = ('selfie.png', pixel_data, 'image/png')
        
        return test_images

    @task(5)
    def health_check(self):
        """Health check endpoint - high frequency"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(3)
    def monitoring_summary(self):
        """Monitoring summary - medium frequency"""
        with self.client.get("/monitoring/performance/summary", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Monitoring failed: {response.status_code}")

    @task(2)
    def document_detection(self):
        """Document detection endpoint"""
        files = {'file': self.test_images['document']}
        
        with self.client.post("/document/detect", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'detection_id' in result:
                        response.success()
                        # Store detection_id for follow-up requests
                        self.detection_id = result['detection_id']
                    else:
                        response.failure("Missing detection_id in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Document detection failed: {response.status_code}")

    @task(2)
    def quality_analysis(self):
        """Quality analysis endpoint"""
        files = {'file': self.test_images['document']}
        
        with self.client.post("/quality/analyze", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'quality_id' in result:
                        response.success()
                    else:
                        response.failure("Missing quality_id in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Quality analysis failed: {response.status_code}")

    @task(2)
    def face_detection(self):
        """Face detection endpoint"""
        files = {'file': self.test_images['face']}
        
        with self.client.post("/face-detection/detect", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'detection_id' in result:
                        response.success()
                    else:
                        response.failure("Missing detection_id in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Face detection failed: {response.status_code}")

    @task(1)
    def face_verification(self):
        """Face verification endpoint"""
        files = {
            'reference_image': self.test_images['face'],
            'target_image': self.test_images['selfie']
        }
        
        with self.client.post("/face-verification/verify", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'verification_id' in result:
                        response.success()
                    else:
                        response.failure("Missing verification_id in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Face verification failed: {response.status_code}")

    @task(1)
    def liveness_detection(self):
        """Liveness detection endpoint"""
        files = {'file': self.test_images['selfie']}
        
        with self.client.post("/liveness/detect", files=files, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'liveness_id' in result:
                        response.success()
                    else:
                        response.failure("Missing liveness_id in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Liveness detection failed: {response.status_code}")

    @task(1)
    def ocr_extraction(self):
        """OCR extraction endpoint"""
        files = {'file': self.test_images['document']}
        data = {'field_types': 'ID_NUMBER,FULL_NAME,DOB'}
        
        with self.client.post("/ocr/extract-fields", files=files, data=data, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'result_id' in result:
                        response.success()
                    else:
                        response.failure("Missing result_id in response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"OCR extraction failed: {response.status_code}")

    @task(1)
    def get_detection_result(self):
        """Get detection result if we have a detection_id"""
        if hasattr(self, 'detection_id'):
            with self.client.get(f"/document/result/{self.detection_id}", catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                elif response.status_code == 404:
                    response.success()  # Expected if detection doesn't exist
                else:
                    response.failure(f"Get detection failed: {response.status_code}")

    @task(1)
    def list_detections(self):
        """List detections with pagination"""
        params = {
            'limit': random.randint(5, 20),
            'offset': random.randint(0, 50)
        }
        
        with self.client.get("/document/detections", params=params, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if 'total' in result and 'results' in result:
                        response.success()
                    else:
                        response.failure("Missing required fields in list response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"List detections failed: {response.status_code}")


class HighLoadUser(FaceVerificationAPIUser):
    """
    High-load user for stress testing
    Generates more requests with shorter wait times
    """
    
    wait_time = between(0.1, 0.5)  # Much shorter wait time
    weight = 1  # Lower weight to limit number of these users


class APIDocsUser(HttpUser):
    """
    User that accesses API documentation
    Simulates developers reading the docs
    """
    
    wait_time = between(5, 15)  # Longer wait times for docs browsing
    weight = 1  # Occasional documentation access

    @task(3)
    def swagger_docs(self):
        """Access Swagger documentation"""
        self.client.get("/docs")

    @task(2)
    def redoc_docs(self):
        """Access ReDoc documentation"""
        self.client.get("/redoc")

    @task(1)
    def openapi_json(self):
        """Access OpenAPI JSON schema"""
        self.client.get("/openapi.json")


# Custom test scenarios
class StressTestUser(FaceVerificationAPIUser):
    """
    Stress test user - high frequency requests
    """
    
    wait_time = between(0.1, 0.2)
    weight = 2
    
    @task(10)
    def rapid_health_checks(self):
        """Rapid health checks for stress testing"""
        self.client.get("/health")


# Test configuration for different scenarios
def create_test_config():
    """Create test configuration for different load patterns"""
    return {
        'normal_load': {
            'users': 10,
            'spawn_rate': 2,
            'run_time': '5m'
        },
        'peak_load': {
            'users': 50,
            'spawn_rate': 5,
            'run_time': '10m'
        },
        'stress_test': {
            'users': 100,
            'spawn_rate': 10,
            'run_time': '15m'
        }
    }
