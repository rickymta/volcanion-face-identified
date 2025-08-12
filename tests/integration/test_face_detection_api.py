import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import os
from presentation.api.face_detection_api import router
from domain.entities.face_detection_result import FaceDetectionResult, FaceDetectionStatus

# Create test app
app = FastAPI()
app.include_router(router)

class TestFaceDetectionAPI:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            # Write minimal JPEG header
            temp_file.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb")
            temp_file.flush()
            return temp_file.name
    
    def teardown_method(self):
        # Clean up any temporary files created during tests
        pass
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_detect_face_success(self, mock_get_use_case, client, sample_image_file):
        """Test successful face detection endpoint"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        # Mock successful face detection result
        mock_result = FaceDetectionResult(
            id="test_id",
            image_path="temp_path.jpg",
            status=FaceDetectionStatus.SUCCESS,
            bbox=[100, 100, 200, 200],
            landmarks=[(120, 130), (180, 130), (150, 160), (130, 180), (170, 180)],
            confidence=0.85,
            face_quality_score=0.75,
            alignment_score=0.80,
            occlusion_detected=False
        )
        
        mock_use_case.detect_and_process_face = AsyncMock(return_value=mock_result)
        
        # Test request
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/face-detection/detect",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"source_type": "selfie"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_id"
        assert data["status"] == "success"
        assert data["confidence"] == 0.85
        assert data["bbox"] == [100, 100, 200, 200]
        assert len(data["landmarks"]) == 5
        assert data["occlusion_detected"] is False
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_detect_face_no_face_detected(self, mock_get_use_case, client, sample_image_file):
        """Test face detection when no face is detected"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        # Mock no face detected result
        mock_result = FaceDetectionResult(
            id="test_id",
            image_path="temp_path.jpg",
            status=FaceDetectionStatus.NO_FACE_DETECTED,
            confidence=0.0
        )
        
        mock_use_case.detect_and_process_face = AsyncMock(return_value=mock_result)
        
        # Test request
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/face-detection/detect",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_face_detected"
        assert data["confidence"] == 0.0
        assert data["bbox"] is None
    
    def test_detect_face_invalid_file_type(self, client):
        """Test face detection with invalid file type"""
        # Create a text file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w") as temp_file:
            temp_file.write("This is not an image")
            temp_file.flush()
            
            with open(temp_file.name, "rb") as f:
                response = client.post(
                    "/api/face-detection/detect",
                    files={"file": ("test.txt", f, "text/plain")}
                )
        
        # Clean up
        os.unlink(temp_file.name)
        
        # Assertions
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_validate_face_quality_success(self, mock_get_use_case, client, sample_image_file):
        """Test successful face quality validation"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        # Mock successful validation result
        mock_face_result = FaceDetectionResult(
            id="test_id",
            image_path="temp_path.jpg",
            status=FaceDetectionStatus.SUCCESS,
            confidence=0.85,
            face_quality_score=0.75,
            alignment_score=0.80
        )
        
        mock_validation_result = {
            'is_valid': True,
            'face_result': mock_face_result,
            'recommendations': ["Good quality face"],
            'quality_score': 0.75,
            'alignment_score': 0.80,
            'confidence': 0.85
        }
        
        mock_use_case.validate_face_quality = AsyncMock(return_value=mock_validation_result)
        
        # Test request
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/face-detection/validate",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"source_type": "selfie"}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is True
        assert data["recommendations"] == ["Good quality face"]
        assert data["quality_score"] == 0.75
        assert data["alignment_score"] == 0.80
        assert data["confidence"] == 0.85
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_validate_face_quality_invalid(self, mock_get_use_case, client, sample_image_file):
        """Test face quality validation with invalid face"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        # Mock invalid validation result
        mock_validation_result = {
            'is_valid': False,
            'face_result': None,
            'recommendations': ["No face detected", "Please ensure face is visible"],
            'quality_score': 0.0,
            'alignment_score': 0.0,
            'confidence': 0.0
        }
        
        mock_use_case.validate_face_quality = AsyncMock(return_value=mock_validation_result)
        
        # Test request
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/face-detection/validate",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["is_valid"] is False
        assert len(data["recommendations"]) == 2
        assert data["quality_score"] == 0.0
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_compare_face_alignment_success(self, mock_get_use_case, client, sample_image_file):
        """Test successful face alignment comparison"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        # Create second image file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file2:
            temp_file2.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb")
            temp_file2.flush()
            second_image_file = temp_file2.name
        
        # Mock comparison result
        mock_comparison_result = {
            'alignment_similarity': 0.85,
            'face1_valid': True,
            'face2_valid': True,
            'face1_result': None,
            'face2_result': None,
            'face1_recommendations': ["Good face"],
            'face2_recommendations': ["Good face"],
            'overall_valid': True
        }
        
        mock_use_case.compare_face_alignment = AsyncMock(return_value=mock_comparison_result)
        
        try:
            # Test request
            with open(sample_image_file, "rb") as f1, open(second_image_file, "rb") as f2:
                response = client.post(
                    "/api/face-detection/compare",
                    files={
                        "selfie_file": ("selfie.jpg", f1, "image/jpeg"),
                        "document_file": ("document.jpg", f2, "image/jpeg")
                    }
                )
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert data["alignment_similarity"] == 0.85
            assert data["face1_valid"] is True
            assert data["face2_valid"] is True
            assert data["overall_valid"] is True
            assert data["face1_recommendations"] == ["Good face"]
            assert data["face2_recommendations"] == ["Good face"]
            
        finally:
            # Clean up
            os.unlink(second_image_file)
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_get_face_detection_result_success(self, mock_get_use_case, client):
        """Test successful retrieval of face detection result"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        # Mock face detection result
        mock_result = FaceDetectionResult(
            id="test_id",
            image_path="test.jpg",
            status=FaceDetectionStatus.SUCCESS,
            confidence=0.85
        )
        
        mock_use_case.get_face_detection_result = AsyncMock(return_value=mock_result)
        
        # Test request
        response = client.get("/api/face-detection/result/test_id")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_id"
        assert data["status"] == "success"
        assert data["confidence"] == 0.85
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_get_face_detection_result_not_found(self, mock_get_use_case, client):
        """Test retrieval of non-existent face detection result"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        mock_use_case.get_face_detection_result = AsyncMock(return_value=None)
        
        # Test request
        response = client.get("/api/face-detection/result/nonexistent_id")
        
        # Assertions
        assert response.status_code == 404
        assert "Face detection result not found" in response.json()["detail"]
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_get_face_detection_statistics_success(self, mock_get_use_case, client):
        """Test successful retrieval of face detection statistics"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        # Mock statistics
        mock_stats = {
            'total_detections': 100,
            'success_rate': 85.5,
            'average_confidence': 0.82,
            'average_quality_score': 0.75,
            'average_alignment_score': 0.78,
            'status_distribution': {'success': 85, 'no_face_detected': 10, 'failed': 5},
            'occlusion_distribution': {'glasses': 15, 'hat': 5, 'none': 80}
        }
        
        mock_use_case.get_face_detection_statistics = AsyncMock(return_value=mock_stats)
        
        # Test request
        response = client.get("/api/face-detection/statistics")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["total_detections"] == 100
        assert data["success_rate"] == 85.5
        assert data["average_confidence"] == 0.82
        assert data["status_distribution"]["success"] == 85
        assert data["occlusion_distribution"]["glasses"] == 15
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/face-detection/health")
        
        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "face-detection"
        assert data["version"] == "1.0.0"
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_detect_face_use_case_exception(self, mock_get_use_case, client, sample_image_file):
        """Test face detection when use case raises exception"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        # Mock use case to raise exception
        mock_use_case.detect_and_process_face = AsyncMock(side_effect=Exception("Use case error"))
        
        # Test request
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/face-detection/detect",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        # Assertions
        assert response.status_code == 500
        assert "Face detection failed" in response.json()["detail"]
    
    @patch('presentation.api.face_detection_api.get_face_detection_use_case')
    def test_validate_face_quality_use_case_exception(self, mock_get_use_case, client, sample_image_file):
        """Test face validation when use case raises exception"""
        # Mock use case
        mock_use_case = Mock()
        mock_get_use_case.return_value = mock_use_case
        
        # Mock use case to raise exception
        mock_use_case.validate_face_quality = AsyncMock(side_effect=Exception("Validation error"))
        
        # Test request
        with open(sample_image_file, "rb") as f:
            response = client.post(
                "/api/face-detection/validate",
                files={"file": ("test.jpg", f, "image/jpeg")}
            )
        
        # Assertions
        assert response.status_code == 500
        assert "Face validation failed" in response.json()["detail"]
    
    def cleanup_method(self, sample_image_file):
        """Clean up test files"""
        if os.path.exists(sample_image_file):
            os.unlink(sample_image_file)
