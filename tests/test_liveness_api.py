import pytest
import tempfile
import json
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import numpy as np
import io

from main import app
from domain.entities.liveness_result import (
    LivenessDetectionResult, 
    LivenessStatus, 
    LivenessResult, 
    SpoofType
)

client = TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple RGB image
    img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

@pytest.fixture
def sample_liveness_result():
    """Create a sample liveness detection result"""
    return LivenessDetectionResult(
        id="test_liveness_123",
        image_path="/path/to/test_image.jpg",
        face_bbox=[100, 100, 200, 200],
        status=LivenessStatus.COMPLETED,
        liveness_result=LivenessResult.REAL,
        confidence=0.95,
        liveness_score=0.85,
        spoof_probability=0.15,
        detected_spoof_types=[],
        primary_spoof_type=None,
        image_quality=0.8,
        face_quality=0.85,
        lighting_quality=0.9,
        pose_quality=0.75,
        processing_time_ms=150.0,
        algorithms_used=["texture_analysis", "frequency_analysis"],
        model_version="v1.0",
        threshold_used=0.5
    )

class TestLivenessAPI:
    
    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.detect_and_save_liveness')
    def test_detect_liveness_success(self, mock_detect, sample_image, sample_liveness_result):
        """Test successful liveness detection"""
        mock_detect.return_value = sample_liveness_result
        
        response = client.post(
            "/api/liveness/detect",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
            data={"face_bbox": "[100, 100, 200, 200]", "use_advanced_analysis": "true"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test_liveness_123"
        assert data["status"] == "COMPLETED"
        assert data["liveness_result"] == "REAL"
        assert data["confidence"] == 0.95
        assert data["is_real"] is True
        assert data["is_fake"] is False

    def test_detect_liveness_invalid_file(self):
        """Test liveness detection with invalid file"""
        # Create a text file instead of image
        text_file = io.BytesIO(b"This is not an image")
        
        response = client.post(
            "/api/liveness/detect",
            files={"file": ("test.txt", text_file, "text/plain")},
            data={"face_bbox": "[100, 100, 200, 200]"}
        )
        
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]

    def test_detect_liveness_invalid_bbox(self, sample_image):
        """Test liveness detection with invalid bounding box"""
        response = client.post(
            "/api/liveness/detect",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
            data={"face_bbox": "invalid_json"}
        )
        
        assert response.status_code == 400
        assert "face_bbox must be a JSON array" in response.json()["detail"]

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.batch_detect_liveness')
    def test_batch_detect_liveness(self, mock_batch_detect, sample_image, sample_liveness_result):
        """Test batch liveness detection"""
        # Mock batch detection to return multiple results
        mock_batch_detect.return_value = [sample_liveness_result, sample_liveness_result]
        
        # Create second image
        sample_image2 = io.BytesIO()
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img.save(sample_image2, format='JPEG')
        sample_image2.seek(0)
        
        response = client.post(
            "/api/liveness/batch-detect",
            files=[
                ("files", ("test1.jpg", sample_image, "image/jpeg")),
                ("files", ("test2.jpg", sample_image2, "image/jpeg"))
            ],
            data={
                "face_bboxes": "[[100, 100, 200, 200], [150, 150, 250, 250]]",
                "use_advanced_analysis": "true"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 2
        assert all(result["status"] == "COMPLETED" for result in data)

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.get_liveness_result')
    def test_get_liveness_result(self, mock_get_result, sample_liveness_result):
        """Test getting liveness result by ID"""
        mock_get_result.return_value = sample_liveness_result
        
        response = client.get("/api/liveness/result/test_liveness_123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["id"] == "test_liveness_123"
        assert data["liveness_result"] == "REAL"

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.get_liveness_result')
    def test_get_liveness_result_not_found(self, mock_get_result):
        """Test getting non-existent liveness result"""
        mock_get_result.return_value = None
        
        response = client.get("/api/liveness/result/non_existent_id")
        
        assert response.status_code == 404
        assert "Liveness result not found" in response.json()["detail"]

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.get_recent_detections')
    def test_get_recent_detections(self, mock_get_recent, sample_liveness_result):
        """Test getting recent detections"""
        mock_get_recent.return_value = [sample_liveness_result]
        
        response = client.get("/api/liveness/results/recent?hours=24&limit=100")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 1
        assert data[0]["id"] == "test_liveness_123"

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.get_fake_detections')
    def test_get_fake_detections(self, mock_get_fake, sample_liveness_result):
        """Test getting fake detections"""
        # Create fake result
        fake_result = sample_liveness_result
        fake_result.liveness_result = LivenessResult.FAKE
        
        mock_get_fake.return_value = [fake_result]
        
        response = client.get("/api/liveness/results/fake?confidence_threshold=0.8&limit=100")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 1
        assert data[0]["liveness_result"] == "FAKE"
        assert data[0]["is_fake"] is True

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.get_spoof_attacks_by_type')
    def test_get_spoof_attacks_by_type(self, mock_get_spoof, sample_liveness_result):
        """Test getting spoof attacks by type"""
        spoof_result = sample_liveness_result
        spoof_result.detected_spoof_types = [SpoofType.PHOTO_ATTACK]
        spoof_result.primary_spoof_type = SpoofType.PHOTO_ATTACK
        
        mock_get_spoof.return_value = [spoof_result]
        
        response = client.get("/api/liveness/results/spoof/PHOTO_ATTACK?limit=100")
        
        assert response.status_code == 200
        data = response.json()
        
        assert len(data) == 1
        assert "PHOTO_ATTACK" in data[0]["detected_spoof_types"]

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.analyze_liveness_patterns')
    def test_analyze_liveness_patterns(self, mock_analyze, sample_image):
        """Test liveness pattern analysis"""
        mock_analysis = {
            "total_analyzed": 2,
            "result_distribution": {"REAL": 1, "FAKE": 1},
            "spoof_type_distribution": {"PHOTO_ATTACK": 1},
            "statistics": {"avg_confidence": 0.8, "avg_score": 0.75},
            "risk_assessment": "Medium",
            "recommendations": ["Use additional verification methods"]
        }
        mock_analyze.return_value = mock_analysis
        
        # Create second image
        sample_image2 = io.BytesIO()
        img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        img.save(sample_image2, format='JPEG')
        sample_image2.seek(0)
        
        response = client.post(
            "/api/liveness/analyze-patterns",
            files=[
                ("files", ("test1.jpg", sample_image, "image/jpeg")),
                ("files", ("test2.jpg", sample_image2, "image/jpeg"))
            ],
            data={"face_bboxes": "[[100, 100, 200, 200], [150, 150, 250, 250]]"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_analyzed"] == 2
        assert "result_distribution" in data
        assert "risk_assessment" in data
        assert data["risk_assessment"] == "Medium"

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.validate_detection_quality')
    def test_validate_detection_quality(self, mock_validate):
        """Test detection quality validation"""
        mock_validation = {
            "is_valid": True,
            "issues": [],
            "quality_score": 0.85,
            "recommendations": []
        }
        mock_validate.return_value = mock_validation
        
        response = client.get("/api/liveness/validate/test_liveness_123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["is_valid"] is True
        assert data["quality_score"] == 0.85

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.compare_liveness_results')
    def test_compare_liveness_results(self, mock_compare):
        """Test comparing liveness results"""
        mock_comparison = {
            "results_match": False,
            "confidence_difference": 0.15,
            "score_difference": 0.05,
            "consistency": "Low",
            "result1": {"id": "test_liveness_123", "confidence": 0.95},
            "result2": {"id": "test_liveness_456", "confidence": 0.8}
        }
        mock_compare.return_value = mock_comparison
        
        response = client.get("/api/liveness/compare/test_liveness_123/test_liveness_456")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["results_match"] is False
        assert data["confidence_difference"] == 0.15
        assert data["consistency"] == "Low"

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.get_liveness_statistics')
    def test_get_liveness_statistics(self, mock_get_stats):
        """Test getting liveness statistics"""
        mock_stats = {
            "basic_statistics": {
                "total_detections": 1000,
                "real_faces": 800,
                "fake_faces": 200,
                "success_rate": 0.95
            },
            "performance_metrics": {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.98,
                "f1_score": 0.95
            },
            "spoof_attack_trends": {
                "photo_attacks": 120,
                "screen_attacks": 50,
                "mask_attacks": 30
            },
            "engine_info": {
                "version": "v1.0",
                "algorithms": ["texture", "frequency", "depth"]
            },
            "supported_formats": ["jpg", "png", "bmp"]
        }
        mock_get_stats.return_value = mock_stats
        
        response = client.get("/api/liveness/statistics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["basic_statistics"]["total_detections"] == 1000
        assert data["performance_metrics"]["accuracy"] == 0.95
        assert "photo_attacks" in data["spoof_attack_trends"]

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.optimize_detection_thresholds')
    def test_optimize_detection_thresholds(self, mock_optimize):
        """Test threshold optimization"""
        mock_optimization = {
            "optimal_threshold": 0.75,
            "performance_metrics": {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.95,
                "f1_score": 0.92
            },
            "training_data_stats": {
                "total_samples": 100,
                "real_samples": 60,
                "fake_samples": 40
            }
        }
        mock_optimize.return_value = mock_optimization
        
        training_data = [
            {"image_path": "/path/1.jpg", "face_bbox": [100, 100, 200, 200], "is_real": True},
            {"image_path": "/path/2.jpg", "face_bbox": [150, 150, 250, 250], "is_real": False}
        ]
        
        response = client.post(
            "/api/liveness/optimize-thresholds",
            data={"training_data": json.dumps(training_data)}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["optimal_threshold"] == 0.75
        assert data["performance_metrics"]["accuracy"] == 0.92

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.cleanup_old_results')
    def test_cleanup_old_results(self, mock_cleanup):
        """Test cleanup of old results"""
        mock_cleanup_result = {
            "deleted_count": 50,
            "cutoff_date": "2024-01-01T00:00:00",
            "days_kept": 30,
            "success": True
        }
        mock_cleanup.return_value = mock_cleanup_result
        
        response = client.delete("/api/liveness/cleanup?days_to_keep=30")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["deleted_count"] == 50
        assert data["success"] is True

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.delete_liveness_result')
    def test_delete_liveness_result_success(self, mock_delete):
        """Test successful deletion of liveness result"""
        mock_delete.return_value = True
        
        response = client.delete("/api/liveness/result/test_liveness_123")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["result_id"] == "test_liveness_123"

    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.delete_liveness_result')
    def test_delete_liveness_result_not_found(self, mock_delete):
        """Test deletion of non-existent liveness result"""
        mock_delete.return_value = False
        
        response = client.delete("/api/liveness/result/non_existent_id")
        
        assert response.status_code == 404
        assert "Liveness result not found" in response.json()["detail"]

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/liveness/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "liveness-detection"
        assert data["version"] == "1.0.0"

class TestLivenessAPIErrorHandling:
    
    @patch('application.use_cases.liveness_detection_use_case.LivenessDetectionUseCase.detect_and_save_liveness')
    def test_detect_liveness_internal_error(self, mock_detect, sample_image):
        """Test handling of internal server errors"""
        mock_detect.side_effect = Exception("Internal processing error")
        
        response = client.post(
            "/api/liveness/detect",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
            data={"face_bbox": "[100, 100, 200, 200]"}
        )
        
        assert response.status_code == 500
        assert "Liveness detection failed" in response.json()["detail"]

    def test_analyze_patterns_mismatched_files_and_bboxes(self, sample_image):
        """Test pattern analysis with mismatched files and bboxes"""
        response = client.post(
            "/api/liveness/analyze-patterns",
            files=[("files", ("test.jpg", sample_image, "image/jpeg"))],
            data={"face_bboxes": "[[100, 100, 200, 200], [150, 150, 250, 250]]"}  # 2 bboxes for 1 file
        )
        
        assert response.status_code == 400
        assert "Number of bboxes must match number of files" in response.json()["detail"]

    def test_optimize_thresholds_invalid_json(self):
        """Test threshold optimization with invalid JSON"""
        response = client.post(
            "/api/liveness/optimize-thresholds",
            data={"training_data": "invalid_json"}
        )
        
        assert response.status_code == 400
        assert "Invalid JSON" in response.json()["detail"]

if __name__ == "__main__":
    pytest.main([__file__])
