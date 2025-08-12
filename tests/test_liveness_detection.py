import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta

from domain.entities.liveness_result import (
    LivenessDetectionResult, 
    LivenessStatus, 
    LivenessResult, 
    SpoofType
)
from domain.repositories.liveness_repository import LivenessRepository
from domain.services.liveness_service import LivenessService
from application.use_cases.liveness_detection_use_case import LivenessDetectionUseCase

@pytest.fixture
def mock_image():
    """Create a mock image for testing"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_face_bbox():
    """Mock face bounding box"""
    return [100, 100, 200, 200]

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
        threshold_used=0.5,
        created_at=datetime.now()
    )

@pytest.fixture
def mock_liveness_repository():
    """Mock liveness repository"""
    repo = Mock(spec=LivenessRepository)
    repo.save = AsyncMock()
    repo.find_by_id = AsyncMock()
    repo.get_recent_detections = AsyncMock()
    repo.get_fake_detections = AsyncMock()
    repo.get_spoof_attacks_by_type = AsyncMock()
    repo.get_statistics = AsyncMock()
    repo.count_total = AsyncMock()
    repo.delete_old_results = AsyncMock()
    repo.delete_by_id = AsyncMock()
    return repo

@pytest.fixture
def mock_liveness_service():
    """Mock liveness service"""
    service = Mock(spec=LivenessService)
    service.detect_liveness = Mock()
    service.batch_detect_liveness = Mock()
    service.analyze_patterns = Mock()
    service.validate_quality = Mock()
    service.compare_results = Mock()
    service.get_statistics = Mock()
    service.optimize_thresholds = Mock()
    return service

@pytest.fixture
def liveness_use_case(mock_liveness_service, mock_liveness_repository):
    """Create liveness detection use case with mocked dependencies"""
    return LivenessDetectionUseCase(mock_liveness_service, mock_liveness_repository)

class TestLivenessDetectionResult:
    def test_liveness_result_creation(self, sample_liveness_result):
        """Test creation of liveness detection result"""
        result = sample_liveness_result
        
        assert result.id == "test_liveness_123"
        assert result.image_path == "/path/to/test_image.jpg"
        assert result.face_bbox == [100, 100, 200, 200]
        assert result.status == LivenessStatus.COMPLETED
        assert result.liveness_result == LivenessResult.REAL
        assert result.confidence == 0.95
        assert result.liveness_score == 0.85
        assert result.spoof_probability == 0.15

    def test_is_real_face(self, sample_liveness_result):
        """Test is_real_face method"""
        result = sample_liveness_result
        result.liveness_result = LivenessResult.REAL
        assert result.is_real_face() is True
        
        result.liveness_result = LivenessResult.FAKE
        assert result.is_real_face() is False

    def test_is_fake_face(self, sample_liveness_result):
        """Test is_fake_face method"""
        result = sample_liveness_result
        result.liveness_result = LivenessResult.FAKE
        assert result.is_fake_face() is True
        
        result.liveness_result = LivenessResult.REAL
        assert result.is_fake_face() is False

    def test_get_confidence_level(self, sample_liveness_result):
        """Test confidence level assessment"""
        result = sample_liveness_result
        
        result.confidence = 0.95
        assert result.get_confidence_level() == "Very High"
        
        result.confidence = 0.85
        assert result.get_confidence_level() == "High"
        
        result.confidence = 0.65
        assert result.get_confidence_level() == "Medium"
        
        result.confidence = 0.45
        assert result.get_confidence_level() == "Low"

    def test_get_risk_level(self, sample_liveness_result):
        """Test risk level assessment"""
        result = sample_liveness_result
        
        result.spoof_probability = 0.1
        assert result.get_risk_level() == "Low"
        
        result.spoof_probability = 0.4
        assert result.get_risk_level() == "Medium"
        
        result.spoof_probability = 0.7
        assert result.get_risk_level() == "High"
        
        result.spoof_probability = 0.9
        assert result.get_risk_level() == "Very High"

    def test_get_overall_quality_score(self, sample_liveness_result):
        """Test overall quality score calculation"""
        result = sample_liveness_result
        score = result.get_overall_quality_score()
        
        # Should be weighted average
        expected = (0.8 * 0.25) + (0.85 * 0.3) + (0.9 * 0.25) + (0.75 * 0.2)
        assert abs(score - expected) < 0.01

    def test_has_spoof_type(self, sample_liveness_result):
        """Test spoof type detection"""
        result = sample_liveness_result
        result.detected_spoof_types = [SpoofType.PHOTO_ATTACK, SpoofType.SCREEN_ATTACK]
        
        assert result.has_spoof_type(SpoofType.PHOTO_ATTACK) is True
        assert result.has_spoof_type(SpoofType.MASK_ATTACK) is False

class TestLivenessDetectionUseCase:
    
    @pytest.mark.asyncio
    async def test_detect_and_save_liveness_success(
        self, 
        liveness_use_case, 
        mock_liveness_service, 
        mock_liveness_repository,
        sample_liveness_result,
        mock_face_bbox
    ):
        """Test successful liveness detection and save"""
        image_path = "/path/to/test_image.jpg"
        
        # Mock service response
        mock_liveness_service.detect_liveness.return_value = sample_liveness_result
        mock_liveness_repository.save.return_value = sample_liveness_result.id
        
        result = await liveness_use_case.detect_and_save_liveness(
            image_path, mock_face_bbox, True
        )
        
        # Verify service was called
        mock_liveness_service.detect_liveness.assert_called_once_with(
            image_path, mock_face_bbox, True
        )
        
        # Verify repository save was called
        mock_liveness_repository.save.assert_called_once()
        
        assert result.id == sample_liveness_result.id

    @pytest.mark.asyncio
    async def test_detect_and_save_liveness_failure(
        self, 
        liveness_use_case, 
        mock_liveness_service,
        mock_face_bbox
    ):
        """Test liveness detection failure handling"""
        image_path = "/path/to/test_image.jpg"
        
        # Mock service to raise exception
        mock_liveness_service.detect_liveness.side_effect = Exception("Detection failed")
        
        with pytest.raises(Exception, match="Detection failed"):
            await liveness_use_case.detect_and_save_liveness(
                image_path, mock_face_bbox, True
            )

    @pytest.mark.asyncio
    async def test_batch_detect_liveness(
        self, 
        liveness_use_case, 
        mock_liveness_service, 
        mock_liveness_repository,
        sample_liveness_result
    ):
        """Test batch liveness detection"""
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        face_bboxes = [[100, 100, 200, 200], [150, 150, 250, 250]]
        
        # Mock service to return multiple results
        mock_results = [sample_liveness_result, sample_liveness_result]
        mock_liveness_service.batch_detect_liveness.return_value = mock_results
        
        results = await liveness_use_case.batch_detect_liveness(
            image_paths, face_bboxes, True
        )
        
        # Verify service was called
        mock_liveness_service.batch_detect_liveness.assert_called_once_with(
            image_paths, face_bboxes, True
        )
        
        # Verify all results were saved
        assert mock_liveness_repository.save.call_count == len(mock_results)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_liveness_result(
        self, 
        liveness_use_case, 
        mock_liveness_repository,
        sample_liveness_result
    ):
        """Test getting liveness result by ID"""
        result_id = "test_liveness_123"
        mock_liveness_repository.find_by_id.return_value = sample_liveness_result
        
        result = await liveness_use_case.get_liveness_result(result_id)
        
        mock_liveness_repository.find_by_id.assert_called_once_with(result_id)
        assert result == sample_liveness_result

    @pytest.mark.asyncio
    async def test_get_recent_detections(
        self, 
        liveness_use_case, 
        mock_liveness_repository,
        sample_liveness_result
    ):
        """Test getting recent detections"""
        mock_liveness_repository.get_recent_detections.return_value = [sample_liveness_result]
        
        results = await liveness_use_case.get_recent_detections(24, 100)
        
        mock_liveness_repository.get_recent_detections.assert_called_once_with(24, 100)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_fake_detections(
        self, 
        liveness_use_case, 
        mock_liveness_repository,
        sample_liveness_result
    ):
        """Test getting fake detections"""
        # Create fake result
        fake_result = sample_liveness_result
        fake_result.liveness_result = LivenessResult.FAKE
        
        mock_liveness_repository.get_fake_detections.return_value = [fake_result]
        
        results = await liveness_use_case.get_fake_detections(0.8, 100)
        
        mock_liveness_repository.get_fake_detections.assert_called_once_with(0.8, 100)
        assert len(results) == 1
        assert results[0].is_fake_face()

    @pytest.mark.asyncio
    async def test_get_spoof_attacks_by_type(
        self, 
        liveness_use_case, 
        mock_liveness_repository,
        sample_liveness_result
    ):
        """Test getting spoof attacks by type"""
        spoof_type = "PHOTO_ATTACK"
        mock_liveness_repository.get_spoof_attacks_by_type.return_value = [sample_liveness_result]
        
        results = await liveness_use_case.get_spoof_attacks_by_type(spoof_type, 100)
        
        mock_liveness_repository.get_spoof_attacks_by_type.assert_called_once_with(spoof_type, 100)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_analyze_liveness_patterns(
        self, 
        liveness_use_case, 
        mock_liveness_service
    ):
        """Test liveness pattern analysis"""
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        face_bboxes = [[100, 100, 200, 200], [150, 150, 250, 250]]
        
        mock_analysis = {
            "total_analyzed": 2,
            "result_distribution": {"REAL": 1, "FAKE": 1},
            "spoof_type_distribution": {"PHOTO_ATTACK": 1},
            "statistics": {"avg_confidence": 0.8},
            "risk_assessment": "Medium",
            "recommendations": ["Use additional verification"]
        }
        
        mock_liveness_service.analyze_patterns.return_value = mock_analysis
        
        result = await liveness_use_case.analyze_liveness_patterns(image_paths, face_bboxes)
        
        mock_liveness_service.analyze_patterns.assert_called_once_with(image_paths, face_bboxes)
        assert result["total_analyzed"] == 2

    @pytest.mark.asyncio
    async def test_validate_detection_quality(
        self, 
        liveness_use_case, 
        mock_liveness_service,
        mock_liveness_repository,
        sample_liveness_result
    ):
        """Test detection quality validation"""
        result_id = "test_liveness_123"
        
        mock_liveness_repository.find_by_id.return_value = sample_liveness_result
        mock_validation = {
            "is_valid": True,
            "issues": [],
            "quality_score": 0.85,
            "recommendations": []
        }
        mock_liveness_service.validate_quality.return_value = mock_validation
        
        result = await liveness_use_case.validate_detection_quality(result_id)
        
        mock_liveness_repository.find_by_id.assert_called_once_with(result_id)
        mock_liveness_service.validate_quality.assert_called_once_with(sample_liveness_result)
        assert result["is_valid"] is True

    @pytest.mark.asyncio
    async def test_compare_liveness_results(
        self, 
        liveness_use_case, 
        mock_liveness_service,
        mock_liveness_repository,
        sample_liveness_result
    ):
        """Test comparing liveness results"""
        result_id1 = "test_liveness_123"
        result_id2 = "test_liveness_456"
        
        result2 = sample_liveness_result
        result2.id = result_id2
        result2.confidence = 0.8
        
        mock_liveness_repository.find_by_id.side_effect = [sample_liveness_result, result2]
        
        mock_comparison = {
            "results_match": False,
            "confidence_difference": 0.15,
            "score_difference": 0.05,
            "consistency": "Low",
            "result1": {"id": result_id1, "confidence": 0.95},
            "result2": {"id": result_id2, "confidence": 0.8}
        }
        mock_liveness_service.compare_results.return_value = mock_comparison
        
        result = await liveness_use_case.compare_liveness_results(result_id1, result_id2)
        
        assert mock_liveness_repository.find_by_id.call_count == 2
        mock_liveness_service.compare_results.assert_called_once()
        assert result["results_match"] is False

    @pytest.mark.asyncio
    async def test_get_liveness_statistics(
        self, 
        liveness_use_case, 
        mock_liveness_repository
    ):
        """Test getting liveness statistics"""
        mock_stats = {
            "basic_statistics": {"total_detections": 100},
            "performance_metrics": {"accuracy": 0.95},
            "spoof_attack_trends": {"photo_attacks": 25},
            "engine_info": {"version": "v1.0"},
            "supported_formats": ["jpg", "png"]
        }
        
        mock_liveness_repository.get_statistics.return_value = mock_stats
        
        result = await liveness_use_case.get_liveness_statistics()
        
        mock_liveness_repository.get_statistics.assert_called_once()
        assert result["basic_statistics"]["total_detections"] == 100

    @pytest.mark.asyncio
    async def test_optimize_detection_thresholds(
        self, 
        liveness_use_case, 
        mock_liveness_service
    ):
        """Test threshold optimization"""
        training_samples = [
            {"image_path": "/path/1.jpg", "face_bbox": [100, 100, 200, 200], "is_real": True},
            {"image_path": "/path/2.jpg", "face_bbox": [150, 150, 250, 250], "is_real": False}
        ]
        
        mock_optimization = {
            "optimal_threshold": 0.75,
            "performance_metrics": {"accuracy": 0.92, "precision": 0.89, "recall": 0.95},
            "training_data_stats": {"total_samples": 2, "real_samples": 1, "fake_samples": 1}
        }
        
        mock_liveness_service.optimize_thresholds.return_value = mock_optimization
        
        result = await liveness_use_case.optimize_detection_thresholds(training_samples)
        
        mock_liveness_service.optimize_thresholds.assert_called_once_with(training_samples)
        assert result["optimal_threshold"] == 0.75

    @pytest.mark.asyncio
    async def test_cleanup_old_results(
        self, 
        liveness_use_case, 
        mock_liveness_repository
    ):
        """Test cleanup of old results"""
        days_to_keep = 30
        
        mock_cleanup = {
            "deleted_count": 50,
            "cutoff_date": (datetime.now() - timedelta(days=days_to_keep)).isoformat(),
            "days_kept": days_to_keep,
            "success": True
        }
        
        mock_liveness_repository.delete_old_results.return_value = mock_cleanup
        
        result = await liveness_use_case.cleanup_old_results(days_to_keep)
        
        mock_liveness_repository.delete_old_results.assert_called_once_with(days_to_keep)
        assert result["deleted_count"] == 50
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_delete_liveness_result(
        self, 
        liveness_use_case, 
        mock_liveness_repository
    ):
        """Test deleting liveness result"""
        result_id = "test_liveness_123"
        mock_liveness_repository.delete_by_id.return_value = True
        
        success = await liveness_use_case.delete_liveness_result(result_id)
        
        mock_liveness_repository.delete_by_id.assert_called_once_with(result_id)
        assert success is True

    @pytest.mark.asyncio
    async def test_delete_liveness_result_not_found(
        self, 
        liveness_use_case, 
        mock_liveness_repository
    ):
        """Test deleting non-existent liveness result"""
        result_id = "non_existent_id"
        mock_liveness_repository.delete_by_id.return_value = False
        
        success = await liveness_use_case.delete_liveness_result(result_id)
        
        mock_liveness_repository.delete_by_id.assert_called_once_with(result_id)
        assert success is False

class TestLivenessService:
    
    def test_liveness_service_creation(self):
        """Test liveness service initialization"""
        service = LivenessService()
        
        assert service is not None
        assert hasattr(service, 'detect_liveness')
        assert hasattr(service, 'batch_detect_liveness')
        assert hasattr(service, 'analyze_patterns')

    @patch('domain.services.liveness_service.LivenessEngine')
    def test_detect_liveness(self, mock_engine_class, mock_image, mock_face_bbox):
        """Test liveness detection through service"""
        # Mock engine instance
        mock_engine = Mock()
        mock_engine_class.return_value = mock_engine
        
        # Mock detection result
        mock_detection = {
            "is_real": True,
            "confidence": 0.95,
            "liveness_score": 0.85,
            "spoof_probability": 0.15,
            "detected_spoof_types": [],
            "primary_spoof_type": None,
            "processing_time_ms": 150.0,
            "algorithms_used": ["texture_analysis"],
            "model_version": "v1.0"
        }
        mock_engine.detect_liveness.return_value = mock_detection
        
        # Mock quality analysis
        mock_quality = {
            "image_quality": 0.8,
            "face_quality": 0.85,
            "lighting_quality": 0.9,
            "pose_quality": 0.75
        }
        mock_engine.analyze_quality.return_value = mock_quality
        
        service = LivenessService()
        
        with patch('cv2.imread', return_value=mock_image):
            result = service.detect_liveness("/path/to/test.jpg", mock_face_bbox, True)
            
            assert result.liveness_result == LivenessResult.REAL
            assert result.confidence == 0.95
            assert result.liveness_score == 0.85

    def test_batch_detect_liveness(self, mock_image, mock_face_bbox):
        """Test batch liveness detection"""
        service = LivenessService()
        
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        face_bboxes = [mock_face_bbox, mock_face_bbox]
        
        with patch.object(service, 'detect_liveness') as mock_detect:
            # Mock individual detection results
            mock_result1 = Mock()
            mock_result1.liveness_result = LivenessResult.REAL
            mock_result2 = Mock()
            mock_result2.liveness_result = LivenessResult.FAKE
            
            mock_detect.side_effect = [mock_result1, mock_result2]
            
            results = service.batch_detect_liveness(image_paths, face_bboxes, True)
            
            assert len(results) == 2
            assert mock_detect.call_count == 2

    def test_analyze_patterns(self):
        """Test pattern analysis"""
        service = LivenessService()
        
        image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
        face_bboxes = [[100, 100, 200, 200], [150, 150, 250, 250]]
        
        with patch.object(service, 'batch_detect_liveness') as mock_batch:
            # Mock batch results
            mock_results = [
                Mock(liveness_result=LivenessResult.REAL, confidence=0.9),
                Mock(liveness_result=LivenessResult.FAKE, confidence=0.8)
            ]
            mock_batch.return_value = mock_results
            
            analysis = service.analyze_patterns(image_paths, face_bboxes)
            
            assert analysis["total_analyzed"] == 2
            assert "result_distribution" in analysis
            assert "statistics" in analysis

    def test_validate_quality(self, sample_liveness_result):
        """Test quality validation"""
        service = LivenessService()
        
        validation = service.validate_quality(sample_liveness_result)
        
        assert "is_valid" in validation
        assert "issues" in validation
        assert "quality_score" in validation
        assert "recommendations" in validation

    def test_compare_results(self, sample_liveness_result):
        """Test result comparison"""
        service = LivenessService()
        
        result1 = sample_liveness_result
        result2 = sample_liveness_result
        result2.confidence = 0.8
        
        comparison = service.compare_results(result1, result2)
        
        assert "results_match" in comparison
        assert "confidence_difference" in comparison
        assert "score_difference" in comparison
        assert "consistency" in comparison

if __name__ == "__main__":
    pytest.main([__file__])
