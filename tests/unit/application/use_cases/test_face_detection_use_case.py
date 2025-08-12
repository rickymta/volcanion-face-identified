import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from application.use_cases.face_detection_use_case import FaceDetectionUseCase
from domain.entities.face_detection_result import FaceDetectionResult, FaceDetectionStatus
from domain.services.face_detection_service import FaceDetectionService
from domain.repositories.face_detection_repository import FaceDetectionRepository

class TestFaceDetectionUseCase:
    
    @pytest.fixture
    def mock_face_detection_service(self):
        return Mock(spec=FaceDetectionService)
    
    @pytest.fixture
    def mock_face_detection_repository(self):
        repository = Mock(spec=FaceDetectionRepository)
        # Make all repository methods async
        repository.save = AsyncMock()
        repository.find_by_id = AsyncMock()
        repository.find_by_image_path = AsyncMock()
        repository.find_all = AsyncMock()
        repository.find_by_status = AsyncMock()
        repository.delete_by_id = AsyncMock()
        return repository
    
    @pytest.fixture
    def face_detection_use_case(self, mock_face_detection_service, mock_face_detection_repository):
        return FaceDetectionUseCase(mock_face_detection_service, mock_face_detection_repository)
    
    @pytest.fixture
    def sample_face_result(self):
        return FaceDetectionResult(
            image_path="test.jpg",
            status=FaceDetectionStatus.SUCCESS,
            bbox=[100, 100, 200, 200],
            landmarks=[(120, 130), (180, 130), (150, 160), (130, 180), (170, 180)],
            confidence=0.85,
            face_quality_score=0.75,
            alignment_score=0.80,
            occlusion_detected=False
        )
    
    @pytest.mark.asyncio
    async def test_detect_and_process_face_success(self, face_detection_use_case, 
                                                  mock_face_detection_service,
                                                  mock_face_detection_repository,
                                                  sample_face_result):
        """Test successful face detection and processing"""
        # Setup mocks
        mock_face_detection_service.detect_and_align_face.return_value = sample_face_result
        mock_face_detection_repository.save.return_value = sample_face_result
        
        # Test
        result = await face_detection_use_case.detect_and_process_face("test.jpg", "selfie")
        
        # Assertions
        assert result == sample_face_result
        mock_face_detection_service.detect_and_align_face.assert_called_once_with("test.jpg", "selfie")
        mock_face_detection_repository.save.assert_called_once_with(sample_face_result)
    
    @pytest.mark.asyncio
    async def test_detect_and_process_face_service_exception(self, face_detection_use_case,
                                                           mock_face_detection_service,
                                                           mock_face_detection_repository):
        """Test face detection when service raises exception"""
        # Setup mocks
        mock_face_detection_service.detect_and_align_face.side_effect = Exception("Service error")
        mock_face_detection_repository.save.return_value = Mock()
        
        # Test
        result = await face_detection_use_case.detect_and_process_face("test.jpg", "selfie")
        
        # Assertions
        assert result.status == "failed"
        assert result.confidence == 0.0
        mock_face_detection_repository.save.assert_called()
    
    @pytest.mark.asyncio
    async def test_get_face_detection_result_success(self, face_detection_use_case,
                                                    mock_face_detection_repository,
                                                    sample_face_result):
        """Test successful retrieval of face detection result"""
        # Setup mock
        mock_face_detection_repository.find_by_id.return_value = sample_face_result
        
        # Test
        result = await face_detection_use_case.get_face_detection_result("test_id")
        
        # Assertions
        assert result == sample_face_result
        mock_face_detection_repository.find_by_id.assert_called_once_with("test_id")
    
    @pytest.mark.asyncio
    async def test_get_face_detection_result_not_found(self, face_detection_use_case,
                                                      mock_face_detection_repository):
        """Test retrieval of non-existent face detection result"""
        # Setup mock
        mock_face_detection_repository.find_by_id.return_value = None
        
        # Test
        result = await face_detection_use_case.get_face_detection_result("nonexistent_id")
        
        # Assertions
        assert result is None
        mock_face_detection_repository.find_by_id.assert_called_once_with("nonexistent_id")
    
    @pytest.mark.asyncio
    async def test_get_face_detections_by_image(self, face_detection_use_case,
                                               mock_face_detection_repository,
                                               sample_face_result):
        """Test retrieval of face detections by image path"""
        # Setup mock
        mock_face_detection_repository.find_by_image_path.return_value = [sample_face_result]
        
        # Test
        results = await face_detection_use_case.get_face_detections_by_image("test.jpg")
        
        # Assertions
        assert len(results) == 1
        assert results[0] == sample_face_result
        mock_face_detection_repository.find_by_image_path.assert_called_once_with("test.jpg")
    
    @pytest.mark.asyncio
    async def test_validate_face_quality_success(self, face_detection_use_case,
                                                mock_face_detection_service,
                                                sample_face_result):
        """Test successful face quality validation"""
        # Setup mocks
        mock_face_detection_service.detect_and_align_face.return_value = sample_face_result
        mock_face_detection_service.get_face_recommendations.return_value = ["Good quality face"]
        mock_face_detection_service.validate_face_detection.return_value = True
        
        # Test
        result = await face_detection_use_case.validate_face_quality("test.jpg", "selfie")
        
        # Assertions
        assert result['is_valid'] is True
        assert result['face_result'] == sample_face_result
        assert result['recommendations'] == ["Good quality face"]
        assert result['quality_score'] == 0.75
        assert result['alignment_score'] == 0.80
        assert result['confidence'] == 0.85
    
    @pytest.mark.asyncio
    async def test_validate_face_quality_invalid_face(self, face_detection_use_case,
                                                     mock_face_detection_service):
        """Test face quality validation with invalid face"""
        # Create invalid face result
        invalid_face_result = FaceDetectionResult(
            image_path="test.jpg",
            status=FaceDetectionStatus.NO_FACE_DETECTED,
            confidence=0.0
        )
        
        # Setup mocks
        mock_face_detection_service.detect_and_align_face.return_value = invalid_face_result
        mock_face_detection_service.get_face_recommendations.return_value = ["No face detected"]
        mock_face_detection_service.validate_face_detection.return_value = False
        
        # Test
        result = await face_detection_use_case.validate_face_quality("test.jpg", "selfie")
        
        # Assertions
        assert result['is_valid'] is False
        assert result['face_result'] == invalid_face_result
        assert result['recommendations'] == ["No face detected"]
        assert result['confidence'] == 0.0
    
    @pytest.mark.asyncio
    async def test_compare_face_alignment_success(self, face_detection_use_case,
                                                 mock_face_detection_service,
                                                 sample_face_result):
        """Test successful face alignment comparison"""
        # Create second face result
        face_result_2 = FaceDetectionResult(
            image_path="test2.jpg",
            status=FaceDetectionStatus.SUCCESS,
            bbox=[110, 110, 210, 210],
            landmarks=[(125, 135), (185, 135), (155, 165), (135, 185), (175, 185)],
            confidence=0.80,
            face_quality_score=0.70,
            alignment_score=0.75,
            occlusion_detected=False
        )
        
        # Setup mocks
        mock_face_detection_service.detect_and_align_face.side_effect = [sample_face_result, face_result_2]
        mock_face_detection_service.compare_face_alignment.return_value = 0.85
        mock_face_detection_service.validate_face_detection.side_effect = [True, True]
        mock_face_detection_service.get_face_recommendations.side_effect = [
            ["Good face 1"], ["Good face 2"]
        ]
        
        # Test
        result = await face_detection_use_case.compare_face_alignment("test1.jpg", "test2.jpg")
        
        # Assertions
        assert result['alignment_similarity'] == 0.85
        assert result['face1_valid'] is True
        assert result['face2_valid'] is True
        assert result['overall_valid'] is True
        assert result['face1_result'] == sample_face_result
        assert result['face2_result'] == face_result_2
        assert result['face1_recommendations'] == ["Good face 1"]
        assert result['face2_recommendations'] == ["Good face 2"]
    
    @pytest.mark.asyncio
    async def test_compare_face_alignment_low_similarity(self, face_detection_use_case,
                                                        mock_face_detection_service,
                                                        sample_face_result):
        """Test face alignment comparison with low similarity"""
        # Create second face result
        face_result_2 = FaceDetectionResult(
            image_path="test2.jpg",
            status=FaceDetectionStatus.SUCCESS,
            confidence=0.80
        )
        
        # Setup mocks
        mock_face_detection_service.detect_and_align_face.side_effect = [sample_face_result, face_result_2]
        mock_face_detection_service.compare_face_alignment.return_value = 0.3  # Low similarity
        mock_face_detection_service.validate_face_detection.side_effect = [True, True]
        mock_face_detection_service.get_face_recommendations.side_effect = [[], []]
        
        # Test
        result = await face_detection_use_case.compare_face_alignment("test1.jpg", "test2.jpg")
        
        # Assertions
        assert result['alignment_similarity'] == 0.3
        assert result['overall_valid'] is False  # Low similarity should make overall invalid
    
    @pytest.mark.asyncio
    async def test_get_face_detection_statistics_with_data(self, face_detection_use_case,
                                                          mock_face_detection_repository):
        """Test face detection statistics with sample data"""
        # Create sample face detection results
        sample_results = [
            FaceDetectionResult(
                image_path="test1.jpg",
                status=FaceDetectionStatus.SUCCESS,
                confidence=0.85,
                face_quality_score=0.75,
                alignment_score=0.80,
                occlusion_detected=False
            ),
            FaceDetectionResult(
                image_path="test2.jpg",
                status=FaceDetectionStatus.SUCCESS,
                confidence=0.90,
                face_quality_score=0.80,
                alignment_score=0.85,
                occlusion_detected=True,
                occlusion_type="glasses"
            ),
            FaceDetectionResult(
                image_path="test3.jpg",
                status=FaceDetectionStatus.NO_FACE_DETECTED,
                confidence=0.0
            )
        ]
        
        # Setup mock
        mock_face_detection_repository.find_all.return_value = sample_results
        
        # Test
        stats = await face_detection_use_case.get_face_detection_statistics()
        
        # Assertions
        assert stats['total_detections'] == 3
        assert stats['success_rate'] == 66.67  # 2/3 successful
        assert stats['average_confidence'] == 0.875  # (0.85 + 0.90) / 2
        assert stats['average_quality_score'] == 0.775  # (0.75 + 0.80) / 2
        assert stats['average_alignment_score'] == 0.825  # (0.80 + 0.85) / 2
        assert stats['status_distribution']['success'] == 2
        assert stats['status_distribution']['no_face_detected'] == 1
        assert stats['occlusion_distribution']['glasses'] == 1
    
    @pytest.mark.asyncio
    async def test_get_face_detection_statistics_no_data(self, face_detection_use_case,
                                                        mock_face_detection_repository):
        """Test face detection statistics with no data"""
        # Setup mock
        mock_face_detection_repository.find_all.return_value = []
        
        # Test
        stats = await face_detection_use_case.get_face_detection_statistics()
        
        # Assertions
        assert stats['total_detections'] == 0
        assert stats['success_rate'] == 0.0
        assert stats['average_confidence'] == 0.0
        assert stats['average_quality_score'] == 0.0
        assert stats['average_alignment_score'] == 0.0
        assert stats['status_distribution'] == {}
        assert stats['occlusion_distribution'] == {}
    
    @pytest.mark.asyncio
    async def test_get_face_detection_statistics_exception(self, face_detection_use_case,
                                                          mock_face_detection_repository):
        """Test face detection statistics when repository raises exception"""
        # Setup mock
        mock_face_detection_repository.find_all.side_effect = Exception("Database error")
        
        # Test
        stats = await face_detection_use_case.get_face_detection_statistics()
        
        # Assertions
        assert stats['total_detections'] == 0
        assert stats['success_rate'] == 0.0
        assert 'error' in stats
    
    @pytest.mark.asyncio
    async def test_validate_face_quality_exception(self, face_detection_use_case,
                                                  mock_face_detection_service):
        """Test face quality validation when service raises exception"""
        # Setup mock
        mock_face_detection_service.detect_and_align_face.side_effect = Exception("Service error")
        
        # Test
        result = await face_detection_use_case.validate_face_quality("test.jpg", "selfie")
        
        # Assertions
        assert result['is_valid'] is False
        assert result['face_result'] is None
        assert "Lỗi xử lý ảnh" in result['recommendations'][0]
        assert result['quality_score'] == 0.0
        assert result['alignment_score'] == 0.0
        assert result['confidence'] == 0.0
    
    @pytest.mark.asyncio
    async def test_compare_face_alignment_exception(self, face_detection_use_case,
                                                   mock_face_detection_service):
        """Test face alignment comparison when service raises exception"""
        # Setup mock
        mock_face_detection_service.detect_and_align_face.side_effect = Exception("Service error")
        
        # Test
        result = await face_detection_use_case.compare_face_alignment("test1.jpg", "test2.jpg")
        
        # Assertions
        assert result['alignment_similarity'] == 0.0
        assert result['face1_valid'] is False
        assert result['face2_valid'] is False
        assert result['overall_valid'] is False
        assert "Lỗi xử lý ảnh" in result['face1_recommendations'][0]
        assert "Lỗi xử lý ảnh" in result['face2_recommendations'][0]
