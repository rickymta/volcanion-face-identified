import pytest
from unittest.mock import Mock, patch
from application.use_cases.document_quality_check_usecase import DocumentQualityCheckUseCase
from domain.services.document_quality_service import DocumentQualityService
from domain.repositories.document_quality_repository import DocumentQualityRepository
from domain.entities.document_quality import DocumentQuality, QualityStatus, TamperType

class TestDocumentQualityCheckUseCase:
    
    @pytest.fixture
    def mock_service(self):
        return Mock(spec=DocumentQualityService)
    
    @pytest.fixture
    def mock_repository(self):
        return Mock(spec=DocumentQualityRepository)
    
    @pytest.fixture
    def usecase(self, mock_service, mock_repository):
        return DocumentQualityCheckUseCase(mock_service, mock_repository)
    
    def test_execute_success_with_db_save(self, usecase, mock_service, mock_repository):
        """Test thực hiện use case thành công với lưu DB"""
        # Setup mock
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.GOOD,
            quality_score=0.8,
            tamper_detected=False,
            tamper_type=TamperType.NONE
        )
        mock_service.analyze_quality.return_value = mock_quality
        mock_service.validate_quality.return_value = True
        mock_service.get_quality_recommendations.return_value = ["Chất lượng tốt"]
        mock_repository.save.return_value = "quality_id_123"
        
        # Execute
        result = usecase.execute("test.jpg", save_to_db=True)
        
        # Assertions
        assert result == mock_quality
        mock_service.analyze_quality.assert_called_once_with("test.jpg", None)
        mock_service.validate_quality.assert_called_once_with(mock_quality)
        mock_repository.save.assert_called_once_with(mock_quality)
    
    def test_execute_success_without_db_save(self, usecase, mock_service, mock_repository):
        """Test thực hiện use case thành công không lưu DB"""
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.FAIR,
            quality_score=0.6,
            tamper_detected=False,
            tamper_type=TamperType.NONE
        )
        mock_service.analyze_quality.return_value = mock_quality
        mock_service.validate_quality.return_value = True
        mock_service.get_quality_recommendations.return_value = ["Cần cải thiện độ sắc nét"]
        
        result = usecase.execute("test.jpg", save_to_db=False)
        
        assert result == mock_quality
        mock_service.analyze_quality.assert_called_once_with("test.jpg", None)
        mock_repository.save.assert_not_called()
    
    def test_execute_with_bbox(self, usecase, mock_service, mock_repository):
        """Test thực hiện use case với bbox"""
        bbox = [10, 10, 100, 100]
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.GOOD,
            quality_score=0.9
        )
        mock_service.analyze_quality.return_value = mock_quality
        mock_service.validate_quality.return_value = True
        mock_service.get_quality_recommendations.return_value = []
        
        result = usecase.execute("test.jpg", bbox=bbox)
        
        mock_service.analyze_quality.assert_called_once_with("test.jpg", bbox)
    
    def test_execute_with_repository_save_error(self, usecase, mock_service, mock_repository):
        """Test xử lý lỗi khi lưu vào repository"""
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.GOOD,
            quality_score=0.8
        )
        mock_service.analyze_quality.return_value = mock_quality
        mock_service.validate_quality.return_value = True
        mock_service.get_quality_recommendations.return_value = []
        mock_repository.save.side_effect = Exception("Database error")
        
        # Không nên raise exception, chỉ log error
        result = usecase.execute("test.jpg", save_to_db=True)
        
        assert result == mock_quality
        mock_repository.save.assert_called_once_with(mock_quality)
    
    def test_execute_with_service_error(self, usecase, mock_service, mock_repository):
        """Test xử lý lỗi từ service"""
        mock_service.analyze_quality.side_effect = Exception("Service error")
        
        with pytest.raises(Exception):
            usecase.execute("test.jpg")
    
    def test_execute_without_repository(self, mock_service):
        """Test use case không có repository"""
        usecase = DocumentQualityCheckUseCase(mock_service, None)
        
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.GOOD,
            quality_score=0.8
        )
        mock_service.analyze_quality.return_value = mock_quality
        mock_service.validate_quality.return_value = True
        mock_service.get_quality_recommendations.return_value = []
        
        result = usecase.execute("test.jpg", save_to_db=True)
        
        assert result == mock_quality
        mock_service.analyze_quality.assert_called_once_with("test.jpg", None)
    
    def test_get_recommendations_success(self, usecase, mock_service):
        """Test lấy recommendations thành công"""
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            blur_score=0.3,
            glare_score=0.8
        )
        mock_service.analyze_quality.return_value = mock_quality
        mock_service.get_quality_recommendations.return_value = [
            "Ảnh bị mờ, hãy chụp lại với camera ổn định hơn",
            "Có ánh sáng chói, hãy tránh ánh sáng trực tiếp"
        ]
        
        recommendations = usecase.get_recommendations("test.jpg")
        
        assert len(recommendations) == 2
        assert "mờ" in recommendations[0]
        assert "chói" in recommendations[1]
        mock_service.analyze_quality.assert_called_once_with("test.jpg", None)
        mock_service.get_quality_recommendations.assert_called_once_with(mock_quality)
    
    def test_get_recommendations_with_bbox(self, usecase, mock_service):
        """Test lấy recommendations với bbox"""
        bbox = [20, 20, 200, 200]
        mock_quality = DocumentQuality(image_path="test.jpg")
        mock_service.analyze_quality.return_value = mock_quality
        mock_service.get_quality_recommendations.return_value = ["Test recommendation"]
        
        recommendations = usecase.get_recommendations("test.jpg", bbox)
        
        mock_service.analyze_quality.assert_called_once_with("test.jpg", bbox)
        assert recommendations == ["Test recommendation"]
    
    def test_get_recommendations_error(self, usecase, mock_service):
        """Test xử lý lỗi khi lấy recommendations"""
        mock_service.analyze_quality.side_effect = Exception("Analysis error")
        
        recommendations = usecase.get_recommendations("test.jpg")
        
        assert len(recommendations) == 1
        assert "lỗi" in recommendations[0]
    
    def test_execute_tamper_detected(self, usecase, mock_service, mock_repository):
        """Test với tamper được phát hiện"""
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.REJECTED,
            quality_score=0.3,
            tamper_detected=True,
            tamper_type=TamperType.DIGITAL_MANIPULATION,
            tamper_confidence=0.9
        )
        mock_service.analyze_quality.return_value = mock_quality
        mock_service.validate_quality.return_value = False  # Invalid vì có tamper
        mock_service.get_quality_recommendations.return_value = [
            "Phát hiện dấu hiệu chỉnh sửa (digital_manipulation)"
        ]
        
        result = usecase.execute("test.jpg")
        
        assert result.tamper_detected == True
        assert result.tamper_type == TamperType.DIGITAL_MANIPULATION
        assert result.overall_quality == QualityStatus.REJECTED
        mock_service.validate_quality.assert_called_once_with(mock_quality)
