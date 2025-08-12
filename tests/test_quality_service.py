import pytest
from unittest.mock import Mock, patch
from domain.entities.document_quality import DocumentQuality, QualityStatus, TamperType
from domain.services.document_quality_service import DocumentQualityService

class TestDocumentQualityService:
    
    @pytest.fixture
    def service(self):
        return DocumentQualityService()
    
    @patch('domain.services.document_quality_service.QualityAnalyzer')
    @patch('domain.services.document_quality_service.TamperDetector')
    def test_analyze_quality_success(self, mock_tamper_detector_class, mock_quality_analyzer_class, service):
        """Test phân tích chất lượng thành công"""
        # Setup mocks
        mock_analyzer = Mock()
        mock_detector = Mock()
        
        mock_analyzer.analyze_image_quality.return_value = {
            'overall_score': 0.8,
            'blur_score': 0.7,
            'glare_score': 0.2,
            'contrast_score': 0.8,
            'brightness_score': 0.9,
            'noise_score': 0.1,
            'edge_sharpness': 0.8,
            'watermark_present': False
        }
        
        mock_detector.detect_tampering.return_value = {
            'is_tampered': False,
            'tamper_type': 'none',
            'confidence': 0.1,
            'metadata_analysis': {'suspicious_indicators': []}
        }
        
        mock_quality_analyzer_class.return_value = mock_analyzer
        mock_tamper_detector_class.return_value = mock_detector
        
        # Test
        result = service.analyze_quality("test_image.jpg")
        
        # Assertions
        assert isinstance(result, DocumentQuality)
        assert result.overall_quality == QualityStatus.GOOD
        assert result.quality_score == 0.8
        assert result.tamper_detected == False
        assert result.tamper_type == TamperType.NONE
        assert result.blur_score == 0.7
        assert result.glare_score == 0.2
        assert result.contrast_score == 0.8
        assert result.brightness_score == 0.9
        assert result.noise_score == 0.1
        assert result.edge_sharpness == 0.8
        assert result.watermark_present == False
    
    @patch('domain.services.document_quality_service.QualityAnalyzer')
    @patch('domain.services.document_quality_service.TamperDetector')
    def test_analyze_quality_with_tamper(self, mock_tamper_detector_class, mock_quality_analyzer_class, service):
        """Test phân tích với tamper detected"""
        mock_analyzer = Mock()
        mock_detector = Mock()
        
        mock_analyzer.analyze_image_quality.return_value = {
            'overall_score': 0.6,
            'blur_score': 0.6,
            'glare_score': 0.3,
            'contrast_score': 0.6,
            'brightness_score': 0.7,
            'noise_score': 0.4,
            'edge_sharpness': 0.5,
            'watermark_present': True
        }
        
        mock_detector.detect_tampering.return_value = {
            'is_tampered': True,
            'tamper_type': 'digital_manipulation',
            'confidence': 0.8,
            'metadata_analysis': {'suspicious_indicators': ['editing_software_detected']}
        }
        
        mock_quality_analyzer_class.return_value = mock_analyzer
        mock_tamper_detector_class.return_value = mock_detector
        
        result = service.analyze_quality("test_image.jpg")
        
        assert result.tamper_detected == True
        assert result.tamper_type == TamperType.DIGITAL_MANIPULATION
        assert result.tamper_confidence == 0.8
        assert result.watermark_present == True
    
    def test_calculate_overall_quality(self, service):
        """Test tính toán overall quality"""
        # Test GOOD quality
        good_metrics = {'overall_score': 0.85}
        assert service._calculate_overall_quality(good_metrics) == QualityStatus.GOOD
        
        # Test FAIR quality
        fair_metrics = {'overall_score': 0.65}
        assert service._calculate_overall_quality(fair_metrics) == QualityStatus.FAIR
        
        # Test POOR quality
        poor_metrics = {'overall_score': 0.45}
        assert service._calculate_overall_quality(poor_metrics) == QualityStatus.POOR
        
        # Test REJECTED quality
        rejected_metrics = {'overall_score': 0.25}
        assert service._calculate_overall_quality(rejected_metrics) == QualityStatus.REJECTED
    
    def test_validate_quality_valid(self, service):
        """Test validation với quality hợp lệ"""
        quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.GOOD,
            quality_score=0.8,
            tamper_detected=False,
            tamper_type=TamperType.NONE,
            tamper_confidence=0.1
        )
        
        assert service.validate_quality(quality) == True
    
    def test_validate_quality_invalid_low_score(self, service):
        """Test validation với quality score thấp"""
        quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.POOR,
            quality_score=0.3,
            tamper_detected=False,
            tamper_type=TamperType.NONE
        )
        
        assert service.validate_quality(quality) == False
    
    def test_validate_quality_invalid_tamper(self, service):
        """Test validation với tamper detected"""
        quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.GOOD,
            quality_score=0.8,
            tamper_detected=True,
            tamper_type=TamperType.DIGITAL_MANIPULATION
        )
        
        assert service.validate_quality(quality) == False
    
    def test_get_quality_recommendations_blur(self, service):
        """Test khuyến nghị cho ảnh bị mờ"""
        quality = DocumentQuality(
            image_path="test.jpg",
            blur_score=0.3,
            glare_score=0.2,
            brightness_score=0.7,
            contrast_score=0.6,
            edge_sharpness=0.4,
            noise_score=0.3,
            tamper_detected=False
        )
        
        recommendations = service.get_quality_recommendations(quality)
        
        assert any("mờ" in rec for rec in recommendations)
        assert any("ổn định" in rec for rec in recommendations)
    
    def test_get_quality_recommendations_glare(self, service):
        """Test khuyến nghị cho ảnh có glare"""
        quality = DocumentQuality(
            image_path="test.jpg",
            blur_score=0.7,
            glare_score=0.8,
            brightness_score=0.7,
            contrast_score=0.6,
            edge_sharpness=0.6,
            noise_score=0.3,
            tamper_detected=False
        )
        
        recommendations = service.get_quality_recommendations(quality)
        
        assert any("chói" in rec for rec in recommendations)
        assert any("ánh sáng trực tiếp" in rec for rec in recommendations)
    
    def test_get_quality_recommendations_brightness(self, service):
        """Test khuyến nghị cho brightness"""
        # Test ảnh quá tối
        dark_quality = DocumentQuality(
            image_path="test.jpg",
            brightness_score=0.2,
            tamper_detected=False
        )
        
        dark_recommendations = service.get_quality_recommendations(dark_quality)
        assert any("tối" in rec for rec in dark_recommendations)
        
        # Test ảnh quá sáng
        bright_quality = DocumentQuality(
            image_path="test.jpg",
            brightness_score=0.9,
            tamper_detected=False
        )
        
        bright_recommendations = service.get_quality_recommendations(bright_quality)
        assert any("sáng" in rec for rec in bright_recommendations)
    
    def test_get_quality_recommendations_tamper(self, service):
        """Test khuyến nghị khi có tamper"""
        quality = DocumentQuality(
            image_path="test.jpg",
            tamper_detected=True,
            tamper_type=TamperType.DIGITAL_MANIPULATION
        )
        
        recommendations = service.get_quality_recommendations(quality)
        
        assert any("chỉnh sửa" in rec for rec in recommendations)
        assert any("digital_manipulation" in rec for rec in recommendations)
    
    def test_get_quality_recommendations_multiple_issues(self, service):
        """Test khuyến nghị với nhiều vấn đề"""
        quality = DocumentQuality(
            image_path="test.jpg",
            blur_score=0.3,
            glare_score=0.8,
            brightness_score=0.2,
            contrast_score=0.3,
            edge_sharpness=0.4,
            noise_score=0.7,
            tamper_detected=True,
            tamper_type=TamperType.COPY_PASTE
        )
        
        recommendations = service.get_quality_recommendations(quality)
        
        # Nên có nhiều khuyến nghị
        assert len(recommendations) >= 5
        assert any("mờ" in rec for rec in recommendations)
        assert any("chói" in rec for rec in recommendations)
        assert any("tối" in rec for rec in recommendations)
        assert any("tương phản" in rec for rec in recommendations)
        assert any("nhiễu" in rec for rec in recommendations)
        assert any("chỉnh sửa" in rec for rec in recommendations)
    
    @patch('domain.services.document_quality_service.QualityAnalyzer')
    @patch('domain.services.document_quality_service.TamperDetector')
    def test_analyze_quality_exception_handling(self, mock_tamper_detector_class, mock_quality_analyzer_class, service):
        """Test xử lý exception"""
        mock_quality_analyzer_class.side_effect = Exception("Test error")
        
        result = service.analyze_quality("test_image.jpg")
        
        assert result.overall_quality == QualityStatus.REJECTED
        assert result.quality_score == 0.0
        assert result.tamper_detected == True
        assert result.tamper_confidence == 0.0
