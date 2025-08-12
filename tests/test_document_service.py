import pytest
from unittest.mock import Mock, patch
from domain.entities.document import Document, DocumentType
from domain.services.document_service import DocumentService

class TestDocumentService:
    
    @pytest.fixture
    def service(self):
        return DocumentService()
    
    @patch('domain.services.document_service.DocumentDetector')
    def test_classify_document_success(self, mock_detector_class, service):
        """Test phân loại giấy tờ thành công"""
        # Setup mock
        mock_detector = Mock()
        mock_detector.detect.return_value = [10, 10, 100, 100]
        mock_detector.classify.return_value = 'cmnd'
        mock_detector.get_document_confidence.return_value = 0.8
        mock_detector_class.return_value = mock_detector
        
        # Test
        result = service.classify_document("test_image.jpg")
        
        # Assertions
        assert isinstance(result, Document)
        assert result.doc_type == DocumentType.CMND
        assert result.bbox == [10, 10, 100, 100]
        assert result.confidence == 0.8
        assert result.image_path == "test_image.jpg"
    
    @patch('domain.services.document_service.DocumentDetector')
    def test_classify_document_unknown_type(self, mock_detector_class, service):
        """Test với loại giấy tờ không xác định"""
        mock_detector = Mock()
        mock_detector.detect.return_value = None
        mock_detector.classify.return_value = 'invalid_type'
        mock_detector.get_document_confidence.return_value = 0.1
        mock_detector_class.return_value = mock_detector
        
        result = service.classify_document("test_image.jpg")
        
        assert result.doc_type == DocumentType.UNKNOWN
        assert result.bbox is None
        assert result.confidence == 0.1
    
    @patch('domain.services.document_service.DocumentDetector')
    def test_classify_document_exception(self, mock_detector_class, service):
        """Test xử lý exception"""
        mock_detector_class.side_effect = Exception("Test error")
        
        result = service.classify_document("test_image.jpg")
        
        assert result.doc_type == DocumentType.UNKNOWN
        assert result.bbox is None
        assert result.confidence == 0.0
    
    def test_validate_document_valid(self, service):
        """Test validation với document hợp lệ"""
        doc = Document(
            image_path="test.jpg",
            doc_type=DocumentType.CMND,
            bbox=[10, 10, 100, 100],
            confidence=0.8
        )
        
        assert service.validate_document(doc) is True
    
    def test_validate_document_low_confidence(self, service):
        """Test validation với confidence thấp"""
        doc = Document(
            image_path="test.jpg",
            doc_type=DocumentType.CMND,
            bbox=[10, 10, 100, 100],
            confidence=0.2
        )
        
        assert service.validate_document(doc) is False
    
    def test_validate_document_no_bbox(self, service):
        """Test validation không có bbox"""
        doc = Document(
            image_path="test.jpg",
            doc_type=DocumentType.CMND,
            bbox=None,
            confidence=0.8
        )
        
        assert service.validate_document(doc) is False
    
    def test_validate_document_unknown_type(self, service):
        """Test validation với loại unknown"""
        doc = Document(
            image_path="test.jpg",
            doc_type=DocumentType.UNKNOWN,
            bbox=[10, 10, 100, 100],
            confidence=0.8
        )
        
        assert service.validate_document(doc) is False
