import pytest
from unittest.mock import Mock, patch
from application.use_cases.document_detection_usecase import DocumentDetectionUseCase
from domain.services.document_service import DocumentService
from domain.repositories.document_repository import DocumentRepository
from domain.entities.document import Document, DocumentType

class TestDocumentDetectionUseCase:
    
    @pytest.fixture
    def mock_service(self):
        return Mock(spec=DocumentService)
    
    @pytest.fixture
    def mock_repository(self):
        return Mock(spec=DocumentRepository)
    
    @pytest.fixture
    def usecase(self, mock_service, mock_repository):
        return DocumentDetectionUseCase(mock_service, mock_repository)
    
    def test_execute_success_with_db_save(self, usecase, mock_service, mock_repository):
        """Test thực hiện use case thành công với lưu DB"""
        # Setup mock
        mock_document = Document(
            image_path="test.jpg",
            doc_type=DocumentType.CMND,
            bbox=[10, 10, 100, 100],
            confidence=0.8
        )
        mock_service.classify_document.return_value = mock_document
        mock_service.validate_document.return_value = True
        mock_repository.save.return_value = "doc_id_123"
        
        # Execute
        result = usecase.execute("test.jpg", save_to_db=True)
        
        # Assertions
        assert result == mock_document
        mock_service.classify_document.assert_called_once_with("test.jpg")
        mock_service.validate_document.assert_called_once_with(mock_document)
        mock_repository.save.assert_called_once_with(mock_document)
    
    def test_execute_success_without_db_save(self, usecase, mock_service, mock_repository):
        """Test thực hiện use case thành công không lưu DB"""
        mock_document = Document(
            image_path="test.jpg",
            doc_type=DocumentType.PASSPORT,
            bbox=[20, 20, 200, 200],
            confidence=0.9
        )
        mock_service.classify_document.return_value = mock_document
        mock_service.validate_document.return_value = True
        
        result = usecase.execute("test.jpg", save_to_db=False)
        
        assert result == mock_document
        mock_service.classify_document.assert_called_once_with("test.jpg")
        mock_repository.save.assert_not_called()
    
    def test_execute_with_repository_save_error(self, usecase, mock_service, mock_repository):
        """Test xử lý lỗi khi lưu vào repository"""
        mock_document = Document(
            image_path="test.jpg",
            doc_type=DocumentType.CMND,
            bbox=[10, 10, 100, 100],
            confidence=0.8
        )
        mock_service.classify_document.return_value = mock_document
        mock_service.validate_document.return_value = True
        mock_repository.save.side_effect = Exception("Database error")
        
        # Không nên raise exception, chỉ log error
        result = usecase.execute("test.jpg", save_to_db=True)
        
        assert result == mock_document
        mock_repository.save.assert_called_once_with(mock_document)
    
    def test_execute_with_service_error(self, usecase, mock_service, mock_repository):
        """Test xử lý lỗi từ service"""
        mock_service.classify_document.side_effect = Exception("Service error")
        
        with pytest.raises(Exception):
            usecase.execute("test.jpg")
    
    def test_execute_without_repository(self, mock_service):
        """Test use case không có repository"""
        usecase = DocumentDetectionUseCase(mock_service, None)
        
        mock_document = Document(
            image_path="test.jpg",
            doc_type=DocumentType.CMND,
            bbox=[10, 10, 100, 100],
            confidence=0.8
        )
        mock_service.classify_document.return_value = mock_document
        mock_service.validate_document.return_value = True
        
        result = usecase.execute("test.jpg", save_to_db=True)
        
        assert result == mock_document
        mock_service.classify_document.assert_called_once_with("test.jpg")
