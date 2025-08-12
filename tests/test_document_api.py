import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import UploadFile
from io import BytesIO
from main import app
from domain.entities.document import Document, DocumentType

# Test client
client = TestClient(app)

class TestDocumentAPI:
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_document_health_endpoint(self):
        """Test document health endpoint"""
        response = client.get("/api/v1/document/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @patch('presentation.api.document_api.DocumentDetectionUseCase')
    def test_detect_document_success(self, mock_usecase_class):
        """Test phát hiện giấy tờ thành công"""
        # Setup mock
        mock_usecase = Mock()
        mock_document = Document(
            image_path="test.jpg",
            doc_type=DocumentType.CMND,
            bbox=[10, 10, 100, 100],
            confidence=0.8
        )
        mock_usecase.execute.return_value = mock_document
        mock_usecase_class.return_value = mock_usecase
        
        # Tạo file giả
        test_file = BytesIO(b"fake image content")
        
        response = client.post(
            "/api/v1/document/detect",
            files={"file": ("test.jpg", test_file, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_type"] == "cmnd"
        assert data["bbox"] == [10, 10, 100, 100]
        assert data["confidence"] == 0.8
        assert "message" in data
    
    def test_detect_document_invalid_file_type(self):
        """Test với file không phải ảnh"""
        test_file = BytesIO(b"not an image")
        
        response = client.post(
            "/api/v1/document/detect",
            files={"file": ("test.txt", test_file, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]
    
    @patch('presentation.api.document_api.DocumentRepositoryImpl')
    def test_list_documents(self, mock_repo_class):
        """Test lấy danh sách documents"""
        # Setup mock
        mock_repo = Mock()
        mock_documents = [
            Document(
                image_path="test1.jpg",
                doc_type=DocumentType.CMND,
                bbox=[10, 10, 100, 100],
                confidence=0.8
            ),
            Document(
                image_path="test2.jpg",
                doc_type=DocumentType.PASSPORT,
                bbox=[20, 20, 200, 200],
                confidence=0.9
            )
        ]
        mock_repo.get_all.return_value = mock_documents
        mock_repo_class.return_value = mock_repo
        
        response = client.get("/api/v1/document/list")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["documents"]) == 2
        assert data["documents"][0]["doc_type"] == "cmnd"
        assert data["documents"][1]["doc_type"] == "passport"
    
    @patch('presentation.api.document_api.DocumentRepositoryImpl')
    def test_delete_document_success(self, mock_repo_class):
        """Test xóa document thành công"""
        mock_repo = Mock()
        mock_repo.delete_by_id.return_value = True
        mock_repo_class.return_value = mock_repo
        
        response = client.delete("/api/v1/document/test_id")
        
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
    
    @patch('presentation.api.document_api.DocumentRepositoryImpl')
    def test_delete_document_not_found(self, mock_repo_class):
        """Test xóa document không tồn tại"""
        mock_repo = Mock()
        mock_repo.delete_by_id.return_value = False
        mock_repo_class.return_value = mock_repo
        
        response = client.delete("/api/v1/document/nonexistent_id")
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]
