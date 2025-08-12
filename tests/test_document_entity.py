import pytest
from datetime import datetime
from domain.entities.document import Document, DocumentType

class TestDocumentEntity:
    
    def test_document_creation_default(self):
        """Test tạo document với giá trị mặc định"""
        doc = Document(image_path="test.jpg")
        
        assert doc.image_path == "test.jpg"
        assert doc.doc_type == DocumentType.UNKNOWN
        assert doc.bbox is None
        assert doc.confidence == 0.0
        assert isinstance(doc.created_at, datetime)
    
    def test_document_creation_full_params(self):
        """Test tạo document với đầy đủ tham số"""
        created_time = datetime.now()
        doc = Document(
            image_path="test.jpg",
            doc_type=DocumentType.CMND,
            bbox=[10, 10, 100, 100],
            confidence=0.8,
            created_at=created_time
        )
        
        assert doc.image_path == "test.jpg"
        assert doc.doc_type == DocumentType.CMND
        assert doc.bbox == [10, 10, 100, 100]
        assert doc.confidence == 0.8
        assert doc.created_at == created_time
    
    def test_to_dict(self):
        """Test chuyển đổi document thành dictionary"""
        created_time = datetime.now()
        doc = Document(
            image_path="test.jpg",
            doc_type=DocumentType.PASSPORT,
            bbox=[20, 20, 200, 200],
            confidence=0.9,
            created_at=created_time
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["image_path"] == "test.jpg"
        assert doc_dict["doc_type"] == "passport"
        assert doc_dict["bbox"] == [20, 20, 200, 200]
        assert doc_dict["confidence"] == 0.9
        assert doc_dict["created_at"] == created_time
    
    def test_from_dict(self):
        """Test tạo document từ dictionary"""
        created_time = datetime.now()
        doc_dict = {
            "image_path": "test.jpg",
            "doc_type": "cmnd",
            "bbox": [10, 10, 100, 100],
            "confidence": 0.7,
            "created_at": created_time
        }
        
        doc = Document.from_dict(doc_dict)
        
        assert doc.image_path == "test.jpg"
        assert doc.doc_type == DocumentType.CMND
        assert doc.bbox == [10, 10, 100, 100]
        assert doc.confidence == 0.7
        assert doc.created_at == created_time
    
    def test_from_dict_minimal(self):
        """Test tạo document từ dictionary với thông tin tối thiểu"""
        doc_dict = {
            "image_path": "test.jpg",
            "doc_type": "unknown"
        }
        
        doc = Document.from_dict(doc_dict)
        
        assert doc.image_path == "test.jpg"
        assert doc.doc_type == DocumentType.UNKNOWN
        assert doc.bbox is None
        assert doc.confidence == 0.0
        assert doc.created_at is None
    
    def test_document_type_enum_values(self):
        """Test các giá trị của DocumentType enum"""
        assert DocumentType.CMND.value == "cmnd"
        assert DocumentType.PASSPORT.value == "passport"
        assert DocumentType.CCCD.value == "cccd"
        assert DocumentType.UNKNOWN.value == "unknown"
        
        # Test tạo DocumentType từ string
        assert DocumentType("cmnd") == DocumentType.CMND
        assert DocumentType("passport") == DocumentType.PASSPORT
        assert DocumentType("cccd") == DocumentType.CCCD
        assert DocumentType("unknown") == DocumentType.UNKNOWN
    
    def test_document_type_invalid_value(self):
        """Test tạo DocumentType với giá trị không hợp lệ"""
        with pytest.raises(ValueError):
            DocumentType("invalid_type")
