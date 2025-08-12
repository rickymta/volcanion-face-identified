from typing import List, Optional
from pymongo import MongoClient
from bson import ObjectId
import logging
from domain.repositories.document_repository import DocumentRepository
from domain.entities.document import Document

class DocumentRepositoryImpl(DocumentRepository):
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", 
                 database_name: str = "volcanion_face_db"):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db.documents
        self.logger = logging.getLogger(__name__)
        
    def save(self, document: Document) -> str:
        """Lưu document vào MongoDB và trả về ID"""
        try:
            doc_dict = document.to_dict()
            result = self.collection.insert_one(doc_dict)
            return str(result.inserted_id)
        except Exception as e:
            self.logger.error(f"Error saving document: {e}")
            raise

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Lấy document từ MongoDB theo ID"""
        try:
            doc_data = self.collection.find_one({"_id": ObjectId(doc_id)})
            if doc_data:
                # Loại bỏ _id từ dict
                doc_data.pop('_id', None)
                return Document.from_dict(doc_data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting document by ID: {e}")
            return None
    
    def get_all(self) -> List[Document]:
        """Lấy tất cả documents từ MongoDB"""
        try:
            documents = []
            cursor = self.collection.find()
            for doc_data in cursor:
                doc_data.pop('_id', None)
                documents.append(Document.from_dict(doc_data))
            return documents
        except Exception as e:
            self.logger.error(f"Error getting all documents: {e}")
            return []
    
    def delete_by_id(self, doc_id: str) -> bool:
        """Xóa document theo ID"""
        try:
            result = self.collection.delete_one({"_id": ObjectId(doc_id)})
            return result.deleted_count > 0
        except Exception as e:
            self.logger.error(f"Error deleting document: {e}")
            return False
            
    def close(self):
        """Đóng kết nối database"""
        if self.client:
            self.client.close()
