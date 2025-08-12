from typing import List, Optional
from pymongo import MongoClient
from bson import ObjectId
import logging
from domain.repositories.document_quality_repository import DocumentQualityRepository
from domain.entities.document_quality import DocumentQuality

class DocumentQualityRepositoryImpl(DocumentQualityRepository):
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", 
                 database_name: str = "volcanion_face_db"):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.collection = self.db.document_qualities
        self.logger = logging.getLogger(__name__)
        
    def save(self, quality: DocumentQuality) -> str:
        """Lưu document quality vào MongoDB và trả về ID"""
        try:
            quality_dict = quality.to_dict()
            result = self.collection.insert_one(quality_dict)
            return str(result.inserted_id)
        except Exception as e:
            self.logger.error(f"Error saving document quality: {e}")
            raise

    def get_by_id(self, quality_id: str) -> Optional[DocumentQuality]:
        """Lấy document quality từ MongoDB theo ID"""
        try:
            quality_data = self.collection.find_one({"_id": ObjectId(quality_id)})
            if quality_data:
                # Loại bỏ _id từ dict
                quality_data.pop('_id', None)
                return DocumentQuality.from_dict(quality_data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting quality by ID: {e}")
            return None
    
    def get_by_image_path(self, image_path: str) -> Optional[DocumentQuality]:
        """Lấy document quality theo đường dẫn ảnh"""
        try:
            quality_data = self.collection.find_one({"image_path": image_path})
            if quality_data:
                quality_data.pop('_id', None)
                return DocumentQuality.from_dict(quality_data)
            return None
        except Exception as e:
            self.logger.error(f"Error getting quality by image path: {e}")
            return None
    
    def get_all(self) -> List[DocumentQuality]:
        """Lấy tất cả document qualities từ MongoDB"""
        try:
            qualities = []
            cursor = self.collection.find()
            for quality_data in cursor:
                quality_data.pop('_id', None)
                qualities.append(DocumentQuality.from_dict(quality_data))
            return qualities
        except Exception as e:
            self.logger.error(f"Error getting all qualities: {e}")
            return []
    
    def delete_by_id(self, quality_id: str) -> bool:
        """Xóa document quality theo ID"""
        try:
            result = self.collection.delete_one({"_id": ObjectId(quality_id)})
            return result.deleted_count > 0
        except Exception as e:
            self.logger.error(f"Error deleting quality: {e}")
            return False
            
    def close(self):
        """Đóng kết nối database"""
        if self.client:
            self.client.close()
