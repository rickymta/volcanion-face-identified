from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entities.document_quality import DocumentQuality

class DocumentQualityRepository(ABC):
    @abstractmethod
    def save(self, quality: DocumentQuality) -> str:
        """Lưu document quality và trả về ID"""
        pass

    @abstractmethod
    def get_by_id(self, quality_id: str) -> Optional[DocumentQuality]:
        """Lấy document quality theo ID"""
        pass
    
    @abstractmethod
    def get_by_image_path(self, image_path: str) -> Optional[DocumentQuality]:
        """Lấy document quality theo đường dẫn ảnh"""
        pass
    
    @abstractmethod
    def get_all(self) -> List[DocumentQuality]:
        """Lấy tất cả document qualities"""
        pass
    
    @abstractmethod
    def delete_by_id(self, quality_id: str) -> bool:
        """Xóa document quality theo ID"""
        pass
