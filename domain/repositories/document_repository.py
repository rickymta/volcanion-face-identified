from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entities.document import Document

class DocumentRepository(ABC):
    @abstractmethod
    def save(self, document: Document) -> str:
        """Lưu document và trả về ID"""
        pass

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Lấy document theo ID"""
        pass
    
    @abstractmethod
    def get_all(self) -> List[Document]:
        """Lấy tất cả documents"""
        pass
    
    @abstractmethod
    def delete_by_id(self, doc_id: str) -> bool:
        """Xóa document theo ID"""
        pass
