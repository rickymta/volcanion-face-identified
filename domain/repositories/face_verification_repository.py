from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entities.face_verification_result import FaceEmbedding, FaceVerificationResult

class FaceEmbeddingRepository(ABC):
    """Repository interface cho face embeddings"""
    
    @abstractmethod
    async def save_embedding(self, embedding: FaceEmbedding) -> FaceEmbedding:
        """Lưu face embedding"""
        pass
    
    @abstractmethod
    async def find_embedding_by_id(self, embedding_id: str) -> Optional[FaceEmbedding]:
        """Tìm embedding theo ID"""
        pass
    
    @abstractmethod
    async def find_embeddings_by_image_path(self, image_path: str) -> List[FaceEmbedding]:
        """Tìm embeddings theo đường dẫn ảnh"""
        pass
    
    @abstractmethod
    async def find_embeddings_by_model(self, model_name: str) -> List[FaceEmbedding]:
        """Tìm embeddings theo model"""
        pass
    
    @abstractmethod
    async def delete_embedding(self, embedding_id: str) -> bool:
        """Xóa embedding"""
        pass
    
    @abstractmethod
    async def get_all_embeddings(self) -> List[FaceEmbedding]:
        """Lấy tất cả embeddings"""
        pass

class FaceVerificationRepository(ABC):
    """Repository interface cho face verification results"""
    
    @abstractmethod
    async def save_verification(self, verification: FaceVerificationResult) -> FaceVerificationResult:
        """Lưu verification result"""
        pass
    
    @abstractmethod
    async def find_verification_by_id(self, verification_id: str) -> Optional[FaceVerificationResult]:
        """Tìm verification result theo ID"""
        pass
    
    @abstractmethod
    async def find_verifications_by_images(self, ref_image: str, target_image: str) -> List[FaceVerificationResult]:
        """Tìm verification results theo cặp ảnh"""
        pass
    
    @abstractmethod
    async def find_verifications_by_status(self, status: str) -> List[FaceVerificationResult]:
        """Tìm verification results theo status"""
        pass
    
    @abstractmethod
    async def find_verifications_by_result(self, result: str) -> List[FaceVerificationResult]:
        """Tìm verification results theo kết quả"""
        pass
    
    @abstractmethod
    async def get_verification_statistics(self) -> dict:
        """Lấy thống kê verification"""
        pass
    
    @abstractmethod
    async def delete_verification(self, verification_id: str) -> bool:
        """Xóa verification result"""
        pass
    
    @abstractmethod
    async def get_recent_verifications(self, limit: int = 50) -> List[FaceVerificationResult]:
        """Lấy verification results gần đây"""
        pass
