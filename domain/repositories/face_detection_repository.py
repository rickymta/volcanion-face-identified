from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entities.face_detection_result import FaceDetectionResult

class FaceDetectionRepository(ABC):
    @abstractmethod
    def save(self, face_result: FaceDetectionResult) -> str:
        """Lưu face detection result và trả về ID"""
        pass

    @abstractmethod
    def get_by_id(self, result_id: str) -> Optional[FaceDetectionResult]:
        """Lấy face detection result theo ID"""
        pass
    
    @abstractmethod
    def get_by_image_path(self, image_path: str) -> Optional[FaceDetectionResult]:
        """Lấy face detection result theo đường dẫn ảnh"""
        pass
    
    @abstractmethod
    def get_all(self) -> List[FaceDetectionResult]:
        """Lấy tất cả face detection results"""
        pass
    
    @abstractmethod
    def delete_by_id(self, result_id: str) -> bool:
        """Xóa face detection result theo ID"""
        pass
    
    @abstractmethod
    def get_by_status(self, status: str) -> List[FaceDetectionResult]:
        """Lấy face detection results theo status"""
        pass
