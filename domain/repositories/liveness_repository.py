from abc import ABC, abstractmethod
from typing import List, Optional
from domain.entities.liveness_result import LivenessDetectionResult

class LivenessRepository(ABC):
    """Repository interface cho liveness detection results"""
    
    @abstractmethod
    async def save(self, result: LivenessDetectionResult) -> LivenessDetectionResult:
        """Lưu liveness detection result"""
        pass
    
    @abstractmethod
    async def find_by_id(self, result_id: str) -> Optional[LivenessDetectionResult]:
        """Tìm liveness result theo ID"""
        pass
    
    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[LivenessDetectionResult]:
        """Lấy tất cả liveness results"""
        pass
    
    @abstractmethod
    async def find_by_image_path(self, image_path: str) -> List[LivenessDetectionResult]:
        """Tìm liveness results theo image path"""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: str, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm liveness results theo status"""
        pass
    
    @abstractmethod
    async def find_by_result(self, liveness_result: str, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm liveness results theo kết quả (REAL/FAKE/UNCERTAIN)"""
        pass
    
    @abstractmethod
    async def find_real_faces(self, confidence_threshold: float = 0.8, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm các khuôn mặt thật với confidence cao"""
        pass
    
    @abstractmethod
    async def find_fake_faces(self, confidence_threshold: float = 0.8, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm các khuôn mặt giả được phát hiện"""
        pass
    
    @abstractmethod
    async def find_by_spoof_type(self, spoof_type: str, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm liveness results theo loại spoof attack"""
        pass
    
    @abstractmethod
    async def find_recent_results(self, hours: int = 24, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm liveness results gần đây"""
        pass
    
    @abstractmethod
    async def update(self, result: LivenessDetectionResult) -> LivenessDetectionResult:
        """Cập nhật liveness detection result"""
        pass
    
    @abstractmethod
    async def delete_by_id(self, result_id: str) -> bool:
        """Xóa liveness result theo ID"""
        pass
    
    @abstractmethod
    async def delete_old_results(self, days: int = 30) -> int:
        """Xóa liveness results cũ"""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Đếm tổng số liveness results"""
        pass
    
    @abstractmethod
    async def count_by_result(self, liveness_result: str) -> int:
        """Đếm số liveness results theo kết quả"""
        pass
    
    @abstractmethod
    async def count_by_spoof_type(self, spoof_type: str) -> int:
        """Đếm số liveness results theo loại spoof"""
        pass
    
    @abstractmethod
    async def get_statistics(self) -> dict:
        """Lấy thống kê liveness detection"""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self, start_date: Optional[str] = None, 
                                    end_date: Optional[str] = None) -> dict:
        """Lấy performance metrics trong khoảng thời gian"""
        pass
    
    @abstractmethod
    async def get_spoof_attack_trends(self, days: int = 30) -> dict:
        """Lấy xu hướng spoof attacks"""
        pass
