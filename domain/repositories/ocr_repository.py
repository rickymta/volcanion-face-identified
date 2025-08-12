from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

from domain.entities.ocr_result import OCRResult, DocumentType, OCRStatus, FieldType

class OCRRepository(ABC):
    """Repository interface cho OCR results"""
    
    @abstractmethod
    async def save(self, ocr_result: OCRResult) -> str:
        """
        Lưu OCR result
        
        Args:
            ocr_result: OCR result để lưu
            
        Returns:
            str: ID của OCR result đã lưu
        """
        pass
    
    @abstractmethod
    async def find_by_id(self, ocr_id: str) -> Optional[OCRResult]:
        """
        Tìm OCR result theo ID
        
        Args:
            ocr_id: ID của OCR result
            
        Returns:
            Optional[OCRResult]: OCR result nếu tìm thấy
        """
        pass
    
    @abstractmethod
    async def find_by_image_path(self, image_path: str) -> List[OCRResult]:
        """
        Tìm OCR results theo image path
        
        Args:
            image_path: Đường dẫn ảnh
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        pass
    
    @abstractmethod
    async def find_by_document_type(self, document_type: DocumentType, limit: int = 100) -> List[OCRResult]:
        """
        Tìm OCR results theo document type
        
        Args:
            document_type: Loại document
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        pass
    
    @abstractmethod
    async def find_by_status(self, status: OCRStatus, limit: int = 100) -> List[OCRResult]:
        """
        Tìm OCR results theo status
        
        Args:
            status: Status của OCR
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        pass
    
    @abstractmethod
    async def get_recent_results(self, hours: int = 24, limit: int = 100) -> List[OCRResult]:
        """
        Lấy OCR results gần đây
        
        Args:
            hours: Số giờ look back
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        pass
    
    @abstractmethod
    async def get_successful_results(self, limit: int = 100) -> List[OCRResult]:
        """
        Lấy OCR results thành công
        
        Args:
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results thành công
        """
        pass
    
    @abstractmethod
    async def get_failed_results(self, limit: int = 100) -> List[OCRResult]:
        """
        Lấy OCR results thất bại
        
        Args:
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results thất bại
        """
        pass
    
    @abstractmethod
    async def find_by_field_type(self, field_type: FieldType, limit: int = 100) -> List[OCRResult]:
        """
        Tìm OCR results có chứa field type cụ thể
        
        Args:
            field_type: Loại field cần tìm
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        pass
    
    @abstractmethod
    async def find_by_confidence_range(
        self, 
        min_confidence: float, 
        max_confidence: float = 1.0, 
        limit: int = 100
    ) -> List[OCRResult]:
        """
        Tìm OCR results theo confidence range
        
        Args:
            min_confidence: Confidence tối thiểu
            max_confidence: Confidence tối đa
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        pass
    
    @abstractmethod
    async def search_by_text(self, search_text: str, limit: int = 100) -> List[OCRResult]:
        """
        Tìm kiếm OCR results theo text content
        
        Args:
            search_text: Text cần tìm
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Lấy statistics tổng quan về OCR
        
        Returns:
            Dict[str, Any]: Thống kê OCR
        """
        pass
    
    @abstractmethod
    async def get_statistics_by_document_type(self, document_type: DocumentType) -> Dict[str, Any]:
        """
        Lấy statistics theo document type
        
        Args:
            document_type: Loại document
            
        Returns:
            Dict[str, Any]: Thống kê theo document type
        """
        pass
    
    @abstractmethod
    async def get_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Lấy performance metrics trong khoảng thời gian
        
        Args:
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        pass
    
    @abstractmethod
    async def count_total(self) -> int:
        """
        Đếm tổng số OCR results
        
        Returns:
            int: Tổng số OCR results
        """
        pass
    
    @abstractmethod
    async def count_by_status(self, status: OCRStatus) -> int:
        """
        Đếm số OCR results theo status
        
        Args:
            status: Status cần đếm
            
        Returns:
            int: Số lượng OCR results
        """
        pass
    
    @abstractmethod
    async def count_by_document_type(self, document_type: DocumentType) -> int:
        """
        Đếm số OCR results theo document type
        
        Args:
            document_type: Loại document
            
        Returns:
            int: Số lượng OCR results
        """
        pass
    
    @abstractmethod
    async def update_status(self, ocr_id: str, status: OCRStatus, error_message: str = None) -> bool:
        """
        Cập nhật status của OCR result
        
        Args:
            ocr_id: ID của OCR result
            status: Status mới
            error_message: Error message (nếu có)
            
        Returns:
            bool: True nếu update thành công
        """
        pass
    
    @abstractmethod
    async def delete_by_id(self, ocr_id: str) -> bool:
        """
        Xóa OCR result theo ID
        
        Args:
            ocr_id: ID của OCR result cần xóa
            
        Returns:
            bool: True nếu xóa thành công
        """
        pass
    
    @abstractmethod
    async def delete_old_results(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Xóa OCR results cũ
        
        Args:
            days_to_keep: Số ngày data cần giữ lại
            
        Returns:
            Dict[str, Any]: Thông tin về cleanup process
        """
        pass
    
    @abstractmethod
    async def get_duplicate_results(self, image_path: str) -> List[OCRResult]:
        """
        Tìm duplicate OCR results cho cùng một image
        
        Args:
            image_path: Đường dẫn ảnh
            
        Returns:
            List[OCRResult]: Danh sách duplicate results
        """
        pass
    
    @abstractmethod
    async def get_accuracy_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Lấy accuracy trends trong khoảng thời gian
        
        Args:
            days: Số ngày để tính trends
            
        Returns:
            Dict[str, Any]: Accuracy trends data
        """
        pass
    
    @abstractmethod
    async def get_field_extraction_stats(self) -> Dict[str, Any]:
        """
        Lấy statistics về field extraction
        
        Returns:
            Dict[str, Any]: Field extraction statistics
        """
        pass
    
    @abstractmethod
    async def find_similar_results(
        self, 
        ocr_result: OCRResult, 
        similarity_threshold: float = 0.8, 
        limit: int = 10
    ) -> List[OCRResult]:
        """
        Tìm OCR results tương tự
        
        Args:
            ocr_result: OCR result để so sánh
            similarity_threshold: Ngưỡng similarity
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách similar results
        """
        pass
