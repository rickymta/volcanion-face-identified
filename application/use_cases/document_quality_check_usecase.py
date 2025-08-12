from typing import Optional
from domain.services.document_quality_service import DocumentQualityService
from domain.repositories.document_quality_repository import DocumentQualityRepository
from domain.entities.document_quality import DocumentQuality
import logging

class DocumentQualityCheckUseCase:
    def __init__(self, 
                 quality_service: DocumentQualityService,
                 quality_repository: Optional[DocumentQualityRepository] = None):
        self.quality_service = quality_service
        self.quality_repository = quality_repository
        self.logger = logging.getLogger(__name__)

    def execute(self, image_path: str, bbox: list = None, save_to_db: bool = True) -> DocumentQuality:
        """
        Thực hiện kiểm tra chất lượng và tamper detection
        """
        try:
            # Phân tích chất lượng và tamper
            quality = self.quality_service.analyze_quality(image_path, bbox)
            
            # Kiểm tra tính hợp lệ
            is_valid = self.quality_service.validate_quality(quality)
            
            # Lấy khuyến nghị
            recommendations = self.quality_service.get_quality_recommendations(quality)
            
            self.logger.info(f"Quality analysis completed: {quality.overall_quality.value}, "
                           f"score: {quality.quality_score:.2f}, "
                           f"tamper: {quality.tamper_detected}, "
                           f"valid: {is_valid}")
            
            if recommendations:
                self.logger.info(f"Recommendations: {'; '.join(recommendations)}")
            
            # Lưu vào database nếu cần
            if save_to_db and self.quality_repository:
                try:
                    quality_id = self.quality_repository.save(quality)
                    self.logger.info(f"Quality analysis saved with ID: {quality_id}")
                except Exception as e:
                    self.logger.error(f"Failed to save quality analysis: {e}")
                    
            return quality
            
        except Exception as e:
            self.logger.error(f"Error in quality check use case: {e}")
            raise

    def get_recommendations(self, image_path: str, bbox: list = None) -> list:
        """Lấy khuyến nghị cải thiện chất lượng mà không lưu DB"""
        try:
            quality = self.quality_service.analyze_quality(image_path, bbox)
            return self.quality_service.get_quality_recommendations(quality)
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return ["Có lỗi xảy ra khi phân tích ảnh"]
