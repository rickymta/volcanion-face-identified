from domain.entities.document_quality import DocumentQuality, QualityStatus, TamperType
import logging

class DocumentQualityService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def analyze_quality(self, image_path: str, bbox: list = None) -> DocumentQuality:
        """Phân tích chất lượng và tamper của giấy tờ"""
        try:
            from infrastructure.ml_models.quality_analyzer import QualityAnalyzer
            from infrastructure.ml_models.tamper_detector import TamperDetector
            
            quality_analyzer = QualityAnalyzer()
            tamper_detector = TamperDetector()
            
            # Phân tích chất lượng ảnh
            quality_metrics = quality_analyzer.analyze_image_quality(image_path, bbox)
            
            # Phát hiện tamper
            tamper_result = tamper_detector.detect_tampering(image_path, bbox)
            
            # Tính toán overall quality
            overall_quality = self._calculate_overall_quality(quality_metrics)
            
            return DocumentQuality(
                image_path=image_path,
                overall_quality=overall_quality,
                quality_score=quality_metrics['overall_score'],
                tamper_detected=tamper_result['is_tampered'],
                tamper_type=TamperType(tamper_result['tamper_type']),
                tamper_confidence=tamper_result['confidence'],
                blur_score=quality_metrics['blur_score'],
                glare_score=quality_metrics['glare_score'],
                contrast_score=quality_metrics['contrast_score'],
                brightness_score=quality_metrics['brightness_score'],
                noise_score=quality_metrics['noise_score'],
                edge_sharpness=quality_metrics['edge_sharpness'],
                watermark_present=quality_metrics['watermark_present'],
                metadata_analysis=tamper_result['metadata_analysis']
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing document quality: {e}")
            return DocumentQuality(
                image_path=image_path,
                overall_quality=QualityStatus.REJECTED,
                quality_score=0.0,
                tamper_detected=True,
                tamper_type=TamperType.NONE,
                tamper_confidence=0.0
            )
    
    def _calculate_overall_quality(self, metrics: dict) -> QualityStatus:
        """Tính toán chất lượng tổng thể dựa trên các metrics"""
        overall_score = metrics['overall_score']
        
        if overall_score >= 0.8:
            return QualityStatus.GOOD
        elif overall_score >= 0.6:
            return QualityStatus.FAIR
        elif overall_score >= 0.4:
            return QualityStatus.POOR
        else:
            return QualityStatus.REJECTED
    
    def validate_quality(self, quality: DocumentQuality) -> bool:
        """Kiểm tra tính hợp lệ của document quality"""
        return quality.is_acceptable()
    
    def get_quality_recommendations(self, quality: DocumentQuality) -> list:
        """Đưa ra khuyến nghị cải thiện chất lượng"""
        recommendations = []
        
        if quality.blur_score < 0.5:
            recommendations.append("Ảnh bị mờ, hãy chụp lại với camera ổn định hơn")
        
        if quality.glare_score > 0.7:
            recommendations.append("Có ánh sáng chói, hãy tránh ánh sáng trực tiếp")
        
        if quality.brightness_score < 0.3:
            recommendations.append("Ảnh quá tối, hãy tăng độ sáng")
        elif quality.brightness_score > 0.8:
            recommendations.append("Ảnh quá sáng, hãy giảm độ sáng")
        
        if quality.contrast_score < 0.4:
            recommendations.append("Độ tương phản thấp, hãy chụp trong điều kiện ánh sáng tốt hơn")
        
        if quality.edge_sharpness < 0.5:
            recommendations.append("Ảnh thiếu độ sắc nét, hãy focus camera đúng cách")
        
        if quality.noise_score > 0.6:
            recommendations.append("Ảnh có nhiều nhiễu, hãy chụp trong điều kiện ánh sáng tốt hơn")
        
        if quality.tamper_detected:
            recommendations.append(f"Phát hiện dấu hiệu chỉnh sửa ({quality.tamper_type.value})")
        
        return recommendations
