from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime

class QualityStatus(Enum):
    GOOD = 'good'
    FAIR = 'fair'
    POOR = 'poor'
    REJECTED = 'rejected'

class TamperType(Enum):
    NONE = 'none'
    DIGITAL_MANIPULATION = 'digital_manipulation'
    PHYSICAL_TAMPERING = 'physical_tampering'
    WATERMARK_REMOVED = 'watermark_removed'
    OVERLAY_DETECTED = 'overlay_detected'
    COPY_PASTE = 'copy_paste'

class DocumentQuality:
    def __init__(self,
                 image_path: str,
                 overall_quality: QualityStatus = QualityStatus.POOR,
                 quality_score: float = 0.0,
                 tamper_detected: bool = False,
                 tamper_type: TamperType = TamperType.NONE,
                 tamper_confidence: float = 0.0,
                 blur_score: float = 0.0,
                 glare_score: float = 0.0,
                 contrast_score: float = 0.0,
                 brightness_score: float = 0.0,
                 noise_score: float = 0.0,
                 edge_sharpness: float = 0.0,
                 watermark_present: bool = False,
                 metadata_analysis: Optional[Dict[str, Any]] = None,
                 created_at: Optional[datetime] = None):
        self.image_path = image_path
        self.overall_quality = overall_quality
        self.quality_score = quality_score
        self.tamper_detected = tamper_detected
        self.tamper_type = tamper_type
        self.tamper_confidence = tamper_confidence
        self.blur_score = blur_score
        self.glare_score = glare_score
        self.contrast_score = contrast_score
        self.brightness_score = brightness_score
        self.noise_score = noise_score
        self.edge_sharpness = edge_sharpness
        self.watermark_present = watermark_present
        self.metadata_analysis = metadata_analysis or {}
        self.created_at = created_at or datetime.now()
    
    def to_dict(self) -> dict:
        """Chuyển đổi thành dictionary để lưu vào database"""
        return {
            'image_path': self.image_path,
            'overall_quality': self.overall_quality.value,
            'quality_score': self.quality_score,
            'tamper_detected': self.tamper_detected,
            'tamper_type': self.tamper_type.value,
            'tamper_confidence': self.tamper_confidence,
            'blur_score': self.blur_score,
            'glare_score': self.glare_score,
            'contrast_score': self.contrast_score,
            'brightness_score': self.brightness_score,
            'noise_score': self.noise_score,
            'edge_sharpness': self.edge_sharpness,
            'watermark_present': self.watermark_present,
            'metadata_analysis': self.metadata_analysis,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DocumentQuality':
        """Tạo DocumentQuality từ dictionary"""
        return cls(
            image_path=data['image_path'],
            overall_quality=QualityStatus(data['overall_quality']),
            quality_score=data.get('quality_score', 0.0),
            tamper_detected=data.get('tamper_detected', False),
            tamper_type=TamperType(data.get('tamper_type', 'none')),
            tamper_confidence=data.get('tamper_confidence', 0.0),
            blur_score=data.get('blur_score', 0.0),
            glare_score=data.get('glare_score', 0.0),
            contrast_score=data.get('contrast_score', 0.0),
            brightness_score=data.get('brightness_score', 0.0),
            noise_score=data.get('noise_score', 0.0),
            edge_sharpness=data.get('edge_sharpness', 0.0),
            watermark_present=data.get('watermark_present', False),
            metadata_analysis=data.get('metadata_analysis', {}),
            created_at=data.get('created_at')
        )
    
    def is_acceptable(self) -> bool:
        """Kiểm tra xem chất lượng có chấp nhận được không"""
        return (self.overall_quality in [QualityStatus.GOOD, QualityStatus.FAIR] and
                not self.tamper_detected and
                self.quality_score >= 0.6)
