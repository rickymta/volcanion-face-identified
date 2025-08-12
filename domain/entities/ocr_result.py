from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import uuid

class OCRStatus(Enum):
    """Status của OCR processing"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class DocumentType(Enum):
    """Loại giấy tờ được OCR"""
    CMND = "CMND"
    CCCD = "CCCD"
    PASSPORT = "PASSPORT"
    DRIVER_LICENSE = "DRIVER_LICENSE"
    OTHER = "OTHER"

class TextConfidenceLevel(Enum):
    """Mức độ tin cậy của text được OCR"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"

class FieldType(Enum):
    """Loại field được trích xuất"""
    ID_NUMBER = "ID_NUMBER"
    FULL_NAME = "FULL_NAME"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    PLACE_OF_BIRTH = "PLACE_OF_BIRTH"
    ADDRESS = "ADDRESS"
    NATIONALITY = "NATIONALITY"
    GENDER = "GENDER"
    ISSUE_DATE = "ISSUE_DATE"
    EXPIRY_DATE = "EXPIRY_DATE"
    ISSUE_PLACE = "ISSUE_PLACE"
    PASSPORT_NUMBER = "PASSPORT_NUMBER"
    OTHER = "OTHER"

@dataclass
class BoundingBox:
    """Bounding box cho text region"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    def get_width(self) -> int:
        return self.x2 - self.x1
    
    def get_height(self) -> int:
        return self.y2 - self.y1
    
    def get_area(self) -> int:
        return self.get_width() * self.get_height()
    
    def get_center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

@dataclass
class TextRegion:
    """Thông tin về một text region được OCR"""
    text: str
    confidence: float
    bbox: BoundingBox
    field_type: Optional[FieldType] = None
    language: Optional[str] = None
    font_size: Optional[float] = None
    is_handwritten: bool = False
    
    def get_confidence_level(self) -> TextConfidenceLevel:
        """Lấy confidence level"""
        if self.confidence >= 0.95:
            return TextConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.85:
            return TextConfidenceLevel.HIGH
        elif self.confidence >= 0.70:
            return TextConfidenceLevel.MEDIUM
        elif self.confidence >= 0.50:
            return TextConfidenceLevel.LOW
        else:
            return TextConfidenceLevel.VERY_LOW
    
    def is_reliable(self) -> bool:
        """Kiểm tra text có đáng tin cậy không"""
        return self.confidence >= 0.70

@dataclass
class ExtractedField:
    """Field được trích xuất từ document"""
    field_type: FieldType
    value: str
    confidence: float
    bbox: BoundingBox
    raw_text: str
    normalized_value: Optional[str] = None
    validation_status: bool = True
    validation_errors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
    
    def get_confidence_level(self) -> TextConfidenceLevel:
        """Lấy confidence level"""
        if self.confidence >= 0.95:
            return TextConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.85:
            return TextConfidenceLevel.HIGH
        elif self.confidence >= 0.70:
            return TextConfidenceLevel.MEDIUM
        elif self.confidence >= 0.50:
            return TextConfidenceLevel.LOW
        else:
            return TextConfidenceLevel.VERY_LOW
    
    def is_valid(self) -> bool:
        """Kiểm tra field có valid không"""
        return self.validation_status and self.confidence >= 0.5

@dataclass
class OCRStatistics:
    """Thống kê về quá trình OCR"""
    total_text_regions: int
    reliable_regions: int
    unreliable_regions: int
    average_confidence: float
    total_characters: int
    processing_time_ms: float
    languages_detected: List[str]
    
    def get_reliability_rate(self) -> float:
        """Tỷ lệ regions đáng tin cậy"""
        if self.total_text_regions == 0:
            return 0.0
        return self.reliable_regions / self.total_text_regions

@dataclass
class OCRResult:
    """Kết quả OCR từ document"""
    id: str
    image_path: str
    document_type: DocumentType
    status: OCRStatus
    text_regions: List[TextRegion]
    extracted_fields: List[ExtractedField]
    full_text: str
    statistics: OCRStatistics
    processing_time_ms: float
    model_version: str
    languages_detected: List[str]
    confidence_threshold: float
    preprocessing_applied: List[str]
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = f"ocr_{uuid.uuid4().hex[:12]}"
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def get_overall_confidence(self) -> float:
        """Tính overall confidence"""
        if not self.text_regions:
            return 0.0
        
        total_confidence = sum(region.confidence for region in self.text_regions)
        return total_confidence / len(self.text_regions)
    
    def get_reliable_text_regions(self) -> List[TextRegion]:
        """Lấy text regions đáng tin cậy"""
        return [region for region in self.text_regions if region.is_reliable()]
    
    def get_field_by_type(self, field_type: FieldType) -> Optional[ExtractedField]:
        """Lấy field theo type"""
        for field in self.extracted_fields:
            if field.field_type == field_type:
                return field
        return None
    
    def get_valid_fields(self) -> List[ExtractedField]:
        """Lấy các fields valid"""
        return [field for field in self.extracted_fields if field.is_valid()]
    
    def get_invalid_fields(self) -> List[ExtractedField]:
        """Lấy các fields invalid"""
        return [field for field in self.extracted_fields if not field.is_valid()]
    
    def is_successful(self) -> bool:
        """Kiểm tra OCR có thành công không"""
        return self.status == OCRStatus.COMPLETED and self.get_overall_confidence() >= 0.5
    
    def is_partial_success(self) -> bool:
        """Kiểm tra có partial success không"""
        return self.status in [OCRStatus.COMPLETED, OCRStatus.PARTIAL] and len(self.get_valid_fields()) > 0
    
    def get_confidence_level(self) -> TextConfidenceLevel:
        """Lấy overall confidence level"""
        confidence = self.get_overall_confidence()
        if confidence >= 0.95:
            return TextConfidenceLevel.VERY_HIGH
        elif confidence >= 0.85:
            return TextConfidenceLevel.HIGH
        elif confidence >= 0.70:
            return TextConfidenceLevel.MEDIUM
        elif confidence >= 0.50:
            return TextConfidenceLevel.LOW
        else:
            return TextConfidenceLevel.VERY_LOW
    
    def get_completion_rate(self) -> float:
        """Tỷ lệ hoàn thành extraction"""
        if not self.extracted_fields:
            return 0.0
        
        valid_fields = len(self.get_valid_fields())
        return valid_fields / len(self.extracted_fields)
    
    def get_text_by_confidence(self, min_confidence: float = 0.7) -> str:
        """Lấy text với confidence tối thiểu"""
        reliable_regions = [
            region for region in self.text_regions 
            if region.confidence >= min_confidence
        ]
        
        return " ".join([region.text for region in reliable_regions])
    
    def has_field_type(self, field_type: FieldType) -> bool:
        """Kiểm tra có field type hay không"""
        return any(field.field_type == field_type for field in self.extracted_fields)
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Tóm tắt kết quả extraction"""
        return {
            "total_fields": len(self.extracted_fields),
            "valid_fields": len(self.get_valid_fields()),
            "invalid_fields": len(self.get_invalid_fields()),
            "overall_confidence": self.get_overall_confidence(),
            "confidence_level": self.get_confidence_level().value,
            "completion_rate": self.get_completion_rate(),
            "is_successful": self.is_successful(),
            "processing_time_ms": self.processing_time_ms,
            "languages_detected": self.languages_detected
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "image_path": self.image_path,
            "document_type": self.document_type.value,
            "status": self.status.value,
            "text_regions": [
                {
                    "text": region.text,
                    "confidence": region.confidence,
                    "bbox": {
                        "x1": region.bbox.x1,
                        "y1": region.bbox.y1,
                        "x2": region.bbox.x2,
                        "y2": region.bbox.y2
                    },
                    "field_type": region.field_type.value if region.field_type else None,
                    "language": region.language,
                    "font_size": region.font_size,
                    "is_handwritten": region.is_handwritten
                }
                for region in self.text_regions
            ],
            "extracted_fields": [
                {
                    "field_type": field.field_type.value,
                    "value": field.value,
                    "confidence": field.confidence,
                    "bbox": {
                        "x1": field.bbox.x1,
                        "y1": field.bbox.y1,
                        "x2": field.bbox.x2,
                        "y2": field.bbox.y2
                    },
                    "raw_text": field.raw_text,
                    "normalized_value": field.normalized_value,
                    "validation_status": field.validation_status,
                    "validation_errors": field.validation_errors
                }
                for field in self.extracted_fields
            ],
            "full_text": self.full_text,
            "statistics": {
                "total_text_regions": self.statistics.total_text_regions,
                "reliable_regions": self.statistics.reliable_regions,
                "unreliable_regions": self.statistics.unreliable_regions,
                "average_confidence": self.statistics.average_confidence,
                "total_characters": self.statistics.total_characters,
                "processing_time_ms": self.statistics.processing_time_ms,
                "languages_detected": self.statistics.languages_detected,
                "reliability_rate": self.statistics.get_reliability_rate()
            },
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "languages_detected": self.languages_detected,
            "confidence_threshold": self.confidence_threshold,
            "preprocessing_applied": self.preprocessing_applied,
            "overall_confidence": self.get_overall_confidence(),
            "confidence_level": self.get_confidence_level().value,
            "completion_rate": self.get_completion_rate(),
            "is_successful": self.is_successful(),
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
