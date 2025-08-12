from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum

class QualityStatusEnum(str, Enum):
    GOOD = 'good'
    FAIR = 'fair'
    POOR = 'poor'
    REJECTED = 'rejected'

class TamperTypeEnum(str, Enum):
    NONE = 'none'
    DIGITAL_MANIPULATION = 'digital_manipulation'
    PHYSICAL_TAMPERING = 'physical_tampering'
    WATERMARK_REMOVED = 'watermark_removed'
    OVERLAY_DETECTED = 'overlay_detected'
    COPY_PASTE = 'copy_paste'

class QualityCheckRequest(BaseModel):
    save_to_db: bool = True
    bbox: Optional[List[int]] = None

class QualityMetrics(BaseModel):
    blur_score: float
    glare_score: float
    contrast_score: float
    brightness_score: float
    noise_score: float
    edge_sharpness: float

class TamperAnalysis(BaseModel):
    tamper_detected: bool
    tamper_type: TamperTypeEnum
    tamper_confidence: float
    metadata_suspicious: bool

class QualityCheckResponse(BaseModel):
    overall_quality: QualityStatusEnum
    quality_score: float
    is_acceptable: bool
    metrics: QualityMetrics
    tamper_analysis: TamperAnalysis
    watermark_present: bool
    recommendations: List[str]
    message: str

class QualityListResponse(BaseModel):
    qualities: List[QualityCheckResponse]
    total: int

class RecommendationsRequest(BaseModel):
    bbox: Optional[List[int]] = None

class RecommendationsResponse(BaseModel):
    recommendations: List[str]
    quality_score: float
