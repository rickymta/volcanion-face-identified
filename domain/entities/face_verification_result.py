from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime
from enum import Enum
import uuid

class VerificationStatus(str, Enum):
    """Status của face verification"""
    SUCCESS = "success"
    FAILED = "failed"
    NO_FACE_DETECTED = "no_face_detected"
    LOW_QUALITY = "low_quality"
    FACES_NOT_COMPARABLE = "faces_not_comparable"
    INSUFFICIENT_FEATURES = "insufficient_features"

class VerificationResult(str, Enum):
    """Kết quả verification"""
    MATCH = "match"
    NO_MATCH = "no_match"
    UNCERTAIN = "uncertain"

@dataclass
class FaceEmbedding:
    """Face embedding entity"""
    id: Optional[str] = None
    image_path: str = ""
    face_bbox: Optional[List[int]] = None
    embedding_vector: Optional[List[float]] = None
    embedding_model: str = "facenet"
    feature_quality: float = 0.0
    extraction_confidence: float = 0.0
    face_alignment_score: float = 0.0
    preprocessing_applied: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def get_embedding_dimension(self) -> int:
        """Lấy chiều của embedding vector"""
        return len(self.embedding_vector) if self.embedding_vector else 0
    
    def is_valid_embedding(self) -> bool:
        """Kiểm tra tính hợp lệ của embedding"""
        return (
            self.embedding_vector is not None and
            len(self.embedding_vector) > 0 and
            self.extraction_confidence > 0.5 and
            self.feature_quality > 0.3
        )

@dataclass
class FaceVerificationResult:
    """Kết quả face verification"""
    id: Optional[str] = None
    reference_image_path: str = ""
    target_image_path: str = ""
    reference_embedding_id: Optional[str] = None
    target_embedding_id: Optional[str] = None
    status: VerificationStatus = VerificationStatus.FAILED
    verification_result: VerificationResult = VerificationResult.NO_MATCH
    similarity_score: float = 0.0
    distance_metric: str = "cosine"
    confidence: float = 0.0
    threshold_used: float = 0.6
    match_probability: float = 0.0
    processing_time_ms: float = 0.0
    model_used: str = "facenet"
    quality_assessment: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.quality_assessment is None:
            self.quality_assessment = {}
    
    def is_successful_verification(self) -> bool:
        """Kiểm tra xem verification có thành công không"""
        return self.status == VerificationStatus.SUCCESS
    
    def is_positive_match(self) -> bool:
        """Kiểm tra xem có phải là positive match không"""
        return (
            self.is_successful_verification() and
            self.verification_result == VerificationResult.MATCH and
            self.similarity_score >= self.threshold_used
        )
    
    def get_confidence_level(self) -> str:
        """Lấy mức độ tin cậy dạng text"""
        if self.confidence >= 0.9:
            return "very_high"
        elif self.confidence >= 0.8:
            return "high"
        elif self.confidence >= 0.6:
            return "medium"
        elif self.confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def get_verification_summary(self) -> dict:
        """Lấy tóm tắt verification"""
        return {
            "status": self.status,
            "result": self.verification_result,
            "similarity": round(self.similarity_score, 3),
            "confidence": round(self.confidence, 3),
            "confidence_level": self.get_confidence_level(),
            "is_match": self.is_positive_match(),
            "threshold": self.threshold_used,
            "processing_time": round(self.processing_time_ms, 2)
        }
