from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
import uuid

class LivenessStatus(Enum):
    """Trạng thái liveness detection"""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ERROR = "ERROR"

class LivenessResult(Enum):
    """Kết quả liveness detection"""
    REAL = "REAL"           # Khuôn mặt thật
    FAKE = "FAKE"           # Khuôn mặt giả
    UNCERTAIN = "UNCERTAIN" # Không chắc chắn
    
class SpoofType(Enum):
    """Loại spoof attack được phát hiện"""
    PHOTO_ATTACK = "PHOTO_ATTACK"       # Tấn công bằng ảnh
    VIDEO_REPLAY = "VIDEO_REPLAY"       # Tấn công bằng video
    MASK_ATTACK = "MASK_ATTACK"         # Tấn công bằng mask 3D
    SCREEN_ATTACK = "SCREEN_ATTACK"     # Tấn công bằng màn hình
    DEEPFAKE = "DEEPFAKE"               # Deepfake attack
    UNKNOWN = "UNKNOWN"                 # Không xác định được loại

@dataclass
class LivenessDetectionResult:
    """Entity cho kết quả liveness detection"""
    
    # Basic information
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    image_path: str = ""
    video_path: Optional[str] = None
    face_bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
    
    # Detection results
    status: LivenessStatus = LivenessStatus.PENDING
    liveness_result: LivenessResult = LivenessResult.UNCERTAIN
    confidence: float = 0.0
    liveness_score: float = 0.0  # 0-1, càng cao càng real
    
    # Spoof detection
    spoof_probability: float = 0.0  # 0-1, càng cao càng fake
    detected_spoof_types: List[SpoofType] = field(default_factory=list)
    primary_spoof_type: Optional[SpoofType] = None
    
    # Analysis details
    texture_analysis: Dict[str, float] = field(default_factory=dict)
    motion_analysis: Optional[Dict[str, float]] = None
    depth_analysis: Optional[Dict[str, float]] = None
    frequency_analysis: Dict[str, float] = field(default_factory=dict)
    eye_blink_analysis: Optional[Dict[str, float]] = None
    
    # Quality metrics
    image_quality: float = 0.0
    face_quality: float = 0.0
    lighting_quality: float = 0.0
    pose_quality: float = 0.0
    
    # Processing info
    processing_time_ms: float = 0.0
    algorithms_used: List[str] = field(default_factory=list)
    model_version: str = "1.0"
    threshold_used: float = 0.5
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    def is_real_face(self) -> bool:
        """Kiểm tra có phải khuôn mặt thật không"""
        return self.liveness_result == LivenessResult.REAL
    
    def is_fake_face(self) -> bool:
        """Kiểm tra có phải khuôn mặt giả không"""
        return self.liveness_result == LivenessResult.FAKE
    
    def is_uncertain(self) -> bool:
        """Kiểm tra có uncertain không"""
        return self.liveness_result == LivenessResult.UNCERTAIN
    
    def get_confidence_level(self) -> str:
        """Lấy mức độ confidence"""
        if self.confidence >= 0.9:
            return "VERY_HIGH"
        elif self.confidence >= 0.8:
            return "HIGH"
        elif self.confidence >= 0.6:
            return "MEDIUM"
        elif self.confidence >= 0.4:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def get_risk_level(self) -> str:
        """Lấy mức độ rủi ro"""
        if self.is_real_face() and self.confidence >= 0.9:
            return "LOW"
        elif self.is_real_face() and self.confidence >= 0.7:
            return "MEDIUM"
        elif self.is_fake_face():
            return "HIGH"
        else:
            return "VERY_HIGH"
    
    def has_spoof_detection(self) -> bool:
        """Kiểm tra có phát hiện spoof không"""
        return len(self.detected_spoof_types) > 0
    
    def get_primary_attack_type(self) -> str:
        """Lấy loại tấn công chính"""
        if self.primary_spoof_type:
            return self.primary_spoof_type.value
        elif self.detected_spoof_types:
            return self.detected_spoof_types[0].value
        return "NONE"
    
    def add_spoof_type(self, spoof_type: SpoofType, confidence: float = 0.0):
        """Thêm loại spoof được phát hiện"""
        if spoof_type not in self.detected_spoof_types:
            self.detected_spoof_types.append(spoof_type)
        
        # Set primary spoof type nếu chưa có hoặc confidence cao hơn
        if not self.primary_spoof_type or confidence > self.spoof_probability:
            self.primary_spoof_type = spoof_type
            self.spoof_probability = confidence
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Lấy tóm tắt analysis"""
        return {
            "liveness_result": self.liveness_result.value,
            "confidence": self.confidence,
            "liveness_score": self.liveness_score,
            "spoof_probability": self.spoof_probability,
            "risk_level": self.get_risk_level(),
            "confidence_level": self.get_confidence_level(),
            "primary_attack": self.get_primary_attack_type(),
            "detected_attacks": [spoof.value for spoof in self.detected_spoof_types],
            "quality_scores": {
                "image_quality": self.image_quality,
                "face_quality": self.face_quality,
                "lighting_quality": self.lighting_quality,
                "pose_quality": self.pose_quality
            },
            "processing_time_ms": self.processing_time_ms
        }
    
    def update_quality_metrics(self, image_q: float, face_q: float, 
                             lighting_q: float, pose_q: float):
        """Cập nhật quality metrics"""
        self.image_quality = max(0.0, min(1.0, image_q))
        self.face_quality = max(0.0, min(1.0, face_q))
        self.lighting_quality = max(0.0, min(1.0, lighting_q))
        self.pose_quality = max(0.0, min(1.0, pose_q))
    
    def update_texture_analysis(self, lbp_score: float, sobel_score: float, 
                              gabor_score: float, variance_score: float):
        """Cập nhật texture analysis"""
        self.texture_analysis.update({
            "lbp_score": lbp_score,
            "sobel_score": sobel_score,
            "gabor_score": gabor_score,
            "variance_score": variance_score,
            "overall_score": (lbp_score + sobel_score + gabor_score + variance_score) / 4
        })
    
    def update_motion_analysis(self, optical_flow_score: float, 
                             frame_diff_score: float, blink_detected: bool,
                             blink_frequency: float):
        """Cập nhật motion analysis (cho video)"""
        self.motion_analysis = {
            "optical_flow_score": optical_flow_score,
            "frame_diff_score": frame_diff_score,
            "blink_detected": blink_detected,
            "blink_frequency": blink_frequency,
            "overall_score": (optical_flow_score + frame_diff_score) / 2
        }
    
    def update_depth_analysis(self, depth_variance: float, edge_density: float,
                            shadow_consistency: float):
        """Cập nhật depth analysis"""
        self.depth_analysis = {
            "depth_variance": depth_variance,
            "edge_density": edge_density,
            "shadow_consistency": shadow_consistency,
            "overall_score": (depth_variance + edge_density + shadow_consistency) / 3
        }
    
    def update_frequency_analysis(self, high_freq_ratio: float, low_freq_ratio: float,
                                dct_features: List[float], fft_features: List[float]):
        """Cập nhật frequency analysis"""
        self.frequency_analysis.update({
            "high_freq_ratio": high_freq_ratio,
            "low_freq_ratio": low_freq_ratio,
            "freq_ratio": high_freq_ratio / (low_freq_ratio + 1e-8),
            "dct_mean": sum(dct_features) / len(dct_features) if dct_features else 0.0,
            "fft_mean": sum(fft_features) / len(fft_features) if fft_features else 0.0
        })
    
    def finalize_detection(self, final_score: float, threshold: float):
        """Hoàn thiện detection result"""
        self.liveness_score = final_score
        self.threshold_used = threshold
        
        # Xác định kết quả cuối cùng
        if final_score >= threshold:
            self.liveness_result = LivenessResult.REAL
            self.confidence = min(0.99, final_score)
        elif final_score <= (threshold - 0.2):
            self.liveness_result = LivenessResult.FAKE
            self.confidence = min(0.99, 1.0 - final_score)
        else:
            self.liveness_result = LivenessResult.UNCERTAIN
            self.confidence = 0.5
        
        self.status = LivenessStatus.COMPLETED
        self.updated_at = datetime.now()
    
    def mark_as_failed(self, error_message: str):
        """Đánh dấu detection thất bại"""
        self.status = LivenessStatus.FAILED
        self.liveness_result = LivenessResult.UNCERTAIN
        self.confidence = 0.0
        self.metadata["error"] = error_message
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "image_path": self.image_path,
            "video_path": self.video_path,
            "face_bbox": self.face_bbox,
            "status": self.status.value,
            "liveness_result": self.liveness_result.value,
            "confidence": self.confidence,
            "liveness_score": self.liveness_score,
            "spoof_probability": self.spoof_probability,
            "detected_spoof_types": [spoof.value for spoof in self.detected_spoof_types],
            "primary_spoof_type": self.primary_spoof_type.value if self.primary_spoof_type else None,
            "texture_analysis": self.texture_analysis,
            "motion_analysis": self.motion_analysis,
            "depth_analysis": self.depth_analysis,
            "frequency_analysis": self.frequency_analysis,
            "eye_blink_analysis": self.eye_blink_analysis,
            "image_quality": self.image_quality,
            "face_quality": self.face_quality,
            "lighting_quality": self.lighting_quality,
            "pose_quality": self.pose_quality,
            "processing_time_ms": self.processing_time_ms,
            "algorithms_used": self.algorithms_used,
            "model_version": self.model_version,
            "threshold_used": self.threshold_used,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LivenessDetectionResult":
        """Create from dictionary"""
        # Parse enums
        status = LivenessStatus(data.get("status", "PENDING"))
        liveness_result = LivenessResult(data.get("liveness_result", "UNCERTAIN"))
        
        detected_spoof_types = []
        for spoof_str in data.get("detected_spoof_types", []):
            try:
                detected_spoof_types.append(SpoofType(spoof_str))
            except ValueError:
                continue
        
        primary_spoof_type = None
        if data.get("primary_spoof_type"):
            try:
                primary_spoof_type = SpoofType(data["primary_spoof_type"])
            except ValueError:
                pass
        
        # Parse dates
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except ValueError:
                pass
        
        updated_at = None
        if data.get("updated_at"):
            try:
                updated_at = datetime.fromisoformat(data["updated_at"])
            except ValueError:
                pass
        
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            image_path=data.get("image_path", ""),
            video_path=data.get("video_path"),
            face_bbox=data.get("face_bbox"),
            status=status,
            liveness_result=liveness_result,
            confidence=data.get("confidence", 0.0),
            liveness_score=data.get("liveness_score", 0.0),
            spoof_probability=data.get("spoof_probability", 0.0),
            detected_spoof_types=detected_spoof_types,
            primary_spoof_type=primary_spoof_type,
            texture_analysis=data.get("texture_analysis", {}),
            motion_analysis=data.get("motion_analysis"),
            depth_analysis=data.get("depth_analysis"),
            frequency_analysis=data.get("frequency_analysis", {}),
            eye_blink_analysis=data.get("eye_blink_analysis"),
            image_quality=data.get("image_quality", 0.0),
            face_quality=data.get("face_quality", 0.0),
            lighting_quality=data.get("lighting_quality", 0.0),
            pose_quality=data.get("pose_quality", 0.0),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            algorithms_used=data.get("algorithms_used", []),
            model_version=data.get("model_version", "1.0"),
            threshold_used=data.get("threshold_used", 0.5),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at
        )
