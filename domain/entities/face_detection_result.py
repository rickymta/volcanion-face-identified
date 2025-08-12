from enum import Enum
from typing import Optional, List, Tuple
from datetime import datetime
import numpy as np

class FaceDetectionStatus(Enum):
    SUCCESS = 'success'
    NO_FACE_DETECTED = 'no_face_detected'
    MULTIPLE_FACES = 'multiple_faces'
    FACE_TOO_SMALL = 'face_too_small'
    FACE_TOO_BLURRY = 'face_too_blurry'
    FACE_OCCLUDED = 'face_occluded'
    FAILED = 'failed'

class OcclusionType(Enum):
    NONE = 'none'
    GLASSES = 'glasses'
    HAT = 'hat'
    HAND = 'hand'
    CHIN_SUPPORT = 'chin_support'
    MASK = 'mask'
    SUNGLASSES = 'sunglasses'

class FaceDetectionResult:
    def __init__(self,
                 image_path: str,
                 status: FaceDetectionStatus = FaceDetectionStatus.FAILED,
                 bbox: Optional[List[int]] = None,
                 landmarks: Optional[List[Tuple[float, float]]] = None,
                 confidence: float = 0.0,
                 face_size: Optional[Tuple[int, int]] = None,
                 occlusion_detected: bool = False,
                 occlusion_type: OcclusionType = OcclusionType.NONE,
                 occlusion_confidence: float = 0.0,
                 alignment_score: float = 0.0,
                 face_quality_score: float = 0.0,
                 pose_angles: Optional[Tuple[float, float, float]] = None,  # yaw, pitch, roll
                 created_at: Optional[datetime] = None):
        self.image_path = image_path
        self.status = status
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.landmarks = landmarks  # List of (x, y) coordinates for facial landmarks
        self.confidence = confidence
        self.face_size = face_size  # (width, height)
        self.occlusion_detected = occlusion_detected
        self.occlusion_type = occlusion_type
        self.occlusion_confidence = occlusion_confidence
        self.alignment_score = alignment_score
        self.face_quality_score = face_quality_score
        self.pose_angles = pose_angles
        self.created_at = created_at or datetime.now()
    
    def to_dict(self) -> dict:
        """Chuyển đổi thành dictionary để lưu vào database"""
        return {
            'image_path': self.image_path,
            'status': self.status.value,
            'bbox': self.bbox,
            'landmarks': self.landmarks,
            'confidence': self.confidence,
            'face_size': list(self.face_size) if self.face_size else None,
            'occlusion_detected': self.occlusion_detected,
            'occlusion_type': self.occlusion_type.value,
            'occlusion_confidence': self.occlusion_confidence,
            'alignment_score': self.alignment_score,
            'face_quality_score': self.face_quality_score,
            'pose_angles': list(self.pose_angles) if self.pose_angles else None,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FaceDetectionResult':
        """Tạo FaceDetectionResult từ dictionary"""
        return cls(
            image_path=data['image_path'],
            status=FaceDetectionStatus(data['status']),
            bbox=data.get('bbox'),
            landmarks=data.get('landmarks'),
            confidence=data.get('confidence', 0.0),
            face_size=tuple(data['face_size']) if data.get('face_size') else None,
            occlusion_detected=data.get('occlusion_detected', False),
            occlusion_type=OcclusionType(data.get('occlusion_type', 'none')),
            occlusion_confidence=data.get('occlusion_confidence', 0.0),
            alignment_score=data.get('alignment_score', 0.0),
            face_quality_score=data.get('face_quality_score', 0.0),
            pose_angles=tuple(data['pose_angles']) if data.get('pose_angles') else None,
            created_at=data.get('created_at')
        )
    
    def is_acceptable(self) -> bool:
        """Kiểm tra xem khuôn mặt có chấp nhận được không"""
        return (self.status == FaceDetectionStatus.SUCCESS and
                self.confidence >= 0.7 and
                not self.occlusion_detected and
                self.face_quality_score >= 0.6 and
                self.alignment_score >= 0.5)
    
    def get_face_center(self) -> Optional[Tuple[float, float]]:
        """Lấy tọa độ trung tâm khuôn mặt"""
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        return None
    
    def get_face_area(self) -> Optional[int]:
        """Tính diện tích khuôn mặt"""
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            return (x2 - x1) * (y2 - y1)
        return None
