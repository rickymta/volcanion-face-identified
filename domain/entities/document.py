from enum import Enum
from typing import Optional
from datetime import datetime

class DocumentType(Enum):
    CMND = 'cmnd'
    PASSPORT = 'passport'
    CCCD = 'cccd'
    UNKNOWN = 'unknown'

class Document:
    def __init__(self, 
                 image_path: str, 
                 doc_type: DocumentType = DocumentType.UNKNOWN, 
                 bbox: Optional[list] = None,
                 confidence: float = 0.0,
                 created_at: Optional[datetime] = None):
        self.image_path = image_path
        self.doc_type = doc_type
        self.bbox = bbox  # [x1, y1, x2, y2] vùng giấy tờ
        self.confidence = confidence
        self.created_at = created_at or datetime.now()
        
    def to_dict(self) -> dict:
        """Chuyển đổi thành dictionary để lưu vào database"""
        return {
            'image_path': self.image_path,
            'doc_type': self.doc_type.value,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'created_at': self.created_at
        }
        
    @classmethod
    def from_dict(cls, data: dict) -> 'Document':
        """Tạo Document từ dictionary"""
        return cls(
            image_path=data['image_path'],
            doc_type=DocumentType(data['doc_type']),
            bbox=data.get('bbox'),
            confidence=data.get('confidence', 0.0),
            created_at=data.get('created_at')
        )
