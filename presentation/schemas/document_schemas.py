from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class DocumentTypeEnum(str, Enum):
    CMND = 'cmnd'
    PASSPORT = 'passport'
    CCCD = 'cccd'
    UNKNOWN = 'unknown'

class DocumentDetectionRequest(BaseModel):
    save_to_db: bool = True

class DocumentDetectionResponse(BaseModel):
    doc_type: DocumentTypeEnum
    bbox: Optional[List[int]] = None
    confidence: float
    is_valid: bool
    message: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentDetectionResponse]
    total: int

class ErrorResponse(BaseModel):
    error: str
    detail: str
