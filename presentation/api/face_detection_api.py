from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from application.use_cases.face_detection_use_case import FaceDetectionUseCase
from domain.services.face_detection_service import FaceDetectionService
from infrastructure.repositories.mongo_face_detection_repository import MongoFaceDetectionRepository
from pydantic import BaseModel
from typing import Optional, List
import tempfile
import os
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/face-detection", tags=["Face Detection"])

# Dependency injection
def get_face_detection_use_case() -> FaceDetectionUseCase:
    face_detection_service = FaceDetectionService()
    face_detection_repository = MongoFaceDetectionRepository()
    return FaceDetectionUseCase(face_detection_service, face_detection_repository)

# Request/Response models
class FaceDetectionResponse(BaseModel):
    id: Optional[str]
    image_path: str
    status: str
    bbox: Optional[List[int]]
    landmarks: Optional[List[List[float]]]
    confidence: float
    face_size: Optional[List[int]]
    occlusion_detected: bool
    occlusion_type: Optional[str]
    occlusion_confidence: Optional[float]
    alignment_score: Optional[float]
    face_quality_score: Optional[float]
    pose_angles: Optional[List[float]]
    created_at: Optional[str]

class FaceValidationResponse(BaseModel):
    is_valid: bool
    face_result: Optional[FaceDetectionResponse]
    recommendations: List[str]
    quality_score: float
    alignment_score: float
    confidence: float

class FaceComparisonResponse(BaseModel):
    alignment_similarity: float
    face1_valid: bool
    face2_valid: bool
    face1_result: Optional[FaceDetectionResponse]
    face2_result: Optional[FaceDetectionResponse]
    face1_recommendations: List[str]
    face2_recommendations: List[str]
    overall_valid: bool

class FaceDetectionStatsResponse(BaseModel):
    total_detections: int
    success_rate: float
    average_confidence: float
    average_quality_score: float
    average_alignment_score: float
    status_distribution: dict
    occlusion_distribution: dict

@router.post("/detect", response_model=FaceDetectionResponse)
async def detect_face(
    file: UploadFile = File(...),
    source_type: str = "selfie",
    use_case: FaceDetectionUseCase = Depends(get_face_detection_use_case)
):
    """
    Phát hiện và phân tích khuôn mặt trong ảnh
    
    - **file**: File ảnh cần phân tích
    - **source_type**: Loại nguồn ảnh ("selfie" hoặc "document")
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name
    
    try:
        # Process face detection
        result = await use_case.detect_and_process_face(temp_path, source_type)
        
        # Convert to response model
        response = FaceDetectionResponse(
            id=result.id,
            image_path=result.image_path,
            status=result.status,
            bbox=result.bbox,
            landmarks=result.landmarks,
            confidence=result.confidence,
            face_size=result.face_size,
            occlusion_detected=result.occlusion_detected,
            occlusion_type=result.occlusion_type,
            occlusion_confidence=result.occlusion_confidence,
            alignment_score=result.alignment_score,
            face_quality_score=result.face_quality_score,
            pose_angles=result.pose_angles,
            created_at=result.created_at.isoformat() if result.created_at else None
        )
        
        logger.info(f"Face detection completed for file: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing face detection: {e}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/validate", response_model=FaceValidationResponse)
async def validate_face_quality(
    file: UploadFile = File(...),
    source_type: str = "selfie",
    use_case: FaceDetectionUseCase = Depends(get_face_detection_use_case)
):
    """
    Validate chất lượng khuôn mặt và đưa ra khuyến nghị cải thiện
    
    - **file**: File ảnh cần validate
    - **source_type**: Loại nguồn ảnh ("selfie" hoặc "document")
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name
    
    try:
        # Validate face quality
        validation_result = await use_case.validate_face_quality(temp_path, source_type)
        
        # Convert face result to response model if exists
        face_result_response = None
        if validation_result['face_result']:
            face_result = validation_result['face_result']
            face_result_response = FaceDetectionResponse(
                id=face_result.id,
                image_path=face_result.image_path,
                status=face_result.status,
                bbox=face_result.bbox,
                landmarks=face_result.landmarks,
                confidence=face_result.confidence,
                face_size=face_result.face_size,
                occlusion_detected=face_result.occlusion_detected,
                occlusion_type=face_result.occlusion_type,
                occlusion_confidence=face_result.occlusion_confidence,
                alignment_score=face_result.alignment_score,
                face_quality_score=face_result.face_quality_score,
                pose_angles=face_result.pose_angles,
                created_at=face_result.created_at.isoformat() if face_result.created_at else None
            )
        
        response = FaceValidationResponse(
            is_valid=validation_result['is_valid'],
            face_result=face_result_response,
            recommendations=validation_result['recommendations'],
            quality_score=validation_result['quality_score'],
            alignment_score=validation_result['alignment_score'],
            confidence=validation_result['confidence']
        )
        
        logger.info(f"Face validation completed for file: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error validating face quality: {e}")
        raise HTTPException(status_code=500, detail=f"Face validation failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/compare", response_model=FaceComparisonResponse)
async def compare_face_alignment(
    selfie_file: UploadFile = File(...),
    document_file: UploadFile = File(...),
    use_case: FaceDetectionUseCase = Depends(get_face_detection_use_case)
):
    """
    So sánh alignment giữa khuôn mặt trong ảnh selfie và ảnh giấy tờ
    
    - **selfie_file**: File ảnh selfie
    - **document_file**: File ảnh giấy tờ
    """
    if not selfie_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Selfie file must be an image")
    
    if not document_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Document file must be an image")
    
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_selfie:
        selfie_contents = await selfie_file.read()
        temp_selfie.write(selfie_contents)
        selfie_path = temp_selfie.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_document:
        document_contents = await document_file.read()
        temp_document.write(document_contents)
        document_path = temp_document.name
    
    try:
        # Compare face alignment
        comparison_result = await use_case.compare_face_alignment(selfie_path, document_path)
        
        # Convert face results to response models
        def convert_face_result(face_result):
            if not face_result:
                return None
            return FaceDetectionResponse(
                id=face_result.id,
                image_path=face_result.image_path,
                status=face_result.status,
                bbox=face_result.bbox,
                landmarks=face_result.landmarks,
                confidence=face_result.confidence,
                face_size=face_result.face_size,
                occlusion_detected=face_result.occlusion_detected,
                occlusion_type=face_result.occlusion_type,
                occlusion_confidence=face_result.occlusion_confidence,
                alignment_score=face_result.alignment_score,
                face_quality_score=face_result.face_quality_score,
                pose_angles=face_result.pose_angles,
                created_at=face_result.created_at.isoformat() if face_result.created_at else None
            )
        
        response = FaceComparisonResponse(
            alignment_similarity=comparison_result['alignment_similarity'],
            face1_valid=comparison_result['face1_valid'],
            face2_valid=comparison_result['face2_valid'],
            face1_result=convert_face_result(comparison_result['face1_result']),
            face2_result=convert_face_result(comparison_result['face2_result']),
            face1_recommendations=comparison_result['face1_recommendations'],
            face2_recommendations=comparison_result['face2_recommendations'],
            overall_valid=comparison_result['overall_valid']
        )
        
        logger.info(f"Face comparison completed between {selfie_file.filename} and {document_file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error comparing face alignment: {e}")
        raise HTTPException(status_code=500, detail=f"Face comparison failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for path in [selfie_path, document_path]:
            if os.path.exists(path):
                os.unlink(path)

@router.get("/result/{result_id}", response_model=FaceDetectionResponse)
async def get_face_detection_result(
    result_id: str,
    use_case: FaceDetectionUseCase = Depends(get_face_detection_use_case)
):
    """
    Lấy kết quả face detection theo ID
    
    - **result_id**: ID của face detection result
    """
    try:
        result = await use_case.get_face_detection_result(result_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Face detection result not found")
        
        response = FaceDetectionResponse(
            id=result.id,
            image_path=result.image_path,
            status=result.status,
            bbox=result.bbox,
            landmarks=result.landmarks,
            confidence=result.confidence,
            face_size=result.face_size,
            occlusion_detected=result.occlusion_detected,
            occlusion_type=result.occlusion_type,
            occlusion_confidence=result.occlusion_confidence,
            alignment_score=result.alignment_score,
            face_quality_score=result.face_quality_score,
            pose_angles=result.pose_angles,
            created_at=result.created_at.isoformat() if result.created_at else None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting face detection result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get result: {str(e)}")

@router.get("/statistics", response_model=FaceDetectionStatsResponse)
async def get_face_detection_statistics(
    use_case: FaceDetectionUseCase = Depends(get_face_detection_use_case)
):
    """
    Lấy thống kê face detection
    """
    try:
        stats = await use_case.get_face_detection_statistics()
        
        response = FaceDetectionStatsResponse(
            total_detections=stats['total_detections'],
            success_rate=stats['success_rate'],
            average_confidence=stats['average_confidence'],
            average_quality_score=stats['average_quality_score'],
            average_alignment_score=stats['average_alignment_score'],
            status_distribution=stats['status_distribution'],
            occlusion_distribution=stats['occlusion_distribution']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting face detection statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "face-detection",
            "version": "1.0.0"
        }
    )
