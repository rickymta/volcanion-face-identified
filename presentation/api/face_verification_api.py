from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from application.use_cases.face_verification_use_case import FaceVerificationUseCase
from domain.services.face_verification_service import FaceVerificationService
from infrastructure.repositories.mongo_face_verification_repository import MongoFaceEmbeddingRepository, MongoFaceVerificationRepository
from pydantic import BaseModel
from typing import Optional, List, Tuple
import tempfile
import os
import json
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/face-verification", tags=["Face Verification"])

# Dependency injection
def get_face_verification_use_case() -> FaceVerificationUseCase:
    face_verification_service = FaceVerificationService()
    embedding_repository = MongoFaceEmbeddingRepository()
    verification_repository = MongoFaceVerificationRepository()
    return FaceVerificationUseCase(face_verification_service, embedding_repository, verification_repository)

# Request/Response models
class FaceEmbeddingResponse(BaseModel):
    id: str
    image_path: str
    face_bbox: Optional[List[int]]
    embedding_dimension: int
    embedding_model: str
    feature_quality: float
    extraction_confidence: float
    face_alignment_score: float
    preprocessing_applied: bool
    created_at: Optional[str]

class FaceVerificationResponse(BaseModel):
    id: str
    reference_image_path: str
    target_image_path: str
    reference_embedding_id: Optional[str]
    target_embedding_id: Optional[str]
    status: str
    verification_result: str
    similarity_score: float
    distance_metric: str
    confidence: float
    threshold_used: float
    match_probability: float
    processing_time_ms: float
    model_used: str
    quality_assessment: Optional[dict]
    is_match: bool
    confidence_level: str
    created_at: Optional[str]

class VerificationStatisticsResponse(BaseModel):
    verification_statistics: dict
    embedding_statistics: dict
    threshold_analysis: dict
    performance_metrics: dict

class ThresholdOptimizationResponse(BaseModel):
    best_threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

@router.post("/extract-embedding", response_model=FaceEmbeddingResponse)
async def extract_face_embedding(
    file: UploadFile = File(...),
    face_bbox: str = Form(...),  # JSON string: [x1, y1, x2, y2]
    model_type: str = Form("facenet"),
    use_case: FaceVerificationUseCase = Depends(get_face_verification_use_case)
):
    """
    Extract face embedding từ ảnh
    
    - **file**: File ảnh chứa khuôn mặt
    - **face_bbox**: Bounding box của khuôn mặt dạng JSON "[x1, y1, x2, y2]"
    - **model_type**: Loại model để extract embedding ("facenet", "dlib")
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Parse face bbox
    try:
        bbox = json.loads(face_bbox)
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Invalid bbox format")
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="face_bbox must be a JSON array with 4 integers: [x1, y1, x2, y2]")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name
    
    try:
        # Extract and save embedding
        embedding = await use_case.extract_and_save_embedding(temp_path, bbox, model_type)
        
        response = FaceEmbeddingResponse(
            id=embedding.id,
            image_path=embedding.image_path,
            face_bbox=embedding.face_bbox,
            embedding_dimension=embedding.get_embedding_dimension(),
            embedding_model=embedding.embedding_model,
            feature_quality=embedding.feature_quality,
            extraction_confidence=embedding.extraction_confidence,
            face_alignment_score=embedding.face_alignment_score,
            preprocessing_applied=embedding.preprocessing_applied,
            created_at=embedding.created_at.isoformat() if embedding.created_at else None
        )
        
        logger.info(f"Embedding extracted for file: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/verify-by-embeddings", response_model=FaceVerificationResponse)
async def verify_faces_by_embeddings(
    reference_embedding_id: str = Form(...),
    target_embedding_id: str = Form(...),
    threshold: Optional[float] = Form(None),
    distance_metric: str = Form("cosine"),
    use_case: FaceVerificationUseCase = Depends(get_face_verification_use_case)
):
    """
    Verify hai khuôn mặt sử dụng existing embeddings
    
    - **reference_embedding_id**: ID của reference embedding
    - **target_embedding_id**: ID của target embedding
    - **threshold**: Ngưỡng similarity (mặc định: 0.6)
    - **distance_metric**: Metric để tính khoảng cách ("cosine", "euclidean", "manhattan")
    """
    try:
        # Perform verification
        verification_result = await use_case.verify_faces_by_embeddings(
            reference_embedding_id, target_embedding_id, threshold, distance_metric
        )
        
        response = FaceVerificationResponse(
            id=verification_result.id,
            reference_image_path=verification_result.reference_image_path,
            target_image_path=verification_result.target_image_path,
            reference_embedding_id=verification_result.reference_embedding_id,
            target_embedding_id=verification_result.target_embedding_id,
            status=verification_result.status.value,
            verification_result=verification_result.verification_result.value,
            similarity_score=verification_result.similarity_score,
            distance_metric=verification_result.distance_metric,
            confidence=verification_result.confidence,
            threshold_used=verification_result.threshold_used,
            match_probability=verification_result.match_probability,
            processing_time_ms=verification_result.processing_time_ms,
            model_used=verification_result.model_used,
            quality_assessment=verification_result.quality_assessment,
            is_match=verification_result.is_positive_match(),
            confidence_level=verification_result.get_confidence_level(),
            created_at=verification_result.created_at.isoformat() if verification_result.created_at else None
        )
        
        logger.info(f"Face verification completed: {verification_result.verification_result.value}")
        return response
        
    except Exception as e:
        logger.error(f"Error verifying faces by embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")

@router.post("/verify-by-images", response_model=FaceVerificationResponse)
async def verify_faces_by_images(
    reference_file: UploadFile = File(...),
    target_file: UploadFile = File(...),
    reference_bbox: str = Form(...),  # JSON string
    target_bbox: str = Form(...),     # JSON string
    threshold: Optional[float] = Form(None),
    distance_metric: str = Form("cosine"),
    model_type: str = Form("facenet"),
    use_case: FaceVerificationUseCase = Depends(get_face_verification_use_case)
):
    """
    Verify hai khuôn mặt trực tiếp từ images
    
    - **reference_file**: File ảnh reference
    - **target_file**: File ảnh target
    - **reference_bbox**: Bounding box của khuôn mặt reference
    - **target_bbox**: Bounding box của khuôn mặt target
    - **threshold**: Ngưỡng similarity
    - **distance_metric**: Metric để tính khoảng cách
    - **model_type**: Loại model embedding
    """
    if not reference_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Reference file must be an image")
    
    if not target_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Target file must be an image")
    
    # Parse bounding boxes
    try:
        ref_bbox = json.loads(reference_bbox)
        tar_bbox = json.loads(target_bbox)
        if not (isinstance(ref_bbox, list) and len(ref_bbox) == 4 and
                isinstance(tar_bbox, list) and len(tar_bbox) == 4):
            raise ValueError("Invalid bbox format")
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Bounding boxes must be JSON arrays with 4 integers")
    
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_ref:
        ref_contents = await reference_file.read()
        temp_ref.write(ref_contents)
        ref_path = temp_ref.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_tar:
        tar_contents = await target_file.read()
        temp_tar.write(tar_contents)
        tar_path = temp_tar.name
    
    try:
        # Perform verification
        verification_result = await use_case.verify_faces_by_images(
            ref_path, ref_bbox, tar_path, tar_bbox, threshold, distance_metric, model_type
        )
        
        response = FaceVerificationResponse(
            id=verification_result.id,
            reference_image_path=verification_result.reference_image_path,
            target_image_path=verification_result.target_image_path,
            reference_embedding_id=verification_result.reference_embedding_id,
            target_embedding_id=verification_result.target_embedding_id,
            status=verification_result.status.value,
            verification_result=verification_result.verification_result.value,
            similarity_score=verification_result.similarity_score,
            distance_metric=verification_result.distance_metric,
            confidence=verification_result.confidence,
            threshold_used=verification_result.threshold_used,
            match_probability=verification_result.match_probability,
            processing_time_ms=verification_result.processing_time_ms,
            model_used=verification_result.model_used,
            quality_assessment=verification_result.quality_assessment,
            is_match=verification_result.is_positive_match(),
            confidence_level=verification_result.get_confidence_level(),
            created_at=verification_result.created_at.isoformat() if verification_result.created_at else None
        )
        
        logger.info(f"Face verification completed: {verification_result.verification_result.value}")
        return response
        
    except Exception as e:
        logger.error(f"Error verifying faces by images: {e}")
        raise HTTPException(status_code=500, detail=f"Face verification failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for path in [ref_path, tar_path]:
            if os.path.exists(path):
                os.unlink(path)

@router.post("/find-matches", response_model=List[FaceVerificationResponse])
async def find_matches_in_gallery(
    target_file: UploadFile = File(...),
    target_bbox: str = Form(...),
    gallery_image_paths: Optional[str] = Form(None),  # JSON array of image paths
    top_k: int = Form(5),
    threshold: Optional[float] = Form(None),
    distance_metric: str = Form("cosine"),
    model_type: str = Form("facenet"),
    use_case: FaceVerificationUseCase = Depends(get_face_verification_use_case)
):
    """
    Tìm matching faces trong gallery
    
    - **target_file**: File ảnh target để tìm matches
    - **target_bbox**: Bounding box của khuôn mặt target
    - **gallery_image_paths**: JSON array các đường dẫn ảnh trong gallery (optional)
    - **top_k**: Số lượng matches tốt nhất trả về
    - **threshold**: Ngưỡng similarity
    - **distance_metric**: Metric để tính khoảng cách
    - **model_type**: Loại model embedding
    """
    if not target_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Target file must be an image")
    
    # Parse target bbox
    try:
        tar_bbox = json.loads(target_bbox)
        if not (isinstance(tar_bbox, list) and len(tar_bbox) == 4):
            raise ValueError("Invalid bbox format")
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="target_bbox must be a JSON array with 4 integers")
    
    # Parse gallery image paths
    gallery_paths = None
    if gallery_image_paths:
        try:
            gallery_paths = json.loads(gallery_image_paths)
            if not isinstance(gallery_paths, list):
                raise ValueError("Invalid gallery paths format")
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="gallery_image_paths must be a JSON array of strings")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_tar:
        tar_contents = await target_file.read()
        temp_tar.write(tar_contents)
        tar_path = temp_tar.name
    
    try:
        # Find matches
        matches = await use_case.find_matches_in_gallery(
            tar_path, tar_bbox, gallery_paths, top_k, threshold, distance_metric, model_type
        )
        
        # Convert to response format
        responses = []
        for match in matches:
            response = FaceVerificationResponse(
                id=match.id,
                reference_image_path=match.reference_image_path,
                target_image_path=match.target_image_path,
                reference_embedding_id=match.reference_embedding_id,
                target_embedding_id=match.target_embedding_id,
                status=match.status.value,
                verification_result=match.verification_result.value,
                similarity_score=match.similarity_score,
                distance_metric=match.distance_metric,
                confidence=match.confidence,
                threshold_used=match.threshold_used,
                match_probability=match.match_probability,
                processing_time_ms=match.processing_time_ms,
                model_used=match.model_used,
                quality_assessment=match.quality_assessment,
                is_match=match.is_positive_match(),
                confidence_level=match.get_confidence_level(),
                created_at=match.created_at.isoformat() if match.created_at else None
            )
            responses.append(response)
        
        logger.info(f"Found {len(matches)} matches for target image")
        return responses
        
    except Exception as e:
        logger.error(f"Error finding matches: {e}")
        raise HTTPException(status_code=500, detail=f"Match finding failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(tar_path):
            os.unlink(tar_path)

@router.get("/embedding/{embedding_id}", response_model=FaceEmbeddingResponse)
async def get_face_embedding(
    embedding_id: str,
    use_case: FaceVerificationUseCase = Depends(get_face_verification_use_case)
):
    """
    Lấy face embedding theo ID
    
    - **embedding_id**: ID của embedding
    """
    try:
        embedding = await use_case.get_embedding(embedding_id)
        
        if not embedding:
            raise HTTPException(status_code=404, detail="Face embedding not found")
        
        response = FaceEmbeddingResponse(
            id=embedding.id,
            image_path=embedding.image_path,
            face_bbox=embedding.face_bbox,
            embedding_dimension=embedding.get_embedding_dimension(),
            embedding_model=embedding.embedding_model,
            feature_quality=embedding.feature_quality,
            extraction_confidence=embedding.extraction_confidence,
            face_alignment_score=embedding.face_alignment_score,
            preprocessing_applied=embedding.preprocessing_applied,
            created_at=embedding.created_at.isoformat() if embedding.created_at else None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get embedding: {str(e)}")

@router.get("/verification/{verification_id}", response_model=FaceVerificationResponse)
async def get_verification_result(
    verification_id: str,
    use_case: FaceVerificationUseCase = Depends(get_face_verification_use_case)
):
    """
    Lấy verification result theo ID
    
    - **verification_id**: ID của verification result
    """
    try:
        verification = await use_case.get_verification_result(verification_id)
        
        if not verification:
            raise HTTPException(status_code=404, detail="Verification result not found")
        
        response = FaceVerificationResponse(
            id=verification.id,
            reference_image_path=verification.reference_image_path,
            target_image_path=verification.target_image_path,
            reference_embedding_id=verification.reference_embedding_id,
            target_embedding_id=verification.target_embedding_id,
            status=verification.status.value,
            verification_result=verification.verification_result.value,
            similarity_score=verification.similarity_score,
            distance_metric=verification.distance_metric,
            confidence=verification.confidence,
            threshold_used=verification.threshold_used,
            match_probability=verification.match_probability,
            processing_time_ms=verification.processing_time_ms,
            model_used=verification.model_used,
            quality_assessment=verification.quality_assessment,
            is_match=verification.is_positive_match(),
            confidence_level=verification.get_confidence_level(),
            created_at=verification.created_at.isoformat() if verification.created_at else None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting verification result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get verification result: {str(e)}")

@router.get("/statistics", response_model=VerificationStatisticsResponse)
async def get_verification_statistics(
    use_case: FaceVerificationUseCase = Depends(get_face_verification_use_case)
):
    """
    Lấy thống kê face verification
    """
    try:
        stats = await use_case.get_verification_statistics()
        
        response = VerificationStatisticsResponse(
            verification_statistics=stats.get('verification_statistics', {}),
            embedding_statistics=stats.get('embedding_statistics', {}),
            threshold_analysis=stats.get('threshold_analysis', {}),
            performance_metrics=stats.get('performance_metrics', {})
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting verification statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/optimize-threshold", response_model=ThresholdOptimizationResponse)
async def optimize_threshold(
    positive_pairs: str = Form(...),  # JSON array of [ref_image, target_image] pairs
    negative_pairs: str = Form(...),  # JSON array of [ref_image, target_image] pairs
    distance_metric: str = Form("cosine"),
    use_case: FaceVerificationUseCase = Depends(get_face_verification_use_case)
):
    """
    Optimize threshold sử dụng positive và negative pairs
    
    - **positive_pairs**: JSON array của positive pairs [[ref_img, tar_img], ...]
    - **negative_pairs**: JSON array của negative pairs [[ref_img, tar_img], ...]
    - **distance_metric**: Metric để tính khoảng cách
    """
    try:
        # Parse pairs
        pos_pairs = json.loads(positive_pairs)
        neg_pairs = json.loads(negative_pairs)
        
        if not (isinstance(pos_pairs, list) and isinstance(neg_pairs, list)):
            raise ValueError("Invalid pairs format")
        
        # Optimize threshold
        optimization_result = await use_case.optimize_threshold(
            pos_pairs, neg_pairs, distance_metric
        )
        
        response = ThresholdOptimizationResponse(
            best_threshold=optimization_result['best_threshold'],
            accuracy=optimization_result.get('accuracy', 0.0),
            precision=optimization_result.get('precision', 0.0),
            recall=optimization_result.get('recall', 0.0),
            f1_score=optimization_result.get('f1_score', 0.0)
        )
        
        return response
        
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid JSON format for pairs")
    except Exception as e:
        logger.error(f"Error optimizing threshold: {e}")
        raise HTTPException(status_code=500, detail=f"Threshold optimization failed: {str(e)}")

@router.delete("/cleanup")
async def cleanup_old_data(
    days_to_keep: int = 30,
    use_case: FaceVerificationUseCase = Depends(get_face_verification_use_case)
):
    """
    Clean up old embeddings và verification results
    
    - **days_to_keep**: Số ngày data cần giữ lại (mặc định: 30)
    """
    try:
        cleanup_result = await use_case.cleanup_old_data(days_to_keep)
        
        return JSONResponse(content={
            "message": "Cleanup completed successfully",
            "deleted_verifications": cleanup_result['deleted_verifications'],
            "deleted_embeddings": cleanup_result['deleted_embeddings'],
            "cutoff_date": cleanup_result.get('cutoff_date'),
            "success": cleanup_result['success']
        })
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "face-verification",
            "version": "1.0.0"
        }
    )
