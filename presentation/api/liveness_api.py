from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from application.use_cases.liveness_detection_use_case import LivenessDetectionUseCase
from domain.services.liveness_service import LivenessService
from infrastructure.repositories.mongo_liveness_repository import MongoLivenessRepository
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import tempfile
import os
import json
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/liveness", tags=["Liveness Detection"])

# Dependency injection
def get_liveness_use_case() -> LivenessDetectionUseCase:
    liveness_service = LivenessService()
    liveness_repository = MongoLivenessRepository()
    return LivenessDetectionUseCase(liveness_service, liveness_repository)

# Request/Response models
class LivenessDetectionResponse(BaseModel):
    id: str
    image_path: str
    face_bbox: Optional[List[int]]
    status: str
    liveness_result: str
    confidence: float
    liveness_score: float
    spoof_probability: float
    detected_spoof_types: List[str]
    primary_spoof_type: Optional[str]
    image_quality: float
    face_quality: float
    lighting_quality: float
    pose_quality: float
    processing_time_ms: float
    algorithms_used: List[str]
    model_version: str
    threshold_used: float
    is_real: bool
    is_fake: bool
    confidence_level: str
    risk_level: str
    created_at: Optional[str]

class LivenessStatisticsResponse(BaseModel):
    basic_statistics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    spoof_attack_trends: Dict[str, Any]
    engine_info: Dict[str, Any]
    supported_formats: List[str]

class LivenessAnalysisResponse(BaseModel):
    total_analyzed: int
    result_distribution: Dict[str, int]
    spoof_type_distribution: Dict[str, int]
    statistics: Dict[str, float]
    risk_assessment: str
    recommendations: List[str]

class LivenessValidationResponse(BaseModel):
    is_valid: bool
    issues: List[str]
    quality_score: float
    recommendations: List[str]

class LivenessComparisonResponse(BaseModel):
    results_match: bool
    confidence_difference: float
    score_difference: float
    consistency: str
    result1: Dict[str, Any]
    result2: Dict[str, Any]

class ThresholdOptimizationResponse(BaseModel):
    optimal_threshold: float
    performance_metrics: Dict[str, float]
    training_data_stats: Dict[str, Any]

@router.post("/detect", response_model=LivenessDetectionResponse)
async def detect_liveness(
    file: UploadFile = File(...),
    face_bbox: str = Form(...),  # JSON string: [x1, y1, x2, y2]
    use_advanced_analysis: bool = Form(True),
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Detect liveness từ ảnh
    
    - **file**: File ảnh chứa khuôn mặt
    - **face_bbox**: Bounding box của khuôn mặt dạng JSON "[x1, y1, x2, y2]"
    - **use_advanced_analysis**: Có sử dụng advanced analysis không (mặc định: true)
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
        # Detect liveness
        result = await use_case.detect_and_save_liveness(temp_path, bbox, use_advanced_analysis)
        
        response = LivenessDetectionResponse(
            id=result.id,
            image_path=result.image_path,
            face_bbox=result.face_bbox,
            status=result.status.value,
            liveness_result=result.liveness_result.value,
            confidence=result.confidence,
            liveness_score=result.liveness_score,
            spoof_probability=result.spoof_probability,
            detected_spoof_types=[spoof.value for spoof in result.detected_spoof_types],
            primary_spoof_type=result.primary_spoof_type.value if result.primary_spoof_type else None,
            image_quality=result.image_quality,
            face_quality=result.face_quality,
            lighting_quality=result.lighting_quality,
            pose_quality=result.pose_quality,
            processing_time_ms=result.processing_time_ms,
            algorithms_used=result.algorithms_used,
            model_version=result.model_version,
            threshold_used=result.threshold_used,
            is_real=result.is_real_face(),
            is_fake=result.is_fake_face(),
            confidence_level=result.get_confidence_level(),
            risk_level=result.get_risk_level(),
            created_at=result.created_at.isoformat() if result.created_at else None
        )
        
        logger.info(f"Liveness detection completed for file: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error detecting liveness: {e}")
        raise HTTPException(status_code=500, detail=f"Liveness detection failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/batch-detect", response_model=List[LivenessDetectionResponse])
async def batch_detect_liveness(
    files: List[UploadFile] = File(...),
    face_bboxes: str = Form(...),  # JSON string: [[x1, y1, x2, y2], ...]
    use_advanced_analysis: bool = Form(True),
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Batch liveness detection từ multiple images
    
    - **files**: List file ảnh
    - **face_bboxes**: JSON array các bounding boxes tương ứng với images
    - **use_advanced_analysis**: Có sử dụng advanced analysis không
    """
    # Validate files
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    # Parse face bboxes
    try:
        bboxes = json.loads(face_bboxes)
        if not isinstance(bboxes, list) or len(bboxes) != len(files):
            raise ValueError("Number of bboxes must match number of files")
        
        for i, bbox in enumerate(bboxes):
            if bbox and (not isinstance(bbox, list) or len(bbox) != 4):
                raise ValueError(f"Bbox at index {i} must be null or array with 4 integers")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid face_bboxes: {str(e)}")
    
    # Save uploaded files temporarily
    temp_paths = []
    try:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                contents = await file.read()
                temp_file.write(contents)
                temp_paths.append(temp_file.name)
        
        # Batch detect liveness
        results = await use_case.batch_detect_liveness(temp_paths, bboxes, use_advanced_analysis)
        
        # Convert to response format
        responses = []
        for result in results:
            response = LivenessDetectionResponse(
                id=result.id,
                image_path=result.image_path,
                face_bbox=result.face_bbox,
                status=result.status.value,
                liveness_result=result.liveness_result.value,
                confidence=result.confidence,
                liveness_score=result.liveness_score,
                spoof_probability=result.spoof_probability,
                detected_spoof_types=[spoof.value for spoof in result.detected_spoof_types],
                primary_spoof_type=result.primary_spoof_type.value if result.primary_spoof_type else None,
                image_quality=result.image_quality,
                face_quality=result.face_quality,
                lighting_quality=result.lighting_quality,
                pose_quality=result.pose_quality,
                processing_time_ms=result.processing_time_ms,
                algorithms_used=result.algorithms_used,
                model_version=result.model_version,
                threshold_used=result.threshold_used,
                is_real=result.is_real_face(),
                is_fake=result.is_fake_face(),
                confidence_level=result.get_confidence_level(),
                risk_level=result.get_risk_level(),
                created_at=result.created_at.isoformat() if result.created_at else None
            )
            responses.append(response)
        
        logger.info(f"Batch liveness detection completed for {len(files)} files")
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch liveness detection: {e}")
        raise HTTPException(status_code=500, detail=f"Batch liveness detection failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)

@router.get("/result/{result_id}", response_model=LivenessDetectionResponse)
async def get_liveness_result(
    result_id: str,
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Lấy liveness detection result theo ID
    
    - **result_id**: ID của liveness detection result
    """
    try:
        result = await use_case.get_liveness_result(result_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Liveness result not found")
        
        response = LivenessDetectionResponse(
            id=result.id,
            image_path=result.image_path,
            face_bbox=result.face_bbox,
            status=result.status.value,
            liveness_result=result.liveness_result.value,
            confidence=result.confidence,
            liveness_score=result.liveness_score,
            spoof_probability=result.spoof_probability,
            detected_spoof_types=[spoof.value for spoof in result.detected_spoof_types],
            primary_spoof_type=result.primary_spoof_type.value if result.primary_spoof_type else None,
            image_quality=result.image_quality,
            face_quality=result.face_quality,
            lighting_quality=result.lighting_quality,
            pose_quality=result.pose_quality,
            processing_time_ms=result.processing_time_ms,
            algorithms_used=result.algorithms_used,
            model_version=result.model_version,
            threshold_used=result.threshold_used,
            is_real=result.is_real_face(),
            is_fake=result.is_fake_face(),
            confidence_level=result.get_confidence_level(),
            risk_level=result.get_risk_level(),
            created_at=result.created_at.isoformat() if result.created_at else None
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting liveness result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get liveness result: {str(e)}")

@router.get("/results/recent", response_model=List[LivenessDetectionResponse])
async def get_recent_detections(
    hours: int = 24,
    limit: int = 100,
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Lấy recent liveness detections
    
    - **hours**: Số giờ look back (mặc định: 24)
    - **limit**: Số lượng results tối đa (mặc định: 100)
    """
    try:
        results = await use_case.get_recent_detections(hours, limit)
        
        # Convert to response format
        responses = []
        for result in results:
            response = LivenessDetectionResponse(
                id=result.id,
                image_path=result.image_path,
                face_bbox=result.face_bbox,
                status=result.status.value,
                liveness_result=result.liveness_result.value,
                confidence=result.confidence,
                liveness_score=result.liveness_score,
                spoof_probability=result.spoof_probability,
                detected_spoof_types=[spoof.value for spoof in result.detected_spoof_types],
                primary_spoof_type=result.primary_spoof_type.value if result.primary_spoof_type else None,
                image_quality=result.image_quality,
                face_quality=result.face_quality,
                lighting_quality=result.lighting_quality,
                pose_quality=result.pose_quality,
                processing_time_ms=result.processing_time_ms,
                algorithms_used=result.algorithms_used,
                model_version=result.model_version,
                threshold_used=result.threshold_used,
                is_real=result.is_real_face(),
                is_fake=result.is_fake_face(),
                confidence_level=result.get_confidence_level(),
                risk_level=result.get_risk_level(),
                created_at=result.created_at.isoformat() if result.created_at else None
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error getting recent detections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent detections: {str(e)}")

@router.get("/results/fake", response_model=List[LivenessDetectionResponse])
async def get_fake_detections(
    confidence_threshold: float = 0.8,
    limit: int = 100,
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Lấy fake face detections
    
    - **confidence_threshold**: Minimum confidence threshold (mặc định: 0.8)
    - **limit**: Số lượng results tối đa (mặc định: 100)
    """
    try:
        results = await use_case.get_fake_detections(confidence_threshold, limit)
        
        # Convert to response format
        responses = []
        for result in results:
            response = LivenessDetectionResponse(
                id=result.id,
                image_path=result.image_path,
                face_bbox=result.face_bbox,
                status=result.status.value,
                liveness_result=result.liveness_result.value,
                confidence=result.confidence,
                liveness_score=result.liveness_score,
                spoof_probability=result.spoof_probability,
                detected_spoof_types=[spoof.value for spoof in result.detected_spoof_types],
                primary_spoof_type=result.primary_spoof_type.value if result.primary_spoof_type else None,
                image_quality=result.image_quality,
                face_quality=result.face_quality,
                lighting_quality=result.lighting_quality,
                pose_quality=result.pose_quality,
                processing_time_ms=result.processing_time_ms,
                algorithms_used=result.algorithms_used,
                model_version=result.model_version,
                threshold_used=result.threshold_used,
                is_real=result.is_real_face(),
                is_fake=result.is_fake_face(),
                confidence_level=result.get_confidence_level(),
                risk_level=result.get_risk_level(),
                created_at=result.created_at.isoformat() if result.created_at else None
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error getting fake detections: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get fake detections: {str(e)}")

@router.get("/results/spoof/{spoof_type}", response_model=List[LivenessDetectionResponse])
async def get_spoof_attacks_by_type(
    spoof_type: str,
    limit: int = 100,
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Lấy spoof attacks theo type
    
    - **spoof_type**: Loại spoof attack (PHOTO_ATTACK, SCREEN_ATTACK, MASK_ATTACK, etc.)
    - **limit**: Số lượng results tối đa
    """
    try:
        results = await use_case.get_spoof_attacks_by_type(spoof_type, limit)
        
        # Convert to response format
        responses = []
        for result in results:
            response = LivenessDetectionResponse(
                id=result.id,
                image_path=result.image_path,
                face_bbox=result.face_bbox,
                status=result.status.value,
                liveness_result=result.liveness_result.value,
                confidence=result.confidence,
                liveness_score=result.liveness_score,
                spoof_probability=result.spoof_probability,
                detected_spoof_types=[spoof.value for spoof in result.detected_spoof_types],
                primary_spoof_type=result.primary_spoof_type.value if result.primary_spoof_type else None,
                image_quality=result.image_quality,
                face_quality=result.face_quality,
                lighting_quality=result.lighting_quality,
                pose_quality=result.pose_quality,
                processing_time_ms=result.processing_time_ms,
                algorithms_used=result.algorithms_used,
                model_version=result.model_version,
                threshold_used=result.threshold_used,
                is_real=result.is_real_face(),
                is_fake=result.is_fake_face(),
                confidence_level=result.get_confidence_level(),
                risk_level=result.get_risk_level(),
                created_at=result.created_at.isoformat() if result.created_at else None
            )
            responses.append(response)
        
        return responses
        
    except Exception as e:
        logger.error(f"Error getting spoof attacks by type: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get spoof attacks: {str(e)}")

@router.post("/analyze-patterns", response_model=LivenessAnalysisResponse)
async def analyze_liveness_patterns(
    files: List[UploadFile] = File(...),
    face_bboxes: str = Form(...),  # JSON string
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Phân tích liveness patterns cho multiple images
    
    - **files**: List file ảnh để phân tích
    - **face_bboxes**: JSON array các bounding boxes
    """
    # Validate files
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    # Parse face bboxes
    try:
        bboxes = json.loads(face_bboxes)
        if not isinstance(bboxes, list) or len(bboxes) != len(files):
            raise ValueError("Number of bboxes must match number of files")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid face_bboxes: {str(e)}")
    
    # Save uploaded files temporarily
    temp_paths = []
    try:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                contents = await file.read()
                temp_file.write(contents)
                temp_paths.append(temp_file.name)
        
        # Analyze patterns
        analysis = await use_case.analyze_liveness_patterns(temp_paths, bboxes)
        
        if "error" in analysis:
            raise HTTPException(status_code=500, detail=analysis["error"])
        
        response = LivenessAnalysisResponse(
            total_analyzed=analysis["total_analyzed"],
            result_distribution=analysis["result_distribution"],
            spoof_type_distribution=analysis["spoof_type_distribution"],
            statistics=analysis["statistics"],
            risk_assessment=analysis["risk_assessment"],
            recommendations=analysis["recommendations"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing liveness patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)

@router.get("/validate/{result_id}", response_model=LivenessValidationResponse)
async def validate_detection_quality(
    result_id: str,
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Validate chất lượng detection result
    
    - **result_id**: ID của liveness detection result
    """
    try:
        validation = await use_case.validate_detection_quality(result_id)
        
        if "error" in validation:
            raise HTTPException(status_code=404, detail=validation["error"])
        
        response = LivenessValidationResponse(
            is_valid=validation["is_valid"],
            issues=validation["issues"],
            quality_score=validation["quality_score"],
            recommendations=validation["recommendations"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating detection quality: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/compare/{result_id1}/{result_id2}", response_model=LivenessComparisonResponse)
async def compare_liveness_results(
    result_id1: str,
    result_id2: str,
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    So sánh 2 liveness detection results
    
    - **result_id1**: ID của result đầu tiên
    - **result_id2**: ID của result thứ hai
    """
    try:
        comparison = await use_case.compare_liveness_results(result_id1, result_id2)
        
        if "error" in comparison:
            raise HTTPException(status_code=404, detail=comparison["error"])
        
        response = LivenessComparisonResponse(
            results_match=comparison["results_match"],
            confidence_difference=comparison["confidence_difference"],
            score_difference=comparison["score_difference"],
            consistency=comparison["consistency"],
            result1=comparison["result1"],
            result2=comparison["result2"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing liveness results: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/statistics", response_model=LivenessStatisticsResponse)
async def get_liveness_statistics(
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Lấy comprehensive liveness detection statistics
    """
    try:
        stats = await use_case.get_liveness_statistics()
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        
        response = LivenessStatisticsResponse(
            basic_statistics=stats["basic_statistics"],
            performance_metrics=stats["performance_metrics"],
            spoof_attack_trends=stats["spoof_attack_trends"],
            engine_info=stats["engine_info"],
            supported_formats=stats["supported_formats"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting liveness statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/optimize-thresholds", response_model=ThresholdOptimizationResponse)
async def optimize_detection_thresholds(
    training_data: str = Form(...),  # JSON string with training samples
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Optimize detection thresholds với training data
    
    - **training_data**: JSON array với format [{"image_path": str, "face_bbox": [int], "is_real": bool}, ...]
    """
    try:
        # Parse training data
        training_samples = json.loads(training_data)
        if not isinstance(training_samples, list):
            raise ValueError("Training data must be an array")
        
        # Optimize thresholds
        optimization = await use_case.optimize_detection_thresholds(training_samples)
        
        if "error" in optimization:
            raise HTTPException(status_code=400, detail=optimization["error"])
        
        response = ThresholdOptimizationResponse(
            optimal_threshold=optimization["optimal_threshold"],
            performance_metrics=optimization["performance_metrics"],
            training_data_stats=optimization["training_data_stats"]
        )
        
        return response
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in training_data")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing thresholds: {e}")
        raise HTTPException(status_code=500, detail=f"Threshold optimization failed: {str(e)}")

@router.delete("/cleanup")
async def cleanup_old_results(
    days_to_keep: int = 30,
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Clean up old liveness detection results
    
    - **days_to_keep**: Số ngày data cần giữ lại (mặc định: 30)
    """
    try:
        cleanup_result = await use_case.cleanup_old_results(days_to_keep)
        
        return JSONResponse(content={
            "message": "Cleanup completed successfully",
            "deleted_count": cleanup_result["deleted_count"],
            "cutoff_date": cleanup_result.get("cutoff_date"),
            "days_kept": cleanup_result.get("days_kept"),
            "success": cleanup_result["success"]
        })
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.delete("/result/{result_id}")
async def delete_liveness_result(
    result_id: str,
    use_case: LivenessDetectionUseCase = Depends(get_liveness_use_case)
):
    """
    Xóa liveness detection result
    
    - **result_id**: ID của result cần xóa
    """
    try:
        success = await use_case.delete_liveness_result(result_id)
        
        if success:
            return JSONResponse(content={
                "message": "Liveness result deleted successfully",
                "result_id": result_id,
                "success": True
            })
        else:
            raise HTTPException(status_code=404, detail="Liveness result not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting liveness result: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "liveness-detection",
            "version": "1.0.0"
        }
    )
