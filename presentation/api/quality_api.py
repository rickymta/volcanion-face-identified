import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from application.use_cases.document_quality_check_usecase import DocumentQualityCheckUseCase
from domain.services.document_quality_service import DocumentQualityService
from infrastructure.adapters.document_quality_repository_impl import DocumentQualityRepositoryImpl
from presentation.schemas.quality_schemas import (
    QualityCheckRequest,
    QualityCheckResponse,
    QualityListResponse,
    RecommendationsRequest,
    RecommendationsResponse,
    QualityStatusEnum,
    TamperTypeEnum,
    QualityMetrics,
    TamperAnalysis
)
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/quality", tags=["Document Quality Check"])

# Dependency injection
def get_quality_service():
    return DocumentQualityService()

def get_quality_repository():
    return DocumentQualityRepositoryImpl()

def get_quality_usecase(
    service: DocumentQualityService = Depends(get_quality_service),
    repository: DocumentQualityRepositoryImpl = Depends(get_quality_repository)
):
    return DocumentQualityCheckUseCase(service, repository)

@router.post('/check', response_model=QualityCheckResponse)
async def check_document_quality(
    file: UploadFile = File(...),
    save_to_db: bool = True,
    bbox: str = None,  # JSON string for bbox
    usecase: DocumentQualityCheckUseCase = Depends(get_quality_usecase)
):
    """
    Kiểm tra chất lượng và tamper detection cho giấy tờ
    """
    # Kiểm tra file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Tạo đường dẫn lưu file tạm
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        # Lưu file tạm
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse bbox nếu có
        bbox_list = None
        if bbox:
            try:
                import json
                bbox_list = json.loads(bbox)
            except:
                logger.warning(f"Invalid bbox format: {bbox}")
        
        logger.info(f"Checking quality for file: {file.filename}")
        
        # Thực hiện kiểm tra chất lượng
        quality = usecase.execute(file_path, bbox_list, save_to_db)
        
        # Lấy khuyến nghị
        service = DocumentQualityService()
        recommendations = service.get_quality_recommendations(quality)
        
        # Tạo response
        response = QualityCheckResponse(
            overall_quality=QualityStatusEnum(quality.overall_quality.value),
            quality_score=quality.quality_score,
            is_acceptable=quality.is_acceptable(),
            metrics=QualityMetrics(
                blur_score=quality.blur_score,
                glare_score=quality.glare_score,
                contrast_score=quality.contrast_score,
                brightness_score=quality.brightness_score,
                noise_score=quality.noise_score,
                edge_sharpness=quality.edge_sharpness
            ),
            tamper_analysis=TamperAnalysis(
                tamper_detected=quality.tamper_detected,
                tamper_type=TamperTypeEnum(quality.tamper_type.value),
                tamper_confidence=quality.tamper_confidence,
                metadata_suspicious=len(quality.metadata_analysis.get('suspicious_indicators', [])) > 0
            ),
            watermark_present=quality.watermark_present,
            recommendations=recommendations,
            message=f"Quality: {quality.overall_quality.value}, Score: {quality.quality_score:.2f}"
        )
        
        logger.info(f"Quality check completed: {quality.overall_quality.value}, "
                   f"score: {quality.quality_score:.2f}, tamper: {quality.tamper_detected}")
        return response
        
    except Exception as e:
        logger.error(f"Error checking document quality: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking quality: {str(e)}")
    
    finally:
        # Xóa file tạm
        if os.path.exists(file_path):
            os.remove(file_path)

@router.post('/recommendations', response_model=RecommendationsResponse)
async def get_quality_recommendations(
    file: UploadFile = File(...),
    bbox: str = None,
    usecase: DocumentQualityCheckUseCase = Depends(get_quality_usecase)
):
    """
    Lấy khuyến nghị cải thiện chất lượng mà không lưu vào DB
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        bbox_list = None
        if bbox:
            try:
                import json
                bbox_list = json.loads(bbox)
            except:
                pass
        
        recommendations = usecase.get_recommendations(file_path, bbox_list)
        
        # Cũng lấy quality score để trả về
        service = DocumentQualityService()
        quality = service.analyze_quality(file_path, bbox_list)
        
        return RecommendationsResponse(
            recommendations=recommendations,
            quality_score=quality.quality_score
        )
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@router.get('/list', response_model=QualityListResponse)
async def list_quality_checks(
    repository: DocumentQualityRepositoryImpl = Depends(get_quality_repository)
):
    """
    Lấy danh sách tất cả quality checks đã lưu
    """
    try:
        qualities = repository.get_all()
        service = DocumentQualityService()
        
        quality_responses = []
        for quality in qualities:
            recommendations = service.get_quality_recommendations(quality)
            
            quality_response = QualityCheckResponse(
                overall_quality=QualityStatusEnum(quality.overall_quality.value),
                quality_score=quality.quality_score,
                is_acceptable=quality.is_acceptable(),
                metrics=QualityMetrics(
                    blur_score=quality.blur_score,
                    glare_score=quality.glare_score,
                    contrast_score=quality.contrast_score,
                    brightness_score=quality.brightness_score,
                    noise_score=quality.noise_score,
                    edge_sharpness=quality.edge_sharpness
                ),
                tamper_analysis=TamperAnalysis(
                    tamper_detected=quality.tamper_detected,
                    tamper_type=TamperTypeEnum(quality.tamper_type.value),
                    tamper_confidence=quality.tamper_confidence,
                    metadata_suspicious=len(quality.metadata_analysis.get('suspicious_indicators', [])) > 0
                ),
                watermark_present=quality.watermark_present,
                recommendations=recommendations,
                message=f"Quality: {quality.overall_quality.value}"
            )
            quality_responses.append(quality_response)
        
        return QualityListResponse(
            qualities=quality_responses,
            total=len(quality_responses)
        )
        
    except Exception as e:
        logger.error(f"Error listing quality checks: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing quality checks: {str(e)}")

@router.get('/health')
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "document_quality_check"}

@router.delete('/{quality_id}')
async def delete_quality_check(
    quality_id: str,
    repository: DocumentQualityRepositoryImpl = Depends(get_quality_repository)
):
    """
    Xóa quality check theo ID
    """
    try:
        success = repository.delete_by_id(quality_id)
        if success:
            return {"message": f"Quality check {quality_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Quality check not found")
            
    except Exception as e:
        logger.error(f"Error deleting quality check: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting quality check: {str(e)}")

@router.get('/stats')
async def get_quality_statistics(
    repository: DocumentQualityRepositoryImpl = Depends(get_quality_repository)
):
    """
    Lấy thống kê về quality checks
    """
    try:
        qualities = repository.get_all()
        
        if not qualities:
            return {
                "total_checks": 0,
                "quality_distribution": {},
                "tamper_detection_rate": 0.0,
                "average_quality_score": 0.0
            }
        
        # Thống kê phân phối quality
        quality_counts = {}
        tamper_count = 0
        total_score = 0.0
        
        for quality in qualities:
            status = quality.overall_quality.value
            quality_counts[status] = quality_counts.get(status, 0) + 1
            
            if quality.tamper_detected:
                tamper_count += 1
            
            total_score += quality.quality_score
        
        return {
            "total_checks": len(qualities),
            "quality_distribution": quality_counts,
            "tamper_detection_rate": tamper_count / len(qualities),
            "average_quality_score": total_score / len(qualities),
            "acceptable_rate": sum(1 for q in qualities if q.is_acceptable()) / len(qualities)
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")
