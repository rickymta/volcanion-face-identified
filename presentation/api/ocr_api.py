from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Query
from fastapi.responses import JSONResponse
from application.use_cases.ocr_extraction_use_case import OCRExtractionUseCase
from domain.services.ocr_service import OCRService
from infrastructure.repositories.mongo_ocr_repository import MongoOCRRepository
from domain.entities.ocr_result import DocumentType, OCRStatus, FieldType
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import tempfile
import os
import json
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/api/ocr", tags=["OCR Extraction"])

# Dependency injection
def get_ocr_use_case() -> OCRExtractionUseCase:
    ocr_service = OCRService()
    ocr_repository = MongoOCRRepository()
    return OCRExtractionUseCase(ocr_service, ocr_repository)

# Request/Response models
class BoundingBoxResponse(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class TextRegionResponse(BaseModel):
    text: str
    confidence: float
    bbox: BoundingBoxResponse
    field_type: Optional[str]
    language: Optional[str]
    font_size: Optional[float]
    is_handwritten: bool

class ExtractedFieldResponse(BaseModel):
    field_type: str
    value: str
    confidence: float
    bbox: BoundingBoxResponse
    raw_text: str
    normalized_value: Optional[str]
    validation_status: bool
    validation_errors: List[str]

class OCRStatisticsResponse(BaseModel):
    total_text_regions: int
    reliable_regions: int
    unreliable_regions: int
    average_confidence: float
    total_characters: int
    processing_time_ms: float
    languages_detected: List[str]
    reliability_rate: float

class OCRResultResponse(BaseModel):
    id: str
    image_path: str
    document_type: str
    status: str
    text_regions: List[TextRegionResponse]
    extracted_fields: List[ExtractedFieldResponse]
    full_text: str
    statistics: OCRStatisticsResponse
    processing_time_ms: float
    model_version: str
    languages_detected: List[str]
    confidence_threshold: float
    preprocessing_applied: List[str]
    overall_confidence: float
    confidence_level: str
    completion_rate: float
    is_successful: bool
    error_message: Optional[str]
    created_at: Optional[str]

class OCRExtractionStatisticsResponse(BaseModel):
    basic_statistics: Dict[str, Any]
    confidence_statistics: Dict[str, Any]
    performance_statistics: Dict[str, Any]
    document_type_distribution: Dict[str, int]
    field_extraction_statistics: Dict[str, Any]
    languages_detected: List[str]

class OCRValidationResponse(BaseModel):
    is_valid: bool
    quality_score: float
    issues: List[str]
    recommendations: List[str]
    statistics: Dict[str, Any]

class OCRComparisonResponse(BaseModel):
    similarity: float
    confidence_difference: float
    field_count_difference: int
    field_matches: int
    field_differences: List[Dict[str, Any]]
    consistency: str
    result1_summary: Dict[str, Any]
    result2_summary: Dict[str, Any]

class PatternAnalysisResponse(BaseModel):
    basic_statistics: Dict[str, Any]
    confidence_statistics: Dict[str, Any]
    performance_statistics: Dict[str, Any]
    document_type_distribution: Dict[str, int]
    field_extraction_statistics: Dict[str, Any]
    languages_detected: List[str]
    pattern_insights: Dict[str, Any]

@router.post("/extract", response_model=OCRResultResponse)
async def extract_text(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(DocumentType.OTHER),
    confidence_threshold: float = Form(0.5),
    use_preprocessing: bool = Form(True),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Extract text từ document image
    
    - **file**: File ảnh document
    - **document_type**: Loại document (CMND, CCCD, PASSPORT, etc.)
    - **confidence_threshold**: Ngưỡng confidence (0.0-1.0)
    - **use_preprocessing**: Có sử dụng image preprocessing không
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name
    
    try:
        # Extract text
        result = await use_case.extract_and_save_text(
            temp_path, document_type, confidence_threshold, use_preprocessing
        )
        
        response = _convert_to_response(result)
        
        logger.info(f"OCR extraction completed for file: {file.filename}")
        return response
        
    except Exception as e:
        logger.error(f"Error in OCR extraction: {e}")
        raise HTTPException(status_code=500, detail=f"OCR extraction failed: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/batch-extract", response_model=List[OCRResultResponse])
async def batch_extract_text(
    files: List[UploadFile] = File(...),
    document_types: str = Form(...),  # JSON string
    confidence_threshold: float = Form(0.5),
    use_preprocessing: bool = Form(True),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Batch OCR extraction từ multiple images
    
    - **files**: List file ảnh documents
    - **document_types**: JSON array các document types tương ứng
    - **confidence_threshold**: Ngưỡng confidence
    - **use_preprocessing**: Có sử dụng preprocessing không
    """
    # Validate files
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    # Parse document types
    try:
        doc_types_list = json.loads(document_types)
        if len(doc_types_list) != len(files):
            raise ValueError("Number of document types must match number of files")
        
        parsed_doc_types = [DocumentType(dt) for dt in doc_types_list]
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid document_types: {str(e)}")
    
    # Save uploaded files temporarily
    temp_paths = []
    try:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                contents = await file.read()
                temp_file.write(contents)
                temp_paths.append(temp_file.name)
        
        # Batch extract
        results = await use_case.batch_extract_and_save_text(
            temp_paths, parsed_doc_types, confidence_threshold, use_preprocessing
        )
        
        # Convert to response format
        responses = [_convert_to_response(result) for result in results]
        
        logger.info(f"Batch OCR extraction completed for {len(files)} files")
        return responses
        
    except Exception as e:
        logger.error(f"Error in batch OCR extraction: {e}")
        raise HTTPException(status_code=500, detail=f"Batch OCR extraction failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)

@router.get("/result/{result_id}", response_model=OCRResultResponse)
async def get_ocr_result(
    result_id: str,
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy OCR result theo ID
    
    - **result_id**: ID của OCR result
    """
    try:
        result = await use_case.get_ocr_result(result_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="OCR result not found")
        
        response = _convert_to_response(result)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting OCR result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get OCR result: {str(e)}")

@router.get("/results/recent", response_model=List[OCRResultResponse])
async def get_recent_extractions(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=1000),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy recent OCR extractions
    
    - **hours**: Số giờ look back (1-168)
    - **limit**: Số lượng results tối đa (1-1000)
    """
    try:
        results = await use_case.get_recent_extractions(hours, limit)
        responses = [_convert_to_response(result) for result in results]
        return responses
        
    except Exception as e:
        logger.error(f"Error getting recent extractions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent extractions: {str(e)}")

@router.get("/results/successful", response_model=List[OCRResultResponse])
async def get_successful_extractions(
    limit: int = Query(100, ge=1, le=1000),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy successful OCR extractions
    
    - **limit**: Số lượng results tối đa
    """
    try:
        results = await use_case.get_successful_extractions(limit)
        responses = [_convert_to_response(result) for result in results]
        return responses
        
    except Exception as e:
        logger.error(f"Error getting successful extractions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get successful extractions: {str(e)}")

@router.get("/results/failed", response_model=List[OCRResultResponse])
async def get_failed_extractions(
    limit: int = Query(100, ge=1, le=1000),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy failed OCR extractions
    
    - **limit**: Số lượng results tối đa
    """
    try:
        results = await use_case.get_failed_extractions(limit)
        responses = [_convert_to_response(result) for result in results]
        return responses
        
    except Exception as e:
        logger.error(f"Error getting failed extractions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get failed extractions: {str(e)}")

@router.get("/results/document-type/{document_type}", response_model=List[OCRResultResponse])
async def get_extractions_by_document_type(
    document_type: DocumentType,
    limit: int = Query(100, ge=1, le=1000),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy OCR extractions theo document type
    
    - **document_type**: Loại document
    - **limit**: Số lượng results tối đa
    """
    try:
        results = await use_case.get_extractions_by_document_type(document_type, limit)
        responses = [_convert_to_response(result) for result in results]
        return responses
        
    except Exception as e:
        logger.error(f"Error getting extractions by document type: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get extractions: {str(e)}")

@router.get("/results/field-type/{field_type}", response_model=List[OCRResultResponse])
async def get_extractions_by_field_type(
    field_type: FieldType,
    limit: int = Query(100, ge=1, le=1000),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy OCR extractions có chứa field type cụ thể
    
    - **field_type**: Loại field
    - **limit**: Số lượng results tối đa
    """
    try:
        results = await use_case.get_extractions_by_field_type(field_type, limit)
        responses = [_convert_to_response(result) for result in results]
        return responses
        
    except Exception as e:
        logger.error(f"Error getting extractions by field type: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get extractions: {str(e)}")

@router.get("/results/confidence", response_model=List[OCRResultResponse])
async def get_extractions_by_confidence(
    min_confidence: float = Query(0.0, ge=0.0, le=1.0),
    max_confidence: float = Query(1.0, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=1000),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy OCR extractions theo confidence range
    
    - **min_confidence**: Confidence tối thiểu
    - **max_confidence**: Confidence tối đa
    - **limit**: Số lượng results tối đa
    """
    if min_confidence > max_confidence:
        raise HTTPException(status_code=400, detail="min_confidence must be <= max_confidence")
    
    try:
        results = await use_case.get_extractions_by_confidence(min_confidence, max_confidence, limit)
        responses = [_convert_to_response(result) for result in results]
        return responses
        
    except Exception as e:
        logger.error(f"Error getting extractions by confidence: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get extractions: {str(e)}")

@router.get("/search", response_model=List[OCRResultResponse])
async def search_by_text(
    q: str = Query(..., min_length=1),
    limit: int = Query(100, ge=1, le=1000),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Tìm kiếm OCR results theo text content
    
    - **q**: Text cần tìm kiếm
    - **limit**: Số lượng results tối đa
    """
    try:
        results = await use_case.search_by_text(q, limit)
        responses = [_convert_to_response(result) for result in results]
        return responses
        
    except Exception as e:
        logger.error(f"Error searching by text: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.post("/analyze-patterns", response_model=PatternAnalysisResponse)
async def analyze_extraction_patterns(
    files: List[UploadFile] = File(...),
    document_types: str = Form(...),  # JSON string
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Phân tích extraction patterns cho multiple images
    
    - **files**: List file ảnh để phân tích
    - **document_types**: JSON array các document types
    """
    # Validate files
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    # Parse document types
    try:
        doc_types_list = json.loads(document_types)
        if len(doc_types_list) != len(files):
            raise ValueError("Number of document types must match number of files")
        
        parsed_doc_types = [DocumentType(dt) for dt in doc_types_list]
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid document_types: {str(e)}")
    
    # Save uploaded files temporarily
    temp_paths = []
    try:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                contents = await file.read()
                temp_file.write(contents)
                temp_paths.append(temp_file.name)
        
        # Analyze patterns
        analysis = await use_case.analyze_extraction_patterns(temp_paths, parsed_doc_types)
        
        if "error" in analysis:
            raise HTTPException(status_code=500, detail=analysis["error"])
        
        response = PatternAnalysisResponse(**analysis)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")
    
    finally:
        # Clean up temporary files
        for path in temp_paths:
            if os.path.exists(path):
                os.unlink(path)

@router.get("/validate/{result_id}", response_model=OCRValidationResponse)
async def validate_extraction_quality(
    result_id: str,
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Validate chất lượng OCR extraction
    
    - **result_id**: ID của OCR result
    """
    try:
        validation = await use_case.validate_extraction_quality(result_id)
        
        if "error" in validation:
            raise HTTPException(status_code=404, detail=validation["error"])
        
        response = OCRValidationResponse(**validation)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating extraction quality: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/compare/{result_id1}/{result_id2}", response_model=OCRComparisonResponse)
async def compare_extractions(
    result_id1: str,
    result_id2: str,
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    So sánh 2 OCR extractions
    
    - **result_id1**: ID của result đầu tiên
    - **result_id2**: ID của result thứ hai
    """
    try:
        comparison = await use_case.compare_extractions(result_id1, result_id2)
        
        if "error" in comparison:
            raise HTTPException(status_code=404, detail=comparison["error"])
        
        response = OCRComparisonResponse(**comparison)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing extractions: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.get("/statistics", response_model=OCRExtractionStatisticsResponse)
async def get_ocr_statistics(
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy comprehensive OCR statistics
    """
    try:
        stats = await use_case.get_ocr_statistics()
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        
        # Extract extraction statistics for response
        extraction_stats = stats.get("extraction_statistics", {})
        response = OCRExtractionStatisticsResponse(**extraction_stats)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting OCR statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.get("/optimize/{document_type}")
async def optimize_for_document_type(
    document_type: DocumentType,
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Optimize OCR settings cho document type cụ thể
    
    - **document_type**: Loại document cần optimize
    """
    try:
        optimization = await use_case.optimize_for_document_type(document_type)
        
        if "error" in optimization:
            raise HTTPException(status_code=400, detail=optimization["error"])
        
        return JSONResponse(content=optimization)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error optimizing for document type: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get("/duplicates/{image_path:path}")
async def get_duplicate_extractions(
    image_path: str,
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Tìm duplicate OCR extractions cho cùng một image
    
    - **image_path**: Đường dẫn ảnh (encoded)
    """
    try:
        results = await use_case.get_duplicate_extractions(image_path)
        responses = [_convert_to_response(result) for result in results]
        
        return JSONResponse(content={
            "image_path": image_path,
            "duplicate_count": len(responses),
            "duplicates": [response.dict() for response in responses]
        })
        
    except Exception as e:
        logger.error(f"Error getting duplicate extractions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get duplicates: {str(e)}")

@router.get("/trends/accuracy")
async def get_accuracy_trends(
    days: int = Query(30, ge=1, le=365),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy accuracy trends
    
    - **days**: Số ngày để tính trends (1-365)
    """
    try:
        trends = await use_case.get_accuracy_trends(days)
        
        if "error" in trends:
            raise HTTPException(status_code=500, detail=trends["error"])
        
        return JSONResponse(content=trends)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting accuracy trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")

@router.get("/field-statistics")
async def get_field_extraction_statistics(
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Lấy field extraction statistics
    """
    try:
        stats = await use_case.get_field_extraction_statistics()
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        
        return JSONResponse(content=stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting field statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get field statistics: {str(e)}")

@router.delete("/cleanup")
async def cleanup_old_extractions(
    days_to_keep: int = Query(30, ge=1, le=365),
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Clean up old OCR extractions
    
    - **days_to_keep**: Số ngày data cần giữ lại (1-365)
    """
    try:
        cleanup_result = await use_case.cleanup_old_extractions(days_to_keep)
        
        return JSONResponse(content={
            "message": "Cleanup completed successfully",
            **cleanup_result
        })
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.delete("/result/{result_id}")
async def delete_ocr_result(
    result_id: str,
    use_case: OCRExtractionUseCase = Depends(get_ocr_use_case)
):
    """
    Xóa OCR result
    
    - **result_id**: ID của result cần xóa
    """
    try:
        success = await use_case.delete_ocr_result(result_id)
        
        if success:
            return JSONResponse(content={
                "message": "OCR result deleted successfully",
                "result_id": result_id,
                "success": True
            })
        else:
            raise HTTPException(status_code=404, detail="OCR result not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting OCR result: {e}")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        content={
            "status": "healthy",
            "service": "ocr-extraction",
            "version": "1.0.0",
            "supported_formats": ["jpg", "jpeg", "png", "bmp"],
            "supported_languages": ["vi", "en"],
            "supported_document_types": [dt.value for dt in DocumentType],
            "supported_field_types": [ft.value for ft in FieldType]
        }
    )

# Helper function
def _convert_to_response(ocr_result) -> OCRResultResponse:
    """Convert OCRResult to response model"""
    # Convert text regions
    text_regions = []
    for region in ocr_result.text_regions:
        bbox = BoundingBoxResponse(
            x1=region.bbox.x1,
            y1=region.bbox.y1,
            x2=region.bbox.x2,
            y2=region.bbox.y2
        )
        
        text_region = TextRegionResponse(
            text=region.text,
            confidence=region.confidence,
            bbox=bbox,
            field_type=region.field_type.value if region.field_type else None,
            language=region.language,
            font_size=region.font_size,
            is_handwritten=region.is_handwritten
        )
        text_regions.append(text_region)
    
    # Convert extracted fields
    extracted_fields = []
    for field in ocr_result.extracted_fields:
        bbox = BoundingBoxResponse(
            x1=field.bbox.x1,
            y1=field.bbox.y1,
            x2=field.bbox.x2,
            y2=field.bbox.y2
        )
        
        extracted_field = ExtractedFieldResponse(
            field_type=field.field_type.value,
            value=field.value,
            confidence=field.confidence,
            bbox=bbox,
            raw_text=field.raw_text,
            normalized_value=field.normalized_value,
            validation_status=field.validation_status,
            validation_errors=field.validation_errors
        )
        extracted_fields.append(extracted_field)
    
    # Convert statistics
    statistics = OCRStatisticsResponse(
        total_text_regions=ocr_result.statistics.total_text_regions,
        reliable_regions=ocr_result.statistics.reliable_regions,
        unreliable_regions=ocr_result.statistics.unreliable_regions,
        average_confidence=ocr_result.statistics.average_confidence,
        total_characters=ocr_result.statistics.total_characters,
        processing_time_ms=ocr_result.statistics.processing_time_ms,
        languages_detected=ocr_result.statistics.languages_detected,
        reliability_rate=ocr_result.statistics.get_reliability_rate()
    )
    
    return OCRResultResponse(
        id=ocr_result.id,
        image_path=ocr_result.image_path,
        document_type=ocr_result.document_type.value,
        status=ocr_result.status.value,
        text_regions=text_regions,
        extracted_fields=extracted_fields,
        full_text=ocr_result.full_text,
        statistics=statistics,
        processing_time_ms=ocr_result.processing_time_ms,
        model_version=ocr_result.model_version,
        languages_detected=ocr_result.languages_detected,
        confidence_threshold=ocr_result.confidence_threshold,
        preprocessing_applied=ocr_result.preprocessing_applied,
        overall_confidence=ocr_result.get_overall_confidence(),
        confidence_level=ocr_result.get_confidence_level().value,
        completion_rate=ocr_result.get_completion_rate(),
        is_successful=ocr_result.is_successful(),
        error_message=ocr_result.error_message,
        created_at=ocr_result.created_at.isoformat() if ocr_result.created_at else None
    )
