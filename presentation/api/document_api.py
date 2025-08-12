import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from application.use_cases.document_detection_usecase import DocumentDetectionUseCase
from domain.services.document_service import DocumentService
from infrastructure.adapters.document_repository_impl import DocumentRepositoryImpl
from presentation.schemas.document_schemas import (
    DocumentDetectionRequest, 
    DocumentDetectionResponse, 
    DocumentListResponse,
    ErrorResponse,
    DocumentTypeEnum
)
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/document", tags=["Document Detection"])

# Dependency injection
def get_document_service():
    return DocumentService()

def get_document_repository():
    return DocumentRepositoryImpl()

def get_document_usecase(
    service: DocumentService = Depends(get_document_service),
    repository: DocumentRepositoryImpl = Depends(get_document_repository)
):
    return DocumentDetectionUseCase(service, repository)

@router.post('/detect', response_model=DocumentDetectionResponse)
async def detect_document(
    file: UploadFile = File(...),
    save_to_db: bool = True,
    usecase: DocumentDetectionUseCase = Depends(get_document_usecase)
):
    """
    Phát hiện và phân loại giấy tờ từ ảnh upload
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
        
        logger.info(f"Processing file: {file.filename}")
        
        # Thực hiện phát hiện giấy tờ
        document = usecase.execute(file_path, save_to_db)
        
        # Kiểm tra tính hợp lệ
        service = DocumentService()
        is_valid = service.validate_document(document)
        
        # Tạo response
        response = DocumentDetectionResponse(
            doc_type=DocumentTypeEnum(document.doc_type.value),
            bbox=document.bbox,
            confidence=document.confidence,
            is_valid=is_valid,
            message=f"Document detected as {document.doc_type.value} with confidence {document.confidence:.2f}"
        )
        
        logger.info(f"Detection completed: {document.doc_type.value}, confidence: {document.confidence:.2f}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        # Xóa file tạm
        if os.path.exists(file_path):
            os.remove(file_path)

@router.get('/list', response_model=DocumentListResponse)
async def list_documents(
    repository: DocumentRepositoryImpl = Depends(get_document_repository)
):
    """
    Lấy danh sách tất cả documents đã lưu
    """
    try:
        documents = repository.get_all()
        service = DocumentService()
        
        doc_responses = []
        for doc in documents:
            is_valid = service.validate_document(doc)
            doc_response = DocumentDetectionResponse(
                doc_type=DocumentTypeEnum(doc.doc_type.value),
                bbox=doc.bbox,
                confidence=doc.confidence,
                is_valid=is_valid,
                message=f"Document: {doc.doc_type.value}"
            )
            doc_responses.append(doc_response)
        
        return DocumentListResponse(
            documents=doc_responses,
            total=len(doc_responses)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@router.get('/health')
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "service": "document_detection"}

@router.delete('/{doc_id}')
async def delete_document(
    doc_id: str,
    repository: DocumentRepositoryImpl = Depends(get_document_repository)
):
    """
    Xóa document theo ID
    """
    try:
        success = repository.delete_by_id(doc_id)
        if success:
            return {"message": f"Document {doc_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
