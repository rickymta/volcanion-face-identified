from typing import Optional
from domain.services.document_service import DocumentService
from domain.repositories.document_repository import DocumentRepository
from domain.entities.document import Document
import logging

class DocumentDetectionUseCase:
    def __init__(self, 
                 document_service: DocumentService,
                 document_repository: Optional[DocumentRepository] = None):
        self.document_service = document_service
        self.document_repository = document_repository
        self.logger = logging.getLogger(__name__)

    def execute(self, image_path: str, save_to_db: bool = True) -> Document:
        """
        Thực hiện phát hiện và phân loại giấy tờ
        """
        try:
            # Phát hiện và phân loại giấy tờ
            document = self.document_service.classify_document(image_path)
            
            # Kiểm tra tính hợp lệ
            is_valid = self.document_service.validate_document(document)
            self.logger.info(f"Document detected: {document.doc_type.value}, "
                           f"confidence: {document.confidence:.2f}, "
                           f"valid: {is_valid}")
            
            # Lưu vào database nếu cần
            if save_to_db and self.document_repository:
                try:
                    doc_id = self.document_repository.save(document)
                    self.logger.info(f"Document saved with ID: {doc_id}")
                except Exception as e:
                    self.logger.error(f"Failed to save document: {e}")
                    
            return document
            
        except Exception as e:
            self.logger.error(f"Error in document detection use case: {e}")
            raise
