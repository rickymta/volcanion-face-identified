from domain.entities.document import Document, DocumentType
import logging

class DocumentService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def classify_document(self, image_path: str) -> Document:
        """Phân loại và phát hiện giấy tờ"""
        try:
            from infrastructure.ml_models.document_detector import DocumentDetector
            detector = DocumentDetector()
            
            # Phát hiện vùng giấy tờ
            bbox = detector.detect(image_path)
            
            # Phân loại loại giấy tờ
            doc_type_str = detector.classify(image_path, bbox)
            try:
                doc_type = DocumentType(doc_type_str)
            except ValueError:
                doc_type = DocumentType.UNKNOWN
                
            # Tính confidence
            confidence = detector.get_document_confidence(image_path, bbox)
            
            return Document(
                image_path=image_path, 
                doc_type=doc_type, 
                bbox=bbox,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return Document(
                image_path=image_path, 
                doc_type=DocumentType.UNKNOWN, 
                bbox=None,
                confidence=0.0
            )
    
    def validate_document(self, document: Document) -> bool:
        """Kiểm tra tính hợp lệ của document"""
        if document.confidence < 0.3:
            return False
        if document.bbox is None:
            return False
        if document.doc_type == DocumentType.UNKNOWN:
            return False
        return True
