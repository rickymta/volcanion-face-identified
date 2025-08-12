from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from domain.entities.ocr_result import OCRResult, DocumentType, OCRStatus, FieldType
from domain.repositories.ocr_repository import OCRRepository
from domain.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

class OCRExtractionUseCase:
    """Use case cho OCR text extraction"""
    
    def __init__(self, ocr_service: OCRService, ocr_repository: OCRRepository):
        self.ocr_service = ocr_service
        self.ocr_repository = ocr_repository
    
    async def extract_and_save_text(
        self, 
        image_path: str, 
        document_type: DocumentType = DocumentType.OTHER,
        confidence_threshold: float = 0.5,
        use_preprocessing: bool = True
    ) -> OCRResult:
        """
        Extract text và save result
        
        Args:
            image_path: Đường dẫn ảnh
            document_type: Loại document
            confidence_threshold: Ngưỡng confidence
            use_preprocessing: Có sử dụng preprocessing không
            
        Returns:
            OCRResult: Kết quả OCR đã được lưu
        """
        try:
            # Extract text
            ocr_result = self.ocr_service.extract_text_from_image(
                image_path, document_type, confidence_threshold, use_preprocessing
            )
            
            # Save to repository
            saved_id = await self.ocr_repository.save(ocr_result)
            ocr_result.id = saved_id
            
            logger.info(f"OCR extraction completed and saved: {saved_id}")
            return ocr_result
            
        except Exception as e:
            logger.error(f"Error in extract and save text: {e}")
            raise
    
    async def batch_extract_and_save_text(
        self, 
        image_paths: List[str], 
        document_types: List[DocumentType] = None,
        confidence_threshold: float = 0.5,
        use_preprocessing: bool = True
    ) -> List[OCRResult]:
        """
        Batch extract text và save results
        
        Args:
            image_paths: Danh sách đường dẫn ảnh
            document_types: Danh sách loại document
            confidence_threshold: Ngưỡng confidence
            use_preprocessing: Có sử dụng preprocessing không
            
        Returns:
            List[OCRResult]: Danh sách kết quả OCR đã được lưu
        """
        try:
            # Batch extract
            ocr_results = self.ocr_service.batch_extract_text(
                image_paths, document_types, confidence_threshold, use_preprocessing
            )
            
            # Save all results
            saved_results = []
            for result in ocr_results:
                try:
                    saved_id = await self.ocr_repository.save(result)
                    result.id = saved_id
                    saved_results.append(result)
                except Exception as e:
                    logger.error(f"Error saving OCR result: {e}")
                    saved_results.append(result)  # Include even if save failed
            
            logger.info(f"Batch OCR extraction completed: {len(saved_results)} results")
            return saved_results
            
        except Exception as e:
            logger.error(f"Error in batch extract and save text: {e}")
            raise
    
    async def get_ocr_result(self, ocr_id: str) -> Optional[OCRResult]:
        """
        Lấy OCR result theo ID
        
        Args:
            ocr_id: ID của OCR result
            
        Returns:
            Optional[OCRResult]: OCR result nếu tìm thấy
        """
        try:
            return await self.ocr_repository.find_by_id(ocr_id)
        except Exception as e:
            logger.error(f"Error getting OCR result {ocr_id}: {e}")
            return None
    
    async def get_recent_extractions(self, hours: int = 24, limit: int = 100) -> List[OCRResult]:
        """
        Lấy OCR extractions gần đây
        
        Args:
            hours: Số giờ look back
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        try:
            return await self.ocr_repository.get_recent_results(hours, limit)
        except Exception as e:
            logger.error(f"Error getting recent extractions: {e}")
            return []
    
    async def get_successful_extractions(self, limit: int = 100) -> List[OCRResult]:
        """
        Lấy successful OCR extractions
        
        Args:
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách successful OCR results
        """
        try:
            return await self.ocr_repository.get_successful_results(limit)
        except Exception as e:
            logger.error(f"Error getting successful extractions: {e}")
            return []
    
    async def get_failed_extractions(self, limit: int = 100) -> List[OCRResult]:
        """
        Lấy failed OCR extractions
        
        Args:
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách failed OCR results
        """
        try:
            return await self.ocr_repository.get_failed_results(limit)
        except Exception as e:
            logger.error(f"Error getting failed extractions: {e}")
            return []
    
    async def get_extractions_by_document_type(
        self, 
        document_type: DocumentType, 
        limit: int = 100
    ) -> List[OCRResult]:
        """
        Lấy OCR extractions theo document type
        
        Args:
            document_type: Loại document
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        try:
            return await self.ocr_repository.find_by_document_type(document_type, limit)
        except Exception as e:
            logger.error(f"Error getting extractions by document type: {e}")
            return []
    
    async def search_by_text(self, search_text: str, limit: int = 100) -> List[OCRResult]:
        """
        Tìm kiếm OCR results theo text content
        
        Args:
            search_text: Text cần tìm
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        try:
            return await self.ocr_repository.search_by_text(search_text, limit)
        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return []
    
    async def get_extractions_by_field_type(
        self, 
        field_type: FieldType, 
        limit: int = 100
    ) -> List[OCRResult]:
        """
        Lấy OCR extractions có chứa field type cụ thể
        
        Args:
            field_type: Loại field cần tìm
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        try:
            return await self.ocr_repository.find_by_field_type(field_type, limit)
        except Exception as e:
            logger.error(f"Error getting extractions by field type: {e}")
            return []
    
    async def get_extractions_by_confidence(
        self, 
        min_confidence: float, 
        max_confidence: float = 1.0, 
        limit: int = 100
    ) -> List[OCRResult]:
        """
        Lấy OCR extractions theo confidence range
        
        Args:
            min_confidence: Confidence tối thiểu
            max_confidence: Confidence tối đa
            limit: Số lượng kết quả tối đa
            
        Returns:
            List[OCRResult]: Danh sách OCR results
        """
        try:
            return await self.ocr_repository.find_by_confidence_range(
                min_confidence, max_confidence, limit
            )
        except Exception as e:
            logger.error(f"Error getting extractions by confidence: {e}")
            return []
    
    async def analyze_extraction_patterns(
        self, 
        image_paths: List[str], 
        document_types: List[DocumentType] = None
    ) -> Dict[str, Any]:
        """
        Phân tích extraction patterns cho multiple images
        
        Args:
            image_paths: Danh sách đường dẫn ảnh
            document_types: Danh sách loại document
            
        Returns:
            Dict[str, Any]: Analysis report
        """
        try:
            # Batch extract
            results = self.ocr_service.batch_extract_text(image_paths, document_types)
            
            # Save results
            for result in results:
                try:
                    await self.ocr_repository.save(result)
                except Exception as e:
                    logger.error(f"Error saving analysis result: {e}")
            
            # Generate analysis
            analysis = self.ocr_service.get_extraction_statistics(results)
            
            # Add pattern-specific insights
            analysis["pattern_insights"] = self._generate_pattern_insights(results)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {"error": str(e)}
    
    async def validate_extraction_quality(self, ocr_id: str) -> Dict[str, Any]:
        """
        Validate chất lượng OCR extraction
        
        Args:
            ocr_id: ID của OCR result
            
        Returns:
            Dict[str, Any]: Validation report
        """
        try:
            ocr_result = await self.ocr_repository.find_by_id(ocr_id)
            if not ocr_result:
                return {"error": "OCR result not found"}
            
            validation = self.ocr_service.validate_extraction_quality(ocr_result)
            return validation
            
        except Exception as e:
            logger.error(f"Error validating extraction quality: {e}")
            return {"error": str(e)}
    
    async def compare_extractions(self, ocr_id1: str, ocr_id2: str) -> Dict[str, Any]:
        """
        So sánh 2 OCR extractions
        
        Args:
            ocr_id1: ID của OCR result đầu tiên
            ocr_id2: ID của OCR result thứ hai
            
        Returns:
            Dict[str, Any]: Comparison report
        """
        try:
            result1 = await self.ocr_repository.find_by_id(ocr_id1)
            result2 = await self.ocr_repository.find_by_id(ocr_id2)
            
            if not result1:
                return {"error": f"OCR result {ocr_id1} not found"}
            if not result2:
                return {"error": f"OCR result {ocr_id2} not found"}
            
            comparison = self.ocr_service.compare_extractions(result1, result2)
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing extractions: {e}")
            return {"error": str(e)}
    
    async def get_ocr_statistics(self) -> Dict[str, Any]:
        """
        Lấy comprehensive OCR statistics
        
        Returns:
            Dict[str, Any]: Statistics report
        """
        try:
            # Get repository statistics
            repo_stats = await self.ocr_repository.get_statistics()
            
            # Get recent results for analysis
            recent_results = await self.ocr_repository.get_recent_results(hours=24, limit=1000)
            
            # Generate extraction statistics
            extraction_stats = self.ocr_service.get_extraction_statistics(recent_results)
            
            # Combine statistics
            combined_stats = {
                "repository_statistics": repo_stats,
                "extraction_statistics": extraction_stats,
                "engine_info": {
                    "version": "v1.0",
                    "supported_languages": ["vi", "en"],
                    "supported_document_types": [dt.value for dt in DocumentType],
                    "supported_field_types": [ft.value for ft in FieldType]
                },
                "performance_summary": self._generate_performance_summary(recent_results)
            }
            
            return combined_stats
            
        except Exception as e:
            logger.error(f"Error getting OCR statistics: {e}")
            return {"error": str(e)}
    
    async def optimize_for_document_type(
        self, 
        document_type: DocumentType,
        sample_image_paths: List[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize OCR settings cho document type cụ thể
        
        Args:
            document_type: Loại document cần optimize
            sample_image_paths: Sample images để test (optional)
            
        Returns:
            Dict[str, Any]: Optimization recommendations
        """
        try:
            # Get existing results for this document type
            existing_results = await self.ocr_repository.find_by_document_type(document_type, 100)
            
            # If sample images provided, extract them too
            if sample_image_paths:
                sample_results = self.ocr_service.batch_extract_text(
                    sample_image_paths, 
                    [document_type] * len(sample_image_paths)
                )
                existing_results.extend(sample_results)
            
            if not existing_results:
                return {"error": "No sample data available for optimization"}
            
            # Generate optimization recommendations
            optimization = self.ocr_service.optimize_for_document_type(document_type, existing_results)
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing for document type: {e}")
            return {"error": str(e)}
    
    async def get_duplicate_extractions(self, image_path: str) -> List[OCRResult]:
        """
        Tìm duplicate OCR extractions cho cùng một image
        
        Args:
            image_path: Đường dẫn ảnh
            
        Returns:
            List[OCRResult]: Danh sách duplicate results
        """
        try:
            return await self.ocr_repository.get_duplicate_results(image_path)
        except Exception as e:
            logger.error(f"Error getting duplicate extractions: {e}")
            return []
    
    async def get_accuracy_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        Lấy accuracy trends
        
        Args:
            days: Số ngày để tính trends
            
        Returns:
            Dict[str, Any]: Accuracy trends data
        """
        try:
            return await self.ocr_repository.get_accuracy_trends(days)
        except Exception as e:
            logger.error(f"Error getting accuracy trends: {e}")
            return {"error": str(e)}
    
    async def get_field_extraction_statistics(self) -> Dict[str, Any]:
        """
        Lấy field extraction statistics
        
        Returns:
            Dict[str, Any]: Field extraction statistics
        """
        try:
            return await self.ocr_repository.get_field_extraction_stats()
        except Exception as e:
            logger.error(f"Error getting field extraction statistics: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_extractions(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old OCR extractions
        
        Args:
            days_to_keep: Số ngày data cần giữ lại
            
        Returns:
            Dict[str, Any]: Cleanup result
        """
        try:
            return await self.ocr_repository.delete_old_results(days_to_keep)
        except Exception as e:
            logger.error(f"Error cleaning up old extractions: {e}")
            return {"error": str(e)}
    
    async def delete_ocr_result(self, ocr_id: str) -> bool:
        """
        Xóa OCR result
        
        Args:
            ocr_id: ID của OCR result cần xóa
            
        Returns:
            bool: True nếu xóa thành công
        """
        try:
            return await self.ocr_repository.delete_by_id(ocr_id)
        except Exception as e:
            logger.error(f"Error deleting OCR result {ocr_id}: {e}")
            return False
    
    async def update_extraction_status(
        self, 
        ocr_id: str, 
        status: OCRStatus, 
        error_message: str = None
    ) -> bool:
        """
        Cập nhật status của OCR extraction
        
        Args:
            ocr_id: ID của OCR result
            status: Status mới
            error_message: Error message (nếu có)
            
        Returns:
            bool: True nếu update thành công
        """
        try:
            return await self.ocr_repository.update_status(ocr_id, status, error_message)
        except Exception as e:
            logger.error(f"Error updating extraction status: {e}")
            return False
    
    def _generate_pattern_insights(self, results: List[OCRResult]) -> Dict[str, Any]:
        """Generate insights về extraction patterns"""
        insights = []
        
        # Analyze success rate by document type
        doc_type_success = {}
        for result in results:
            doc_type = result.document_type.value
            if doc_type not in doc_type_success:
                doc_type_success[doc_type] = {"total": 0, "successful": 0}
            
            doc_type_success[doc_type]["total"] += 1
            if result.is_successful():
                doc_type_success[doc_type]["successful"] += 1
        
        # Find best/worst performing document types
        for doc_type, stats in doc_type_success.items():
            success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
            if success_rate < 0.5:
                insights.append(f"Low success rate for {doc_type}: {success_rate:.2f}")
            elif success_rate > 0.9:
                insights.append(f"High success rate for {doc_type}: {success_rate:.2f}")
        
        # Analyze common failure patterns
        failed_results = [r for r in results if not r.is_successful()]
        if len(failed_results) > len(results) * 0.3:
            insights.append("High failure rate detected - consider image quality improvements")
        
        return {
            "insights": insights,
            "document_type_performance": doc_type_success,
            "total_analyzed": len(results),
            "success_rate": len([r for r in results if r.is_successful()]) / len(results) if results else 0
        }
    
    def _generate_performance_summary(self, results: List[OCRResult]) -> Dict[str, Any]:
        """Generate performance summary"""
        if not results:
            return {"message": "No recent data available"}
        
        processing_times = [r.processing_time_ms for r in results]
        confidences = [r.get_overall_confidence() for r in results]
        
        return {
            "total_extractions": len(results),
            "avg_processing_time_ms": sum(processing_times) / len(processing_times),
            "avg_confidence": sum(confidences) / len(confidences),
            "success_rate": len([r for r in results if r.is_successful()]) / len(results),
            "most_common_document_type": max(
                set([r.document_type.value for r in results]),
                key=[r.document_type.value for r in results].count
            ) if results else None
        }
