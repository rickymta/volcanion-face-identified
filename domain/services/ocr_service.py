import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid

from domain.entities.ocr_result import (
    OCRResult, TextRegion, ExtractedField, OCRStatistics,
    DocumentType, OCRStatus, FieldType
)
from infrastructure.ml.ocr_engine import OCREngine, OCRConfig

logger = logging.getLogger(__name__)

class OCRService:
    """Domain service cho OCR operations"""
    
    def __init__(self):
        self.config = OCRConfig()
        self.engine = OCREngine(self.config)
    
    def extract_text_from_image(
        self, 
        image_path: str, 
        document_type: DocumentType = DocumentType.OTHER,
        confidence_threshold: float = 0.5,
        use_preprocessing: bool = True
    ) -> OCRResult:
        """
        Extract text từ ảnh document
        
        Args:
            image_path: Đường dẫn ảnh
            document_type: Loại document
            confidence_threshold: Ngưỡng confidence
            use_preprocessing: Có sử dụng preprocessing không
            
        Returns:
            OCRResult: Kết quả OCR
        """
        start_time = datetime.now()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image from {image_path}")
            
            # Update config
            self.config.confidence_threshold = confidence_threshold
            self.config.preprocess_image = use_preprocessing
            
            # Extract text regions
            text_regions = self.engine.extract_text(image)
            
            # Extract structured fields
            extracted_fields = self.engine.extract_fields(text_regions)
            
            # Generate full text
            full_text = self._generate_full_text(text_regions)
            
            # Calculate statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            statistics = self.engine.calculate_statistics(text_regions)
            statistics.processing_time_ms = processing_time
            
            # Determine languages
            languages_detected = list(set([
                region.language for region in text_regions 
                if region.language and region.language != 'auto'
            ]))
            if not languages_detected:
                languages_detected = ['auto']
            
            # Determine status
            status = self._determine_status(text_regions, extracted_fields)
            
            # Create result
            result = OCRResult(
                id=f"ocr_{uuid.uuid4().hex[:12]}",
                image_path=image_path,
                document_type=document_type,
                status=status,
                text_regions=text_regions,
                extracted_fields=extracted_fields,
                full_text=full_text,
                statistics=statistics,
                processing_time_ms=processing_time,
                model_version="v1.0",
                languages_detected=languages_detected,
                confidence_threshold=confidence_threshold,
                preprocessing_applied=self._get_preprocessing_steps(use_preprocessing),
                created_at=datetime.now()
            )
            
            logger.info(f"OCR completed for {image_path}: {len(text_regions)} regions, {len(extracted_fields)} fields")
            return result
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            return self._create_error_result(image_path, document_type, str(e))
    
    def batch_extract_text(
        self, 
        image_paths: List[str], 
        document_types: List[DocumentType] = None,
        confidence_threshold: float = 0.5,
        use_preprocessing: bool = True
    ) -> List[OCRResult]:
        """
        Batch OCR extraction
        
        Args:
            image_paths: Danh sách đường dẫn ảnh
            document_types: Danh sách loại document tương ứng
            confidence_threshold: Ngưỡng confidence
            use_preprocessing: Có sử dụng preprocessing không
            
        Returns:
            List[OCRResult]: Danh sách kết quả OCR
        """
        if document_types is None:
            document_types = [DocumentType.OTHER] * len(image_paths)
        
        if len(document_types) != len(image_paths):
            raise ValueError("Number of document types must match number of image paths")
        
        results = []
        for image_path, doc_type in zip(image_paths, document_types):
            try:
                result = self.extract_text_from_image(
                    image_path, doc_type, confidence_threshold, use_preprocessing
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                error_result = self._create_error_result(image_path, doc_type, str(e))
                results.append(error_result)
        
        return results
    
    def extract_specific_fields(
        self, 
        image_path: str, 
        target_fields: List[FieldType],
        document_type: DocumentType = DocumentType.OTHER
    ) -> Dict[FieldType, ExtractedField]:
        """
        Extract specific fields từ document
        
        Args:
            image_path: Đường dẫn ảnh
            target_fields: Danh sách fields cần extract
            document_type: Loại document
            
        Returns:
            Dict[FieldType, ExtractedField]: Mapping field type -> extracted field
        """
        ocr_result = self.extract_text_from_image(image_path, document_type)
        
        field_mapping = {}
        for field in ocr_result.extracted_fields:
            if field.field_type in target_fields:
                field_mapping[field.field_type] = field
        
        return field_mapping
    
    def validate_extraction_quality(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """
        Validate chất lượng OCR result
        
        Args:
            ocr_result: OCR result cần validate
            
        Returns:
            Dict[str, Any]: Validation report
        """
        issues = []
        recommendations = []
        
        # Check overall confidence
        overall_confidence = ocr_result.get_overall_confidence()
        if overall_confidence < 0.7:
            issues.append(f"Low overall confidence: {overall_confidence:.2f}")
            recommendations.append("Consider image preprocessing or higher resolution")
        
        # Check field extraction completeness
        expected_fields = self._get_expected_fields(ocr_result.document_type)
        missing_fields = []
        for field_type in expected_fields:
            if not ocr_result.has_field_type(field_type):
                missing_fields.append(field_type.value)
        
        if missing_fields:
            issues.append(f"Missing expected fields: {', '.join(missing_fields)}")
            recommendations.append("Check image quality and field positioning")
        
        # Check invalid fields
        invalid_fields = ocr_result.get_invalid_fields()
        if invalid_fields:
            field_names = [f.field_type.value for f in invalid_fields]
            issues.append(f"Invalid fields detected: {', '.join(field_names)}")
            recommendations.append("Review field validation rules")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(ocr_result)
        
        return {
            "is_valid": len(issues) == 0,
            "quality_score": quality_score,
            "issues": issues,
            "recommendations": recommendations,
            "statistics": {
                "overall_confidence": overall_confidence,
                "valid_fields": len(ocr_result.get_valid_fields()),
                "invalid_fields": len(invalid_fields),
                "missing_fields": len(missing_fields),
                "completion_rate": ocr_result.get_completion_rate()
            }
        }
    
    def compare_extractions(self, result1: OCRResult, result2: OCRResult) -> Dict[str, Any]:
        """
        So sánh 2 OCR results
        
        Args:
            result1: OCR result đầu tiên
            result2: OCR result thứ hai
            
        Returns:
            Dict[str, Any]: Comparison report
        """
        # Compare overall metrics
        confidence_diff = abs(result1.get_overall_confidence() - result2.get_overall_confidence())
        field_count_diff = abs(len(result1.extracted_fields) - len(result2.extracted_fields))
        
        # Compare fields
        field_matches = 0
        field_differences = []
        
        for field1 in result1.extracted_fields:
            field2 = result2.get_field_by_type(field1.field_type)
            if field2:
                if field1.normalized_value == field2.normalized_value:
                    field_matches += 1
                else:
                    field_differences.append({
                        "field_type": field1.field_type.value,
                        "value1": field1.normalized_value,
                        "value2": field2.normalized_value,
                        "confidence1": field1.confidence,
                        "confidence2": field2.confidence
                    })
        
        # Calculate similarity
        total_fields = max(len(result1.extracted_fields), len(result2.extracted_fields))
        similarity = field_matches / total_fields if total_fields > 0 else 0.0
        
        return {
            "similarity": similarity,
            "confidence_difference": confidence_diff,
            "field_count_difference": field_count_diff,
            "field_matches": field_matches,
            "field_differences": field_differences,
            "consistency": "High" if similarity > 0.8 else "Medium" if similarity > 0.5 else "Low",
            "result1_summary": result1.get_extraction_summary(),
            "result2_summary": result2.get_extraction_summary()
        }
    
    def get_extraction_statistics(self, results: List[OCRResult]) -> Dict[str, Any]:
        """
        Tính statistics cho danh sách OCR results
        
        Args:
            results: Danh sách OCR results
            
        Returns:
            Dict[str, Any]: Statistics report
        """
        if not results:
            return {"error": "No results provided"}
        
        # Basic statistics
        total_results = len(results)
        successful_results = len([r for r in results if r.is_successful()])
        failed_results = total_results - successful_results
        
        # Confidence statistics
        confidences = [r.get_overall_confidence() for r in results]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        min_confidence = np.min(confidences) if confidences else 0.0
        max_confidence = np.max(confidences) if confidences else 0.0
        
        # Processing time statistics
        processing_times = [r.processing_time_ms for r in results]
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        # Document type distribution
        doc_type_counts = {}
        for result in results:
            doc_type = result.document_type.value
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
        
        # Field extraction statistics
        field_stats = self._calculate_field_statistics(results)
        
        return {
            "basic_statistics": {
                "total_results": total_results,
                "successful_results": successful_results,
                "failed_results": failed_results,
                "success_rate": successful_results / total_results if total_results > 0 else 0.0
            },
            "confidence_statistics": {
                "average_confidence": avg_confidence,
                "min_confidence": min_confidence,
                "max_confidence": max_confidence,
                "std_confidence": np.std(confidences) if confidences else 0.0
            },
            "performance_statistics": {
                "average_processing_time_ms": avg_processing_time,
                "min_processing_time_ms": np.min(processing_times) if processing_times else 0.0,
                "max_processing_time_ms": np.max(processing_times) if processing_times else 0.0
            },
            "document_type_distribution": doc_type_counts,
            "field_extraction_statistics": field_stats,
            "languages_detected": list(set([
                lang for result in results 
                for lang in result.languages_detected
            ]))
        }
    
    def optimize_for_document_type(
        self, 
        document_type: DocumentType,
        sample_results: List[OCRResult]
    ) -> Dict[str, Any]:
        """
        Optimize OCR settings cho document type cụ thể
        
        Args:
            document_type: Loại document
            sample_results: Sample results để optimize
            
        Returns:
            Dict[str, Any]: Optimization recommendations
        """
        if not sample_results:
            return {"error": "No sample results provided"}
        
        # Analyze performance by confidence threshold
        thresholds = [0.3, 0.5, 0.7, 0.9]
        threshold_performance = {}
        
        for threshold in thresholds:
            valid_results = []
            for result in sample_results:
                # Simulate filtering with this threshold
                filtered_fields = [
                    f for f in result.extracted_fields 
                    if f.confidence >= threshold
                ]
                if filtered_fields:
                    valid_results.append(len(filtered_fields))
            
            if valid_results:
                threshold_performance[threshold] = {
                    "avg_fields": np.mean(valid_results),
                    "total_results": len(valid_results)
                }
        
        # Find optimal threshold
        optimal_threshold = 0.5
        best_score = 0
        for threshold, performance in threshold_performance.items():
            score = performance["avg_fields"] * performance["total_results"]
            if score > best_score:
                best_score = score
                optimal_threshold = threshold
        
        # Document-specific recommendations
        recommendations = self._get_document_specific_recommendations(document_type, sample_results)
        
        return {
            "optimal_threshold": optimal_threshold,
            "threshold_performance": threshold_performance,
            "document_type": document_type.value,
            "recommendations": recommendations,
            "sample_size": len(sample_results)
        }
    
    def _generate_full_text(self, text_regions: List[TextRegion]) -> str:
        """Generate full text từ text regions"""
        # Sort regions by position (top to bottom, left to right)
        sorted_regions = sorted(text_regions, key=lambda r: (r.bbox.y1, r.bbox.x1))
        return " ".join([region.text for region in sorted_regions])
    
    def _determine_status(self, text_regions: List[TextRegion], extracted_fields: List[ExtractedField]) -> OCRStatus:
        """Determine OCR status"""
        if not text_regions:
            return OCRStatus.FAILED
        
        reliable_regions = [r for r in text_regions if r.is_reliable()]
        if len(reliable_regions) == 0:
            return OCRStatus.FAILED
        elif len(reliable_regions) < len(text_regions) * 0.5:
            return OCRStatus.PARTIAL
        else:
            return OCRStatus.COMPLETED
    
    def _get_preprocessing_steps(self, use_preprocessing: bool) -> List[str]:
        """Get list of preprocessing steps applied"""
        if use_preprocessing:
            return ["enhance_image", "correct_skew", "remove_noise", "binarize"]
        else:
            return []
    
    def _create_error_result(self, image_path: str, document_type: DocumentType, error_message: str) -> OCRResult:
        """Create error OCR result"""
        return OCRResult(
            id=f"ocr_error_{uuid.uuid4().hex[:12]}",
            image_path=image_path,
            document_type=document_type,
            status=OCRStatus.FAILED,
            text_regions=[],
            extracted_fields=[],
            full_text="",
            statistics=OCRStatistics(
                total_text_regions=0,
                reliable_regions=0,
                unreliable_regions=0,
                average_confidence=0.0,
                total_characters=0,
                processing_time_ms=0.0,
                languages_detected=[]
            ),
            processing_time_ms=0.0,
            model_version="v1.0",
            languages_detected=[],
            confidence_threshold=0.5,
            preprocessing_applied=[],
            error_message=error_message,
            created_at=datetime.now()
        )
    
    def _get_expected_fields(self, document_type: DocumentType) -> List[FieldType]:
        """Get expected fields cho document type"""
        field_mapping = {
            DocumentType.CMND: [
                FieldType.ID_NUMBER, FieldType.FULL_NAME, FieldType.DATE_OF_BIRTH,
                FieldType.PLACE_OF_BIRTH, FieldType.ADDRESS, FieldType.GENDER
            ],
            DocumentType.CCCD: [
                FieldType.ID_NUMBER, FieldType.FULL_NAME, FieldType.DATE_OF_BIRTH,
                FieldType.PLACE_OF_BIRTH, FieldType.ADDRESS, FieldType.GENDER,
                FieldType.ISSUE_DATE, FieldType.EXPIRY_DATE
            ],
            DocumentType.PASSPORT: [
                FieldType.PASSPORT_NUMBER, FieldType.FULL_NAME, FieldType.DATE_OF_BIRTH,
                FieldType.NATIONALITY, FieldType.GENDER, FieldType.ISSUE_DATE, FieldType.EXPIRY_DATE
            ],
            DocumentType.DRIVER_LICENSE: [
                FieldType.ID_NUMBER, FieldType.FULL_NAME, FieldType.DATE_OF_BIRTH,
                FieldType.ADDRESS, FieldType.ISSUE_DATE, FieldType.EXPIRY_DATE
            ]
        }
        
        return field_mapping.get(document_type, [])
    
    def _calculate_quality_score(self, ocr_result: OCRResult) -> float:
        """Calculate overall quality score"""
        # Factors: confidence, completion rate, field validity
        confidence_score = ocr_result.get_overall_confidence()
        completion_score = ocr_result.get_completion_rate()
        
        valid_fields = len(ocr_result.get_valid_fields())
        total_fields = len(ocr_result.extracted_fields)
        validity_score = valid_fields / total_fields if total_fields > 0 else 0.0
        
        # Weighted average
        return (confidence_score * 0.4 + completion_score * 0.3 + validity_score * 0.3)
    
    def _calculate_field_statistics(self, results: List[OCRResult]) -> Dict[str, Any]:
        """Calculate field extraction statistics"""
        field_counts = {}
        field_confidences = {}
        
        for result in results:
            for field in result.extracted_fields:
                field_type = field.field_type.value
                
                # Count occurrences
                field_counts[field_type] = field_counts.get(field_type, 0) + 1
                
                # Track confidences
                if field_type not in field_confidences:
                    field_confidences[field_type] = []
                field_confidences[field_type].append(field.confidence)
        
        # Calculate average confidences
        field_avg_confidences = {}
        for field_type, confidences in field_confidences.items():
            field_avg_confidences[field_type] = np.mean(confidences)
        
        return {
            "field_counts": field_counts,
            "field_average_confidences": field_avg_confidences,
            "most_extracted_field": max(field_counts.items(), key=lambda x: x[1])[0] if field_counts else None,
            "highest_confidence_field": max(field_avg_confidences.items(), key=lambda x: x[1])[0] if field_avg_confidences else None
        }
    
    def _get_document_specific_recommendations(
        self, 
        document_type: DocumentType, 
        sample_results: List[OCRResult]
    ) -> List[str]:
        """Get recommendations for specific document type"""
        recommendations = []
        
        # Analyze common issues
        avg_confidence = np.mean([r.get_overall_confidence() for r in sample_results])
        
        if avg_confidence < 0.7:
            recommendations.append("Consider using higher resolution images")
            recommendations.append("Apply more aggressive image preprocessing")
        
        if document_type in [DocumentType.CMND, DocumentType.CCCD]:
            recommendations.append("Ensure proper lighting to avoid shadows on text")
            recommendations.append("Position document flat to minimize perspective distortion")
        
        elif document_type == DocumentType.PASSPORT:
            recommendations.append("Handle machine-readable zone (MRZ) separately")
            recommendations.append("Account for security features that may interfere with text")
        
        return recommendations
