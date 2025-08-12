from typing import List, Optional, Dict, Any
import logging
from domain.services.liveness_service import LivenessService
from domain.repositories.liveness_repository import LivenessRepository
from domain.entities.liveness_result import LivenessDetectionResult, LivenessStatus, LivenessResult
import cv2
import numpy as np
import os
from datetime import datetime, timedelta

# Initialize logger
logger = logging.getLogger(__name__)

class LivenessDetectionUseCase:
    """Use case cho liveness detection operations"""
    
    def __init__(self, liveness_service: LivenessService, liveness_repository: LivenessRepository):
        self.liveness_service = liveness_service
        self.liveness_repository = liveness_repository
    
    async def detect_and_save_liveness(self, image_path: str, face_bbox: Optional[List[int]] = None,
                                     use_advanced_analysis: bool = True) -> LivenessDetectionResult:
        """
        Detect liveness và save result
        
        Args:
            image_path: Đường dẫn đến image file
            face_bbox: Face bounding box [x1, y1, x2, y2]
            use_advanced_analysis: Có sử dụng advanced analysis không
            
        Returns:
            LivenessDetectionResult đã được save
        """
        try:
            # Validate input
            if not image_path or not os.path.exists(image_path):
                raise ValueError("Invalid image path")
            
            if face_bbox and len(face_bbox) != 4:
                raise ValueError("Face bbox must have 4 coordinates [x1, y1, x2, y2]")
            
            # Detect liveness
            result = self.liveness_service.detect_liveness_from_image(
                image_path, face_bbox, use_advanced_analysis
            )
            
            # Save to repository
            saved_result = await self.liveness_repository.save(result)
            
            logger.info(f"Liveness detection completed and saved: {saved_result.id}")
            return saved_result
            
        except Exception as e:
            logger.error(f"Error in detect and save liveness: {e}")
            raise e
    
    async def detect_liveness_from_array(self, image_array: np.ndarray, 
                                       face_bbox: Optional[List[int]] = None,
                                       use_advanced_analysis: bool = True,
                                       save_result: bool = True) -> LivenessDetectionResult:
        """
        Detect liveness từ numpy array
        
        Args:
            image_array: Image array
            face_bbox: Face bounding box
            use_advanced_analysis: Có sử dụng advanced analysis không
            save_result: Có save result không
            
        Returns:
            LivenessDetectionResult
        """
        try:
            # Validate input
            if image_array is None or image_array.size == 0:
                raise ValueError("Invalid image array")
            
            if face_bbox and len(face_bbox) != 4:
                raise ValueError("Face bbox must have 4 coordinates [x1, y1, x2, y2]")
            
            # Detect liveness
            result = self.liveness_service.detect_liveness_from_array(
                image_array, face_bbox, use_advanced_analysis
            )
            
            # Save if requested
            if save_result:
                result = await self.liveness_repository.save(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in detect liveness from array: {e}")
            raise e
    
    async def batch_detect_liveness(self, image_paths: List[str], 
                                  face_bboxes: Optional[List[List[int]]] = None,
                                  use_advanced_analysis: bool = True,
                                  save_results: bool = True) -> List[LivenessDetectionResult]:
        """
        Batch liveness detection
        
        Args:
            image_paths: List đường dẫn image files
            face_bboxes: List face bounding boxes
            use_advanced_analysis: Có sử dụng advanced analysis không
            save_results: Có save results không
            
        Returns:
            List of LivenessDetectionResult
        """
        try:
            if not image_paths:
                raise ValueError("Image paths cannot be empty")
            
            # Validate face_bboxes if provided
            if face_bboxes:
                if len(face_bboxes) != len(image_paths):
                    raise ValueError("Number of face_bboxes must match number of image_paths")
                
                for i, bbox in enumerate(face_bboxes):
                    if bbox and len(bbox) != 4:
                        raise ValueError(f"Face bbox at index {i} must have 4 coordinates")
            
            # Batch detect
            results = self.liveness_service.batch_detect_liveness(
                image_paths, face_bboxes, use_advanced_analysis
            )
            
            # Save results if requested
            if save_results:
                saved_results = []
                for result in results:
                    try:
                        saved_result = await self.liveness_repository.save(result)
                        saved_results.append(saved_result)
                    except Exception as e:
                        logger.error(f"Error saving result {result.id}: {e}")
                        saved_results.append(result)
                
                results = saved_results
            
            logger.info(f"Batch liveness detection completed for {len(image_paths)} images")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch detect liveness: {e}")
            raise e
    
    async def get_liveness_result(self, result_id: str) -> Optional[LivenessDetectionResult]:
        """
        Get liveness result by ID
        
        Args:
            result_id: Result ID
            
        Returns:
            LivenessDetectionResult hoặc None
        """
        try:
            if not result_id:
                raise ValueError("Result ID cannot be empty")
            
            result = await self.liveness_repository.find_by_id(result_id)
            return result
            
        except Exception as e:
            logger.error(f"Error getting liveness result {result_id}: {e}")
            return None
    
    async def get_liveness_results_by_image(self, image_path: str) -> List[LivenessDetectionResult]:
        """
        Get liveness results by image path
        
        Args:
            image_path: Image path
            
        Returns:
            List of LivenessDetectionResult
        """
        try:
            if not image_path:
                raise ValueError("Image path cannot be empty")
            
            results = await self.liveness_repository.find_by_image_path(image_path)
            return results
            
        except Exception as e:
            logger.error(f"Error getting liveness results for image {image_path}: {e}")
            return []
    
    async def get_recent_detections(self, hours: int = 24, limit: int = 100) -> List[LivenessDetectionResult]:
        """
        Get recent liveness detections
        
        Args:
            hours: Number of hours to look back
            limit: Maximum number of results
            
        Returns:
            List of recent LivenessDetectionResult
        """
        try:
            if hours <= 0:
                raise ValueError("Hours must be positive")
            
            if limit <= 0:
                raise ValueError("Limit must be positive")
            
            results = await self.liveness_repository.find_recent_results(hours, limit)
            return results
            
        except Exception as e:
            logger.error(f"Error getting recent detections: {e}")
            return []
    
    async def get_fake_detections(self, confidence_threshold: float = 0.8, 
                                limit: int = 100) -> List[LivenessDetectionResult]:
        """
        Get fake face detections
        
        Args:
            confidence_threshold: Minimum confidence threshold
            limit: Maximum number of results
            
        Returns:
            List of fake LivenessDetectionResult
        """
        try:
            if not (0.0 <= confidence_threshold <= 1.0):
                raise ValueError("Confidence threshold must be between 0 and 1")
            
            if limit <= 0:
                raise ValueError("Limit must be positive")
            
            results = await self.liveness_repository.find_fake_faces(confidence_threshold, limit)
            return results
            
        except Exception as e:
            logger.error(f"Error getting fake detections: {e}")
            return []
    
    async def get_spoof_attacks_by_type(self, spoof_type: str, limit: int = 100) -> List[LivenessDetectionResult]:
        """
        Get spoof attacks by type
        
        Args:
            spoof_type: Type of spoof attack
            limit: Maximum number of results
            
        Returns:
            List of LivenessDetectionResult
        """
        try:
            if not spoof_type:
                raise ValueError("Spoof type cannot be empty")
            
            if limit <= 0:
                raise ValueError("Limit must be positive")
            
            results = await self.liveness_repository.find_by_spoof_type(spoof_type, limit)
            return results
            
        except Exception as e:
            logger.error(f"Error getting spoof attacks by type {spoof_type}: {e}")
            return []
    
    async def analyze_liveness_patterns(self, image_paths: List[str],
                                      face_bboxes: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """
        Analyze liveness patterns for multiple images
        
        Args:
            image_paths: List image paths to analyze
            face_bboxes: List face bounding boxes
            
        Returns:
            Dict chứa pattern analysis
        """
        try:
            if not image_paths:
                raise ValueError("Image paths cannot be empty")
            
            # Detect liveness for all images
            results = await self.batch_detect_liveness(
                image_paths, face_bboxes, use_advanced_analysis=True, save_results=False
            )
            
            # Analyze patterns
            pattern_analysis = self.liveness_service.analyze_spoof_patterns(results)
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing liveness patterns: {e}")
            return {"error": str(e)}
    
    async def validate_detection_quality(self, result_id: str) -> Dict[str, Any]:
        """
        Validate quality of liveness detection
        
        Args:
            result_id: ID của liveness detection result
            
        Returns:
            Dict chứa validation results
        """
        try:
            # Get result
            result = await self.get_liveness_result(result_id)
            if not result:
                return {"error": "Liveness result not found"}
            
            # Validate quality
            validation = self.liveness_service.validate_liveness_quality(result)
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating detection quality: {e}")
            return {"error": str(e)}
    
    async def get_liveness_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive liveness detection statistics
        
        Returns:
            Dict chứa statistics
        """
        try:
            # Get basic statistics from repository
            basic_stats = await self.liveness_repository.get_statistics()
            
            # Get performance metrics
            performance_metrics = await self.liveness_repository.get_performance_metrics()
            
            # Get spoof attack trends
            spoof_trends = await self.liveness_repository.get_spoof_attack_trends()
            
            # Combine all statistics
            comprehensive_stats = {
                "basic_statistics": basic_stats,
                "performance_metrics": performance_metrics,
                "spoof_attack_trends": spoof_trends,
                "engine_info": self.liveness_service.get_engine_info(),
                "supported_formats": self.liveness_service.get_supported_formats()
            }
            
            return comprehensive_stats
            
        except Exception as e:
            logger.error(f"Error getting liveness statistics: {e}")
            return {"error": str(e)}
    
    async def optimize_detection_thresholds(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize detection thresholds based on training data
        
        Args:
            training_data: List of training samples with format:
                          [{"image_path": str, "face_bbox": List[int], "is_real": bool}, ...]
            
        Returns:
            Dict chứa optimization results
        """
        try:
            if not training_data or len(training_data) < 10:
                return {"error": "Insufficient training data (need at least 10 samples)"}
            
            # Process training data
            training_results = []
            
            for sample in training_data:
                image_path = sample.get("image_path")
                face_bbox = sample.get("face_bbox")
                is_real = sample.get("is_real")
                
                if not image_path or is_real is None:
                    continue
                
                # Detect liveness
                result = self.liveness_service.detect_liveness_from_image(
                    image_path, face_bbox, use_advanced_analysis=True
                )
                
                training_results.append((result, is_real))
            
            if len(training_results) < 10:
                return {"error": "Insufficient valid training samples"}
            
            # Optimize parameters
            optimization_result = self.liveness_service.optimize_detection_parameters(training_results)
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error optimizing detection thresholds: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_results(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old liveness detection results
        
        Args:
            days_to_keep: Number of days to keep results
            
        Returns:
            Dict chứa cleanup results
        """
        try:
            if days_to_keep <= 0:
                raise ValueError("Days to keep must be positive")
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Delete old results
            deleted_count = await self.liveness_repository.delete_old_results(days_to_keep)
            
            cleanup_result = {
                "deleted_count": deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "days_kept": days_to_keep,
                "success": True
            }
            
            logger.info(f"Cleanup completed: {deleted_count} old results deleted")
            return cleanup_result
            
        except Exception as e:
            logger.error(f"Error cleaning up old results: {e}")
            return {
                "deleted_count": 0,
                "success": False,
                "error": str(e)
            }
    
    async def compare_liveness_results(self, result_id1: str, result_id2: str) -> Dict[str, Any]:
        """
        Compare two liveness detection results
        
        Args:
            result_id1: First result ID
            result_id2: Second result ID
            
        Returns:
            Dict chứa comparison results
        """
        try:
            # Get both results
            result1 = await self.get_liveness_result(result_id1)
            result2 = await self.get_liveness_result(result_id2)
            
            if not result1:
                return {"error": f"Result {result_id1} not found"}
            
            if not result2:
                return {"error": f"Result {result_id2} not found"}
            
            # Compare results
            comparison = self.liveness_service.compare_liveness_results(result1, result2)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing liveness results: {e}")
            return {"error": str(e)}
    
    async def update_liveness_result(self, result_id: str, 
                                   updates: Dict[str, Any]) -> Optional[LivenessDetectionResult]:
        """
        Update liveness detection result
        
        Args:
            result_id: Result ID to update
            updates: Dict containing fields to update
            
        Returns:
            Updated LivenessDetectionResult hoặc None
        """
        try:
            # Get existing result
            result = await self.get_liveness_result(result_id)
            if not result:
                return None
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(result, field):
                    setattr(result, field, value)
            
            # Update timestamp
            result.updated_at = datetime.now()
            
            # Save updated result
            updated_result = await self.liveness_repository.update(result)
            
            logger.info(f"Liveness result {result_id} updated successfully")
            return updated_result
            
        except Exception as e:
            logger.error(f"Error updating liveness result {result_id}: {e}")
            return None
    
    async def delete_liveness_result(self, result_id: str) -> bool:
        """
        Delete liveness detection result
        
        Args:
            result_id: Result ID to delete
            
        Returns:
            bool indicating success
        """
        try:
            if not result_id:
                raise ValueError("Result ID cannot be empty")
            
            success = await self.liveness_repository.delete_by_id(result_id)
            
            if success:
                logger.info(f"Liveness result {result_id} deleted successfully")
            else:
                logger.warning(f"Liveness result {result_id} not found for deletion")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting liveness result {result_id}: {e}")
            return False
    
    # Validation methods
    def _validate_image_path(self, image_path: str) -> bool:
        """Validate image path"""
        if not image_path:
            return False
        
        if not os.path.exists(image_path):
            return False
        
        # Check supported formats
        supported_formats = self.liveness_service.get_supported_formats()
        _, ext = os.path.splitext(image_path.lower())
        return ext in supported_formats
    
    def _validate_face_bbox(self, face_bbox: List[int]) -> bool:
        """Validate face bounding box"""
        if not face_bbox or len(face_bbox) != 4:
            return False
        
        x1, y1, x2, y2 = face_bbox
        
        # Check coordinates are non-negative and valid
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            return False
        
        return True
