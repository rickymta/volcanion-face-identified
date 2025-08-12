import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging
from domain.ml.liveness_engine import LivenessEngine
from domain.entities.liveness_result import LivenessDetectionResult, SpoofType
import os

# Initialize logger
logger = logging.getLogger(__name__)

class LivenessService:
    """Domain service cho liveness detection"""
    
    def __init__(self):
        self.liveness_engine = LivenessEngine()
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def detect_liveness_from_image(self, image_path: str, face_bbox: Optional[List[int]] = None,
                                 use_advanced_analysis: bool = True) -> LivenessDetectionResult:
        """
        Detect liveness từ image file
        
        Args:
            image_path: Đường dẫn đến image file
            face_bbox: Face bounding box [x1, y1, x2, y2]
            use_advanced_analysis: Có sử dụng advanced analysis không
            
        Returns:
            LivenessDetectionResult
        """
        try:
            # Validate file
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            if not self._is_supported_format(image_path):
                raise ValueError(f"Unsupported image format: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Detect liveness
            result = self.liveness_engine.detect_liveness(image, face_bbox, use_advanced_analysis)
            result.image_path = image_path
            
            logger.info(f"Liveness detection completed for: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error in liveness detection from image: {e}")
            # Create failed result
            result = LivenessDetectionResult(image_path=image_path, face_bbox=face_bbox)
            result.mark_as_failed(str(e))
            return result
    
    def detect_liveness_from_array(self, image: np.ndarray, face_bbox: Optional[List[int]] = None,
                                 use_advanced_analysis: bool = True) -> LivenessDetectionResult:
        """
        Detect liveness từ numpy array
        
        Args:
            image: Image array
            face_bbox: Face bounding box [x1, y1, x2, y2]
            use_advanced_analysis: Có sử dụng advanced analysis không
            
        Returns:
            LivenessDetectionResult
        """
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image array")
            
            # Detect liveness
            result = self.liveness_engine.detect_liveness(image, face_bbox, use_advanced_analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in liveness detection from array: {e}")
            # Create failed result
            result = LivenessDetectionResult(face_bbox=face_bbox)
            result.mark_as_failed(str(e))
            return result
    
    def batch_detect_liveness(self, image_paths: List[str], 
                            face_bboxes: Optional[List[List[int]]] = None,
                            use_advanced_analysis: bool = True) -> List[LivenessDetectionResult]:
        """
        Batch liveness detection từ multiple images
        
        Args:
            image_paths: List đường dẫn đến image files
            face_bboxes: List face bounding boxes
            use_advanced_analysis: Có sử dụng advanced analysis không
            
        Returns:
            List of LivenessDetectionResult
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                # Get corresponding bbox
                bbox = None
                if face_bboxes and i < len(face_bboxes):
                    bbox = face_bboxes[i]
                
                # Detect liveness
                result = self.detect_liveness_from_image(image_path, bbox, use_advanced_analysis)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in batch detection for {image_path}: {e}")
                # Create failed result
                failed_result = LivenessDetectionResult(image_path=image_path)
                failed_result.mark_as_failed(str(e))
                results.append(failed_result)
        
        logger.info(f"Batch liveness detection completed for {len(image_paths)} images")
        return results
    
    def analyze_spoof_patterns(self, results: List[LivenessDetectionResult]) -> Dict[str, Any]:
        """
        Phân tích spoof patterns từ multiple results
        
        Args:
            results: List of LivenessDetectionResult
            
        Returns:
            Dict chứa spoof pattern analysis
        """
        try:
            if not results:
                return {"error": "No results provided"}
            
            # Count results by type
            real_count = sum(1 for r in results if r.is_real_face())
            fake_count = sum(1 for r in results if r.is_fake_face())
            uncertain_count = sum(1 for r in results if r.is_uncertain())
            
            # Count spoof types
            spoof_counts = {}
            for result in results:
                for spoof_type in result.detected_spoof_types:
                    spoof_counts[spoof_type.value] = spoof_counts.get(spoof_type.value, 0) + 1
            
            # Calculate statistics
            if results:
                avg_confidence = sum(r.confidence for r in results) / len(results)
                avg_liveness_score = sum(r.liveness_score for r in results) / len(results)
                avg_processing_time = sum(r.processing_time_ms for r in results) / len(results)
            else:
                avg_confidence = avg_liveness_score = avg_processing_time = 0.0
            
            # Risk assessment
            risk_level = self._assess_overall_risk(results)
            
            return {
                "total_analyzed": len(results),
                "result_distribution": {
                    "real": real_count,
                    "fake": fake_count,
                    "uncertain": uncertain_count
                },
                "spoof_type_distribution": spoof_counts,
                "statistics": {
                    "avg_confidence": avg_confidence,
                    "avg_liveness_score": avg_liveness_score,
                    "avg_processing_time_ms": avg_processing_time,
                    "fake_detection_rate": fake_count / len(results) if results else 0.0
                },
                "risk_assessment": risk_level,
                "recommendations": self._generate_recommendations(results)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spoof patterns: {e}")
            return {"error": str(e)}
    
    def compare_liveness_results(self, result1: LivenessDetectionResult, 
                               result2: LivenessDetectionResult) -> Dict[str, Any]:
        """
        So sánh 2 liveness detection results
        
        Args:
            result1: First result
            result2: Second result
            
        Returns:
            Dict chứa comparison analysis
        """
        try:
            comparison = {
                "results_match": result1.liveness_result == result2.liveness_result,
                "confidence_difference": abs(result1.confidence - result2.confidence),
                "score_difference": abs(result1.liveness_score - result2.liveness_score),
                "result1": {
                    "result": result1.liveness_result.value,
                    "confidence": result1.confidence,
                    "score": result1.liveness_score,
                    "spoof_types": [s.value for s in result1.detected_spoof_types]
                },
                "result2": {
                    "result": result2.liveness_result.value,
                    "confidence": result2.confidence,
                    "score": result2.liveness_score,
                    "spoof_types": [s.value for s in result2.detected_spoof_types]
                }
            }
            
            # Analysis consistency
            if comparison["results_match"]:
                if comparison["confidence_difference"] < 0.1:
                    comparison["consistency"] = "HIGH"
                elif comparison["confidence_difference"] < 0.3:
                    comparison["consistency"] = "MEDIUM"
                else:
                    comparison["consistency"] = "LOW"
            else:
                comparison["consistency"] = "INCONSISTENT"
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing liveness results: {e}")
            return {"error": str(e)}
    
    def validate_liveness_quality(self, result: LivenessDetectionResult) -> Dict[str, Any]:
        """
        Validate chất lượng của liveness detection result
        
        Args:
            result: LivenessDetectionResult to validate
            
        Returns:
            Dict chứa validation results
        """
        try:
            validation = {
                "is_valid": True,
                "issues": [],
                "quality_score": 0.0,
                "recommendations": []
            }
            
            # Check image quality
            if result.image_quality < 0.5:
                validation["issues"].append("Low image quality")
                validation["recommendations"].append("Use higher quality images")
            
            # Check face quality
            if result.face_quality < 0.5:
                validation["issues"].append("Poor face quality")
                validation["recommendations"].append("Ensure clear face visibility")
            
            # Check lighting
            if result.lighting_quality < 0.4:
                validation["issues"].append("Poor lighting conditions")
                validation["recommendations"].append("Improve lighting conditions")
            
            # Check confidence
            if result.confidence < 0.6:
                validation["issues"].append("Low confidence detection")
                validation["recommendations"].append("Retake image or use different angle")
            
            # Check processing
            if result.processing_time_ms > 5000:  # 5 seconds
                validation["issues"].append("Slow processing time")
                validation["recommendations"].append("Optimize image size or quality")
            
            # Overall quality score
            quality_components = [
                result.image_quality,
                result.face_quality,
                result.lighting_quality,
                result.pose_quality,
                result.confidence
            ]
            validation["quality_score"] = sum(quality_components) / len(quality_components)
            
            # Set validity
            validation["is_valid"] = len(validation["issues"]) == 0
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating liveness quality: {e}")
            return {"error": str(e)}
    
    def optimize_detection_parameters(self, training_results: List[Tuple[LivenessDetectionResult, bool]]) -> Dict[str, Any]:
        """
        Optimize detection parameters based on training results
        
        Args:
            training_results: List of (result, ground_truth) pairs
            
        Returns:
            Dict chứa optimized parameters
        """
        try:
            if len(training_results) < 10:
                return {"error": "Insufficient training data (need at least 10 samples)"}
            
            # Separate real và fake results
            real_scores = []
            fake_scores = []
            
            for result, is_real in training_results:
                if is_real:
                    real_scores.append(result.liveness_score)
                else:
                    fake_scores.append(result.liveness_score)
            
            if not real_scores or not fake_scores:
                return {"error": "Need both real and fake samples"}
            
            # Find optimal threshold
            best_threshold = self._find_optimal_threshold(real_scores, fake_scores)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(training_results, best_threshold)
            
            # Update engine parameters
            self.liveness_engine.set_thresholds(best_threshold, best_threshold - 0.2)
            
            return {
                "optimal_threshold": best_threshold,
                "performance_metrics": metrics,
                "training_data_stats": {
                    "total_samples": len(training_results),
                    "real_samples": len(real_scores),
                    "fake_samples": len(fake_scores),
                    "real_score_mean": np.mean(real_scores),
                    "fake_score_mean": np.mean(fake_scores)
                }
            }
            
        except Exception as e:
            logger.error(f"Error optimizing detection parameters: {e}")
            return {"error": str(e)}
    
    def get_supported_formats(self) -> List[str]:
        """Get supported image formats"""
        return self.supported_formats.copy()
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get liveness engine information"""
        return self.liveness_engine.get_model_info()
    
    def _is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        try:
            _, ext = os.path.splitext(file_path.lower())
            return ext in self.supported_formats
        except:
            return False
    
    def _assess_overall_risk(self, results: List[LivenessDetectionResult]) -> str:
        """Assess overall risk level"""
        try:
            if not results:
                return "UNKNOWN"
            
            fake_ratio = sum(1 for r in results if r.is_fake_face()) / len(results)
            uncertain_ratio = sum(1 for r in results if r.is_uncertain()) / len(results)
            avg_confidence = sum(r.confidence for r in results) / len(results)
            
            if fake_ratio > 0.5 or uncertain_ratio > 0.3:
                return "HIGH"
            elif fake_ratio > 0.2 or uncertain_ratio > 0.1 or avg_confidence < 0.7:
                return "MEDIUM"
            else:
                return "LOW"
                
        except:
            return "UNKNOWN"
    
    def _generate_recommendations(self, results: List[LivenessDetectionResult]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        try:
            if not results:
                return ["No data available for recommendations"]
            
            # Check common issues
            low_quality_count = sum(1 for r in results if r.image_quality < 0.5)
            fake_count = sum(1 for r in results if r.is_fake_face())
            uncertain_count = sum(1 for r in results if r.is_uncertain())
            
            if low_quality_count / len(results) > 0.3:
                recommendations.append("Improve image quality (lighting, resolution, focus)")
            
            if fake_count > 0:
                recommendations.append("Implement additional security measures")
                recommendations.append("Consider multi-factor authentication")
            
            if uncertain_count / len(results) > 0.2:
                recommendations.append("Refine detection thresholds")
                recommendations.append("Collect more training data")
            
            # Spoof-specific recommendations
            spoof_types = set()
            for result in results:
                spoof_types.update(result.detected_spoof_types)
            
            if SpoofType.PHOTO_ATTACK in spoof_types:
                recommendations.append("Add motion detection or challenge-response")
            
            if SpoofType.SCREEN_ATTACK in spoof_types:
                recommendations.append("Implement screen detection algorithms")
            
            if SpoofType.MASK_ATTACK in spoof_types:
                recommendations.append("Add 3D depth analysis")
            
            if not recommendations:
                recommendations.append("Current detection performance is satisfactory")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]
    
    def _find_optimal_threshold(self, real_scores: List[float], fake_scores: List[float]) -> float:
        """Find optimal threshold using ROC analysis"""
        try:
            # Test different thresholds
            thresholds = np.linspace(0.1, 0.9, 81)
            best_threshold = 0.5
            best_accuracy = 0.0
            
            for threshold in thresholds:
                # Calculate true positives, false positives, etc.
                tp = sum(1 for score in real_scores if score >= threshold)
                fn = len(real_scores) - tp
                tn = sum(1 for score in fake_scores if score < threshold)
                fp = len(fake_scores) - tn
                
                # Calculate accuracy
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
            
            return best_threshold
            
        except Exception as e:
            logger.error(f"Error finding optimal threshold: {e}")
            return 0.5
    
    def _calculate_performance_metrics(self, training_results: List[Tuple[LivenessDetectionResult, bool]], 
                                     threshold: float) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            tp = tn = fp = fn = 0
            
            for result, is_real in training_results:
                predicted_real = result.liveness_score >= threshold
                
                if is_real and predicted_real:
                    tp += 1
                elif is_real and not predicted_real:
                    fn += 1
                elif not is_real and not predicted_real:
                    tn += 1
                else:  # not is_real and predicted_real
                    fp += 1
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "specificity": specificity,
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "specificity": 0.0,
                "true_positives": 0,
                "true_negatives": 0,
                "false_positives": 0,
                "false_negatives": 0
            }
