import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging
from domain.ml.liveness_detector import TextureAnalyzer, FrequencyAnalyzer, DepthAnalyzer, QualityAnalyzer
from domain.entities.liveness_result import LivenessDetectionResult, LivenessStatus, LivenessResult, SpoofType
import time

# Initialize logger
logger = logging.getLogger(__name__)

class LivenessEngine:
    """Engine chính cho liveness detection"""
    
    def __init__(self):
        self.texture_analyzer = TextureAnalyzer()
        self.frequency_analyzer = FrequencyAnalyzer()
        self.depth_analyzer = DepthAnalyzer()
        self.quality_analyzer = QualityAnalyzer()
        
        # Thresholds
        self.real_threshold = 0.6
        self.fake_threshold = 0.4
        
        # Weights cho các components
        self.weights = {
            "texture": 0.3,
            "frequency": 0.25,
            "depth": 0.25,
            "quality": 0.2
        }
    
    def detect_liveness(self, image: np.ndarray, face_bbox: Optional[List[int]] = None,
                       use_advanced_analysis: bool = True) -> LivenessDetectionResult:
        """
        Thực hiện liveness detection
        
        Args:
            image: Input image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            use_advanced_analysis: Có sử dụng advanced analysis không
            
        Returns:
            LivenessDetectionResult
        """
        start_time = time.time()
        
        # Tạo result object
        result = LivenessDetectionResult(
            image_path="",  # Sẽ được set từ bên ngoài
            face_bbox=face_bbox,
            status=LivenessStatus.PROCESSING
        )
        
        try:
            # Validate input
            if image is None or image.size == 0:
                raise ValueError("Invalid input image")
            
            # Quality analysis
            quality_metrics = self.quality_analyzer.analyze_image_quality(image, face_bbox)
            result.update_quality_metrics(
                quality_metrics["image_quality"],
                quality_metrics["face_quality"],
                quality_metrics["lighting_quality"],
                quality_metrics["pose_quality"]
            )
            
            # Kiểm tra quality threshold
            if quality_metrics["overall_quality"] < 0.3:
                result.mark_as_failed("Image quality too low for reliable liveness detection")
                return result
            
            # Core analysis
            scores = self._perform_core_analysis(image, face_bbox, use_advanced_analysis)
            
            # Update result với analysis details
            self._update_result_with_analysis(result, scores)
            
            # Calculate final score
            final_score = self._calculate_final_score(scores)
            
            # Detect spoof types
            self._detect_spoof_types(result, scores)
            
            # Finalize result
            result.finalize_detection(final_score, self.real_threshold)
            
            # Set processing time
            processing_time = (time.time() - start_time) * 1000
            result.processing_time_ms = processing_time
            
            # Add algorithms used
            result.algorithms_used = [
                "texture_analysis", "frequency_analysis", 
                "depth_analysis", "quality_analysis"
            ]
            
            logger.info(f"Liveness detection completed: {result.liveness_result.value} "
                       f"(score: {final_score:.3f}, confidence: {result.confidence:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in liveness detection: {e}")
            result.mark_as_failed(str(e))
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
    
    def _perform_core_analysis(self, image: np.ndarray, face_bbox: Optional[List[int]] = None,
                             use_advanced: bool = True) -> Dict[str, Dict[str, float]]:
        """Thực hiện core analysis"""
        scores = {}
        
        try:
            # Texture analysis
            texture_scores = self.texture_analyzer.analyze_image_texture(image, face_bbox)
            scores["texture"] = texture_scores
            
            # Frequency analysis
            frequency_scores = self.frequency_analyzer.analyze_frequency_domain(image, face_bbox)
            scores["frequency"] = frequency_scores
            
            if use_advanced:
                # Depth analysis
                depth_scores = self.depth_analyzer.analyze_depth_cues(image, face_bbox)
                scores["depth"] = depth_scores
            else:
                scores["depth"] = {"overall_depth_score": 0.5}
            
            # Quality analysis
            quality_scores = self.quality_analyzer.analyze_image_quality(image, face_bbox)
            scores["quality"] = quality_scores
            
        except Exception as e:
            logger.error(f"Error in core analysis: {e}")
            # Return default scores
            scores = {
                "texture": {"overall_texture_score": 0.0},
                "frequency": {"high_freq_ratio": 0.0},
                "depth": {"overall_depth_score": 0.0},
                "quality": {"overall_quality": 0.0}
            }
        
        return scores
    
    def _update_result_with_analysis(self, result: LivenessDetectionResult, 
                                   scores: Dict[str, Dict[str, float]]):
        """Update result với analysis details"""
        try:
            # Texture analysis
            if "texture" in scores:
                texture = scores["texture"]
                result.update_texture_analysis(
                    texture.get("lbp_score", 0.0),
                    texture.get("sobel_score", 0.0),
                    texture.get("gabor_score", 0.0),
                    texture.get("variance_score", 0.0)
                )
            
            # Frequency analysis
            if "frequency" in scores:
                frequency = scores["frequency"]
                result.update_frequency_analysis(
                    frequency.get("high_freq_ratio", 0.0),
                    frequency.get("low_freq_ratio", 0.0),
                    [frequency.get("dct_energy", 0.0)],
                    [frequency.get("fft_energy", 0.0)]
                )
            
            # Depth analysis
            if "depth" in scores:
                depth = scores["depth"]
                result.update_depth_analysis(
                    depth.get("depth_variance", 0.0),
                    depth.get("edge_density", 0.0),
                    depth.get("shadow_consistency", 0.0)
                )
                
        except Exception as e:
            logger.error(f"Error updating result with analysis: {e}")
    
    def _calculate_final_score(self, scores: Dict[str, Dict[str, float]]) -> float:
        """Tính final liveness score"""
        try:
            # Extract main scores
            texture_score = scores.get("texture", {}).get("overall_texture_score", 0.0)
            frequency_score = self._get_frequency_score(scores.get("frequency", {}))
            depth_score = scores.get("depth", {}).get("overall_depth_score", 0.0)
            quality_score = scores.get("quality", {}).get("overall_quality", 0.0)
            
            # Weighted combination
            final_score = (
                texture_score * self.weights["texture"] +
                frequency_score * self.weights["frequency"] +
                depth_score * self.weights["depth"] +
                quality_score * self.weights["quality"]
            )
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating final score: {e}")
            return 0.0
    
    def _get_frequency_score(self, frequency_data: Dict[str, float]) -> float:
        """Tính frequency score từ frequency analysis"""
        try:
            high_freq = frequency_data.get("high_freq_ratio", 0.0)
            fft_energy = frequency_data.get("fft_energy", 0.0)
            dct_energy = frequency_data.get("dct_energy", 0.0)
            
            # Real faces thường có high frequency content cao hơn
            frequency_score = (high_freq * 0.5 + fft_energy * 0.3 + dct_energy * 0.2)
            
            return min(max(frequency_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating frequency score: {e}")
            return 0.0
    
    def _detect_spoof_types(self, result: LivenessDetectionResult, 
                          scores: Dict[str, Dict[str, float]]):
        """Phát hiện các loại spoof attacks"""
        try:
            texture_data = scores.get("texture", {})
            frequency_data = scores.get("frequency", {})
            depth_data = scores.get("depth", {})
            quality_data = scores.get("quality", {})
            
            # Photo attack detection
            self._detect_photo_attack(result, texture_data, frequency_data)
            
            # Screen attack detection
            self._detect_screen_attack(result, frequency_data, quality_data)
            
            # Mask attack detection
            self._detect_mask_attack(result, depth_data, texture_data)
            
            # General fake detection
            self._detect_general_fake(result, scores)
            
        except Exception as e:
            logger.error(f"Error detecting spoof types: {e}")
    
    def _detect_photo_attack(self, result: LivenessDetectionResult, 
                           texture_data: Dict[str, float], 
                           frequency_data: Dict[str, float]):
        """Phát hiện photo attack"""
        try:
            # Photo attacks thường có:
            # - Low texture variance
            # - Low high frequency content
            # - Regular patterns from printing
            
            texture_score = texture_data.get("overall_texture_score", 1.0)
            high_freq = frequency_data.get("high_freq_ratio", 1.0)
            
            # Thresholds cho photo detection
            if texture_score < 0.3 and high_freq < 0.2:
                confidence = 1.0 - ((texture_score + high_freq) / 2)
                result.add_spoof_type(SpoofType.PHOTO_ATTACK, confidence)
                
        except Exception as e:
            logger.error(f"Error detecting photo attack: {e}")
    
    def _detect_screen_attack(self, result: LivenessDetectionResult,
                            frequency_data: Dict[str, float],
                            quality_data: Dict[str, float]):
        """Phát hiện screen attack"""
        try:
            # Screen attacks thường có:
            # - Moire patterns trong frequency domain
            # - Low image quality từ screen refresh
            
            freq_ratio = frequency_data.get("freq_ratio", 1.0)
            image_quality = quality_data.get("image_quality", 1.0)
            
            # Detect abnormal frequency patterns và low quality
            if freq_ratio > 3.0 or (freq_ratio < 0.5 and image_quality < 0.4):
                confidence = min(0.9, abs(freq_ratio - 1.0) + (1.0 - image_quality))
                result.add_spoof_type(SpoofType.SCREEN_ATTACK, confidence)
                
        except Exception as e:
            logger.error(f"Error detecting screen attack: {e}")
    
    def _detect_mask_attack(self, result: LivenessDetectionResult,
                          depth_data: Dict[str, float],
                          texture_data: Dict[str, float]):
        """Phát hiện mask attack"""
        try:
            # Mask attacks thường có:
            # - Abnormal depth cues
            # - Artificial texture patterns
            
            depth_score = depth_data.get("overall_depth_score", 1.0)
            texture_variance = texture_data.get("variance_score", 1.0)
            
            if depth_score < 0.3 or texture_variance > 0.8:
                confidence = max(1.0 - depth_score, texture_variance - 0.5)
                confidence = min(confidence, 0.9)
                result.add_spoof_type(SpoofType.MASK_ATTACK, confidence)
                
        except Exception as e:
            logger.error(f"Error detecting mask attack: {e}")
    
    def _detect_general_fake(self, result: LivenessDetectionResult,
                           scores: Dict[str, Dict[str, float]]):
        """Phát hiện general fake patterns"""
        try:
            # Tính tổng score để detect unknown attacks
            overall_scores = []
            
            for category, data in scores.items():
                if category == "texture":
                    overall_scores.append(data.get("overall_texture_score", 0.0))
                elif category == "frequency":
                    overall_scores.append(self._get_frequency_score(data))
                elif category == "depth":
                    overall_scores.append(data.get("overall_depth_score", 0.0))
                elif category == "quality":
                    overall_scores.append(data.get("overall_quality", 0.0))
            
            avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            
            # Nếu score thấp mà chưa detect được spoof type cụ thể
            if avg_score < 0.4 and not result.has_spoof_detection():
                confidence = 1.0 - avg_score
                result.add_spoof_type(SpoofType.UNKNOWN, confidence)
                
        except Exception as e:
            logger.error(f"Error detecting general fake: {e}")
    
    def batch_detect_liveness(self, images: List[np.ndarray], 
                            face_bboxes: Optional[List[List[int]]] = None,
                            use_advanced_analysis: bool = True) -> List[LivenessDetectionResult]:
        """
        Batch liveness detection
        
        Args:
            images: List of input images
            face_bboxes: List of face bounding boxes
            use_advanced_analysis: Use advanced analysis
            
        Returns:
            List of LivenessDetectionResult
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                # Get corresponding bbox
                bbox = None
                if face_bboxes and i < len(face_bboxes):
                    bbox = face_bboxes[i]
                
                # Detect liveness
                result = self.detect_liveness(image, bbox, use_advanced_analysis)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error in batch detection for image {i}: {e}")
                # Create failed result
                failed_result = LivenessDetectionResult()
                failed_result.mark_as_failed(str(e))
                results.append(failed_result)
        
        logger.info(f"Batch liveness detection completed for {len(images)} images")
        return results
    
    def set_thresholds(self, real_threshold: float, fake_threshold: float):
        """Set detection thresholds"""
        if 0.0 <= real_threshold <= 1.0:
            self.real_threshold = real_threshold
        
        if 0.0 <= fake_threshold <= 1.0:
            self.fake_threshold = fake_threshold
    
    def set_weights(self, texture: float, frequency: float, depth: float, quality: float):
        """Set component weights"""
        total = texture + frequency + depth + quality
        if total > 0:
            self.weights = {
                "texture": texture / total,
                "frequency": frequency / total,
                "depth": depth / total,
                "quality": quality / total
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "version": "1.0",
            "components": ["texture", "frequency", "depth", "quality"],
            "thresholds": {
                "real_threshold": self.real_threshold,
                "fake_threshold": self.fake_threshold
            },
            "weights": self.weights,
            "supported_attacks": [
                "PHOTO_ATTACK",
                "SCREEN_ATTACK", 
                "MASK_ATTACK",
                "VIDEO_REPLAY",
                "UNKNOWN"
            ]
        }
