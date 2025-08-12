import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import logging
from skimage.feature import local_binary_pattern
from scipy import ndimage
from scipy.fft import fft2, fftshift
import math

# Initialize logger
logger = logging.getLogger(__name__)

class TextureAnalyzer:
    """Phân tích texture để phát hiện spoof attacks"""
    
    def __init__(self):
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius
    
    def analyze_image_texture(self, image: np.ndarray, face_bbox: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Phân tích texture của ảnh để phát hiện spoof
        
        Args:
            image: Input image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Dict chứa các texture metrics
        """
        try:
            # Crop face region nếu có bbox
            if face_bbox and len(face_bbox) == 4:
                x1, y1, x2, y2 = face_bbox
                face_region = image[y1:y2, x1:x2]
            else:
                face_region = image
            
            # Convert to grayscale nếu cần
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region.copy()
            
            # Resize để chuẩn hóa
            gray = cv2.resize(gray, (128, 128))
            
            # Các phân tích texture
            lbp_score = self._analyze_lbp(gray)
            sobel_score = self._analyze_edges(gray)
            gabor_score = self._analyze_gabor(gray)
            variance_score = self._analyze_texture_variance(gray)
            
            return {
                "lbp_score": lbp_score,
                "sobel_score": sobel_score,
                "gabor_score": gabor_score,
                "variance_score": variance_score,
                "overall_texture_score": (lbp_score + sobel_score + gabor_score + variance_score) / 4
            }
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return {
                "lbp_score": 0.0,
                "sobel_score": 0.0,
                "gabor_score": 0.0,
                "variance_score": 0.0,
                "overall_texture_score": 0.0
            }
    
    def _analyze_lbp(self, gray: np.ndarray) -> float:
        """Phân tích Local Binary Pattern"""
        try:
            # Tính LBP
            lbp = local_binary_pattern(gray, self.lbp_n_points, self.lbp_radius, method='uniform')
            
            # Tính histogram
            hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_n_points + 2, 
                                 range=(0, self.lbp_n_points + 2), density=True)
            
            # Tính uniformity của LBP
            uniformity = 1.0 - np.sum(hist**2)
            
            # Tính contrast
            contrast = np.var(lbp)
            
            # Score cao hơn cho real faces (có texture phức tạp hơn)
            lbp_score = (uniformity * 0.6 + min(contrast / 1000.0, 1.0) * 0.4)
            
            return min(max(lbp_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in LBP analysis: {e}")
            return 0.0
    
    def _analyze_edges(self, gray: np.ndarray) -> float:
        """Phân tích edge density"""
        try:
            # Sobel edge detection
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Magnitude
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Edge density
            edge_threshold = np.mean(magnitude) + np.std(magnitude)
            edge_density = np.sum(magnitude > edge_threshold) / magnitude.size
            
            # Edge strength
            edge_strength = np.mean(magnitude) / 255.0
            
            # Combine metrics
            edge_score = (edge_density * 0.7 + edge_strength * 0.3)
            
            return min(max(edge_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in edge analysis: {e}")
            return 0.0
    
    def _analyze_gabor(self, gray: np.ndarray) -> float:
        """Phân tích Gabor filters"""
        try:
            # Multiple Gabor filters với các frequency và orientation khác nhau
            ksize = 31
            sigma = 4
            gabor_responses = []
            
            for theta in [0, 45, 90, 135]:  # Các orientation
                for frequency in [0.1, 0.3, 0.5]:  # Các frequency
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, 
                                              np.radians(theta), 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                    response = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                    gabor_responses.append(np.var(response))
            
            # Tính mean và std của responses
            mean_response = np.mean(gabor_responses)
            std_response = np.std(gabor_responses)
            
            # Score cao hơn cho real faces
            gabor_score = (mean_response / 10000.0 + std_response / 5000.0) / 2
            
            return min(max(gabor_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in Gabor analysis: {e}")
            return 0.0
    
    def _analyze_texture_variance(self, gray: np.ndarray) -> float:
        """Phân tích texture variance"""
        try:
            # Local variance
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
            local_variance = local_sqr_mean - local_mean**2
            
            # Statistics
            mean_variance = np.mean(local_variance)
            std_variance = np.std(local_variance)
            
            # Texture richness
            texture_richness = mean_variance / 10000.0
            texture_uniformity = 1.0 - (std_variance / (mean_variance + 1e-8))
            
            # Combined score
            variance_score = (texture_richness * 0.6 + texture_uniformity * 0.4)
            
            return min(max(variance_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in variance analysis: {e}")
            return 0.0

class FrequencyAnalyzer:
    """Phân tích frequency domain để phát hiện spoof"""
    
    def analyze_frequency_domain(self, image: np.ndarray, face_bbox: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Phân tích frequency domain của ảnh
        
        Args:
            image: Input image
            face_bbox: Face bounding box
            
        Returns:
            Dict chứa frequency analysis metrics
        """
        try:
            # Crop face region
            if face_bbox and len(face_bbox) == 4:
                x1, y1, x2, y2 = face_bbox
                face_region = image[y1:y2, x1:x2]
            else:
                face_region = image
            
            # Convert to grayscale
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region.copy()
            
            # Resize
            gray = cv2.resize(gray, (128, 128))
            
            # FFT analysis
            fft_features = self._analyze_fft(gray)
            
            # DCT analysis
            dct_features = self._analyze_dct(gray)
            
            # High/Low frequency ratio
            high_freq_ratio = self._calculate_high_freq_ratio(gray)
            low_freq_ratio = 1.0 - high_freq_ratio
            
            return {
                "high_freq_ratio": high_freq_ratio,
                "low_freq_ratio": low_freq_ratio,
                "freq_ratio": high_freq_ratio / (low_freq_ratio + 1e-8),
                "fft_energy": fft_features["energy"],
                "fft_entropy": fft_features["entropy"],
                "dct_energy": dct_features["energy"],
                "dct_concentration": dct_features["concentration"]
            }
            
        except Exception as e:
            logger.error(f"Error in frequency analysis: {e}")
            return {
                "high_freq_ratio": 0.0,
                "low_freq_ratio": 1.0,
                "freq_ratio": 0.0,
                "fft_energy": 0.0,
                "fft_entropy": 0.0,
                "dct_energy": 0.0,
                "dct_concentration": 0.0
            }
    
    def _analyze_fft(self, gray: np.ndarray) -> Dict[str, float]:
        """Phân tích FFT"""
        try:
            # 2D FFT
            f_transform = fft2(gray)
            f_shift = fftshift(f_transform)
            magnitude = np.abs(f_shift)
            
            # Energy
            energy = np.sum(magnitude**2) / magnitude.size
            
            # Entropy
            magnitude_norm = magnitude / (np.sum(magnitude) + 1e-8)
            entropy = -np.sum(magnitude_norm * np.log(magnitude_norm + 1e-8))
            
            return {
                "energy": min(energy / 1e10, 1.0),
                "entropy": min(entropy / 10.0, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error in FFT analysis: {e}")
            return {"energy": 0.0, "entropy": 0.0}
    
    def _analyze_dct(self, gray: np.ndarray) -> Dict[str, float]:
        """Phân tích DCT"""
        try:
            # DCT
            gray_float = np.float32(gray)
            dct = cv2.dct(gray_float)
            
            # Energy
            energy = np.sum(dct**2) / dct.size
            
            # Low frequency concentration (top-left corner)
            h, w = dct.shape
            low_freq_region = dct[:h//4, :w//4]
            total_energy = np.sum(dct**2)
            low_freq_energy = np.sum(low_freq_region**2)
            concentration = low_freq_energy / (total_energy + 1e-8)
            
            return {
                "energy": min(energy / 1e6, 1.0),
                "concentration": concentration
            }
            
        except Exception as e:
            logger.error(f"Error in DCT analysis: {e}")
            return {"energy": 0.0, "concentration": 0.0}
    
    def _calculate_high_freq_ratio(self, gray: np.ndarray) -> float:
        """Tính tỷ lệ high frequency"""
        try:
            # High-pass filter
            kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])
            high_freq = cv2.filter2D(gray, -1, kernel)
            
            # Calculate ratio
            high_freq_energy = np.sum(high_freq**2)
            total_energy = np.sum(gray**2)
            
            ratio = high_freq_energy / (total_energy + 1e-8)
            return min(max(ratio, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating high freq ratio: {e}")
            return 0.0

class DepthAnalyzer:
    """Phân tích depth cues để phát hiện flat attacks"""
    
    def analyze_depth_cues(self, image: np.ndarray, face_bbox: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Phân tích depth cues
        
        Args:
            image: Input image
            face_bbox: Face bounding box
            
        Returns:
            Dict chứa depth analysis metrics
        """
        try:
            # Crop face region
            if face_bbox and len(face_bbox) == 4:
                x1, y1, x2, y2 = face_bbox
                face_region = image[y1:y2, x1:x2]
            else:
                face_region = image
            
            # Convert to grayscale
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region.copy()
            
            # Resize
            gray = cv2.resize(gray, (128, 128))
            
            # Depth analysis
            depth_variance = self._analyze_depth_variance(gray)
            edge_density = self._analyze_edge_density(gray)
            shadow_consistency = self._analyze_shadow_consistency(face_region)
            
            return {
                "depth_variance": depth_variance,
                "edge_density": edge_density,
                "shadow_consistency": shadow_consistency,
                "overall_depth_score": (depth_variance + edge_density + shadow_consistency) / 3
            }
            
        except Exception as e:
            logger.error(f"Error in depth analysis: {e}")
            return {
                "depth_variance": 0.0,
                "edge_density": 0.0,
                "shadow_consistency": 0.0,
                "overall_depth_score": 0.0
            }
    
    def _analyze_depth_variance(self, gray: np.ndarray) -> float:
        """Phân tích variance of gradients (depth indicator)"""
        try:
            # Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Local variance of gradients
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(magnitude, -1, kernel)
            local_variance = cv2.filter2D(magnitude**2, -1, kernel) - local_mean**2
            
            # Overall variance
            depth_variance = np.var(local_variance) / 10000.0
            
            return min(max(depth_variance, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in depth variance analysis: {e}")
            return 0.0
    
    def _analyze_edge_density(self, gray: np.ndarray) -> float:
        """Phân tích edge density (3D faces có more complex edges)"""
        try:
            # Multi-scale edge detection
            edges_scales = []
            for ksize in [3, 5, 7]:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
                edges = np.sqrt(grad_x**2 + grad_y**2)
                edges_scales.append(edges)
            
            # Combine scales
            combined_edges = np.mean(edges_scales, axis=0)
            
            # Edge density
            threshold = np.mean(combined_edges) + 0.5 * np.std(combined_edges)
            edge_density = np.sum(combined_edges > threshold) / combined_edges.size
            
            return min(max(edge_density, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in edge density analysis: {e}")
            return 0.0
    
    def _analyze_shadow_consistency(self, image: np.ndarray) -> float:
        """Phân tích shadow consistency"""
        try:
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
            else:
                l_channel = image.copy()
            
            # Resize
            l_channel = cv2.resize(l_channel, (128, 128))
            
            # Analyze lighting gradients
            grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=5)
            
            # Gradient direction consistency
            angles = np.arctan2(grad_y, grad_x)
            
            # Circular variance (measure of consistency)
            sin_angles = np.sin(2 * angles)
            cos_angles = np.cos(2 * angles)
            
            mean_sin = np.mean(sin_angles)
            mean_cos = np.mean(cos_angles)
            
            consistency = np.sqrt(mean_sin**2 + mean_cos**2)
            
            return min(max(consistency, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in shadow consistency analysis: {e}")
            return 0.0

class QualityAnalyzer:
    """Phân tích chất lượng ảnh cho liveness detection"""
    
    def analyze_image_quality(self, image: np.ndarray, face_bbox: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Phân tích chất lượng ảnh
        
        Args:
            image: Input image
            face_bbox: Face bounding box
            
        Returns:
            Dict chứa quality metrics
        """
        try:
            # Image quality
            image_quality = self._analyze_overall_quality(image)
            
            # Face quality (nếu có bbox)
            face_quality = 0.0
            if face_bbox and len(face_bbox) == 4:
                x1, y1, x2, y2 = face_bbox
                face_region = image[y1:y2, x1:x2]
                face_quality = self._analyze_face_quality(face_region)
            
            # Lighting quality
            lighting_quality = self._analyze_lighting_quality(image, face_bbox)
            
            # Pose quality
            pose_quality = self._analyze_pose_quality(image, face_bbox)
            
            return {
                "image_quality": image_quality,
                "face_quality": face_quality,
                "lighting_quality": lighting_quality,
                "pose_quality": pose_quality,
                "overall_quality": (image_quality + face_quality + lighting_quality + pose_quality) / 4
            }
            
        except Exception as e:
            logger.error(f"Error in quality analysis: {e}")
            return {
                "image_quality": 0.0,
                "face_quality": 0.0,
                "lighting_quality": 0.0,
                "pose_quality": 0.0,
                "overall_quality": 0.0
            }
    
    def _analyze_overall_quality(self, image: np.ndarray) -> float:
        """Phân tích chất lượng tổng thể"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_quality = min(blur_score / 1000.0, 1.0)
            
            # Noise estimation
            noise_score = self._estimate_noise(gray)
            noise_quality = max(0.0, 1.0 - noise_score / 50.0)
            
            # Contrast
            contrast = gray.std()
            contrast_quality = min(contrast / 50.0, 1.0)
            
            # Combined quality
            quality = (blur_quality * 0.4 + noise_quality * 0.3 + contrast_quality * 0.3)
            
            return min(max(quality, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in overall quality analysis: {e}")
            return 0.0
    
    def _analyze_face_quality(self, face_region: np.ndarray) -> float:
        """Phân tích chất lượng vùng khuôn mặt"""
        try:
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region.copy()
            
            # Face sharpness
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_quality = min(sharpness / 500.0, 1.0)
            
            # Face contrast
            contrast = gray.std()
            contrast_quality = min(contrast / 40.0, 1.0)
            
            # Face brightness
            brightness = gray.mean()
            brightness_quality = 1.0 - abs(brightness - 128) / 128.0
            
            # Combined quality
            quality = (sharpness_quality * 0.5 + contrast_quality * 0.3 + brightness_quality * 0.2)
            
            return min(max(quality, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in face quality analysis: {e}")
            return 0.0
    
    def _analyze_lighting_quality(self, image: np.ndarray, face_bbox: Optional[List[int]] = None) -> float:
        """Phân tích chất lượng ánh sáng"""
        try:
            # Use face region if available
            if face_bbox and len(face_bbox) == 4:
                x1, y1, x2, y2 = face_bbox
                roi = image[y1:y2, x1:x2]
            else:
                roi = image
            
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()
            
            # Lighting uniformity
            mean_brightness = gray.mean()
            std_brightness = gray.std()
            uniformity = max(0.0, 1.0 - std_brightness / (mean_brightness + 1e-8))
            
            # Exposure quality
            overexposed = np.sum(gray > 240) / gray.size
            underexposed = np.sum(gray < 15) / gray.size
            exposure_quality = max(0.0, 1.0 - overexposed - underexposed)
            
            # Combined lighting quality
            lighting_quality = (uniformity * 0.6 + exposure_quality * 0.4)
            
            return min(max(lighting_quality, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in lighting quality analysis: {e}")
            return 0.0
    
    def _analyze_pose_quality(self, image: np.ndarray, face_bbox: Optional[List[int]] = None) -> float:
        """Phân tích chất lượng pose (frontal face)"""
        try:
            # Simplified pose analysis based on face symmetry
            if not face_bbox or len(face_bbox) != 4:
                return 0.5  # Unknown pose quality
            
            x1, y1, x2, y2 = face_bbox
            face_region = image[y1:y2, x1:x2]
            
            if len(face_region.shape) == 3:
                gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_region.copy()
            
            # Resize for consistency
            gray = cv2.resize(gray, (64, 64))
            
            # Horizontal symmetry
            left_half = gray[:, :32]
            right_half = gray[:, 32:]
            right_half_flipped = np.fliplr(right_half)
            
            # Calculate symmetry score
            diff = np.abs(left_half.astype(np.float32) - right_half_flipped.astype(np.float32))
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            
            # Aspect ratio check
            h, w = gray.shape
            aspect_ratio = w / h
            aspect_quality = max(0.0, 1.0 - abs(aspect_ratio - 1.0))
            
            # Combined pose quality
            pose_quality = (symmetry_score * 0.7 + aspect_quality * 0.3)
            
            return min(max(pose_quality, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error in pose quality analysis: {e}")
            return 0.0
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Ước tính noise level"""
        try:
            # Noise estimation using median filter
            filtered = cv2.medianBlur(gray, 5)
            noise = gray.astype(np.float32) - filtered.astype(np.float32)
            noise_level = np.std(noise)
            
            return noise_level
            
        except Exception as e:
            logger.error(f"Error in noise estimation: {e}")
            return 0.0
