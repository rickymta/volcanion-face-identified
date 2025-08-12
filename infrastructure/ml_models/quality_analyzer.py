import cv2
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from PIL import Image, ImageStat
from PIL.ExifTags import TAGS
import math

class QualityAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_image_quality(self, image_path: str, bbox: Optional[list] = None) -> Dict:
        """
        Phân tích chất lượng ảnh toàn diện
        Returns: Dictionary chứa các metrics chất lượng
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Crop ảnh theo bbox nếu có
            if bbox:
                x1, y1, x2, y2 = bbox
                image = image[y1:y2, x1:x2]
            
            # Các metrics chất lượng
            blur_score = self._calculate_blur_score(image)
            glare_score = self._detect_glare(image)
            contrast_score = self._calculate_contrast(image)
            brightness_score = self._calculate_brightness(image)
            noise_score = self._calculate_noise(image)
            edge_sharpness = self._calculate_edge_sharpness(image)
            watermark_present = self._detect_watermark(image)
            
            # Tính overall score
            overall_score = self._calculate_overall_score({
                'blur': blur_score,
                'glare': 1.0 - glare_score,  # Glare thấp thì tốt
                'contrast': contrast_score,
                'brightness': brightness_score,
                'noise': 1.0 - noise_score,  # Noise thấp thì tốt
                'sharpness': edge_sharpness
            })
            
            return {
                'overall_score': overall_score,
                'blur_score': blur_score,
                'glare_score': glare_score,
                'contrast_score': contrast_score,
                'brightness_score': brightness_score,
                'noise_score': noise_score,
                'edge_sharpness': edge_sharpness,
                'watermark_present': watermark_present
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing image quality: {e}")
            return self._get_default_metrics()
    
    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """Tính toán độ mờ bằng Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 (higher is better)
        # Threshold based on typical document images
        max_variance = 1000
        blur_score = min(laplacian_var / max_variance, 1.0)
        
        return blur_score
    
    def _detect_glare(self, image: np.ndarray) -> float:
        """Phát hiện ánh sáng chói"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Tìm các vùng có độ sáng cao
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        bright_pixels = np.sum(bright_mask == 255)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        glare_ratio = bright_pixels / total_pixels
        
        # Kiểm tra clustering của bright pixels
        if glare_ratio > 0.05:  # Nếu có nhiều pixel sáng
            contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Tìm contour lớn nhất
                largest_contour = max(contours, key=cv2.contourArea)
                largest_area = cv2.contourArea(largest_contour)
                if largest_area > total_pixels * 0.02:  # Vùng sáng lớn
                    glare_ratio *= 2  # Tăng penalty
        
        return min(glare_ratio, 1.0)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Tính toán độ tương phản"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # RMS contrast
        mean = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean) ** 2))
        
        # Normalize to 0-1
        max_contrast = 127.5  # Theoretical maximum for 8-bit
        contrast_score = min(rms_contrast / max_contrast, 1.0)
        
        return contrast_score
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Tính toán độ sáng tối ưu"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Optimal brightness around 120-140 for documents
        optimal_range = (120, 140)
        
        if optimal_range[0] <= mean_brightness <= optimal_range[1]:
            brightness_score = 1.0
        else:
            # Calculate distance from optimal range
            if mean_brightness < optimal_range[0]:
                distance = optimal_range[0] - mean_brightness
            else:
                distance = mean_brightness - optimal_range[1]
            
            # Normalize distance (max penalty at 0 or 255)
            max_distance = max(optimal_range[0], 255 - optimal_range[1])
            brightness_score = max(0.0, 1.0 - distance / max_distance)
        
        return brightness_score
    
    def _calculate_noise(self, image: np.ndarray) -> float:
        """Tính toán mức độ nhiễu"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sử dụng median filter để ước tính noise
        median_filtered = cv2.medianBlur(gray, 5)
        noise = cv2.absdiff(gray, median_filtered)
        noise_level = np.mean(noise)
        
        # Normalize to 0-1
        max_noise = 50  # Threshold cho noise cao
        noise_score = min(noise_level / max_noise, 1.0)
        
        return noise_score
    
    def _calculate_edge_sharpness(self, image: np.ndarray) -> float:
        """Tính toán độ sắc nét của edges"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Calculate mean edge strength
        edge_strength = np.mean(sobel_magnitude)
        
        # Normalize to 0-1
        max_edge_strength = 100  # Threshold for strong edges
        sharpness_score = min(edge_strength / max_edge_strength, 1.0)
        
        return sharpness_score
    
    def _detect_watermark(self, image: np.ndarray) -> bool:
        """Phát hiện watermark đơn giản"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Tìm các vùng có độ trong suốt hoặc pattern đặc biệt
            # Sử dụng template matching cho một số watermark phổ biến
            
            # Kiểm tra pattern lặp lại (characteristic của watermark)
            # FFT để tìm frequency patterns
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Tìm peaks trong frequency domain
            # Watermark thường tạo ra periodic patterns
            mean_magnitude = np.mean(magnitude_spectrum)
            std_magnitude = np.std(magnitude_spectrum)
            
            # Simple heuristic: high variance in frequency domain might indicate watermark
            if std_magnitude > mean_magnitude * 0.8:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Tính toán điểm chất lượng tổng thể"""
        # Weights cho các metrics
        weights = {
            'blur': 0.25,
            'glare': 0.15,
            'contrast': 0.20,
            'brightness': 0.15,
            'noise': 0.15,
            'sharpness': 0.10
        }
        
        overall_score = sum(metrics[key] * weights[key] for key in weights.keys())
        return min(max(overall_score, 0.0), 1.0)
    
    def _get_default_metrics(self) -> Dict:
        """Trả về metrics mặc định khi có lỗi"""
        return {
            'overall_score': 0.0,
            'blur_score': 0.0,
            'glare_score': 1.0,
            'contrast_score': 0.0,
            'brightness_score': 0.0,
            'noise_score': 1.0,
            'edge_sharpness': 0.0,
            'watermark_present': False
        }
