import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging
from PIL import Image
from PIL.ExifTags import TAGS
import os
import hashlib

class TamperDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_tampering(self, image_path: str, bbox: Optional[list] = None) -> Dict:
        """
        Phát hiện dấu hiệu chỉnh sửa/giả mạo
        Returns: Dictionary chứa kết quả phân tích tamper
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Crop ảnh theo bbox nếu có
            if bbox:
                x1, y1, x2, y2 = bbox
                image = image[y1:y2, x1:x2]
            
            # Các phương pháp phát hiện tamper
            compression_analysis = self._analyze_compression_artifacts(image)
            edge_analysis = self._analyze_edge_inconsistencies(image)
            noise_analysis = self._analyze_noise_patterns(image)
            metadata_analysis = self._analyze_metadata(image_path)
            color_analysis = self._analyze_color_inconsistencies(image)
            geometric_analysis = self._analyze_geometric_distortions(image)
            
            # Tính toán confidence tổng thể
            tamper_indicators = [
                compression_analysis['is_suspicious'],
                edge_analysis['is_suspicious'],
                noise_analysis['is_suspicious'],
                color_analysis['is_suspicious'],
                geometric_analysis['is_suspicious']
            ]
            
            tamper_confidence = sum([
                compression_analysis['confidence'],
                edge_analysis['confidence'],
                noise_analysis['confidence'],
                color_analysis['confidence'],
                geometric_analysis['confidence']
            ]) / 5.0
            
            is_tampered = sum(tamper_indicators) >= 2 or tamper_confidence > 0.7
            
            # Xác định loại tamper chính
            tamper_type = self._determine_tamper_type({
                'compression': compression_analysis,
                'edge': edge_analysis,
                'noise': noise_analysis,
                'color': color_analysis,
                'geometric': geometric_analysis
            })
            
            return {
                'is_tampered': is_tampered,
                'tamper_type': tamper_type,
                'confidence': tamper_confidence,
                'metadata_analysis': metadata_analysis,
                'detailed_analysis': {
                    'compression': compression_analysis,
                    'edge': edge_analysis,
                    'noise': noise_analysis,
                    'color': color_analysis,
                    'geometric': geometric_analysis
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting tampering: {e}")
            return self._get_default_tamper_result()
    
    def _analyze_compression_artifacts(self, image: np.ndarray) -> Dict:
        """Phân tích artifacts từ nén JPEG để phát hiện chỉnh sửa"""
        try:
            # Chuyển đổi sang YUV để phân tích compression
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            y_channel = yuv[:, :, 0]
            
            # Tính DCT để tìm blocking artifacts
            h, w = y_channel.shape
            block_size = 8
            artifacts_score = 0.0
            block_count = 0
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = y_channel[i:i+block_size, j:j+block_size].astype(np.float32)
                    
                    # Tính DCT
                    dct_block = cv2.dct(block)
                    
                    # Phân tích high frequency components
                    high_freq = np.sum(np.abs(dct_block[4:, 4:]))
                    total_energy = np.sum(np.abs(dct_block))
                    
                    if total_energy > 0:
                        hf_ratio = high_freq / total_energy
                        artifacts_score += hf_ratio
                        block_count += 1
            
            if block_count > 0:
                avg_artifacts = artifacts_score / block_count
                is_suspicious = avg_artifacts < 0.1  # Quá ít high freq có thể là do re-compression
                confidence = 1.0 - avg_artifacts if is_suspicious else avg_artifacts
            else:
                is_suspicious = False
                confidence = 0.0
            
            return {
                'is_suspicious': is_suspicious,
                'confidence': min(confidence, 1.0),
                'artifacts_score': avg_artifacts if block_count > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in compression analysis: {e}")
            return {'is_suspicious': False, 'confidence': 0.0, 'artifacts_score': 0.0}
    
    def _analyze_edge_inconsistencies(self, image: np.ndarray) -> Dict:
        """Phân tích sự không nhất quán của edges"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Phân tích edge strength variations
            # Chia ảnh thành các vùng và so sánh edge characteristics
            h, w = edges.shape
            grid_h, grid_w = h // 4, w // 4
            
            edge_densities = []
            for i in range(4):
                for j in range(4):
                    y1, y2 = i * grid_h, (i + 1) * grid_h
                    x1, x2 = j * grid_w, (j + 1) * grid_w
                    
                    region = edges[y1:y2, x1:x2]
                    edge_density = np.sum(region > 0) / (region.shape[0] * region.shape[1])
                    edge_densities.append(edge_density)
            
            # Tính độ biến thiên của edge density
            edge_variance = np.var(edge_densities)
            edge_mean = np.mean(edge_densities)
            
            # Phát hiện discontinuities
            discontinuity_score = self._detect_edge_discontinuities(edges)
            
            # Kết hợp các chỉ số
            is_suspicious = edge_variance > 0.01 or discontinuity_score > 0.3
            confidence = min((edge_variance * 10 + discontinuity_score) / 2, 1.0)
            
            return {
                'is_suspicious': is_suspicious,
                'confidence': confidence,
                'edge_variance': edge_variance,
                'discontinuity_score': discontinuity_score
            }
            
        except Exception as e:
            self.logger.error(f"Error in edge analysis: {e}")
            return {'is_suspicious': False, 'confidence': 0.0}
    
    def _detect_edge_discontinuities(self, edges: np.ndarray) -> float:
        """Phát hiện các điểm gián đoạn bất thường trong edges"""
        # Tìm contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        discontinuity_score = 0.0
        total_contours = len(contours)
        
        if total_contours == 0:
            return 0.0
        
        for contour in contours:
            if len(contour) < 10:  # Bỏ qua contour quá nhỏ
                continue
            
            # Tính độ smooth của contour
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.1:  # Contour không smooth
                    discontinuity_score += 1.0
        
        return min(discontinuity_score / max(total_contours, 1), 1.0)
    
    def _analyze_noise_patterns(self, image: np.ndarray) -> Dict:
        """Phân tích patterns nhiễu để phát hiện vùng được ghép"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Chia ảnh thành lưới và phân tích noise trong từng vùng
            h, w = gray.shape
            grid_size = 64
            noise_variations = []
            
            for i in range(0, h - grid_size, grid_size // 2):
                for j in range(0, w - grid_size, grid_size // 2):
                    region = gray[i:i+grid_size, j:j+grid_size]
                    
                    # Tính noise bằng high-pass filter
                    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                    noise_response = cv2.filter2D(region.astype(np.float32), -1, kernel)
                    noise_level = np.std(noise_response)
                    noise_variations.append(noise_level)
            
            if len(noise_variations) > 0:
                noise_variance = np.var(noise_variations)
                noise_mean = np.mean(noise_variations)
                
                # Normalize
                cv_noise = noise_variance / (noise_mean + 1e-6)
                
                is_suspicious = cv_noise > 0.5  # High variation indicates potential splicing
                confidence = min(cv_noise, 1.0)
            else:
                is_suspicious = False
                confidence = 0.0
            
            return {
                'is_suspicious': is_suspicious,
                'confidence': confidence,
                'noise_variance': noise_variance if len(noise_variations) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in noise analysis: {e}")
            return {'is_suspicious': False, 'confidence': 0.0}
    
    def _analyze_color_inconsistencies(self, image: np.ndarray) -> Dict:
        """Phân tích sự không nhất quán về màu sắc"""
        try:
            # Chuyển sang LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Phân tích distribution của màu sắc
            l_channel = lab[:, :, 0]
            a_channel = lab[:, :, 1]
            b_channel = lab[:, :, 2]
            
            # Chia ảnh thành vùng và phân tích histogram
            h, w = l_channel.shape
            grid_h, grid_w = h // 3, w // 3
            
            color_inconsistencies = []
            
            for i in range(3):
                for j in range(3):
                    y1, y2 = i * grid_h, (i + 1) * grid_h
                    x1, x2 = j * grid_w, (j + 1) * grid_w
                    
                    l_region = l_channel[y1:y2, x1:x2]
                    a_region = a_channel[y1:y2, x1:x2]
                    b_region = b_channel[y1:y2, x1:x2]
                    
                    # Tính statistics cho từng channel
                    l_stats = (np.mean(l_region), np.std(l_region))
                    a_stats = (np.mean(a_region), np.std(a_region))
                    b_stats = (np.mean(b_region), np.std(b_region))
                    
                    color_inconsistencies.append({
                        'l': l_stats,
                        'a': a_stats,
                        'b': b_stats
                    })
            
            # Tính variance của các statistics
            l_means = [stats['l'][0] for stats in color_inconsistencies]
            a_means = [stats['a'][0] for stats in color_inconsistencies]
            b_means = [stats['b'][0] for stats in color_inconsistencies]
            
            l_variance = np.var(l_means)
            a_variance = np.var(a_means)
            b_variance = np.var(b_means)
            
            total_variance = l_variance + a_variance + b_variance
            
            is_suspicious = total_variance > 1000  # Threshold for color inconsistency
            confidence = min(total_variance / 2000, 1.0)
            
            return {
                'is_suspicious': is_suspicious,
                'confidence': confidence,
                'color_variance': total_variance
            }
            
        except Exception as e:
            self.logger.error(f"Error in color analysis: {e}")
            return {'is_suspicious': False, 'confidence': 0.0}
    
    def _analyze_geometric_distortions(self, image: np.ndarray) -> Dict:
        """Phân tích méo hình học"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Tìm các đường thẳng bằng Hough Transform
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) < 4:
                return {'is_suspicious': False, 'confidence': 0.0}
            
            # Phân tích độ song song và vuông góc
            angles = []
            for line in lines:
                rho, theta = line[0]
                angles.append(theta)
            
            # Kiểm tra distribution của angles
            angles = np.array(angles)
            
            # Tìm các nhóm angle chính (horizontal, vertical)
            horizontal_angles = angles[np.abs(angles - np.pi/2) < 0.1]
            vertical_angles = angles[np.abs(angles) < 0.1]
            
            # Tính độ biến thiên trong từng nhóm
            h_variance = np.var(horizontal_angles) if len(horizontal_angles) > 1 else 0
            v_variance = np.var(vertical_angles) if len(vertical_angles) > 1 else 0
            
            total_variance = h_variance + v_variance
            
            is_suspicious = total_variance > 0.01  # High variance indicates distortion
            confidence = min(total_variance * 100, 1.0)
            
            return {
                'is_suspicious': is_suspicious,
                'confidence': confidence,
                'geometric_variance': total_variance
            }
            
        except Exception as e:
            self.logger.error(f"Error in geometric analysis: {e}")
            return {'is_suspicious': False, 'confidence': 0.0}
    
    def _analyze_metadata(self, image_path: str) -> Dict:
        """Phân tích metadata của ảnh"""
        try:
            # Đọc EXIF data
            image = Image.open(image_path)
            exifdata = image.getexif()
            
            metadata_info = {}
            suspicious_indicators = []
            
            for tag_id in exifdata:
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                metadata_info[tag] = str(data)
            
            # Kiểm tra các dấu hiệu suspicious
            if 'Software' in metadata_info:
                software = metadata_info['Software'].lower()
                editing_software = ['photoshop', 'gimp', 'paint', 'editor', 'adobe']
                if any(editor in software for editor in editing_software):
                    suspicious_indicators.append('editing_software_detected')
            
            # Kiểm tra timestamp inconsistencies
            if 'DateTime' in metadata_info and 'DateTimeOriginal' in metadata_info:
                # So sánh timestamps
                if metadata_info['DateTime'] != metadata_info['DateTimeOriginal']:
                    suspicious_indicators.append('timestamp_mismatch')
            
            # Kiểm tra missing metadata
            expected_fields = ['Make', 'Model', 'DateTime', 'ExifVersion']
            missing_fields = [field for field in expected_fields if field not in metadata_info]
            if len(missing_fields) > 2:
                suspicious_indicators.append('missing_metadata')
            
            return {
                'metadata': metadata_info,
                'suspicious_indicators': suspicious_indicators,
                'is_suspicious': len(suspicious_indicators) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing metadata: {e}")
            return {'metadata': {}, 'suspicious_indicators': [], 'is_suspicious': False}
    
    def _determine_tamper_type(self, analysis_results: Dict) -> str:
        """Xác định loại tamper chính dựa trên kết quả phân tích"""
        if analysis_results['compression']['is_suspicious']:
            return 'digital_manipulation'
        elif analysis_results['edge']['is_suspicious'] or analysis_results['noise']['is_suspicious']:
            return 'copy_paste'
        elif analysis_results['color']['is_suspicious']:
            return 'overlay_detected'
        elif analysis_results['geometric']['is_suspicious']:
            return 'physical_tampering'
        else:
            return 'none'
    
    def _get_default_tamper_result(self) -> Dict:
        """Trả về kết quả mặc định khi có lỗi"""
        return {
            'is_tampered': True,  # Conservative approach
            'tamper_type': 'none',
            'confidence': 0.0,
            'metadata_analysis': {},
            'detailed_analysis': {}
        }
