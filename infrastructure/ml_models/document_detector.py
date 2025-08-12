import cv2
import numpy as np
from typing import Tuple, Optional
import logging

class DocumentDetector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect(self, image_path: str) -> Optional[list]:
        """
        Phát hiện vùng giấy tờ trong ảnh
        Returns: [x1, y1, x2, y2] hoặc None nếu không tìm thấy
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Cannot load image: {image_path}")
                return None
                
            # Tiền xử lý ảnh
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 75, 200)
            
            # Tìm contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            doc_contour = None
            max_area = 0
            image_area = image.shape[0] * image.shape[1]
            
            for c in contours:
                # Tính chu vi và xấp xỉ contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                area = cv2.contourArea(c)
                
                # Kiểm tra xem có phải hình chữ nhật và đủ lớn không
                if len(approx) == 4 and area > max_area and area > image_area * 0.1:
                    doc_contour = approx
                    max_area = area
                    
            if doc_contour is not None:
                x, y, w, h = cv2.boundingRect(doc_contour)
                return [x, y, x + w, y + h]
                
            # Fallback: tìm contour lớn nhất có thể là giấy tờ
            if contours:
                largest_contour = contours[0]
                area = cv2.contourArea(largest_contour)
                if area > image_area * 0.1:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    return [x, y, x + w, y + h]
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting document: {e}")
            return None

    def classify(self, image_path: str, bbox: Optional[list] = None) -> str:
        """
        Phân loại loại giấy tờ dựa trên tỷ lệ khung hình và các đặc trưng
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 'unknown'
                
            # Nếu có bbox, crop ảnh theo vùng giấy tờ
            if bbox:
                x1, y1, x2, y2 = bbox
                cropped = image[y1:y2, x1:x2]
                if cropped.size > 0:
                    image = cropped
                    
            h, w = image.shape[:2]
            aspect_ratio = w / h
            
            # Phân loại dựa trên tỷ lệ khung hình
            if 1.35 <= aspect_ratio <= 1.65:
                # Kiểm tra thêm màu sắc và pattern cho passport
                if self._is_passport_pattern(image):
                    return 'passport'
                    
            elif 1.5 <= aspect_ratio <= 1.9:
                # CMND/CCCD có tỷ lệ khác
                if self._is_id_card_pattern(image):
                    return 'cmnd'
                    
            # Thêm logic phân loại chi tiết hơn
            return self._advanced_classification(image, aspect_ratio)
            
        except Exception as e:
            self.logger.error(f"Error classifying document: {e}")
            return 'unknown'
    
    def _is_passport_pattern(self, image: np.ndarray) -> bool:
        """Kiểm tra pattern đặc trưng của passport"""
        # Passport thường có màu xanh đậm hoặc đỏ đậm
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Kiểm tra màu xanh (passport VN)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
        
        # Kiểm tra màu đỏ
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        red_ratio = np.sum(red_mask > 0) / red_mask.size
        
        return blue_ratio > 0.3 or red_ratio > 0.2
    
    def _is_id_card_pattern(self, image: np.ndarray) -> bool:
        """Kiểm tra pattern đặc trưng của CMND/CCCD"""
        # CMND/CCCD VN thường có nền trắng với text đen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Kiểm tra độ sáng trung bình (nền trắng)
        mean_brightness = np.mean(gray)
        
        # Kiểm tra contrast (có text)
        std_brightness = np.std(gray)
        
        return mean_brightness > 150 and std_brightness > 30
    
    def _advanced_classification(self, image: np.ndarray, aspect_ratio: float) -> str:
        """Phân loại nâng cao dựa trên nhiều đặc trưng"""
        
        # Kiểm tra kích thước tương đối
        h, w = image.shape[:2]
        
        if aspect_ratio > 1.8:
            return 'unknown'  # Quá dài
        elif aspect_ratio < 1.2:
            return 'unknown'  # Quá vuông
        elif 1.5 <= aspect_ratio <= 1.9:
            return 'cmnd'  # Tỷ lệ CMND/CCCD
        elif 1.35 <= aspect_ratio <= 1.65:
            return 'passport'  # Tỷ lệ passport
        else:
            return 'unknown'

    def get_document_confidence(self, image_path: str, bbox: Optional[list] = None) -> float:
        """
        Tính độ tin cậy của việc phát hiện giấy tờ
        Returns: confidence score từ 0.0 đến 1.0
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
                
            if bbox is None:
                bbox = self.detect(image_path)
                if bbox is None:
                    return 0.0
                    
            x1, y1, x2, y2 = bbox
            doc_area = (x2 - x1) * (y2 - y1)
            image_area = image.shape[0] * image.shape[1]
            area_ratio = doc_area / image_area
            
            # Tính confidence dựa trên tỷ lệ diện tích
            if area_ratio > 0.7:
                return 0.9
            elif area_ratio > 0.5:
                return 0.8
            elif area_ratio > 0.3:
                return 0.7
            elif area_ratio > 0.1:
                return 0.5
            else:
                return 0.3
                
        except Exception:
            return 0.0
