import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
import logging
from dataclasses import dataclass
from datetime import datetime

from domain.entities.ocr_result import (
    TextRegion, BoundingBox, ExtractedField, OCRStatistics, 
    DocumentType, FieldType, OCRStatus
)

# Configure logging
logger = logging.getLogger(__name__)

# Import optional libraries với fallback
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Using basic OCR.")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    logger.warning("Pytesseract not available.")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Limited image preprocessing.")

@dataclass
class OCRConfig:
    """Configuration cho OCR engine"""
    confidence_threshold: float = 0.5
    languages: List[str] = None
    preprocess_image: bool = True
    use_gpu: bool = False
    use_multiple_engines: bool = True
    enable_field_detection: bool = True
    enable_postprocessing: bool = True
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['vi', 'en']

class ImagePreprocessor:
    """Class xử lý tiền xử lý ảnh cho OCR"""
    
    @staticmethod
    def enhance_image(image: np.ndarray) -> np.ndarray:
        """Enhance ảnh để OCR tốt hơn"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Sharpening
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            logger.error(f"Error in image enhancement: {e}")
            return image
    
    @staticmethod
    def correct_skew(image: np.ndarray) -> np.ndarray:
        """Sửa skew của ảnh"""
        try:
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using HoughLines
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                
                if angles:
                    avg_angle = np.mean(angles)
                    
                    # Rotate image
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h), 
                                           flags=cv2.INTER_CUBIC, 
                                           borderMode=cv2.BORDER_REPLICATE)
                    return rotated
            
            return image
            
        except Exception as e:
            logger.error(f"Error in skew correction: {e}")
            return image
    
    @staticmethod
    def remove_noise(image: np.ndarray) -> np.ndarray:
        """Loại bỏ noise"""
        try:
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error in noise removal: {e}")
            return image
    
    @staticmethod
    def binarize(image: np.ndarray) -> np.ndarray:
        """Binarize ảnh"""
        try:
            gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            return binary
            
        except Exception as e:
            logger.error(f"Error in binarization: {e}")
            return image

class TextFieldDetector:
    """Class phát hiện và classify text fields"""
    
    # Patterns cho các field types
    FIELD_PATTERNS = {
        FieldType.ID_NUMBER: [
            r'\b\d{9}\b',  # CMND 9 digits
            r'\b\d{12}\b',  # CCCD 12 digits
            r'[A-Z]\d{8}',  # Passport pattern
        ],
        FieldType.DATE_OF_BIRTH: [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
        ],
        FieldType.FULL_NAME: [
            r'[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+\s+[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ][a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]*'
        ],
        FieldType.ADDRESS: [
            r'(?i)(số|st|street|đường|phường|quận|huyện|tỉnh|thành phố)'
        ],
        FieldType.GENDER: [
            r'\b(Nam|Nữ|Male|Female|M|F)\b'
        ]
    }
    
    @classmethod
    def detect_field_type(cls, text: str) -> Optional[FieldType]:
        """Phát hiện field type từ text"""
        for field_type, patterns in cls.FIELD_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return field_type
        return None
    
    @classmethod
    def validate_field(cls, field_type: FieldType, value: str) -> Tuple[bool, List[str]]:
        """Validate field value"""
        errors = []
        
        if field_type == FieldType.ID_NUMBER:
            if not re.match(r'^\d{9}$|^\d{12}$|^[A-Z]\d{8}$', value):
                errors.append("Invalid ID number format")
        
        elif field_type == FieldType.DATE_OF_BIRTH:
            if not re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{4}$|^\d{4}[/-]\d{1,2}[/-]\d{1,2}$', value):
                errors.append("Invalid date format")
        
        elif field_type == FieldType.GENDER:
            if value.upper() not in ['NAM', 'NỮ', 'MALE', 'FEMALE', 'M', 'F']:
                errors.append("Invalid gender value")
        
        return len(errors) == 0, errors
    
    @classmethod
    def normalize_field_value(cls, field_type: FieldType, value: str) -> str:
        """Normalize field value"""
        normalized = value.strip()
        
        if field_type == FieldType.FULL_NAME:
            # Proper case for names
            normalized = ' '.join([word.capitalize() for word in normalized.split()])
        
        elif field_type == FieldType.ID_NUMBER:
            # Remove spaces and special characters
            normalized = re.sub(r'[^\dA-Z]', '', normalized.upper())
        
        elif field_type == FieldType.GENDER:
            # Standardize gender
            gender_map = {
                'NAM': 'Nam', 'MALE': 'Nam', 'M': 'Nam',
                'NỮ': 'Nữ', 'FEMALE': 'Nữ', 'F': 'Nữ'
            }
            normalized = gender_map.get(normalized.upper(), normalized)
        
        return normalized

class OCREngine:
    """Main OCR engine"""
    
    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self.preprocessor = ImagePreprocessor()
        self.field_detector = TextFieldDetector()
        
        # Initialize OCR readers
        self.readers = {}
        self._init_readers()
    
    def _init_readers(self):
        """Initialize OCR readers"""
        try:
            if EASYOCR_AVAILABLE:
                self.readers['easyocr'] = easyocr.Reader(
                    self.config.languages, 
                    gpu=self.config.use_gpu
                )
                logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {e}")
        
        # Note: Pytesseract will be initialized per-call if available
    
    def extract_text(self, image: np.ndarray) -> List[TextRegion]:
        """Extract text từ ảnh"""
        start_time = datetime.now()
        
        try:
            # Preprocess image
            if self.config.preprocess_image:
                processed_image = self._preprocess_image(image)
            else:
                processed_image = image
            
            # Extract with multiple engines
            all_regions = []
            
            if 'easyocr' in self.readers:
                regions = self._extract_with_easyocr(processed_image)
                all_regions.extend(regions)
            
            if PYTESSERACT_AVAILABLE:
                regions = self._extract_with_tesseract(processed_image)
                all_regions.extend(regions)
            
            # Merge and filter results
            merged_regions = self._merge_overlapping_regions(all_regions)
            filtered_regions = self._filter_by_confidence(merged_regions)
            
            # Add field type detection
            if self.config.enable_field_detection:
                for region in filtered_regions:
                    region.field_type = self.field_detector.detect_field_type(region.text)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(f"Text extraction completed in {processing_time:.2f}ms")
            
            return filtered_regions
            
        except Exception as e:
            logger.error(f"Error in text extraction: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess ảnh"""
        processed = image.copy()
        
        # Apply preprocessing steps
        processed = self.preprocessor.enhance_image(processed)
        processed = self.preprocessor.correct_skew(processed)
        processed = self.preprocessor.remove_noise(processed)
        
        return processed
    
    def _extract_with_easyocr(self, image: np.ndarray) -> List[TextRegion]:
        """Extract text với EasyOCR"""
        try:
            reader = self.readers['easyocr']
            results = reader.readtext(image)
            
            regions = []
            for bbox_coords, text, confidence in results:
                if confidence >= self.config.confidence_threshold:
                    # Convert bbox format
                    x1 = int(min([point[0] for point in bbox_coords]))
                    y1 = int(min([point[1] for point in bbox_coords]))
                    x2 = int(max([point[0] for point in bbox_coords]))
                    y2 = int(max([point[1] for point in bbox_coords]))
                    
                    bbox = BoundingBox(x1, y1, x2, y2)
                    region = TextRegion(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox,
                        language='auto'
                    )
                    regions.append(region)
            
            return regions
            
        except Exception as e:
            logger.error(f"Error in EasyOCR extraction: {e}")
            return []
    
    def _extract_with_tesseract(self, image: np.ndarray) -> List[TextRegion]:
        """Extract text với Tesseract"""
        try:
            if not PYTESSERACT_AVAILABLE:
                return []
            
            # Convert image for tesseract
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Get detailed results
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            regions = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                confidence = float(data['conf'][i])
                text = data['text'][i].strip()
                
                if confidence >= self.config.confidence_threshold * 100 and text:
                    x1 = data['left'][i]
                    y1 = data['top'][i]
                    x2 = x1 + data['width'][i]
                    y2 = y1 + data['height'][i]
                    
                    bbox = BoundingBox(x1, y1, x2, y2)
                    region = TextRegion(
                        text=text,
                        confidence=confidence / 100,  # Convert to 0-1 range
                        bbox=bbox,
                        language='auto'
                    )
                    regions.append(region)
            
            return regions
            
        except Exception as e:
            logger.error(f"Error in Tesseract extraction: {e}")
            return []
    
    def _merge_overlapping_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Merge overlapping text regions"""
        if not regions:
            return regions
        
        # Sort by position
        sorted_regions = sorted(regions, key=lambda r: (r.bbox.y1, r.bbox.x1))
        merged = []
        
        for region in sorted_regions:
            if not merged:
                merged.append(region)
                continue
            
            # Check overlap with last merged region
            last_region = merged[-1]
            if self._calculate_overlap(region.bbox, last_region.bbox) > 0.5:
                # Merge regions
                merged_region = self._merge_two_regions(last_region, region)
                merged[-1] = merged_region
            else:
                merged.append(region)
        
        return merged
    
    def _calculate_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Tính overlap ratio giữa 2 bounding boxes"""
        # Calculate intersection
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1.get_area()
        area2 = bbox2.get_area()
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_two_regions(self, region1: TextRegion, region2: TextRegion) -> TextRegion:
        """Merge 2 text regions"""
        # Choose region with higher confidence
        primary = region1 if region1.confidence >= region2.confidence else region2
        secondary = region2 if region1.confidence >= region2.confidence else region1
        
        # Merge bounding boxes
        merged_bbox = BoundingBox(
            min(region1.bbox.x1, region2.bbox.x1),
            min(region1.bbox.y1, region2.bbox.y1),
            max(region1.bbox.x2, region2.bbox.x2),
            max(region1.bbox.y2, region2.bbox.y2)
        )
        
        # Merge text
        merged_text = f"{primary.text} {secondary.text}".strip()
        
        return TextRegion(
            text=merged_text,
            confidence=max(region1.confidence, region2.confidence),
            bbox=merged_bbox,
            field_type=primary.field_type,
            language=primary.language
        )
    
    def _filter_by_confidence(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Filter regions theo confidence threshold"""
        return [
            region for region in regions 
            if region.confidence >= self.config.confidence_threshold
        ]
    
    def extract_fields(self, text_regions: List[TextRegion]) -> List[ExtractedField]:
        """Extract structured fields từ text regions"""
        fields = []
        
        for region in text_regions:
            if region.field_type:
                # Validate and normalize
                is_valid, errors = self.field_detector.validate_field(
                    region.field_type, region.text
                )
                
                normalized_value = self.field_detector.normalize_field_value(
                    region.field_type, region.text
                )
                
                field = ExtractedField(
                    field_type=region.field_type,
                    value=normalized_value,
                    confidence=region.confidence,
                    bbox=region.bbox,
                    raw_text=region.text,
                    normalized_value=normalized_value,
                    validation_status=is_valid,
                    validation_errors=errors
                )
                
                fields.append(field)
        
        return fields
    
    def calculate_statistics(self, text_regions: List[TextRegion]) -> OCRStatistics:
        """Tính statistics cho OCR result"""
        if not text_regions:
            return OCRStatistics(
                total_text_regions=0,
                reliable_regions=0,
                unreliable_regions=0,
                average_confidence=0.0,
                total_characters=0,
                processing_time_ms=0.0,
                languages_detected=[]
            )
        
        reliable_regions = [r for r in text_regions if r.is_reliable()]
        unreliable_regions = len(text_regions) - len(reliable_regions)
        
        avg_confidence = sum(r.confidence for r in text_regions) / len(text_regions)
        total_chars = sum(len(r.text) for r in text_regions)
        
        languages = list(set([r.language for r in text_regions if r.language]))
        
        return OCRStatistics(
            total_text_regions=len(text_regions),
            reliable_regions=len(reliable_regions),
            unreliable_regions=unreliable_regions,
            average_confidence=avg_confidence,
            total_characters=total_chars,
            processing_time_ms=0.0,  # Will be set by caller
            languages_detected=languages
        )
