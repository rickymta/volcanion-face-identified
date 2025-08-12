import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class OcclusionDetector:
    """Phát hiện occlusion (che khuất) trên khuôn mặt"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_classifiers()
    
    def _load_classifiers(self):
        """Load các classifier để detect objects che khuất"""
        try:
            # Load Haar cascades for different objects
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            # Glasses detection (sử dụng eye cascade và phân tích pattern)
            self.glasses_patterns = self._init_glasses_patterns()
            
        except Exception as e:
            self.logger.error(f"Failed to load occlusion classifiers: {e}")
    
    def _init_glasses_patterns(self):
        """Initialize patterns để detect glasses"""
        return {
            'horizontal_lines': True,  # Detect horizontal lines across eyes
            'bridge_pattern': True,    # Detect bridge between eyes
            'reflection_pattern': True # Detect lens reflections
        }
    
    def detect_occlusion(self, image_path: str, face_bbox: List[int], 
                        landmarks: Optional[List[Tuple[float, float]]] = None) -> Dict:
        """
        Phát hiện occlusion trên khuôn mặt
        Returns: {
            'is_occluded': bool,
            'occlusion_type': str,
            'confidence': float,
            'occluded_regions': List[str]
        }
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return self._default_result()
            
            x1, y1, x2, y2 = face_bbox
            face_image = image[y1:y2, x1:x2]
            
            if face_image.size == 0:
                return self._default_result()
            
            # Detect different types of occlusion
            glasses_result = self._detect_glasses(face_image, landmarks)
            hat_result = self._detect_hat(image, face_bbox)
            hand_result = self._detect_hand_occlusion(face_image, landmarks)
            mask_result = self._detect_mask(face_image, landmarks)
            chin_support_result = self._detect_chin_support(face_image, landmarks)
            
            # Combine results
            all_results = [glasses_result, hat_result, hand_result, mask_result, chin_support_result]
            occluded_results = [r for r in all_results if r['is_occluded']]
            
            if not occluded_results:
                return self._default_result()
            
            # Get strongest detection
            best_result = max(occluded_results, key=lambda x: x['confidence'])
            
            # Collect all occluded regions
            occluded_regions = []
            for result in occluded_results:
                if result['regions']:
                    occluded_regions.extend(result['regions'])
            
            return {
                'is_occluded': True,
                'occlusion_type': best_result['type'],
                'confidence': best_result['confidence'],
                'occluded_regions': occluded_regions
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting occlusion: {e}")
            return self._default_result()
    
    def _default_result(self) -> Dict:
        """Default result when no occlusion detected"""
        return {
            'is_occluded': False,
            'occlusion_type': 'none',
            'confidence': 0.0,
            'occluded_regions': []
        }
    
    def _detect_glasses(self, face_image: np.ndarray, 
                       landmarks: Optional[List[Tuple[float, float]]]) -> Dict:
        """Detect glasses occlusion"""
        try:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Focus on eye region (upper 60% of face)
            eye_region = gray[:int(h * 0.6), :]
            
            confidence = 0.0
            regions = []
            
            # Method 1: Detect eyes and look for patterns
            eyes = self.eye_cascade.detectMultiScale(eye_region, 1.1, 5)
            
            if len(eyes) >= 2:
                # Sort eyes by x coordinate
                eyes = sorted(eyes, key=lambda x: x[0])
                
                # Analyze area between and around eyes
                left_eye = eyes[0]
                right_eye = eyes[-1]
                
                # Check for horizontal lines (glasses frame)
                horizontal_score = self._detect_horizontal_lines(eye_region, left_eye, right_eye)
                
                # Check for bridge pattern
                bridge_score = self._detect_bridge_pattern(eye_region, left_eye, right_eye)
                
                # Check for reflections (lens)
                reflection_score = self._detect_reflections(eye_region, left_eye, right_eye)
                
                confidence = (horizontal_score + bridge_score + reflection_score) / 3
                
                if confidence > 0.3:
                    regions = ['left_eye', 'right_eye']
            
            # Method 2: Edge detection for rectangular frames
            if confidence < 0.3:
                frame_confidence = self._detect_frame_edges(eye_region)
                confidence = max(confidence, frame_confidence)
                if frame_confidence > 0.3:
                    regions = ['eyes']
            
            return {
                'is_occluded': confidence > 0.3,
                'type': 'glasses',
                'confidence': confidence,
                'regions': regions
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting glasses: {e}")
            return {'is_occluded': False, 'type': 'glasses', 'confidence': 0.0, 'regions': []}
    
    def _detect_horizontal_lines(self, eye_region: np.ndarray, left_eye, right_eye) -> float:
        """Detect horizontal lines indicating glasses frame"""
        try:
            # Apply edge detection
            edges = cv2.Canny(eye_region, 50, 150)
            
            # Look for horizontal lines in eye area
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, 
                                   minLineLength=30, maxLineGap=10)
            
            if lines is None:
                return 0.0
            
            horizontal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Check if line is horizontal (within 15 degrees)
                if angle < 15 or angle > 165:
                    # Check if line is in eye region
                    line_y = (y1 + y2) / 2
                    if left_eye[1] <= line_y <= left_eye[1] + left_eye[3]:
                        horizontal_lines += 1
            
            return min(horizontal_lines / 2.0, 1.0)  # Normalize
            
        except Exception:
            return 0.0
    
    def _detect_bridge_pattern(self, eye_region: np.ndarray, left_eye, right_eye) -> float:
        """Detect bridge pattern between eyes"""
        try:
            # Region between eyes
            bridge_x1 = left_eye[0] + left_eye[2]
            bridge_x2 = right_eye[0]
            bridge_y1 = min(left_eye[1], right_eye[1])
            bridge_y2 = max(left_eye[1] + left_eye[3], right_eye[1] + right_eye[3])
            
            if bridge_x2 <= bridge_x1:
                return 0.0
            
            bridge_region = eye_region[bridge_y1:bridge_y2, bridge_x1:bridge_x2]
            
            if bridge_region.size == 0:
                return 0.0
            
            # Look for vertical edges (bridge)
            edges = cv2.Canny(bridge_region, 50, 150)
            vertical_edges = np.sum(edges) / (bridge_region.shape[0] * bridge_region.shape[1])
            
            return min(vertical_edges * 10, 1.0)  # Scale and normalize
            
        except Exception:
            return 0.0
    
    def _detect_reflections(self, eye_region: np.ndarray, left_eye, right_eye) -> float:
        """Detect lens reflections"""
        try:
            # Look for bright spots (reflections) in eye areas
            _, bright_spots = cv2.threshold(eye_region, 200, 255, cv2.THRESH_BINARY)
            
            reflection_score = 0.0
            
            for eye in [left_eye, right_eye]:
                x, y, w, h = eye
                eye_area = bright_spots[y:y+h, x:x+w]
                
                if eye_area.size > 0:
                    # Count white pixels (potential reflections)
                    white_ratio = np.sum(eye_area == 255) / (w * h)
                    if white_ratio > 0.1:  # At least 10% bright pixels
                        reflection_score += white_ratio
            
            return min(reflection_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _detect_frame_edges(self, eye_region: np.ndarray) -> float:
        """Detect rectangular frame edges"""
        try:
            edges = cv2.Canny(eye_region, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            frame_score = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 100:  # Too small
                    continue
                
                # Approximate contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's rectangular-ish (4-6 vertices)
                if 4 <= len(approx) <= 6:
                    # Check aspect ratio
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Glasses frames are usually wider than tall
                    if 1.5 <= aspect_ratio <= 4.0:
                        frame_score += 0.3
            
            return min(frame_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _detect_hat(self, image: np.ndarray, face_bbox: List[int]) -> Dict:
        """Detect hat/cap occlusion"""
        try:
            x1, y1, x2, y2 = face_bbox
            face_height = y2 - y1
            
            # Check region above face
            hat_y1 = max(0, y1 - int(face_height * 0.5))
            hat_y2 = y1 + int(face_height * 0.3)  # Include forehead
            hat_region = image[hat_y1:hat_y2, x1:x2]
            
            if hat_region.size == 0:
                return {'is_occluded': False, 'type': 'hat', 'confidence': 0.0, 'regions': []}
            
            gray = cv2.cvtColor(hat_region, cv2.COLOR_BGR2GRAY)
            
            # Look for horizontal edges (hat brim)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                   minLineLength=int((x2-x1)*0.3), maxLineGap=10)
            
            hat_confidence = 0.0
            
            if lines is not None:
                horizontal_lines = 0
                for line in lines:
                    x1_l, y1_l, x2_l, y2_l = line[0]
                    angle = abs(np.arctan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi)
                    
                    if angle < 15 or angle > 165:  # Horizontal line
                        horizontal_lines += 1
                
                hat_confidence = min(horizontal_lines / 3.0, 1.0)
            
            # Also check for uniform color regions (hat body)
            std_dev = np.std(gray)
            if std_dev < 30:  # Very uniform region
                hat_confidence = max(hat_confidence, 0.4)
            
            return {
                'is_occluded': hat_confidence > 0.3,
                'type': 'hat',
                'confidence': hat_confidence,
                'regions': ['forehead', 'top_head'] if hat_confidence > 0.3 else []
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting hat: {e}")
            return {'is_occluded': False, 'type': 'hat', 'confidence': 0.0, 'regions': []}
    
    def _detect_hand_occlusion(self, face_image: np.ndarray, 
                              landmarks: Optional[List[Tuple[float, float]]]) -> Dict:
        """Detect hand occlusion"""
        try:
            # Convert to HSV for skin detection
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            
            # Skin color range in HSV
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Remove noise
            kernel = np.ones((3,3), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            hand_confidence = 0.0
            regions = []
            h, w = face_image.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Hand should be reasonably sized
                if area < (w * h * 0.05):  # At least 5% of face
                    continue
                
                # Check shape (hands are not perfectly circular like faces)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    
                    # Hands typically have lower solidity due to fingers
                    if solidity < 0.8:
                        # Check position to determine which part is occluded
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            
                            # Determine occluded region based on position
                            if cy < h * 0.4:  # Upper part
                                regions.append('forehead')
                            elif cy > h * 0.7:  # Lower part
                                regions.append('mouth')
                            else:  # Middle part
                                if cx < w * 0.3:
                                    regions.append('left_cheek')
                                elif cx > w * 0.7:
                                    regions.append('right_cheek')
                                else:
                                    regions.append('nose')
                            
                            hand_confidence += 0.3
            
            return {
                'is_occluded': hand_confidence > 0.25,
                'type': 'hand',
                'confidence': min(hand_confidence, 1.0),
                'regions': regions
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting hand occlusion: {e}")
            return {'is_occluded': False, 'type': 'hand', 'confidence': 0.0, 'regions': []}
    
    def _detect_mask(self, face_image: np.ndarray, 
                    landmarks: Optional[List[Tuple[float, float]]]) -> Dict:
        """Detect face mask"""
        try:
            h, w = face_image.shape[:2]
            
            # Focus on lower half of face (nose and mouth area)
            lower_face = face_image[int(h*0.4):, :]
            
            gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
            
            # Look for edges that might indicate mask
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mask_confidence = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Mask should cover significant area
                if area > (lower_face.shape[0] * lower_face.shape[1] * 0.1):
                    
                    # Check if contour is roughly horizontal
                    rect = cv2.minAreaRect(contour)
                    (x, y), (width, height), angle = rect
                    
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    # Masks are typically wider than tall
                    if aspect_ratio > 1.5:
                        mask_confidence += 0.4
            
            # Also check for uniform color in mouth/nose area
            mouth_region = lower_face[int(lower_face.shape[0]*0.3):, 
                                    int(lower_face.shape[1]*0.2):int(lower_face.shape[1]*0.8)]
            
            if mouth_region.size > 0:
                std_dev = np.std(cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY))
                if std_dev < 25:  # Very uniform color
                    mask_confidence += 0.3
            
            return {
                'is_occluded': mask_confidence > 0.3,
                'type': 'mask',
                'confidence': min(mask_confidence, 1.0),
                'regions': ['nose', 'mouth'] if mask_confidence > 0.3 else []
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting mask: {e}")
            return {'is_occluded': False, 'type': 'mask', 'confidence': 0.0, 'regions': []}
    
    def _detect_chin_support(self, face_image: np.ndarray, 
                            landmarks: Optional[List[Tuple[float, float]]]) -> Dict:
        """Detect chin support (hand supporting chin)"""
        try:
            h, w = face_image.shape[:2]
            
            # Focus on lower part of face and below
            chin_region = face_image[int(h*0.6):, :]
            
            # Use skin detection similar to hand detection
            hsv = cv2.cvtColor(chin_region, cv2.COLOR_BGR2HSV)
            
            # Skin color range
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Find contours
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            chin_support_confidence = 0.0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Should be significant area in chin region
                if area > (chin_region.shape[0] * chin_region.shape[1] * 0.2):
                    
                    # Check if contour extends beyond expected chin area
                    x, y, cont_w, cont_h = cv2.boundingRect(contour)
                    
                    # If contour is wide and positioned at bottom, likely chin support
                    if cont_w > chin_region.shape[1] * 0.4 and y > chin_region.shape[0] * 0.3:
                        chin_support_confidence += 0.5
            
            return {
                'is_occluded': chin_support_confidence > 0.3,
                'type': 'chin_support',
                'confidence': min(chin_support_confidence, 1.0),
                'regions': ['chin', 'jaw'] if chin_support_confidence > 0.3 else []
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting chin support: {e}")
            return {'is_occluded': False, 'type': 'chin_support', 'confidence': 0.0, 'regions': []}
