import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

class FaceDetector:
    """Face detector sử dụng OpenCV Haar Cascades và DNN models"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.face_cascade = None
        self.dnn_net = None
        self._load_models()
    
    def _load_models(self):
        """Load các models để detect face"""
        try:
            # Load Haar Cascade (backup method)
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            # Load DNN model (primary method) - OpenCV DNN face detection
            # Sử dụng pre-trained model từ OpenCV
            try:
                # Model files cần được download và đặt trong thư mục models
                model_path = Path(__file__).parent.parent.parent / "models" / "opencv_face_detector_uint8.pb"
                config_path = Path(__file__).parent.parent.parent / "models" / "opencv_face_detector.pbtxt"
                
                if model_path.exists() and config_path.exists():
                    self.dnn_net = cv2.dnn.readNetFromTensorflow(str(model_path), str(config_path))
                    self.logger.info("DNN face detection model loaded successfully")
                else:
                    self.logger.warning("DNN model files not found, using Haar Cascade only")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load DNN model: {e}")
                
        except Exception as e:
            self.logger.error(f"Failed to load face detection models: {e}")
    
    def detect_faces(self, image_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Phát hiện khuôn mặt trong ảnh
        Returns: {
            'faces': [{'bbox': [x1, y1, x2, y2], 'confidence': float}],
            'image_shape': (height, width, channels)
        }
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'faces': [], 'image_shape': None}
            
            height, width = image.shape[:2]
            
            # Thử DNN trước (nếu có)
            if self.dnn_net is not None:
                faces = self._detect_faces_dnn(image, confidence_threshold)
            else:
                faces = self._detect_faces_haar(image)
            
            return {
                'faces': faces,
                'image_shape': (height, width, image.shape[2] if len(image.shape) > 2 else 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting faces: {e}")
            return {'faces': [], 'image_shape': None}
    
    def _detect_faces_dnn(self, image: np.ndarray, confidence_threshold: float) -> List[Dict]:
        """Detect faces using DNN model"""
        try:
            height, width = image.shape[:2]
            
            # Prepare blob for DNN
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    
                    # Validate bbox
                    if self._validate_bbox([x1, y1, x2, y2], width, height):
                        faces.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence)
                        })
            
            return faces
            
        except Exception as e:
            self.logger.error(f"DNN face detection failed: {e}")
            return self._detect_faces_haar(image)
    
    def _detect_faces_haar(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using Haar Cascade (fallback method)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces_rect = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            faces = []
            for (x, y, w, h) in faces_rect:
                bbox = [x, y, x + w, y + h]
                if self._validate_bbox(bbox, image.shape[1], image.shape[0]):
                    faces.append({
                        'bbox': bbox,
                        'confidence': 0.8  # Default confidence for Haar
                    })
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Haar face detection failed: {e}")
            return []
    
    def _validate_bbox(self, bbox: List[int], image_width: int, image_height: int) -> bool:
        """Validate bounding box"""
        x1, y1, x2, y2 = bbox
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            return False
        
        # Check size
        width = x2 - x1
        height = y2 - y1
        if width < 20 or height < 20:
            return False
        
        # Check aspect ratio (faces should be roughly rectangular)
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        return True
    
    def detect_landmarks(self, image_path: str, bbox: List[int]) -> Optional[List[Tuple[float, float]]]:
        """
        Phát hiện landmarks trên khuôn mặt
        Returns: List of (x, y) coordinates for key landmarks
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            x1, y1, x2, y2 = bbox
            face_image = image[y1:y2, x1:x2]
            
            if face_image.size == 0:
                return None
            
            # Sử dụng method đơn giản để estimate landmarks
            landmarks = self._estimate_landmarks_simple(face_image, bbox)
            
            return landmarks
            
        except Exception as e:
            self.logger.error(f"Error detecting landmarks: {e}")
            return None
    
    def _estimate_landmarks_simple(self, face_image: np.ndarray, bbox: List[int]) -> List[Tuple[float, float]]:
        """
        Estimate landmarks sử dụng method đơn giản
        Returns landmarks for: left_eye, right_eye, nose, left_mouth, right_mouth
        """
        try:
            x1, y1, x2, y2 = bbox
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Convert to grayscale for feature detection
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Use Shi-Tomasi corner detection để tìm key points
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=50,
                qualityLevel=0.01,
                minDistance=10,
                blockSize=3
            )
            
            if corners is None or len(corners) < 5:
                # Fallback: estimate landmarks based on face proportions
                return self._estimate_landmarks_proportional(x1, y1, face_width, face_height)
            
            # Convert corners to absolute coordinates
            corners = corners.reshape(-1, 2)
            absolute_corners = [(x1 + x, y1 + y) for x, y in corners]
            
            # Select 5 best landmarks based on typical face proportions
            landmarks = self._select_best_landmarks(absolute_corners, x1, y1, face_width, face_height)
            
            return landmarks
            
        except Exception as e:
            self.logger.error(f"Error estimating landmarks: {e}")
            return self._estimate_landmarks_proportional(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
    
    def _estimate_landmarks_proportional(self, x1: int, y1: int, width: int, height: int) -> List[Tuple[float, float]]:
        """Estimate landmarks based on typical face proportions"""
        # Standard face proportions
        left_eye = (x1 + width * 0.3, y1 + height * 0.35)
        right_eye = (x1 + width * 0.7, y1 + height * 0.35)
        nose = (x1 + width * 0.5, y1 + height * 0.55)
        left_mouth = (x1 + width * 0.4, y1 + height * 0.75)
        right_mouth = (x1 + width * 0.6, y1 + height * 0.75)
        
        return [left_eye, right_eye, nose, left_mouth, right_mouth]
    
    def _select_best_landmarks(self, corners: List[Tuple[float, float]], 
                              x1: int, y1: int, width: int, height: int) -> List[Tuple[float, float]]:
        """Select 5 best corners as landmarks"""
        try:
            # Define regions for each landmark
            regions = {
                'left_eye': (x1 + width * 0.2, y1 + height * 0.25, x1 + width * 0.4, y1 + height * 0.45),
                'right_eye': (x1 + width * 0.6, y1 + height * 0.25, x1 + width * 0.8, y1 + height * 0.45),
                'nose': (x1 + width * 0.4, y1 + height * 0.45, x1 + width * 0.6, y1 + height * 0.65),
                'left_mouth': (x1 + width * 0.3, y1 + height * 0.65, x1 + width * 0.5, y1 + height * 0.85),
                'right_mouth': (x1 + width * 0.5, y1 + height * 0.65, x1 + width * 0.7, y1 + height * 0.85)
            }
            
            landmarks = []
            
            for region_name, (rx1, ry1, rx2, ry2) in regions.items():
                # Find corners in this region
                region_corners = []
                for x, y in corners:
                    if rx1 <= x <= rx2 and ry1 <= y <= ry2:
                        region_corners.append((x, y))
                
                if region_corners:
                    # Select corner closest to region center
                    center_x, center_y = (rx1 + rx2) / 2, (ry1 + ry2) / 2
                    best_corner = min(region_corners, 
                                    key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
                    landmarks.append(best_corner)
                else:
                    # Use region center as fallback
                    landmarks.append((center_x, center_y))
            
            return landmarks
            
        except Exception:
            # Final fallback
            return self._estimate_landmarks_proportional(x1, y1, width, height)
    
    def align_face(self, image_path: str, landmarks: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        """
        Căn chỉnh khuôn mặt dựa trên landmarks
        """
        try:
            image = cv2.imread(image_path)
            if image is None or len(landmarks) < 2:
                return None
            
            # Use eyes for alignment
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Calculate angle
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.arctan2(dy, dx) * 180.0 / np.pi
            
            # Calculate center point
            center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            
            # Get rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            return aligned
            
        except Exception as e:
            self.logger.error(f"Error aligning face: {e}")
            return None
    
    def crop_aligned_face(self, aligned_image: np.ndarray, landmarks: List[Tuple[float, float]], 
                         output_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Crop khuôn mặt đã căn chỉnh với kích thước chuẩn
        """
        try:
            if len(landmarks) < 2:
                return None
            
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            
            # Calculate face dimensions
            eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
            
            # Face size based on eye distance
            face_size = int(eye_distance * 2.5)
            
            # Face center
            face_center_x = (left_eye[0] + right_eye[0]) / 2
            face_center_y = (left_eye[1] + right_eye[1]) / 2 - eye_distance * 0.2  # Slightly above eyes
            
            # Crop coordinates
            x1 = int(face_center_x - face_size / 2)
            y1 = int(face_center_y - face_size / 2)
            x2 = int(face_center_x + face_size / 2)
            y2 = int(face_center_y + face_size / 2)
            
            # Ensure bounds
            h, w = aligned_image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Crop face
            cropped_face = aligned_image[y1:y2, x1:x2]
            
            if cropped_face.size == 0:
                return None
            
            # Resize to output size
            resized_face = cv2.resize(cropped_face, output_size)
            
            return resized_face
            
        except Exception as e:
            self.logger.error(f"Error cropping aligned face: {e}")
            return None
