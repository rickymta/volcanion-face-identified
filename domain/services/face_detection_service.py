from domain.entities.face_detection_result import FaceDetectionResult, FaceDetectionStatus, OcclusionType
from typing import List, Tuple, Optional
import logging

class FaceDetectionService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_and_align_face(self, image_path: str, source_type: str = 'selfie') -> FaceDetectionResult:
        """
        Phát hiện và căn chỉnh khuôn mặt
        source_type: 'selfie' hoặc 'document'
        """
        try:
            from infrastructure.ml_models.face_detector import FaceDetector
            from infrastructure.ml_models.occlusion_detector import OcclusionDetector
            
            face_detector = FaceDetector()
            occlusion_detector = OcclusionDetector()
            
            # Phát hiện khuôn mặt
            detection_result = face_detector.detect_faces(image_path)
            
            if not detection_result['faces']:
                return FaceDetectionResult(
                    image_path=image_path,
                    status=FaceDetectionStatus.NO_FACE_DETECTED,
                    confidence=0.0
                )
            
            # Lấy khuôn mặt có confidence cao nhất
            best_face = max(detection_result['faces'], key=lambda x: x['confidence'])
            
            # Kiểm tra số lượng khuôn mặt
            if len(detection_result['faces']) > 1:
                status = FaceDetectionStatus.MULTIPLE_FACES
            else:
                status = FaceDetectionStatus.SUCCESS
            
            # Phát hiện landmarks
            landmarks = face_detector.detect_landmarks(image_path, best_face['bbox'])
            
            # Tính toán alignment score
            alignment_score = self._calculate_alignment_score(landmarks) if landmarks else 0.0
            
            # Tính toán face quality
            face_quality_score = self._calculate_face_quality(image_path, best_face['bbox'])
            
            # Phát hiện occlusion
            occlusion_result = occlusion_detector.detect_occlusion(image_path, best_face['bbox'], landmarks)
            
            # Tính toán pose angles
            pose_angles = self._calculate_pose_angles(landmarks) if landmarks else None
            
            # Kiểm tra kích thước khuôn mặt
            face_area = (best_face['bbox'][2] - best_face['bbox'][0]) * (best_face['bbox'][3] - best_face['bbox'][1])
            if face_area < 2500:  # Khuôn mặt quá nhỏ (50x50 pixels)
                status = FaceDetectionStatus.FACE_TOO_SMALL
            
            # Kiểm tra độ mờ của khuôn mặt
            if face_quality_score < 0.3:
                status = FaceDetectionStatus.FACE_TOO_BLURRY
            
            # Kiểm tra occlusion
            if occlusion_result['is_occluded']:
                status = FaceDetectionStatus.FACE_OCCLUDED
            
            return FaceDetectionResult(
                image_path=image_path,
                status=status,
                bbox=best_face['bbox'],
                landmarks=landmarks,
                confidence=best_face['confidence'],
                face_size=(best_face['bbox'][2] - best_face['bbox'][0], 
                          best_face['bbox'][3] - best_face['bbox'][1]),
                occlusion_detected=occlusion_result['is_occluded'],
                occlusion_type=OcclusionType(occlusion_result['occlusion_type']),
                occlusion_confidence=occlusion_result['confidence'],
                alignment_score=alignment_score,
                face_quality_score=face_quality_score,
                pose_angles=pose_angles
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting face: {e}")
            return FaceDetectionResult(
                image_path=image_path,
                status=FaceDetectionStatus.FAILED,
                confidence=0.0
            )
    
    def _calculate_alignment_score(self, landmarks: List[Tuple[float, float]]) -> float:
        """Tính toán điểm alignment dựa trên landmarks"""
        if not landmarks or len(landmarks) < 5:
            return 0.0
        
        try:
            # Landmarks thường theo thứ tự: left_eye, right_eye, nose, left_mouth, right_mouth
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            # Tính độ nghiêng của mắt
            eye_angle = abs((right_eye[1] - left_eye[1]) / (right_eye[0] - left_eye[0] + 1e-6))
            
            # Tính đối xứng của khuôn mặt
            face_center_x = (left_eye[0] + right_eye[0]) / 2
            nose_deviation = abs(nose[0] - face_center_x) / abs(right_eye[0] - left_eye[0] + 1e-6)
            
            # Tính đối xứng của miệng
            mouth_center_x = (left_mouth[0] + right_mouth[0]) / 2
            mouth_deviation = abs(mouth_center_x - face_center_x) / abs(right_eye[0] - left_eye[0] + 1e-6)
            
            # Tính alignment score (higher is better)
            alignment_score = 1.0 - min(1.0, eye_angle + nose_deviation + mouth_deviation)
            
            return max(0.0, alignment_score)
            
        except Exception:
            return 0.0
    
    def _calculate_face_quality(self, image_path: str, bbox: List[int]) -> float:
        """Tính toán chất lượng khuôn mặt"""
        try:
            import cv2
            import numpy as np
            
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
            
            # Crop khuôn mặt
            x1, y1, x2, y2 = bbox
            face_image = image[y1:y2, x1:x2]
            
            if face_image.size == 0:
                return 0.0
            
            # Tính độ sắc nét bằng Laplacian variance
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize (threshold based on typical face images)
            sharpness_score = min(laplacian_var / 500.0, 1.0)
            
            # Tính contrast
            contrast = gray.std() / 255.0
            
            # Tính brightness (optimal around 0.4-0.7)
            brightness = gray.mean() / 255.0
            brightness_score = 1.0 - abs(brightness - 0.55) / 0.45
            brightness_score = max(0.0, brightness_score)
            
            # Kết hợp các scores
            quality_score = (sharpness_score * 0.5 + contrast * 0.3 + brightness_score * 0.2)
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_pose_angles(self, landmarks: List[Tuple[float, float]]) -> Optional[Tuple[float, float, float]]:
        """Tính toán góc pose (yaw, pitch, roll) từ landmarks"""
        if not landmarks or len(landmarks) < 5:
            return None
        
        try:
            import numpy as np
            
            # Landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            left_eye = np.array(landmarks[0])
            right_eye = np.array(landmarks[1])
            nose = np.array(landmarks[2])
            left_mouth = np.array(landmarks[3])
            right_mouth = np.array(landmarks[4])
            
            # Tính roll angle (góc nghiêng)
            eye_vector = right_eye - left_eye
            roll = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
            
            # Tính yaw angle (góc quay trái/phải)
            face_center = (left_eye + right_eye) / 2
            nose_deviation = nose[0] - face_center[0]
            eye_distance = np.linalg.norm(eye_vector)
            yaw = np.arctan2(nose_deviation, eye_distance) * 180 / np.pi
            
            # Tính pitch angle (góc ngước/cúi)
            mouth_center = (left_mouth + right_mouth) / 2
            vertical_distance = mouth_center[1] - face_center[1]
            face_height = abs(mouth_center[1] - face_center[1]) + eye_distance * 0.5
            pitch = np.arctan2(vertical_distance, face_height) * 180 / np.pi
            
            return (float(yaw), float(pitch), float(roll))
            
        except Exception:
            return None
    
    def validate_face_detection(self, face_result: FaceDetectionResult) -> bool:
        """Kiểm tra tính hợp lệ của face detection result"""
        return face_result.is_acceptable()
    
    def compare_face_alignment(self, face1: FaceDetectionResult, face2: FaceDetectionResult) -> float:
        """So sánh alignment giữa hai khuôn mặt"""
        if not face1.landmarks or not face2.landmarks:
            return 0.0
        
        try:
            import numpy as np
            
            # Normalize landmarks theo kích thước khuôn mặt
            def normalize_landmarks(landmarks, bbox):
                x1, y1, x2, y2 = bbox
                width, height = x2 - x1, y2 - y1
                normalized = []
                for x, y in landmarks:
                    norm_x = (x - x1) / width
                    norm_y = (y - y1) / height
                    normalized.append((norm_x, norm_y))
                return normalized
            
            norm_landmarks1 = normalize_landmarks(face1.landmarks, face1.bbox)
            norm_landmarks2 = normalize_landmarks(face2.landmarks, face2.bbox)
            
            # Tính khoảng cách Euclidean giữa các landmarks
            total_distance = 0.0
            for (x1, y1), (x2, y2) in zip(norm_landmarks1, norm_landmarks2):
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                total_distance += distance
            
            # Normalize distance (càng nhỏ càng giống)
            avg_distance = total_distance / len(norm_landmarks1)
            similarity_score = max(0.0, 1.0 - avg_distance * 2)  # Scale factor
            
            return similarity_score
            
        except Exception:
            return 0.0
    
    def get_face_recommendations(self, face_result: FaceDetectionResult) -> List[str]:
        """Đưa ra khuyến nghị cải thiện khuôn mặt"""
        recommendations = []
        
        if face_result.status == FaceDetectionStatus.NO_FACE_DETECTED:
            recommendations.append("Không phát hiện được khuôn mặt. Hãy đảm bảo khuôn mặt hiện rõ trong ảnh")
        
        elif face_result.status == FaceDetectionStatus.MULTIPLE_FACES:
            recommendations.append("Phát hiện nhiều khuôn mặt. Hãy chụp ảnh chỉ có một người")
        
        elif face_result.status == FaceDetectionStatus.FACE_TOO_SMALL:
            recommendations.append("Khuôn mặt quá nhỏ. Hãy di chuyển gần camera hơn")
        
        elif face_result.status == FaceDetectionStatus.FACE_TOO_BLURRY:
            recommendations.append("Khuôn mặt bị mờ. Hãy giữ camera ổn định và đảm bảo ánh sáng đủ")
        
        elif face_result.status == FaceDetectionStatus.FACE_OCCLUDED:
            if face_result.occlusion_type == OcclusionType.GLASSES:
                recommendations.append("Phát hiện kính mắt. Hãy tháo kính nếu có thể")
            elif face_result.occlusion_type == OcclusionType.HAT:
                recommendations.append("Phát hiện mũ/nón. Hãy tháo mũ để lộ rõ khuôn mặt")
            elif face_result.occlusion_type == OcclusionType.HAND:
                recommendations.append("Phát hiện tay che khuôn mặt. Hãy để tay xuống")
            elif face_result.occlusion_type == OcclusionType.CHIN_SUPPORT:
                recommendations.append("Phát hiện tay chống cằm. Hãy để tay xuống")
            elif face_result.occlusion_type == OcclusionType.MASK:
                recommendations.append("Phát hiện khẩu trang. Hãy tháo khẩu trang")
            else:
                recommendations.append("Khuôn mặt bị che khuất. Hãy đảm bảo khuôn mặt hiện rõ hoàn toàn")
        
        if face_result.confidence < 0.7:
            recommendations.append("Độ tin cậy nhận diện thấp. Hãy cải thiện điều kiện ánh sáng")
        
        if face_result.alignment_score < 0.5:
            recommendations.append("Khuôn mặt không thẳng. Hãy nhìn thẳng vào camera")
        
        if face_result.face_quality_score < 0.6:
            recommendations.append("Chất lượng khuôn mặt thấp. Hãy cải thiện ánh sáng và độ sắc nét")
        
        if face_result.pose_angles:
            yaw, pitch, roll = face_result.pose_angles
            if abs(yaw) > 15:
                recommendations.append("Khuôn mặt quay nghiêng. Hãy nhìn thẳng vào camera")
            if abs(pitch) > 15:
                recommendations.append("Khuôn mặt ngước/cúi quá nhiều. Hãy giữ đầu thẳng")
            if abs(roll) > 10:
                recommendations.append("Khuôn mặt nghiêng. Hãy giữ đầu thẳng")
        
        return recommendations
