from domain.entities.face_detection_result import FaceDetectionResult
from domain.repositories.face_detection_repository import FaceDetectionRepository
from domain.services.face_detection_service import FaceDetectionService
from typing import Optional
import logging

class FaceDetectionUseCase:
    """Use case cho face detection và alignment"""
    
    def __init__(self, 
                 face_detection_service: FaceDetectionService,
                 face_detection_repository: FaceDetectionRepository):
        self.face_detection_service = face_detection_service
        self.face_detection_repository = face_detection_repository
        self.logger = logging.getLogger(__name__)
    
    async def detect_and_process_face(self, image_path: str, source_type: str = 'selfie') -> FaceDetectionResult:
        """
        Phát hiện và xử lý khuôn mặt
        source_type: 'selfie' hoặc 'document'
        """
        try:
            self.logger.info(f"Processing face detection for: {image_path}")
            
            # Detect face
            face_result = self.face_detection_service.detect_and_align_face(image_path, source_type)
            
            # Save result to repository
            saved_result = await self.face_detection_repository.save(face_result)
            
            self.logger.info(f"Face detection completed with status: {face_result.status}")
            
            return saved_result
            
        except Exception as e:
            self.logger.error(f"Error in face detection use case: {e}")
            # Return failed result
            failed_result = FaceDetectionResult(
                image_path=image_path,
                status="failed",
                confidence=0.0
            )
            
            # Try to save failed result
            try:
                await self.face_detection_repository.save(failed_result)
            except:
                pass
            
            return failed_result
    
    async def get_face_detection_result(self, result_id: str) -> Optional[FaceDetectionResult]:
        """Lấy kết quả face detection theo ID"""
        try:
            return await self.face_detection_repository.find_by_id(result_id)
        except Exception as e:
            self.logger.error(f"Error getting face detection result: {e}")
            return None
    
    async def get_face_detections_by_image(self, image_path: str) -> list[FaceDetectionResult]:
        """Lấy tất cả face detection results của một ảnh"""
        try:
            return await self.face_detection_repository.find_by_image_path(image_path)
        except Exception as e:
            self.logger.error(f"Error getting face detections by image: {e}")
            return []
    
    async def validate_face_quality(self, image_path: str, source_type: str = 'selfie') -> dict:
        """
        Validate chất lượng khuôn mặt và đưa ra khuyến nghị
        """
        try:
            # Detect face
            face_result = self.face_detection_service.detect_and_align_face(image_path, source_type)
            
            # Get recommendations
            recommendations = self.face_detection_service.get_face_recommendations(face_result)
            
            # Validate face
            is_valid = self.face_detection_service.validate_face_detection(face_result)
            
            return {
                'is_valid': is_valid,
                'face_result': face_result,
                'recommendations': recommendations,
                'quality_score': face_result.face_quality_score,
                'alignment_score': face_result.alignment_score,
                'confidence': face_result.confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error validating face quality: {e}")
            return {
                'is_valid': False,
                'face_result': None,
                'recommendations': ['Lỗi xử lý ảnh. Vui lòng thử lại.'],
                'quality_score': 0.0,
                'alignment_score': 0.0,
                'confidence': 0.0
            }
    
    async def compare_face_alignment(self, image1_path: str, image2_path: str) -> dict:
        """So sánh alignment giữa hai khuôn mặt"""
        try:
            # Detect faces in both images
            face1 = self.face_detection_service.detect_and_align_face(image1_path, 'selfie')
            face2 = self.face_detection_service.detect_and_align_face(image2_path, 'document')
            
            # Compare alignment
            alignment_similarity = self.face_detection_service.compare_face_alignment(face1, face2)
            
            # Check if both faces are valid
            face1_valid = self.face_detection_service.validate_face_detection(face1)
            face2_valid = self.face_detection_service.validate_face_detection(face2)
            
            # Get recommendations for both faces
            face1_recommendations = self.face_detection_service.get_face_recommendations(face1)
            face2_recommendations = self.face_detection_service.get_face_recommendations(face2)
            
            return {
                'alignment_similarity': alignment_similarity,
                'face1_valid': face1_valid,
                'face2_valid': face2_valid,
                'face1_result': face1,
                'face2_result': face2,
                'face1_recommendations': face1_recommendations,
                'face2_recommendations': face2_recommendations,
                'overall_valid': face1_valid and face2_valid and alignment_similarity > 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing face alignment: {e}")
            return {
                'alignment_similarity': 0.0,
                'face1_valid': False,
                'face2_valid': False,
                'face1_result': None,
                'face2_result': None,
                'face1_recommendations': ['Lỗi xử lý ảnh'],
                'face2_recommendations': ['Lỗi xử lý ảnh'],
                'overall_valid': False
            }
    
    async def get_face_detection_statistics(self) -> dict:
        """Lấy thống kê face detection"""
        try:
            # Get all face detection results
            all_results = await self.face_detection_repository.find_all()
            
            if not all_results:
                return {
                    'total_detections': 0,
                    'success_rate': 0.0,
                    'average_confidence': 0.0,
                    'average_quality_score': 0.0,
                    'average_alignment_score': 0.0,
                    'status_distribution': {},
                    'occlusion_distribution': {}
                }
            
            # Calculate statistics
            total_detections = len(all_results)
            successful_detections = len([r for r in all_results if r.status == 'success'])
            success_rate = successful_detections / total_detections
            
            # Average scores (only for successful detections)
            successful_results = [r for r in all_results if r.status == 'success']
            
            if successful_results:
                avg_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
                avg_quality = sum(r.face_quality_score or 0 for r in successful_results) / len(successful_results)
                avg_alignment = sum(r.alignment_score or 0 for r in successful_results) / len(successful_results)
            else:
                avg_confidence = avg_quality = avg_alignment = 0.0
            
            # Status distribution
            status_counts = {}
            for result in all_results:
                status = result.status
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Occlusion distribution
            occlusion_counts = {}
            for result in all_results:
                if result.occlusion_detected:
                    occlusion_type = result.occlusion_type
                    occlusion_counts[occlusion_type] = occlusion_counts.get(occlusion_type, 0) + 1
            
            return {
                'total_detections': total_detections,
                'success_rate': round(success_rate * 100, 2),
                'average_confidence': round(avg_confidence, 3),
                'average_quality_score': round(avg_quality, 3),
                'average_alignment_score': round(avg_alignment, 3),
                'status_distribution': status_counts,
                'occlusion_distribution': occlusion_counts
            }
            
        except Exception as e:
            self.logger.error(f"Error getting face detection statistics: {e}")
            return {
                'total_detections': 0,
                'success_rate': 0.0,
                'average_confidence': 0.0,
                'average_quality_score': 0.0,
                'average_alignment_score': 0.0,
                'status_distribution': {},
                'occlusion_distribution': {},
                'error': str(e)
            }
