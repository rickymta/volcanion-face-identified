import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from domain.entities.face_detection_result import FaceDetectionResult, FaceDetectionStatus, OcclusionType
from domain.services.face_detection_service import FaceDetectionService

class TestFaceDetectionService:
    
    @pytest.fixture
    def face_detection_service(self):
        return FaceDetectionService()
    
    @pytest.fixture
    def sample_image_path(self):
        # Create a temporary image file for testing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            # Write minimal image data (just for testing)
            temp_file.write(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # JPEG header
            return temp_file.name
    
    def teardown_method(self):
        # Clean up any temporary files
        pass
    
    @patch('domain.services.face_detection_service.FaceDetector')
    @patch('domain.services.face_detection_service.OcclusionDetector')
    def test_detect_and_align_face_success(self, mock_occlusion_detector, mock_face_detector, 
                                          face_detection_service, sample_image_path):
        """Test successful face detection"""
        # Mock face detector
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance
        
        # Mock detection result
        mock_detector_instance.detect_faces.return_value = {
            'faces': [
                {
                    'bbox': [100, 100, 200, 200],
                    'confidence': 0.95
                }
            ]
        }
        
        # Mock landmarks
        mock_detector_instance.detect_landmarks.return_value = [
            (120, 130),  # left_eye
            (180, 130),  # right_eye
            (150, 160),  # nose
            (130, 180),  # left_mouth
            (170, 180)   # right_mouth
        ]
        
        # Mock occlusion detector
        mock_occlusion_instance = Mock()
        mock_occlusion_detector.return_value = mock_occlusion_instance
        mock_occlusion_instance.detect_occlusion.return_value = {
            'is_occluded': False,
            'occlusion_type': 'none',
            'confidence': 0.0
        }
        
        # Test detection
        result = face_detection_service.detect_and_align_face(sample_image_path, 'selfie')
        
        # Assertions
        assert result.image_path == sample_image_path
        assert result.status == FaceDetectionStatus.SUCCESS
        assert result.bbox == [100, 100, 200, 200]
        assert result.confidence == 0.95
        assert result.landmarks is not None
        assert len(result.landmarks) == 5
        assert not result.occlusion_detected
        assert result.occlusion_type == OcclusionType.NONE
    
    @patch('domain.services.face_detection_service.FaceDetector')
    @patch('domain.services.face_detection_service.OcclusionDetector')
    def test_detect_and_align_face_no_face_detected(self, mock_occlusion_detector, mock_face_detector,
                                                   face_detection_service, sample_image_path):
        """Test when no face is detected"""
        # Mock face detector
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance
        
        # Mock no faces detected
        mock_detector_instance.detect_faces.return_value = {'faces': []}
        
        # Test detection
        result = face_detection_service.detect_and_align_face(sample_image_path, 'selfie')
        
        # Assertions
        assert result.status == FaceDetectionStatus.NO_FACE_DETECTED
        assert result.confidence == 0.0
        assert result.bbox is None
        assert result.landmarks is None
    
    @patch('domain.services.face_detection_service.FaceDetector')
    @patch('domain.services.face_detection_service.OcclusionDetector')
    def test_detect_and_align_face_multiple_faces(self, mock_occlusion_detector, mock_face_detector,
                                                 face_detection_service, sample_image_path):
        """Test when multiple faces are detected"""
        # Mock face detector
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance
        
        # Mock multiple faces detected
        mock_detector_instance.detect_faces.return_value = {
            'faces': [
                {'bbox': [100, 100, 200, 200], 'confidence': 0.85},
                {'bbox': [300, 100, 400, 200], 'confidence': 0.90}
            ]
        }
        
        mock_detector_instance.detect_landmarks.return_value = [
            (120, 130), (180, 130), (150, 160), (130, 180), (170, 180)
        ]
        
        # Mock occlusion detector
        mock_occlusion_instance = Mock()
        mock_occlusion_detector.return_value = mock_occlusion_instance
        mock_occlusion_instance.detect_occlusion.return_value = {
            'is_occluded': False,
            'occlusion_type': 'none',
            'confidence': 0.0
        }
        
        # Test detection
        result = face_detection_service.detect_and_align_face(sample_image_path, 'selfie')
        
        # Assertions
        assert result.status == FaceDetectionStatus.MULTIPLE_FACES
        assert result.bbox == [300, 100, 400, 200]  # Best confidence face
        assert result.confidence == 0.90
    
    @patch('domain.services.face_detection_service.FaceDetector')
    @patch('domain.services.face_detection_service.OcclusionDetector')
    def test_detect_and_align_face_with_occlusion(self, mock_occlusion_detector, mock_face_detector,
                                                  face_detection_service, sample_image_path):
        """Test face detection with occlusion"""
        # Mock face detector
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance
        
        mock_detector_instance.detect_faces.return_value = {
            'faces': [{'bbox': [100, 100, 200, 200], 'confidence': 0.80}]
        }
        
        mock_detector_instance.detect_landmarks.return_value = [
            (120, 130), (180, 130), (150, 160), (130, 180), (170, 180)
        ]
        
        # Mock occlusion detected
        mock_occlusion_instance = Mock()
        mock_occlusion_detector.return_value = mock_occlusion_instance
        mock_occlusion_instance.detect_occlusion.return_value = {
            'is_occluded': True,
            'occlusion_type': 'glasses',
            'confidence': 0.75
        }
        
        # Test detection
        result = face_detection_service.detect_and_align_face(sample_image_path, 'selfie')
        
        # Assertions
        assert result.status == FaceDetectionStatus.FACE_OCCLUDED
        assert result.occlusion_detected
        assert result.occlusion_type == OcclusionType.GLASSES
        assert result.occlusion_confidence == 0.75
    
    def test_calculate_alignment_score(self, face_detection_service):
        """Test alignment score calculation"""
        # Perfect alignment landmarks
        perfect_landmarks = [
            (100, 130),  # left_eye
            (200, 130),  # right_eye (same y as left)
            (150, 160),  # nose (centered)
            (130, 180),  # left_mouth
            (170, 180)   # right_mouth (same y as left)
        ]
        
        score = face_detection_service._calculate_alignment_score(perfect_landmarks)
        assert score > 0.8  # Should be high for well-aligned face
        
        # Poor alignment landmarks
        poor_landmarks = [
            (100, 130),  # left_eye
            (200, 150),  # right_eye (different y)
            (170, 160),  # nose (off-center)
            (130, 180),  # left_mouth
            (180, 190)   # right_mouth (different y)
        ]
        
        poor_score = face_detection_service._calculate_alignment_score(poor_landmarks)
        assert poor_score < score  # Should be lower than perfect alignment
    
    @patch('cv2.imread')
    @patch('cv2.cvtColor')
    @patch('cv2.Laplacian')
    def test_calculate_face_quality(self, mock_laplacian, mock_cvtcolor, mock_imread, face_detection_service):
        """Test face quality calculation"""
        import numpy as np
        
        # Mock image data
        mock_image = np.ones((300, 300, 3), dtype=np.uint8) * 128  # Gray image
        mock_imread.return_value = mock_image
        
        mock_gray = np.ones((100, 100), dtype=np.uint8) * 128
        mock_cvtcolor.return_value = mock_gray
        
        # Mock high variance (sharp image)
        mock_laplacian.return_value = Mock()
        mock_laplacian.return_value.var.return_value = 1000.0
        
        bbox = [100, 100, 200, 200]
        quality_score = face_detection_service._calculate_face_quality("test.jpg", bbox)
        
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.5  # Should be decent quality
    
    def test_calculate_pose_angles(self, face_detection_service):
        """Test pose angle calculation"""
        # Straight face landmarks
        straight_landmarks = [
            (100, 130),  # left_eye
            (200, 130),  # right_eye
            (150, 160),  # nose
            (130, 180),  # left_mouth
            (170, 180)   # right_mouth
        ]
        
        pose_angles = face_detection_service._calculate_pose_angles(straight_landmarks)
        assert pose_angles is not None
        assert len(pose_angles) == 3  # yaw, pitch, roll
        
        yaw, pitch, roll = pose_angles
        assert abs(yaw) < 10  # Should be close to 0 for straight face
        assert abs(roll) < 10  # Should be close to 0 for upright face
    
    def test_validate_face_detection(self, face_detection_service):
        """Test face detection validation"""
        # Valid face result
        valid_result = FaceDetectionResult(
            image_path="test.jpg",
            status=FaceDetectionStatus.SUCCESS,
            bbox=[100, 100, 200, 200],
            confidence=0.85,
            face_quality_score=0.7,
            alignment_score=0.8,
            occlusion_detected=False
        )
        
        assert face_detection_service.validate_face_detection(valid_result)
        
        # Invalid face result (failed status)
        invalid_result = FaceDetectionResult(
            image_path="test.jpg",
            status=FaceDetectionStatus.FAILED,
            confidence=0.0
        )
        
        assert not face_detection_service.validate_face_detection(invalid_result)
    
    def test_compare_face_alignment(self, face_detection_service):
        """Test face alignment comparison"""
        landmarks1 = [(100, 130), (200, 130), (150, 160), (130, 180), (170, 180)]
        landmarks2 = [(105, 135), (195, 135), (155, 165), (125, 185), (175, 185)]
        
        face1 = FaceDetectionResult(
            image_path="test1.jpg",
            status=FaceDetectionStatus.SUCCESS,
            bbox=[50, 50, 250, 250],
            landmarks=landmarks1,
            confidence=0.9
        )
        
        face2 = FaceDetectionResult(
            image_path="test2.jpg",
            status=FaceDetectionStatus.SUCCESS,
            bbox=[60, 60, 240, 240],
            landmarks=landmarks2,
            confidence=0.85
        )
        
        similarity = face_detection_service.compare_face_alignment(face1, face2)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Similar landmarks should have high similarity
    
    def test_get_face_recommendations(self, face_detection_service):
        """Test face recommendations generation"""
        # No face detected
        no_face_result = FaceDetectionResult(
            image_path="test.jpg",
            status=FaceDetectionStatus.NO_FACE_DETECTED,
            confidence=0.0
        )
        
        recommendations = face_detection_service.get_face_recommendations(no_face_result)
        assert len(recommendations) > 0
        assert any("không phát hiện được khuôn mặt" in rec.lower() for rec in recommendations)
        
        # Face with glasses occlusion
        glasses_result = FaceDetectionResult(
            image_path="test.jpg",
            status=FaceDetectionStatus.FACE_OCCLUDED,
            occlusion_detected=True,
            occlusion_type=OcclusionType.GLASSES,
            confidence=0.8
        )
        
        recommendations = face_detection_service.get_face_recommendations(glasses_result)
        assert any("kính" in rec.lower() for rec in recommendations)
        
        # Low quality face
        low_quality_result = FaceDetectionResult(
            image_path="test.jpg",
            status=FaceDetectionStatus.SUCCESS,
            confidence=0.6,
            face_quality_score=0.3,
            alignment_score=0.4
        )
        
        recommendations = face_detection_service.get_face_recommendations(low_quality_result)
        assert len(recommendations) > 0
        assert any("chất lượng" in rec.lower() for rec in recommendations)
    
    @patch('domain.services.face_detection_service.FaceDetector')
    def test_detect_and_align_face_exception_handling(self, mock_face_detector, 
                                                     face_detection_service, sample_image_path):
        """Test exception handling in face detection"""
        # Mock face detector to raise exception
        mock_face_detector.side_effect = Exception("Test exception")
        
        result = face_detection_service.detect_and_align_face(sample_image_path, 'selfie')
        
        # Should return failed result
        assert result.status == FaceDetectionStatus.FAILED
        assert result.confidence == 0.0
    
    def test_face_detection_service_initialization(self):
        """Test service initialization"""
        service = FaceDetectionService()
        assert service.logger is not None
    
    def cleanup_method(self, sample_image_path):
        """Clean up test files"""
        if os.path.exists(sample_image_path):
            os.unlink(sample_image_path)
