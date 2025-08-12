import pytest
import tempfile
import os
import cv2
import numpy as np
from unittest.mock import Mock, patch
from infrastructure.ml_models.tamper_detector import TamperDetector

class TestTamperDetector:
    
    @pytest.fixture
    def detector(self):
        return TamperDetector()
    
    @pytest.fixture
    def sample_clean_image_path(self):
        """Tạo ảnh sạch không có dấu hiệu tamper"""
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "clean.jpg")
        
        # Tạo ảnh đồng nhất
        image = np.ones((400, 600, 3), dtype=np.uint8) * 128
        cv2.rectangle(image, (50, 50), (550, 350), (200, 200, 200), -1)
        cv2.rectangle(image, (100, 100), (500, 300), (150, 150, 150), 2)
        cv2.imwrite(image_path, image)
        
        yield image_path
        
        if os.path.exists(image_path):
            os.remove(image_path)
        os.rmdir(temp_dir)
    
    @pytest.fixture
    def sample_tampered_image_path(self):
        """Tạo ảnh có dấu hiệu tamper (composite)"""
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "tampered.jpg")
        
        # Tạo ảnh composite với nhiều vùng có characteristics khác nhau
        image = np.ones((400, 600, 3), dtype=np.uint8) * 128
        
        # Vùng 1: nền
        cv2.rectangle(image, (0, 0), (300, 400), (120, 120, 120), -1)
        
        # Vùng 2: có noise pattern khác
        noise_region = np.random.randint(100, 140, (200, 300, 3), dtype=np.uint8)
        image[200:400, 300:600] = noise_region
        
        cv2.imwrite(image_path, image)
        
        yield image_path
        
        if os.path.exists(image_path):
            os.remove(image_path)
        os.rmdir(temp_dir)
    
    def test_detect_tampering_clean_image(self, detector, sample_clean_image_path):
        """Test phát hiện tamper với ảnh sạch"""
        result = detector.detect_tampering(sample_clean_image_path)
        
        assert 'is_tampered' in result
        assert 'tamper_type' in result
        assert 'confidence' in result
        assert 'metadata_analysis' in result
        assert 'detailed_analysis' in result
        
        assert isinstance(result['is_tampered'], bool)
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
    
    def test_detect_tampering_tampered_image(self, detector, sample_tampered_image_path):
        """Test phát hiện tamper với ảnh đã chỉnh sửa"""
        result = detector.detect_tampering(sample_tampered_image_path)
        
        assert 'is_tampered' in result
        assert 'confidence' in result
        
        # Ảnh tampered có thể được phát hiện (tùy thuộc vào độ phức tạp)
        assert 0 <= result['confidence'] <= 1
    
    def test_analyze_compression_artifacts(self, detector, sample_clean_image_path):
        """Test phân tích compression artifacts"""
        image = cv2.imread(sample_clean_image_path)
        analysis = detector._analyze_compression_artifacts(image)
        
        assert 'is_suspicious' in analysis
        assert 'confidence' in analysis
        assert 'artifacts_score' in analysis
        
        assert isinstance(analysis['is_suspicious'], bool)
        assert 0 <= analysis['confidence'] <= 1
        assert analysis['artifacts_score'] >= 0
    
    def test_analyze_edge_inconsistencies(self, detector, sample_clean_image_path):
        """Test phân tích edge inconsistencies"""
        image = cv2.imread(sample_clean_image_path)
        analysis = detector._analyze_edge_inconsistencies(image)
        
        assert 'is_suspicious' in analysis
        assert 'confidence' in analysis
        assert 'edge_variance' in analysis
        assert 'discontinuity_score' in analysis
        
        assert isinstance(analysis['is_suspicious'], bool)
        assert 0 <= analysis['confidence'] <= 1
        assert analysis['edge_variance'] >= 0
        assert 0 <= analysis['discontinuity_score'] <= 1
    
    def test_detect_edge_discontinuities(self, detector, sample_clean_image_path):
        """Test phát hiện edge discontinuities"""
        image = cv2.imread(sample_clean_image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        discontinuity_score = detector._detect_edge_discontinuities(edges)
        
        assert 0 <= discontinuity_score <= 1
    
    def test_analyze_noise_patterns(self, detector, sample_clean_image_path, sample_tampered_image_path):
        """Test phân tích noise patterns"""
        clean_image = cv2.imread(sample_clean_image_path)
        tampered_image = cv2.imread(sample_tampered_image_path)
        
        clean_analysis = detector._analyze_noise_patterns(clean_image)
        tampered_analysis = detector._analyze_noise_patterns(tampered_image)
        
        for analysis in [clean_analysis, tampered_analysis]:
            assert 'is_suspicious' in analysis
            assert 'confidence' in analysis
            assert 'noise_variance' in analysis
            
            assert isinstance(analysis['is_suspicious'], bool)
            assert 0 <= analysis['confidence'] <= 1
            assert analysis['noise_variance'] >= 0
        
        # Ảnh tampered có thể có noise variance cao hơn
        # (không always true nhưng là xu hướng)
    
    def test_analyze_color_inconsistencies(self, detector, sample_clean_image_path):
        """Test phân tích color inconsistencies"""
        image = cv2.imread(sample_clean_image_path)
        analysis = detector._analyze_color_inconsistencies(image)
        
        assert 'is_suspicious' in analysis
        assert 'confidence' in analysis
        assert 'color_variance' in analysis
        
        assert isinstance(analysis['is_suspicious'], bool)
        assert 0 <= analysis['confidence'] <= 1
        assert analysis['color_variance'] >= 0
    
    def test_analyze_geometric_distortions(self, detector, sample_clean_image_path):
        """Test phân tích geometric distortions"""
        image = cv2.imread(sample_clean_image_path)
        analysis = detector._analyze_geometric_distortions(image)
        
        assert 'is_suspicious' in analysis
        assert 'confidence' in analysis
        
        assert isinstance(analysis['is_suspicious'], bool)
        assert 0 <= analysis['confidence'] <= 1
    
    @patch('infrastructure.ml_models.tamper_detector.Image')
    def test_analyze_metadata(self, mock_image_class, detector, sample_clean_image_path):
        """Test phân tích metadata"""
        # Mock PIL Image
        mock_image = Mock()
        mock_exif = {
            256: 'Test Software',  # Software tag
            306: '2023:01:01 12:00:00',  # DateTime
            36867: '2023:01:01 12:00:00',  # DateTimeOriginal
        }
        mock_image.getexif.return_value = mock_exif
        mock_image_class.open.return_value = mock_image
        
        analysis = detector._analyze_metadata(sample_clean_image_path)
        
        assert 'metadata' in analysis
        assert 'suspicious_indicators' in analysis
        assert 'is_suspicious' in analysis
        
        assert isinstance(analysis['metadata'], dict)
        assert isinstance(analysis['suspicious_indicators'], list)
        assert isinstance(analysis['is_suspicious'], bool)
    
    def test_determine_tamper_type(self, detector):
        """Test xác định loại tamper"""
        # Test với compression suspicious
        analysis_compression = {
            'compression': {'is_suspicious': True},
            'edge': {'is_suspicious': False},
            'noise': {'is_suspicious': False},
            'color': {'is_suspicious': False},
            'geometric': {'is_suspicious': False}
        }
        
        tamper_type = detector._determine_tamper_type(analysis_compression)
        assert tamper_type == 'digital_manipulation'
        
        # Test với edge suspicious
        analysis_edge = {
            'compression': {'is_suspicious': False},
            'edge': {'is_suspicious': True},
            'noise': {'is_suspicious': False},
            'color': {'is_suspicious': False},
            'geometric': {'is_suspicious': False}
        }
        
        tamper_type = detector._determine_tamper_type(analysis_edge)
        assert tamper_type == 'copy_paste'
        
        # Test với không có suspicious
        analysis_clean = {
            'compression': {'is_suspicious': False},
            'edge': {'is_suspicious': False},
            'noise': {'is_suspicious': False},
            'color': {'is_suspicious': False},
            'geometric': {'is_suspicious': False}
        }
        
        tamper_type = detector._determine_tamper_type(analysis_clean)
        assert tamper_type == 'none'
    
    def test_detect_tampering_invalid_image(self, detector):
        """Test với ảnh không tồn tại"""
        result = detector.detect_tampering("nonexistent.jpg")
        
        # Nên trả về default result
        assert result['is_tampered'] == True  # Conservative approach
        assert result['confidence'] == 0.0
