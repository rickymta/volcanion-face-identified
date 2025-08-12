import pytest
import tempfile
import os
import cv2
import numpy as np
from infrastructure.ml_models.quality_analyzer import QualityAnalyzer

class TestQualityAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return QualityAnalyzer()
    
    @pytest.fixture
    def sample_good_image_path(self):
        """Tạo ảnh chất lượng tốt cho test"""
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "good_quality.jpg")
        
        # Tạo ảnh sharp với contrast tốt
        image = np.ones((400, 600, 3), dtype=np.uint8) * 128
        cv2.rectangle(image, (50, 50), (550, 350), (255, 255, 255), -1)
        cv2.rectangle(image, (100, 100), (500, 300), (0, 0, 0), 2)
        cv2.putText(image, "SAMPLE DOCUMENT", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite(image_path, image)
        
        yield image_path
        
        if os.path.exists(image_path):
            os.remove(image_path)
        os.rmdir(temp_dir)
    
    @pytest.fixture
    def sample_blurry_image_path(self):
        """Tạo ảnh mờ cho test"""
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "blurry.jpg")
        
        # Tạo ảnh và làm mờ
        image = np.ones((400, 600, 3), dtype=np.uint8) * 128
        cv2.rectangle(image, (50, 50), (550, 350), (255, 255, 255), -1)
        cv2.rectangle(image, (100, 100), (500, 300), (0, 0, 0), 2)
        
        # Blur image
        image = cv2.GaussianBlur(image, (15, 15), 0)
        cv2.imwrite(image_path, image)
        
        yield image_path
        
        if os.path.exists(image_path):
            os.remove(image_path)
        os.rmdir(temp_dir)
    
    def test_analyze_good_quality_image(self, analyzer, sample_good_image_path):
        """Test phân tích ảnh chất lượng tốt"""
        metrics = analyzer.analyze_image_quality(sample_good_image_path)
        
        assert 'overall_score' in metrics
        assert 'blur_score' in metrics
        assert 'glare_score' in metrics
        assert 'contrast_score' in metrics
        assert 'brightness_score' in metrics
        assert 'noise_score' in metrics
        assert 'edge_sharpness' in metrics
        assert 'watermark_present' in metrics
        
        # Ảnh tốt nên có scores cao
        assert metrics['overall_score'] > 0.5
        assert metrics['blur_score'] > 0.3
        assert metrics['contrast_score'] > 0.3
    
    def test_analyze_blurry_image(self, analyzer, sample_blurry_image_path):
        """Test phân tích ảnh mờ"""
        metrics = analyzer.analyze_image_quality(sample_blurry_image_path)
        
        # Ảnh mờ nên có blur score thấp
        assert metrics['blur_score'] < 0.5
        assert metrics['overall_score'] < 0.7
    
    def test_calculate_blur_score(self, analyzer, sample_good_image_path, sample_blurry_image_path):
        """Test tính blur score"""
        good_image = cv2.imread(sample_good_image_path)
        blurry_image = cv2.imread(sample_blurry_image_path)
        
        good_blur = analyzer._calculate_blur_score(good_image)
        blurry_blur = analyzer._calculate_blur_score(blurry_image)
        
        # Ảnh tốt nên có blur score cao hơn ảnh mờ
        assert good_blur > blurry_blur
        assert 0 <= good_blur <= 1
        assert 0 <= blurry_blur <= 1
    
    def test_detect_glare(self, analyzer):
        """Test phát hiện glare"""
        # Tạo ảnh có vùng sáng chói
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "glare.jpg")
        
        image = np.ones((400, 600, 3), dtype=np.uint8) * 100
        # Tạo vùng sáng chói
        cv2.rectangle(image, (200, 150), (400, 250), (255, 255, 255), -1)
        cv2.imwrite(image_path, image)
        
        try:
            glare_image = cv2.imread(image_path)
            glare_score = analyzer._detect_glare(glare_image)
            
            assert 0 <= glare_score <= 1
            assert glare_score > 0.1  # Nên phát hiện được glare
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
            os.rmdir(temp_dir)
    
    def test_calculate_contrast(self, analyzer, sample_good_image_path):
        """Test tính contrast"""
        image = cv2.imread(sample_good_image_path)
        contrast_score = analyzer._calculate_contrast(image)
        
        assert 0 <= contrast_score <= 1
    
    def test_calculate_brightness(self, analyzer):
        """Test tính brightness score"""
        # Tạo ảnh tối
        dark_image = np.ones((100, 100, 3), dtype=np.uint8) * 50
        dark_score = analyzer._calculate_brightness(dark_image)
        
        # Tạo ảnh sáng
        bright_image = np.ones((100, 100, 3), dtype=np.uint8) * 200
        bright_score = analyzer._calculate_brightness(bright_image)
        
        # Tạo ảnh brightness optimal
        optimal_image = np.ones((100, 100, 3), dtype=np.uint8) * 130
        optimal_score = analyzer._calculate_brightness(optimal_image)
        
        assert 0 <= dark_score <= 1
        assert 0 <= bright_score <= 1
        assert 0 <= optimal_score <= 1
        assert optimal_score >= max(dark_score, bright_score)
    
    def test_calculate_noise(self, analyzer, sample_good_image_path):
        """Test tính noise"""
        image = cv2.imread(sample_good_image_path)
        noise_score = analyzer._calculate_noise(image)
        
        assert 0 <= noise_score <= 1
    
    def test_calculate_edge_sharpness(self, analyzer, sample_good_image_path, sample_blurry_image_path):
        """Test tính edge sharpness"""
        good_image = cv2.imread(sample_good_image_path)
        blurry_image = cv2.imread(sample_blurry_image_path)
        
        good_sharpness = analyzer._calculate_edge_sharpness(good_image)
        blurry_sharpness = analyzer._calculate_edge_sharpness(blurry_image)
        
        assert 0 <= good_sharpness <= 1
        assert 0 <= blurry_sharpness <= 1
        assert good_sharpness > blurry_sharpness
    
    def test_detect_watermark(self, analyzer, sample_good_image_path):
        """Test phát hiện watermark"""
        image = cv2.imread(sample_good_image_path)
        has_watermark = analyzer._detect_watermark(image)
        
        assert isinstance(has_watermark, bool)
    
    def test_calculate_overall_score(self, analyzer):
        """Test tính overall score"""
        # Test với metrics tốt
        good_metrics = {
            'blur': 0.8,
            'glare': 0.9,
            'contrast': 0.7,
            'brightness': 0.8,
            'noise': 0.9,
            'sharpness': 0.7
        }
        
        good_score = analyzer._calculate_overall_score(good_metrics)
        assert 0 <= good_score <= 1
        assert good_score > 0.7
        
        # Test với metrics kém
        poor_metrics = {
            'blur': 0.2,
            'glare': 0.3,
            'contrast': 0.2,
            'brightness': 0.3,
            'noise': 0.2,
            'sharpness': 0.1
        }
        
        poor_score = analyzer._calculate_overall_score(poor_metrics)
        assert 0 <= poor_score <= 1
        assert poor_score < 0.4
    
    def test_analyze_invalid_image(self, analyzer):
        """Test với ảnh không tồn tại"""
        metrics = analyzer.analyze_image_quality("nonexistent.jpg")
        
        # Nên trả về default metrics
        assert metrics['overall_score'] == 0.0
        assert metrics['blur_score'] == 0.0
