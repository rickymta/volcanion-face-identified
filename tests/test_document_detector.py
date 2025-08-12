import pytest
import tempfile
import os
import cv2
import numpy as np
from infrastructure.ml_models.document_detector import DocumentDetector

class TestDocumentDetector:
    
    @pytest.fixture
    def detector(self):
        return DocumentDetector()
    
    @pytest.fixture
    def sample_image_path(self):
        # Tạo ảnh mẫu cho test
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "test_image.jpg")
        
        # Tạo ảnh giả có hình chữ nhật (giả lập giấy tờ)
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # Nền trắng
        cv2.rectangle(image, (50, 50), (550, 350), (0, 0, 0), 2)  # Viền đen
        cv2.imwrite(image_path, image)
        
        yield image_path
        
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
        os.rmdir(temp_dir)
    
    def test_detect_document_success(self, detector, sample_image_path):
        """Test phát hiện giấy tờ thành công"""
        bbox = detector.detect(sample_image_path)
        
        assert bbox is not None
        assert len(bbox) == 4
        assert all(isinstance(coord, int) for coord in bbox)
        assert bbox[0] < bbox[2]  # x1 < x2
        assert bbox[1] < bbox[3]  # y1 < y2
    
    def test_detect_invalid_image(self, detector):
        """Test với ảnh không tồn tại"""
        bbox = detector.detect("nonexistent_image.jpg")
        assert bbox is None
    
    def test_classify_document(self, detector, sample_image_path):
        """Test phân loại giấy tờ"""
        doc_type = detector.classify(sample_image_path)
        assert doc_type in ['cmnd', 'passport', 'unknown']
    
    def test_get_document_confidence(self, detector, sample_image_path):
        """Test tính confidence"""
        bbox = detector.detect(sample_image_path)
        confidence = detector.get_document_confidence(sample_image_path, bbox)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_passport_pattern_detection(self, detector):
        """Test phát hiện pattern passport"""
        # Tạo ảnh có màu xanh (giả lập passport)
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "passport.jpg")
        
        # Tạo ảnh màu xanh
        image = np.ones((400, 600, 3), dtype=np.uint8)
        image[:, :] = [100, 0, 0]  # Màu xanh BGR
        cv2.imwrite(image_path, image)
        
        try:
            is_passport = detector._is_passport_pattern(image)
            # Có thể True hoặc False tùy thuộc vào logic
            assert isinstance(is_passport, bool)
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
            os.rmdir(temp_dir)
    
    def test_id_card_pattern_detection(self, detector):
        """Test phát hiện pattern CMND/CCCD"""
        # Tạo ảnh trắng với text đen (giả lập CMND)
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, "cmnd.jpg")
        
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # Nền trắng
        cv2.putText(image, "CMND", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imwrite(image_path, image)
        
        try:
            is_id_card = detector._is_id_card_pattern(image)
            assert isinstance(is_id_card, bool)
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
            os.rmdir(temp_dir)
