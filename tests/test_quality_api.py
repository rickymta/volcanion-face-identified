import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from io import BytesIO
from main import app
from domain.entities.document_quality import DocumentQuality, QualityStatus, TamperType

# Test client
client = TestClient(app)

class TestQualityAPI:
    
    def test_quality_health_endpoint(self):
        """Test quality health endpoint"""
        response = client.get("/api/v1/quality/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["service"] == "document_quality_check"
    
    @patch('presentation.api.quality_api.DocumentQualityCheckUseCase')
    @patch('presentation.api.quality_api.DocumentQualityService')
    def test_check_document_quality_success(self, mock_service_class, mock_usecase_class):
        """Test kiểm tra chất lượng giấy tờ thành công"""
        # Setup mock
        mock_usecase = Mock()
        mock_service = Mock()
        
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.GOOD,
            quality_score=0.85,
            tamper_detected=False,
            tamper_type=TamperType.NONE,
            tamper_confidence=0.1,
            blur_score=0.8,
            glare_score=0.2,
            contrast_score=0.9,
            brightness_score=0.8,
            noise_score=0.1,
            edge_sharpness=0.8,
            watermark_present=False
        )
        
        mock_usecase.execute.return_value = mock_quality
        mock_service.get_quality_recommendations.return_value = ["Chất lượng tốt"]
        mock_usecase_class.return_value = mock_usecase
        mock_service_class.return_value = mock_service
        
        # Tạo file giả
        test_file = BytesIO(b"fake image content")
        
        response = client.post(
            "/api/v1/quality/check",
            files={"file": ("test.jpg", test_file, "image/jpeg")},
            data={"save_to_db": "true"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_quality"] == "good"
        assert data["quality_score"] == 0.85
        assert data["is_acceptable"] == True
        assert data["tamper_analysis"]["tamper_detected"] == False
        assert data["tamper_analysis"]["tamper_type"] == "none"
        assert data["watermark_present"] == False
        assert "Chất lượng tốt" in data["recommendations"]
    
    @patch('presentation.api.quality_api.DocumentQualityCheckUseCase')
    @patch('presentation.api.quality_api.DocumentQualityService')
    def test_check_document_quality_with_tamper(self, mock_service_class, mock_usecase_class):
        """Test kiểm tra chất lượng với tamper detected"""
        mock_usecase = Mock()
        mock_service = Mock()
        
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.REJECTED,
            quality_score=0.3,
            tamper_detected=True,
            tamper_type=TamperType.DIGITAL_MANIPULATION,
            tamper_confidence=0.9,
            blur_score=0.5,
            glare_score=0.3,
            contrast_score=0.4,
            brightness_score=0.6,
            noise_score=0.6,
            edge_sharpness=0.3,
            watermark_present=True,
            metadata_analysis={'suspicious_indicators': ['editing_software_detected']}
        )
        
        mock_usecase.execute.return_value = mock_quality
        mock_service.get_quality_recommendations.return_value = [
            "Phát hiện dấu hiệu chỉnh sửa (digital_manipulation)",
            "Ảnh bị mờ, hãy chụp lại với camera ổn định hơn"
        ]
        mock_usecase_class.return_value = mock_usecase
        mock_service_class.return_value = mock_service
        
        test_file = BytesIO(b"fake tampered image")
        
        response = client.post(
            "/api/v1/quality/check",
            files={"file": ("tampered.jpg", test_file, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_quality"] == "rejected"
        assert data["quality_score"] == 0.3
        assert data["is_acceptable"] == False
        assert data["tamper_analysis"]["tamper_detected"] == True
        assert data["tamper_analysis"]["tamper_type"] == "digital_manipulation"
        assert data["tamper_analysis"]["tamper_confidence"] == 0.9
        assert data["tamper_analysis"]["metadata_suspicious"] == True
        assert data["watermark_present"] == True
        assert len(data["recommendations"]) == 2
    
    def test_check_document_quality_invalid_file_type(self):
        """Test với file không phải ảnh"""
        test_file = BytesIO(b"not an image")
        
        response = client.post(
            "/api/v1/quality/check",
            files={"file": ("test.txt", test_file, "text/plain")}
        )
        
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]
    
    @patch('presentation.api.quality_api.DocumentQualityCheckUseCase')
    def test_check_document_quality_with_bbox(self, mock_usecase_class):
        """Test kiểm tra chất lượng với bbox"""
        mock_usecase = Mock()
        mock_quality = DocumentQuality(
            image_path="test.jpg",
            overall_quality=QualityStatus.FAIR,
            quality_score=0.6
        )
        mock_usecase.execute.return_value = mock_quality
        mock_usecase_class.return_value = mock_usecase
        
        test_file = BytesIO(b"fake image")
        bbox_json = '[10, 10, 100, 100]'
        
        response = client.post(
            "/api/v1/quality/check",
            files={"file": ("test.jpg", test_file, "image/jpeg")},
            data={"bbox": bbox_json}
        )
        
        assert response.status_code == 200
        # Verify bbox was parsed and passed correctly
        mock_usecase.execute.assert_called_once()
        call_args = mock_usecase.execute.call_args
        assert call_args[0][1] == [10, 10, 100, 100]  # bbox parameter
    
    @patch('presentation.api.quality_api.DocumentQualityCheckUseCase')
    def test_get_quality_recommendations_success(self, mock_usecase_class):
        """Test lấy recommendations thành công"""
        mock_usecase = Mock()
        mock_usecase.get_recommendations.return_value = [
            "Ảnh bị mờ, hãy chụp lại với camera ổn định hơn",
            "Có ánh sáng chói, hãy tránh ánh sáng trực tiếp"
        ]
        mock_usecase_class.return_value = mock_usecase
        
        # Mock service for quality score
        with patch('presentation.api.quality_api.DocumentQualityService') as mock_service_class:
            mock_service = Mock()
            mock_quality = DocumentQuality(image_path="test.jpg", quality_score=0.4)
            mock_service.analyze_quality.return_value = mock_quality
            mock_service_class.return_value = mock_service
            
            test_file = BytesIO(b"fake image")
            
            response = client.post(
                "/api/v1/quality/recommendations",
                files={"file": ("test.jpg", test_file, "image/jpeg")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["recommendations"]) == 2
            assert "mờ" in data["recommendations"][0]
            assert "chói" in data["recommendations"][1]
            assert data["quality_score"] == 0.4
    
    @patch('presentation.api.quality_api.DocumentQualityRepositoryImpl')
    def test_list_quality_checks(self, mock_repo_class):
        """Test lấy danh sách quality checks"""
        mock_repo = Mock()
        mock_qualities = [
            DocumentQuality(
                image_path="test1.jpg",
                overall_quality=QualityStatus.GOOD,
                quality_score=0.9,
                tamper_detected=False,
                tamper_type=TamperType.NONE,
                blur_score=0.8,
                glare_score=0.1,
                contrast_score=0.9,
                brightness_score=0.8,
                noise_score=0.1,
                edge_sharpness=0.9,
                watermark_present=False
            ),
            DocumentQuality(
                image_path="test2.jpg",
                overall_quality=QualityStatus.POOR,
                quality_score=0.3,
                tamper_detected=True,
                tamper_type=TamperType.COPY_PASTE,
                blur_score=0.2,
                glare_score=0.8,
                contrast_score=0.3,
                brightness_score=0.2,
                noise_score=0.9,
                edge_sharpness=0.2,
                watermark_present=True
            )
        ]
        mock_repo.get_all.return_value = mock_qualities
        mock_repo_class.return_value = mock_repo
        
        with patch('presentation.api.quality_api.DocumentQualityService') as mock_service_class:
            mock_service = Mock()
            mock_service.get_quality_recommendations.side_effect = [
                ["Chất lượng tốt"],
                ["Chất lượng kém", "Phát hiện tamper"]
            ]
            mock_service_class.return_value = mock_service
            
            response = client.get("/api/v1/quality/list")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 2
            assert len(data["qualities"]) == 2
            assert data["qualities"][0]["overall_quality"] == "good"
            assert data["qualities"][1]["overall_quality"] == "poor"
            assert data["qualities"][1]["tamper_analysis"]["tamper_detected"] == True
    
    @patch('presentation.api.quality_api.DocumentQualityRepositoryImpl')
    def test_delete_quality_check_success(self, mock_repo_class):
        """Test xóa quality check thành công"""
        mock_repo = Mock()
        mock_repo.delete_by_id.return_value = True
        mock_repo_class.return_value = mock_repo
        
        response = client.delete("/api/v1/quality/test_id")
        
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
    
    @patch('presentation.api.quality_api.DocumentQualityRepositoryImpl')
    def test_delete_quality_check_not_found(self, mock_repo_class):
        """Test xóa quality check không tồn tại"""
        mock_repo = Mock()
        mock_repo.delete_by_id.return_value = False
        mock_repo_class.return_value = mock_repo
        
        response = client.delete("/api/v1/quality/nonexistent_id")
        
        assert response.status_code == 404
        assert "Quality check not found" in response.json()["detail"]
    
    @patch('presentation.api.quality_api.DocumentQualityRepositoryImpl')
    def test_get_quality_statistics(self, mock_repo_class):
        """Test lấy thống kê quality checks"""
        mock_repo = Mock()
        mock_qualities = [
            DocumentQuality(
                image_path="test1.jpg",
                overall_quality=QualityStatus.GOOD,
                quality_score=0.9,
                tamper_detected=False
            ),
            DocumentQuality(
                image_path="test2.jpg",
                overall_quality=QualityStatus.FAIR,
                quality_score=0.7,
                tamper_detected=False
            ),
            DocumentQuality(
                image_path="test3.jpg",
                overall_quality=QualityStatus.POOR,
                quality_score=0.4,
                tamper_detected=True
            )
        ]
        
        # Mock is_acceptable method
        for quality in mock_qualities:
            quality.is_acceptable = Mock(return_value=quality.overall_quality in [QualityStatus.GOOD, QualityStatus.FAIR] and not quality.tamper_detected)
        
        mock_repo.get_all.return_value = mock_qualities
        mock_repo_class.return_value = mock_repo
        
        response = client.get("/api/v1/quality/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_checks"] == 3
        assert data["quality_distribution"]["good"] == 1
        assert data["quality_distribution"]["fair"] == 1
        assert data["quality_distribution"]["poor"] == 1
        assert data["tamper_detection_rate"] == 1/3
        assert data["average_quality_score"] == (0.9 + 0.7 + 0.4) / 3
        assert data["acceptable_rate"] == 2/3  # 2 acceptable out of 3
    
    @patch('presentation.api.quality_api.DocumentQualityRepositoryImpl')
    def test_get_quality_statistics_empty(self, mock_repo_class):
        """Test thống kê với danh sách rỗng"""
        mock_repo = Mock()
        mock_repo.get_all.return_value = []
        mock_repo_class.return_value = mock_repo
        
        response = client.get("/api/v1/quality/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_checks"] == 0
        assert data["quality_distribution"] == {}
        assert data["tamper_detection_rate"] == 0.0
        assert data["average_quality_score"] == 0.0
