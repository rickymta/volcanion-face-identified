import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from presentation.api.face_verification_api import router
from domain.entities.face_verification_result import FaceEmbedding, FaceVerificationResult, VerificationStatus, VerificationResult

# Create test app
app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.fixture
def mock_use_case():
    """Mock face verification use case"""
    use_case = Mock()
    
    # Mock embedding
    mock_embedding = FaceEmbedding(
        id="test_embedding_id",
        image_path="/path/to/image.jpg",
        face_bbox=[10, 10, 100, 100],
        embedding_vector=[0.1] * 512,
        embedding_model="facenet",
        feature_quality=0.9,
        extraction_confidence=0.95,
        face_alignment_score=0.8,
        preprocessing_applied=True
    )
    
    # Mock verification result
    mock_verification = FaceVerificationResult(
        id="test_verification_id",
        reference_image_path="/path/to/ref.jpg",
        target_image_path="/path/to/target.jpg",
        reference_embedding_id="ref_embedding_id",
        target_embedding_id="target_embedding_id",
        status=VerificationStatus.COMPLETED,
        verification_result=VerificationResult.MATCH,
        similarity_score=0.85,
        distance_metric="cosine",
        confidence=0.9,
        threshold_used=0.6,
        match_probability=0.92,
        processing_time_ms=150.0,
        model_used="facenet",
        quality_assessment={"overall_quality": 0.85}
    )
    
    # Configure async methods
    use_case.extract_and_save_embedding = AsyncMock(return_value=mock_embedding)
    use_case.verify_faces_by_embeddings = AsyncMock(return_value=mock_verification)
    use_case.verify_faces_by_images = AsyncMock(return_value=mock_verification)
    use_case.find_matches_in_gallery = AsyncMock(return_value=[mock_verification])
    use_case.get_embedding = AsyncMock(return_value=mock_embedding)
    use_case.get_verification_result = AsyncMock(return_value=mock_verification)
    use_case.get_verification_statistics = AsyncMock(return_value={
        'verification_statistics': {'total': 100},
        'embedding_statistics': {'total': 200},
        'threshold_analysis': {'optimal': 0.6},
        'performance_metrics': {'accuracy': 0.95}
    })
    use_case.optimize_threshold = AsyncMock(return_value={
        'best_threshold': 0.65,
        'accuracy': 0.92,
        'precision': 0.90,
        'recall': 0.94,
        'f1_score': 0.92
    })
    use_case.cleanup_old_data = AsyncMock(return_value={
        'deleted_verifications': 10,
        'deleted_embeddings': 5,
        'success': True
    })
    
    return use_case

class TestFaceVerificationAPI:
    """Test Face Verification API endpoints"""
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_extract_face_embedding_success(self, mock_get_use_case, mock_use_case):
        """Test successful face embedding extraction"""
        mock_get_use_case.return_value = mock_use_case
        
        # Create test image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(b"fake image data")
            temp_file_path = temp_file.name
        
        try:
            # Test request
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/face-verification/extract-embedding",
                    files={"file": ("test.jpg", f, "image/jpeg")},
                    data={
                        "face_bbox": "[10, 10, 100, 100]",
                        "model_type": "facenet"
                    }
                )
            
            # Assertions
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test_embedding_id"
            assert data["embedding_model"] == "facenet"
            assert data["feature_quality"] == 0.9
            assert data["extraction_confidence"] == 0.95
            
        finally:
            os.unlink(temp_file_path)
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_extract_face_embedding_invalid_bbox(self, mock_get_use_case, mock_use_case):
        """Test embedding extraction with invalid bbox"""
        mock_get_use_case.return_value = mock_use_case
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(b"fake image data")
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/face-verification/extract-embedding",
                    files={"file": ("test.jpg", f, "image/jpeg")},
                    data={
                        "face_bbox": "[10, 10]",  # Invalid bbox
                        "model_type": "facenet"
                    }
                )
            
            assert response.status_code == 400
            assert "face_bbox must be a JSON array with 4 integers" in response.json()["detail"]
            
        finally:
            os.unlink(temp_file_path)
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_extract_face_embedding_invalid_file_type(self, mock_get_use_case, mock_use_case):
        """Test embedding extraction with invalid file type"""
        mock_get_use_case.return_value = mock_use_case
        
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"not an image")
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/face-verification/extract-embedding",
                    files={"file": ("test.txt", f, "text/plain")},
                    data={
                        "face_bbox": "[10, 10, 100, 100]",
                        "model_type": "facenet"
                    }
                )
            
            assert response.status_code == 400
            assert "File must be an image" in response.json()["detail"]
            
        finally:
            os.unlink(temp_file_path)
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_verify_faces_by_embeddings_success(self, mock_get_use_case, mock_use_case):
        """Test successful face verification by embeddings"""
        mock_get_use_case.return_value = mock_use_case
        
        response = client.post(
            "/api/face-verification/verify-by-embeddings",
            data={
                "reference_embedding_id": "ref_embedding_id",
                "target_embedding_id": "target_embedding_id",
                "threshold": 0.6,
                "distance_metric": "cosine"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_verification_id"
        assert data["verification_result"] == "MATCH"
        assert data["similarity_score"] == 0.85
        assert data["confidence"] == 0.9
        assert data["is_match"] == True
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_verify_faces_by_images_success(self, mock_get_use_case, mock_use_case):
        """Test successful face verification by images"""
        mock_get_use_case.return_value = mock_use_case
        
        # Create test image files
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as ref_file:
            ref_file.write(b"fake ref image")
            ref_file_path = ref_file.name
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as target_file:
            target_file.write(b"fake target image")
            target_file_path = target_file.name
        
        try:
            with open(ref_file_path, "rb") as ref_f, open(target_file_path, "rb") as target_f:
                response = client.post(
                    "/api/face-verification/verify-by-images",
                    files={
                        "reference_file": ("ref.jpg", ref_f, "image/jpeg"),
                        "target_file": ("target.jpg", target_f, "image/jpeg")
                    },
                    data={
                        "reference_bbox": "[10, 10, 100, 100]",
                        "target_bbox": "[15, 15, 105, 105]",
                        "threshold": 0.6,
                        "distance_metric": "cosine",
                        "model_type": "facenet"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert data["verification_result"] == "MATCH"
            assert data["similarity_score"] == 0.85
            
        finally:
            os.unlink(ref_file_path)
            os.unlink(target_file_path)
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_find_matches_in_gallery_success(self, mock_get_use_case, mock_use_case):
        """Test successful gallery search"""
        mock_get_use_case.return_value = mock_use_case
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(b"fake target image")
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/face-verification/find-matches",
                    files={"target_file": ("target.jpg", f, "image/jpeg")},
                    data={
                        "target_bbox": "[10, 10, 100, 100]",
                        "gallery_image_paths": '["gallery1.jpg", "gallery2.jpg"]',
                        "top_k": 5,
                        "threshold": 0.6,
                        "distance_metric": "cosine",
                        "model_type": "facenet"
                    }
                )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["verification_result"] == "MATCH"
            
        finally:
            os.unlink(temp_file_path)
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_get_face_embedding_success(self, mock_get_use_case, mock_use_case):
        """Test successful embedding retrieval"""
        mock_get_use_case.return_value = mock_use_case
        
        response = client.get("/api/face-verification/embedding/test_embedding_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_embedding_id"
        assert data["embedding_model"] == "facenet"
        assert data["feature_quality"] == 0.9
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_get_face_embedding_not_found(self, mock_get_use_case, mock_use_case):
        """Test embedding retrieval with non-existent ID"""
        mock_get_use_case.return_value = mock_use_case
        mock_use_case.get_embedding.return_value = None
        
        response = client.get("/api/face-verification/embedding/non_existent_id")
        
        assert response.status_code == 404
        assert "Face embedding not found" in response.json()["detail"]
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_get_verification_result_success(self, mock_get_use_case, mock_use_case):
        """Test successful verification result retrieval"""
        mock_get_use_case.return_value = mock_use_case
        
        response = client.get("/api/face-verification/verification/test_verification_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_verification_id"
        assert data["verification_result"] == "MATCH"
        assert data["similarity_score"] == 0.85
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_get_verification_result_not_found(self, mock_get_use_case, mock_use_case):
        """Test verification result retrieval with non-existent ID"""
        mock_get_use_case.return_value = mock_use_case
        mock_use_case.get_verification_result.return_value = None
        
        response = client.get("/api/face-verification/verification/non_existent_id")
        
        assert response.status_code == 404
        assert "Verification result not found" in response.json()["detail"]
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_get_verification_statistics_success(self, mock_get_use_case, mock_use_case):
        """Test successful statistics retrieval"""
        mock_get_use_case.return_value = mock_use_case
        
        response = client.get("/api/face-verification/statistics")
        
        assert response.status_code == 200
        data = response.json()
        assert "verification_statistics" in data
        assert "embedding_statistics" in data
        assert "threshold_analysis" in data
        assert "performance_metrics" in data
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_optimize_threshold_success(self, mock_get_use_case, mock_use_case):
        """Test successful threshold optimization"""
        mock_get_use_case.return_value = mock_use_case
        
        response = client.post(
            "/api/face-verification/optimize-threshold",
            data={
                "positive_pairs": '[["ref1.jpg", "tar1.jpg"]]',
                "negative_pairs": '[["ref2.jpg", "tar2.jpg"]]',
                "distance_metric": "cosine"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["best_threshold"] == 0.65
        assert data["accuracy"] == 0.92
        assert data["precision"] == 0.90
        assert data["recall"] == 0.94
        assert data["f1_score"] == 0.92
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_optimize_threshold_invalid_json(self, mock_get_use_case, mock_use_case):
        """Test threshold optimization with invalid JSON"""
        mock_get_use_case.return_value = mock_use_case
        
        response = client.post(
            "/api/face-verification/optimize-threshold",
            data={
                "positive_pairs": "invalid json",
                "negative_pairs": '[["ref2.jpg", "tar2.jpg"]]',
                "distance_metric": "cosine"
            }
        )
        
        assert response.status_code == 400
        assert "Invalid JSON format for pairs" in response.json()["detail"]
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_cleanup_old_data_success(self, mock_get_use_case, mock_use_case):
        """Test successful data cleanup"""
        mock_get_use_case.return_value = mock_use_case
        
        response = client.delete("/api/face-verification/cleanup?days_to_keep=30")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Cleanup completed successfully"
        assert data["deleted_verifications"] == 10
        assert data["deleted_embeddings"] == 5
        assert data["success"] == True
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/api/face-verification/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "face-verification"
        assert data["version"] == "1.0.0"

class TestFaceVerificationAPIErrorHandling:
    """Test error handling in Face Verification API"""
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_extract_embedding_internal_error(self, mock_get_use_case, mock_use_case):
        """Test embedding extraction with internal error"""
        mock_get_use_case.return_value = mock_use_case
        mock_use_case.extract_and_save_embedding.side_effect = Exception("Internal error")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(b"fake image data")
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/face-verification/extract-embedding",
                    files={"file": ("test.jpg", f, "image/jpeg")},
                    data={
                        "face_bbox": "[10, 10, 100, 100]",
                        "model_type": "facenet"
                    }
                )
            
            assert response.status_code == 500
            assert "Embedding extraction failed" in response.json()["detail"]
            
        finally:
            os.unlink(temp_file_path)
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_verify_by_embeddings_internal_error(self, mock_get_use_case, mock_use_case):
        """Test verification by embeddings with internal error"""
        mock_get_use_case.return_value = mock_use_case
        mock_use_case.verify_faces_by_embeddings.side_effect = Exception("Verification error")
        
        response = client.post(
            "/api/face-verification/verify-by-embeddings",
            data={
                "reference_embedding_id": "ref_id",
                "target_embedding_id": "target_id"
            }
        )
        
        assert response.status_code == 500
        assert "Face verification failed" in response.json()["detail"]
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_verify_by_images_internal_error(self, mock_get_use_case, mock_use_case):
        """Test verification by images with internal error"""
        mock_get_use_case.return_value = mock_use_case
        mock_use_case.verify_faces_by_images.side_effect = Exception("Image verification error")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as ref_file:
            ref_file.write(b"fake ref image")
            ref_file_path = ref_file.name
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as target_file:
            target_file.write(b"fake target image")
            target_file_path = target_file.name
        
        try:
            with open(ref_file_path, "rb") as ref_f, open(target_file_path, "rb") as target_f:
                response = client.post(
                    "/api/face-verification/verify-by-images",
                    files={
                        "reference_file": ("ref.jpg", ref_f, "image/jpeg"),
                        "target_file": ("target.jpg", target_f, "image/jpeg")
                    },
                    data={
                        "reference_bbox": "[10, 10, 100, 100]",
                        "target_bbox": "[15, 15, 105, 105]"
                    }
                )
            
            assert response.status_code == 500
            assert "Face verification failed" in response.json()["detail"]
            
        finally:
            os.unlink(ref_file_path)
            os.unlink(target_file_path)
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_find_matches_internal_error(self, mock_get_use_case, mock_use_case):
        """Test gallery search with internal error"""
        mock_get_use_case.return_value = mock_use_case
        mock_use_case.find_matches_in_gallery.side_effect = Exception("Gallery search error")
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(b"fake target image")
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/face-verification/find-matches",
                    files={"target_file": ("target.jpg", f, "image/jpeg")},
                    data={
                        "target_bbox": "[10, 10, 100, 100]",
                        "top_k": 5
                    }
                )
            
            assert response.status_code == 500
            assert "Match finding failed" in response.json()["detail"]
            
        finally:
            os.unlink(temp_file_path)
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_get_embedding_internal_error(self, mock_get_use_case, mock_use_case):
        """Test embedding retrieval with internal error"""
        mock_get_use_case.return_value = mock_use_case
        mock_use_case.get_embedding.side_effect = Exception("Database error")
        
        response = client.get("/api/face-verification/embedding/test_id")
        
        assert response.status_code == 500
        assert "Failed to get embedding" in response.json()["detail"]
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_get_statistics_internal_error(self, mock_get_use_case, mock_use_case):
        """Test statistics retrieval with internal error"""
        mock_get_use_case.return_value = mock_use_case
        mock_use_case.get_verification_statistics.side_effect = Exception("Statistics error")
        
        response = client.get("/api/face-verification/statistics")
        
        assert response.status_code == 500
        assert "Failed to get statistics" in response.json()["detail"]
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_cleanup_internal_error(self, mock_get_use_case, mock_use_case):
        """Test cleanup with internal error"""
        mock_get_use_case.return_value = mock_use_case
        mock_use_case.cleanup_old_data.side_effect = Exception("Cleanup error")
        
        response = client.delete("/api/face-verification/cleanup")
        
        assert response.status_code == 500
        assert "Cleanup failed" in response.json()["detail"]

class TestFaceVerificationAPIValidation:
    """Test input validation in Face Verification API"""
    
    def test_invalid_bbox_formats(self):
        """Test various invalid bbox formats"""
        invalid_bboxes = [
            "[10, 10]",           # Too few coordinates
            "[10, 10, 100]",      # Too few coordinates
            "[10, 10, 100, 100, 50]",  # Too many coordinates
            "not json",           # Invalid JSON
            "[10, 'ten', 100, 100]",   # Non-numeric values
            "[]"                  # Empty array
        ]
        
        for invalid_bbox in invalid_bboxes:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                temp_file.write(b"fake image data")
                temp_file_path = temp_file.name
            
            try:
                with open(temp_file_path, "rb") as f:
                    response = client.post(
                        "/api/face-verification/extract-embedding",
                        files={"file": ("test.jpg", f, "image/jpeg")},
                        data={
                            "face_bbox": invalid_bbox,
                            "model_type": "facenet"
                        }
                    )
                
                assert response.status_code == 400
                
            finally:
                os.unlink(temp_file_path)
    
    def test_invalid_gallery_paths_format(self):
        """Test invalid gallery paths format"""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file.write(b"fake target image")
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/api/face-verification/find-matches",
                    files={"target_file": ("target.jpg", f, "image/jpeg")},
                    data={
                        "target_bbox": "[10, 10, 100, 100]",
                        "gallery_image_paths": "not json array",
                        "top_k": 5
                    }
                )
            
            assert response.status_code == 400
            assert "gallery_image_paths must be a JSON array" in response.json()["detail"]
            
        finally:
            os.unlink(temp_file_path)

class TestFaceVerificationAPIIntegration:
    """Integration tests for Face Verification API"""
    
    @patch('presentation.api.face_verification_api.get_face_verification_use_case')
    def test_full_verification_workflow_api(self, mock_get_use_case, mock_use_case):
        """Test complete verification workflow through API"""
        mock_get_use_case.return_value = mock_use_case
        
        # Step 1: Extract reference embedding
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as ref_file:
            ref_file.write(b"fake ref image")
            ref_file_path = ref_file.name
        
        try:
            with open(ref_file_path, "rb") as f:
                ref_response = client.post(
                    "/api/face-verification/extract-embedding",
                    files={"file": ("ref.jpg", f, "image/jpeg")},
                    data={
                        "face_bbox": "[10, 10, 100, 100]",
                        "model_type": "facenet"
                    }
                )
            
            assert ref_response.status_code == 200
            ref_embedding_id = ref_response.json()["id"]
            
            # Step 2: Extract target embedding
            with open(ref_file_path, "rb") as f:  # Use same file for simplicity
                target_response = client.post(
                    "/api/face-verification/extract-embedding",
                    files={"file": ("target.jpg", f, "image/jpeg")},
                    data={
                        "face_bbox": "[15, 15, 105, 105]",
                        "model_type": "facenet"
                    }
                )
            
            assert target_response.status_code == 200
            target_embedding_id = target_response.json()["id"]
            
            # Step 3: Verify faces
            verify_response = client.post(
                "/api/face-verification/verify-by-embeddings",
                data={
                    "reference_embedding_id": ref_embedding_id,
                    "target_embedding_id": target_embedding_id,
                    "threshold": 0.6,
                    "distance_metric": "cosine"
                }
            )
            
            assert verify_response.status_code == 200
            verification_data = verify_response.json()
            assert verification_data["verification_result"] == "MATCH"
            assert verification_data["similarity_score"] == 0.85
            
            # Step 4: Get verification result by ID
            verification_id = verification_data["id"]
            get_response = client.get(f"/api/face-verification/verification/{verification_id}")
            
            assert get_response.status_code == 200
            get_data = get_response.json()
            assert get_data["id"] == verification_id
            
        finally:
            os.unlink(ref_file_path)
