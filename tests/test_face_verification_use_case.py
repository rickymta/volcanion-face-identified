import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from domain.entities.face_verification_result import FaceEmbedding, FaceVerificationResult, VerificationStatus, VerificationResult
from domain.services.face_verification_service import FaceVerificationService
from application.use_cases.face_verification_use_case import FaceVerificationUseCase

@pytest.fixture
def mock_face_verification_service():
    """Mock face verification service"""
    service = Mock(spec=FaceVerificationService)
    
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
        preprocessing_applied=True,
        metadata={},
        created_at=datetime.now()
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
        quality_assessment={"overall_quality": 0.85},
        metadata={},
        created_at=datetime.now()
    )
    
    # Configure mocks
    service.extract_face_embedding.return_value = mock_embedding
    service.verify_faces.return_value = mock_verification
    service.calculate_similarity.return_value = (0.85, 0.9)
    service.find_similar_faces.return_value = [mock_verification]
    service.validate_embedding_quality.return_value = True
    
    return service

@pytest.fixture
def mock_embedding_repository():
    """Mock embedding repository"""
    repository = Mock()
    
    mock_embedding = FaceEmbedding(
        id="test_embedding_id",
        image_path="/path/to/image.jpg",
        face_bbox=[10, 10, 100, 100],
        embedding_vector=[0.1] * 512,
        embedding_model="facenet",
        feature_quality=0.9,
        extraction_confidence=0.95,
        face_alignment_score=0.8,
        preprocessing_applied=True,
        metadata={},
        created_at=datetime.now()
    )
    
    repository.save.return_value = mock_embedding
    repository.find_by_id.return_value = mock_embedding
    repository.find_all.return_value = [mock_embedding]
    repository.find_by_image_path.return_value = mock_embedding
    repository.delete_by_id.return_value = True
    repository.count.return_value = 1
    repository.get_statistics.return_value = {
        'total_embeddings': 1,
        'avg_quality': 0.9,
        'models_used': ['facenet']
    }
    
    return repository

@pytest.fixture
def mock_verification_repository():
    """Mock verification repository"""
    repository = Mock()
    
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
        quality_assessment={"overall_quality": 0.85},
        metadata={},
        created_at=datetime.now()
    )
    
    repository.save.return_value = mock_verification
    repository.find_by_id.return_value = mock_verification
    repository.find_all.return_value = [mock_verification]
    repository.delete_by_id.return_value = True
    repository.count.return_value = 1
    repository.get_statistics.return_value = {
        'total_verifications': 1,
        'avg_similarity': 0.85,
        'match_rate': 0.8
    }
    
    return repository

@pytest.fixture
def face_verification_use_case(mock_face_verification_service, mock_embedding_repository, mock_verification_repository):
    """Face verification use case fixture"""
    return FaceVerificationUseCase(
        mock_face_verification_service,
        mock_embedding_repository,
        mock_verification_repository
    )

class TestFaceVerificationUseCase:
    
    @pytest.mark.asyncio
    async def test_extract_and_save_embedding_success(self, face_verification_use_case):
        """Test successful embedding extraction and saving"""
        # Test
        result = await face_verification_use_case.extract_and_save_embedding(
            "/path/to/image.jpg", [10, 10, 100, 100], "facenet"
        )
        
        # Assertions
        assert result is not None
        assert result.id == "test_embedding_id"
        assert result.embedding_model == "facenet"
        assert result.feature_quality == 0.9
        assert result.extraction_confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_extract_and_save_embedding_invalid_bbox(self, face_verification_use_case):
        """Test embedding extraction with invalid bbox"""
        with pytest.raises(ValueError, match="Invalid face bounding box"):
            await face_verification_use_case.extract_and_save_embedding(
                "/path/to/image.jpg", [10, 10], "facenet"
            )
    
    @pytest.mark.asyncio
    async def test_extract_and_save_embedding_unsupported_model(self, face_verification_use_case):
        """Test embedding extraction with unsupported model"""
        with pytest.raises(ValueError, match="Unsupported embedding model"):
            await face_verification_use_case.extract_and_save_embedding(
                "/path/to/image.jpg", [10, 10, 100, 100], "unsupported_model"
            )
    
    @pytest.mark.asyncio
    async def test_verify_faces_by_embeddings_success(self, face_verification_use_case):
        """Test successful face verification by embeddings"""
        # Test
        result = await face_verification_use_case.verify_faces_by_embeddings(
            "ref_embedding_id", "target_embedding_id", 0.6, "cosine"
        )
        
        # Assertions
        assert result is not None
        assert result.id == "test_verification_id"
        assert result.verification_result == VerificationResult.MATCH
        assert result.similarity_score == 0.85
        assert result.confidence == 0.9
        assert result.is_positive_match() == True
    
    @pytest.mark.asyncio
    async def test_verify_faces_by_embeddings_not_found(self, face_verification_use_case, mock_embedding_repository):
        """Test face verification with non-existent embeddings"""
        # Mock embedding not found
        mock_embedding_repository.find_by_id.return_value = None
        
        with pytest.raises(ValueError, match="Reference embedding not found"):
            await face_verification_use_case.verify_faces_by_embeddings(
                "non_existent_id", "target_embedding_id", 0.6, "cosine"
            )
    
    @pytest.mark.asyncio
    async def test_verify_faces_by_images_success(self, face_verification_use_case):
        """Test successful face verification by images"""
        # Test
        result = await face_verification_use_case.verify_faces_by_images(
            "/path/to/ref.jpg", [10, 10, 100, 100],
            "/path/to/target.jpg", [15, 15, 105, 105],
            0.6, "cosine", "facenet"
        )
        
        # Assertions
        assert result is not None
        assert result.verification_result == VerificationResult.MATCH
        assert result.similarity_score == 0.85
        assert result.distance_metric == "cosine"
        assert result.model_used == "facenet"
    
    @pytest.mark.asyncio
    async def test_verify_faces_by_images_invalid_threshold(self, face_verification_use_case):
        """Test face verification with invalid threshold"""
        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            await face_verification_use_case.verify_faces_by_images(
                "/path/to/ref.jpg", [10, 10, 100, 100],
                "/path/to/target.jpg", [15, 15, 105, 105],
                1.5, "cosine", "facenet"
            )
    
    @pytest.mark.asyncio
    async def test_find_matches_in_gallery_success(self, face_verification_use_case):
        """Test successful gallery search"""
        # Test
        result = await face_verification_use_case.find_matches_in_gallery(
            "/path/to/target.jpg", [10, 10, 100, 100],
            ["/path/to/gallery1.jpg", "/path/to/gallery2.jpg"],
            top_k=5, threshold=0.6, distance_metric="cosine", model_type="facenet"
        )
        
        # Assertions
        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].verification_result == VerificationResult.MATCH
        assert result[0].similarity_score == 0.85
    
    @pytest.mark.asyncio
    async def test_find_matches_in_gallery_invalid_k(self, face_verification_use_case):
        """Test gallery search with invalid top_k"""
        with pytest.raises(ValueError, match="top_k must be positive"):
            await face_verification_use_case.find_matches_in_gallery(
                "/path/to/target.jpg", [10, 10, 100, 100],
                None, top_k=0
            )
    
    @pytest.mark.asyncio
    async def test_get_embedding_success(self, face_verification_use_case):
        """Test successful embedding retrieval"""
        # Test
        result = await face_verification_use_case.get_embedding("test_embedding_id")
        
        # Assertions
        assert result is not None
        assert result.id == "test_embedding_id"
        assert result.embedding_model == "facenet"
    
    @pytest.mark.asyncio
    async def test_get_embedding_not_found(self, face_verification_use_case, mock_embedding_repository):
        """Test embedding retrieval with non-existent ID"""
        # Mock embedding not found
        mock_embedding_repository.find_by_id.return_value = None
        
        result = await face_verification_use_case.get_embedding("non_existent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_verification_result_success(self, face_verification_use_case):
        """Test successful verification result retrieval"""
        # Test
        result = await face_verification_use_case.get_verification_result("test_verification_id")
        
        # Assertions
        assert result is not None
        assert result.id == "test_verification_id"
        assert result.verification_result == VerificationResult.MATCH
    
    @pytest.mark.asyncio
    async def test_get_verification_result_not_found(self, face_verification_use_case, mock_verification_repository):
        """Test verification result retrieval with non-existent ID"""
        # Mock verification not found
        mock_verification_repository.find_by_id.return_value = None
        
        result = await face_verification_use_case.get_verification_result("non_existent_id")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_verification_statistics_success(self, face_verification_use_case):
        """Test successful statistics retrieval"""
        # Test
        result = await face_verification_use_case.get_verification_statistics()
        
        # Assertions
        assert isinstance(result, dict)
        assert 'verification_statistics' in result
        assert 'embedding_statistics' in result
        assert 'threshold_analysis' in result
        assert 'performance_metrics' in result
    
    @pytest.mark.asyncio
    async def test_optimize_threshold_success(self, face_verification_use_case, mock_face_verification_service):
        """Test successful threshold optimization"""
        # Mock optimization result
        mock_face_verification_service.optimize_threshold_for_pairs.return_value = {
            'best_threshold': 0.65,
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.94,
            'f1_score': 0.92
        }
        
        # Test data
        positive_pairs = [["/path/to/ref1.jpg", "/path/to/tar1.jpg"]]
        negative_pairs = [["/path/to/ref2.jpg", "/path/to/tar2.jpg"]]
        
        # Test
        result = await face_verification_use_case.optimize_threshold(
            positive_pairs, negative_pairs, "cosine"
        )
        
        # Assertions
        assert isinstance(result, dict)
        assert result['best_threshold'] == 0.65
        assert result['accuracy'] == 0.92
        assert result['precision'] == 0.90
        assert result['recall'] == 0.94
        assert result['f1_score'] == 0.92
    
    @pytest.mark.asyncio
    async def test_optimize_threshold_empty_pairs(self, face_verification_use_case):
        """Test threshold optimization with empty pairs"""
        with pytest.raises(ValueError, match="Positive and negative pairs cannot be empty"):
            await face_verification_use_case.optimize_threshold([], [], "cosine")
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data_success(self, face_verification_use_case):
        """Test successful data cleanup"""
        # Test
        result = await face_verification_use_case.cleanup_old_data(30)
        
        # Assertions
        assert isinstance(result, dict)
        assert 'deleted_verifications' in result
        assert 'deleted_embeddings' in result
        assert 'success' in result
        assert result['success'] == True
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data_invalid_days(self, face_verification_use_case):
        """Test data cleanup with invalid days"""
        with pytest.raises(ValueError, match="days_to_keep must be positive"):
            await face_verification_use_case.cleanup_old_data(0)
    
    def test_validate_face_bbox_valid(self, face_verification_use_case):
        """Test valid face bbox validation"""
        # Valid bbox
        result = face_verification_use_case._validate_face_bbox([10, 10, 100, 100])
        assert result == True
    
    def test_validate_face_bbox_invalid_length(self, face_verification_use_case):
        """Test invalid face bbox validation - wrong length"""
        # Invalid length
        result = face_verification_use_case._validate_face_bbox([10, 10, 100])
        assert result == False
    
    def test_validate_face_bbox_invalid_coordinates(self, face_verification_use_case):
        """Test invalid face bbox validation - invalid coordinates"""
        # Invalid coordinates (x2 < x1)
        result = face_verification_use_case._validate_face_bbox([100, 10, 50, 100])
        assert result == False
    
    def test_validate_face_bbox_negative_coordinates(self, face_verification_use_case):
        """Test invalid face bbox validation - negative coordinates"""
        # Negative coordinates
        result = face_verification_use_case._validate_face_bbox([-10, 10, 100, 100])
        assert result == False
    
    def test_validate_threshold_valid(self, face_verification_use_case):
        """Test valid threshold validation"""
        result = face_verification_use_case._validate_threshold(0.6)
        assert result == True
    
    def test_validate_threshold_invalid_low(self, face_verification_use_case):
        """Test invalid threshold validation - too low"""
        result = face_verification_use_case._validate_threshold(-0.1)
        assert result == False
    
    def test_validate_threshold_invalid_high(self, face_verification_use_case):
        """Test invalid threshold validation - too high"""
        result = face_verification_use_case._validate_threshold(1.1)
        assert result == False
    
    def test_validate_distance_metric_valid(self, face_verification_use_case):
        """Test valid distance metric validation"""
        for metric in ["cosine", "euclidean", "manhattan"]:
            result = face_verification_use_case._validate_distance_metric(metric)
            assert result == True
    
    def test_validate_distance_metric_invalid(self, face_verification_use_case):
        """Test invalid distance metric validation"""
        result = face_verification_use_case._validate_distance_metric("invalid_metric")
        assert result == False
    
    def test_validate_model_type_valid(self, face_verification_use_case):
        """Test valid model type validation"""
        for model in ["facenet", "dlib", "opencv"]:
            result = face_verification_use_case._validate_model_type(model)
            assert result == True
    
    def test_validate_model_type_invalid(self, face_verification_use_case):
        """Test invalid model type validation"""
        result = face_verification_use_case._validate_model_type("invalid_model")
        assert result == False

class TestFaceVerificationUseCaseIntegration:
    """Integration tests for face verification use case"""
    
    @pytest.mark.asyncio
    async def test_full_verification_workflow(self, face_verification_use_case):
        """Test complete verification workflow"""
        # Step 1: Extract embeddings
        ref_embedding = await face_verification_use_case.extract_and_save_embedding(
            "/path/to/ref.jpg", [10, 10, 100, 100], "facenet"
        )
        
        target_embedding = await face_verification_use_case.extract_and_save_embedding(
            "/path/to/target.jpg", [15, 15, 105, 105], "facenet"
        )
        
        # Step 2: Verify faces
        verification = await face_verification_use_case.verify_faces_by_embeddings(
            ref_embedding.id, target_embedding.id, 0.6, "cosine"
        )
        
        # Step 3: Check results
        assert verification.verification_result == VerificationResult.MATCH
        assert verification.similarity_score > 0.6
        assert verification.is_positive_match() == True
    
    @pytest.mark.asyncio
    async def test_gallery_search_workflow(self, face_verification_use_case):
        """Test gallery search workflow"""
        # Find matches in gallery
        matches = await face_verification_use_case.find_matches_in_gallery(
            "/path/to/target.jpg", [10, 10, 100, 100],
            gallery_image_paths=["/path/to/gallery1.jpg", "/path/to/gallery2.jpg"],
            top_k=3, threshold=0.6
        )
        
        # Check results
        assert isinstance(matches, list)
        assert len(matches) > 0
        for match in matches:
            assert match.similarity_score >= 0.6
            assert match.verification_result in [VerificationResult.MATCH, VerificationResult.NO_MATCH]
    
    @pytest.mark.asyncio
    async def test_statistics_and_optimization_workflow(self, face_verification_use_case, mock_face_verification_service):
        """Test statistics and optimization workflow"""
        # Mock optimization
        mock_face_verification_service.optimize_threshold_for_pairs.return_value = {
            'best_threshold': 0.65,
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.94,
            'f1_score': 0.92
        }
        
        # Get statistics
        stats = await face_verification_use_case.get_verification_statistics()
        assert isinstance(stats, dict)
        assert 'verification_statistics' in stats
        
        # Optimize threshold
        positive_pairs = [["/path/to/ref1.jpg", "/path/to/tar1.jpg"]]
        negative_pairs = [["/path/to/ref2.jpg", "/path/to/tar2.jpg"]]
        
        optimization = await face_verification_use_case.optimize_threshold(
            positive_pairs, negative_pairs, "cosine"
        )
        
        assert optimization['best_threshold'] == 0.65
        assert optimization['accuracy'] > 0.9
    
    @pytest.mark.asyncio
    async def test_cleanup_workflow(self, face_verification_use_case):
        """Test cleanup workflow"""
        # Perform cleanup
        cleanup_result = await face_verification_use_case.cleanup_old_data(30)
        
        assert cleanup_result['success'] == True
        assert 'deleted_verifications' in cleanup_result
        assert 'deleted_embeddings' in cleanup_result
