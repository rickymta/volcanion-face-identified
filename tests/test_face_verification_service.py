import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from domain.services.face_verification_service import FaceVerificationService
from domain.ml.face_embedding_extractor import FaceEmbeddingExtractor
from domain.ml.face_verification_engine import FaceVerificationEngine
from domain.entities.face_verification_result import FaceEmbedding, FaceVerificationResult, VerificationStatus, VerificationResult

@pytest.fixture
def mock_embedding_extractor():
    """Mock face embedding extractor"""
    extractor = Mock(spec=FaceEmbeddingExtractor)
    
    # Mock embedding result
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
    
    extractor.extract_embedding.return_value = mock_embedding
    extractor.is_model_available.return_value = True
    extractor.get_supported_models.return_value = ["facenet", "dlib", "opencv"]
    extractor.validate_face_region.return_value = True
    extractor.preprocess_face_region.return_value = "processed_image"
    
    return extractor

@pytest.fixture
def mock_verification_engine():
    """Mock face verification engine"""
    engine = Mock(spec=FaceVerificationEngine)
    
    engine.calculate_similarity.return_value = (0.85, 0.9)
    engine.verify_faces.return_value = (True, 0.85, 0.9)
    engine.get_optimal_threshold.return_value = 0.6
    engine.batch_verify.return_value = [(True, 0.85, 0.9)]
    engine.optimize_threshold.return_value = {
        'best_threshold': 0.65,
        'accuracy': 0.92,
        'precision': 0.90,
        'recall': 0.94,
        'f1_score': 0.92
    }
    
    return engine

@pytest.fixture
def face_verification_service(mock_embedding_extractor, mock_verification_engine):
    """Face verification service fixture"""
    service = FaceVerificationService()
    service.embedding_extractor = mock_embedding_extractor
    service.verification_engine = mock_verification_engine
    return service

class TestFaceVerificationService:
    
    def test_extract_face_embedding_success(self, face_verification_service):
        """Test successful face embedding extraction"""
        # Test
        result = face_verification_service.extract_face_embedding(
            "/path/to/image.jpg", [10, 10, 100, 100], "facenet"
        )
        
        # Assertions
        assert result is not None
        assert result.id == "test_embedding_id"
        assert result.embedding_model == "facenet"
        assert result.feature_quality == 0.9
        assert result.extraction_confidence == 0.95
    
    def test_extract_face_embedding_invalid_image_path(self, face_verification_service):
        """Test embedding extraction with invalid image path"""
        with pytest.raises(ValueError, match="Image path cannot be empty"):
            face_verification_service.extract_face_embedding("", [10, 10, 100, 100], "facenet")
    
    def test_extract_face_embedding_invalid_bbox(self, face_verification_service):
        """Test embedding extraction with invalid bbox"""
        with pytest.raises(ValueError, match="Invalid face bounding box"):
            face_verification_service.extract_face_embedding(
                "/path/to/image.jpg", [10, 10], "facenet"
            )
    
    def test_extract_face_embedding_unsupported_model(self, face_verification_service, mock_embedding_extractor):
        """Test embedding extraction with unsupported model"""
        # Mock model not available
        mock_embedding_extractor.is_model_available.return_value = False
        
        with pytest.raises(ValueError, match="Unsupported embedding model"):
            face_verification_service.extract_face_embedding(
                "/path/to/image.jpg", [10, 10, 100, 100], "unsupported_model"
            )
    
    def test_verify_faces_success(self, face_verification_service):
        """Test successful face verification"""
        # Mock embeddings
        ref_embedding = FaceEmbedding(
            id="ref_id",
            image_path="/path/to/ref.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet"
        )
        
        target_embedding = FaceEmbedding(
            id="target_id",
            image_path="/path/to/target.jpg",
            face_bbox=[15, 15, 105, 105],
            embedding_vector=[0.12] * 512,
            embedding_model="facenet"
        )
        
        # Test
        result = face_verification_service.verify_faces(
            ref_embedding, target_embedding, 0.6, "cosine"
        )
        
        # Assertions
        assert result is not None
        assert result.verification_result == VerificationResult.MATCH
        assert result.similarity_score == 0.85
        assert result.confidence == 0.9
        assert result.distance_metric == "cosine"
        assert result.threshold_used == 0.6
    
    def test_verify_faces_incompatible_models(self, face_verification_service):
        """Test face verification with incompatible embedding models"""
        # Mock embeddings with different models
        ref_embedding = FaceEmbedding(
            id="ref_id",
            image_path="/path/to/ref.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet"
        )
        
        target_embedding = FaceEmbedding(
            id="target_id",
            image_path="/path/to/target.jpg",
            face_bbox=[15, 15, 105, 105],
            embedding_vector=[0.12] * 128,
            embedding_model="dlib"
        )
        
        with pytest.raises(ValueError, match="Embeddings must use the same model"):
            face_verification_service.verify_faces(
                ref_embedding, target_embedding, 0.6, "cosine"
            )
    
    def test_verify_faces_incompatible_dimensions(self, face_verification_service):
        """Test face verification with incompatible embedding dimensions"""
        # Mock embeddings with different dimensions
        ref_embedding = FaceEmbedding(
            id="ref_id",
            image_path="/path/to/ref.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet"
        )
        
        target_embedding = FaceEmbedding(
            id="target_id",
            image_path="/path/to/target.jpg",
            face_bbox=[15, 15, 105, 105],
            embedding_vector=[0.12] * 256,
            embedding_model="facenet"
        )
        
        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            face_verification_service.verify_faces(
                ref_embedding, target_embedding, 0.6, "cosine"
            )
    
    def test_calculate_similarity_success(self, face_verification_service):
        """Test successful similarity calculation"""
        # Mock embeddings
        ref_embedding = FaceEmbedding(
            id="ref_id",
            image_path="/path/to/ref.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet"
        )
        
        target_embedding = FaceEmbedding(
            id="target_id",
            image_path="/path/to/target.jpg",
            face_bbox=[15, 15, 105, 105],
            embedding_vector=[0.12] * 512,
            embedding_model="facenet"
        )
        
        # Test
        similarity, confidence = face_verification_service.calculate_similarity(
            ref_embedding, target_embedding, "cosine"
        )
        
        # Assertions
        assert similarity == 0.85
        assert confidence == 0.9
    
    def test_find_similar_faces_success(self, face_verification_service):
        """Test successful similar faces finding"""
        # Mock target embedding
        target_embedding = FaceEmbedding(
            id="target_id",
            image_path="/path/to/target.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet"
        )
        
        # Mock gallery embeddings
        gallery_embeddings = [
            FaceEmbedding(
                id="gallery_1",
                image_path="/path/to/gallery1.jpg",
                face_bbox=[15, 15, 105, 105],
                embedding_vector=[0.12] * 512,
                embedding_model="facenet"
            )
        ]
        
        # Test
        results = face_verification_service.find_similar_faces(
            target_embedding, gallery_embeddings, top_k=5, threshold=0.6, distance_metric="cosine"
        )
        
        # Assertions
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].verification_result == VerificationResult.MATCH
        assert results[0].similarity_score == 0.85
    
    def test_find_similar_faces_invalid_top_k(self, face_verification_service):
        """Test similar faces finding with invalid top_k"""
        target_embedding = FaceEmbedding(
            id="target_id",
            image_path="/path/to/target.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet"
        )
        
        with pytest.raises(ValueError, match="top_k must be positive"):
            face_verification_service.find_similar_faces(
                target_embedding, [], top_k=0
            )
    
    def test_batch_verify_faces_success(self, face_verification_service):
        """Test successful batch face verification"""
        # Mock embedding pairs
        embedding_pairs = [
            (
                FaceEmbedding(
                    id="ref_1",
                    image_path="/path/to/ref1.jpg",
                    face_bbox=[10, 10, 100, 100],
                    embedding_vector=[0.1] * 512,
                    embedding_model="facenet"
                ),
                FaceEmbedding(
                    id="target_1",
                    image_path="/path/to/target1.jpg",
                    face_bbox=[15, 15, 105, 105],
                    embedding_vector=[0.12] * 512,
                    embedding_model="facenet"
                )
            )
        ]
        
        # Test
        results = face_verification_service.batch_verify_faces(
            embedding_pairs, threshold=0.6, distance_metric="cosine"
        )
        
        # Assertions
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].verification_result == VerificationResult.MATCH
        assert results[0].similarity_score == 0.85
    
    def test_batch_verify_faces_empty_pairs(self, face_verification_service):
        """Test batch verification with empty pairs"""
        with pytest.raises(ValueError, match="Embedding pairs cannot be empty"):
            face_verification_service.batch_verify_faces([])
    
    def test_optimize_threshold_for_pairs_success(self, face_verification_service, mock_verification_engine):
        """Test successful threshold optimization"""
        # Mock optimization result
        mock_verification_engine.optimize_threshold.return_value = {
            'best_threshold': 0.65,
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.94,
            'f1_score': 0.92
        }
        
        # Test data
        positive_pairs = [
            (
                FaceEmbedding(id="pos_ref", image_path="", face_bbox=[], embedding_vector=[0.1] * 512, embedding_model="facenet"),
                FaceEmbedding(id="pos_tar", image_path="", face_bbox=[], embedding_vector=[0.11] * 512, embedding_model="facenet")
            )
        ]
        
        negative_pairs = [
            (
                FaceEmbedding(id="neg_ref", image_path="", face_bbox=[], embedding_vector=[0.1] * 512, embedding_model="facenet"),
                FaceEmbedding(id="neg_tar", image_path="", face_bbox=[], embedding_vector=[0.9] * 512, embedding_model="facenet")
            )
        ]
        
        # Test
        result = face_verification_service.optimize_threshold_for_pairs(
            positive_pairs, negative_pairs, "cosine"
        )
        
        # Assertions
        assert isinstance(result, dict)
        assert result['best_threshold'] == 0.65
        assert result['accuracy'] == 0.92
        assert result['precision'] == 0.90
        assert result['recall'] == 0.94
        assert result['f1_score'] == 0.92
    
    def test_optimize_threshold_empty_pairs(self, face_verification_service):
        """Test threshold optimization with empty pairs"""
        with pytest.raises(ValueError, match="Positive and negative pairs cannot be empty"):
            face_verification_service.optimize_threshold_for_pairs([], [], "cosine")
    
    def test_validate_embedding_quality_high_quality(self, face_verification_service):
        """Test embedding quality validation - high quality"""
        embedding = FaceEmbedding(
            id="test_id",
            image_path="/path/to/image.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet",
            feature_quality=0.9,
            extraction_confidence=0.95
        )
        
        result = face_verification_service.validate_embedding_quality(embedding)
        assert result == True
    
    def test_validate_embedding_quality_low_quality(self, face_verification_service):
        """Test embedding quality validation - low quality"""
        embedding = FaceEmbedding(
            id="test_id",
            image_path="/path/to/image.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet",
            feature_quality=0.3,
            extraction_confidence=0.4
        )
        
        result = face_verification_service.validate_embedding_quality(embedding)
        assert result == False
    
    def test_get_quality_assessment_success(self, face_verification_service):
        """Test quality assessment generation"""
        embedding = FaceEmbedding(
            id="test_id",
            image_path="/path/to/image.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet",
            feature_quality=0.85,
            extraction_confidence=0.92,
            face_alignment_score=0.78
        )
        
        assessment = face_verification_service.get_quality_assessment(embedding)
        
        # Assertions
        assert isinstance(assessment, dict)
        assert 'overall_quality' in assessment
        assert 'feature_quality' in assessment
        assert 'extraction_confidence' in assessment
        assert 'face_alignment_score' in assessment
        assert 'quality_level' in assessment
        assert assessment['feature_quality'] == 0.85
        assert assessment['extraction_confidence'] == 0.92
    
    def test_get_supported_models_success(self, face_verification_service, mock_embedding_extractor):
        """Test getting supported models"""
        # Test
        models = face_verification_service.get_supported_models()
        
        # Assertions
        assert isinstance(models, list)
        assert "facenet" in models
        assert "dlib" in models
        assert "opencv" in models
    
    def test_get_supported_distance_metrics_success(self, face_verification_service):
        """Test getting supported distance metrics"""
        # Test
        metrics = face_verification_service.get_supported_distance_metrics()
        
        # Assertions
        assert isinstance(metrics, list)
        assert "cosine" in metrics
        assert "euclidean" in metrics
        assert "manhattan" in metrics

class TestFaceVerificationServiceEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_verify_faces_with_extraction_failure(self, face_verification_service, mock_embedding_extractor):
        """Test verification when embedding extraction fails"""
        # Mock extraction failure
        mock_embedding_extractor.extract_embedding.side_effect = Exception("Extraction failed")
        
        with pytest.raises(Exception, match="Extraction failed"):
            face_verification_service.extract_face_embedding(
                "/path/to/image.jpg", [10, 10, 100, 100], "facenet"
            )
    
    def test_verify_faces_with_similarity_calculation_failure(self, face_verification_service, mock_verification_engine):
        """Test verification when similarity calculation fails"""
        # Mock similarity calculation failure
        mock_verification_engine.calculate_similarity.side_effect = Exception("Similarity calculation failed")
        
        ref_embedding = FaceEmbedding(
            id="ref_id",
            image_path="/path/to/ref.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet"
        )
        
        target_embedding = FaceEmbedding(
            id="target_id",
            image_path="/path/to/target.jpg",
            face_bbox=[15, 15, 105, 105],
            embedding_vector=[0.12] * 512,
            embedding_model="facenet"
        )
        
        with pytest.raises(Exception, match="Similarity calculation failed"):
            face_verification_service.verify_faces(
                ref_embedding, target_embedding, 0.6, "cosine"
            )
    
    def test_find_similar_faces_with_empty_gallery(self, face_verification_service):
        """Test finding similar faces with empty gallery"""
        target_embedding = FaceEmbedding(
            id="target_id",
            image_path="/path/to/target.jpg",
            face_bbox=[10, 10, 100, 100],
            embedding_vector=[0.1] * 512,
            embedding_model="facenet"
        )
        
        # Test with empty gallery
        results = face_verification_service.find_similar_faces(
            target_embedding, [], top_k=5
        )
        
        # Should return empty list
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_batch_verify_with_mixed_models(self, face_verification_service):
        """Test batch verification with mixed embedding models"""
        # Mixed model embeddings
        embedding_pairs = [
            (
                FaceEmbedding(
                    id="ref_1",
                    image_path="/path/to/ref1.jpg",
                    face_bbox=[10, 10, 100, 100],
                    embedding_vector=[0.1] * 512,
                    embedding_model="facenet"
                ),
                FaceEmbedding(
                    id="target_1",
                    image_path="/path/to/target1.jpg",
                    face_bbox=[15, 15, 105, 105],
                    embedding_vector=[0.12] * 128,
                    embedding_model="dlib"
                )
            )
        ]
        
        # Should handle individual pair errors gracefully
        results = face_verification_service.batch_verify_faces(embedding_pairs)
        
        # Check that errors are handled
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].status == VerificationStatus.FAILED

class TestFaceVerificationServiceIntegration:
    """Integration tests combining multiple service methods"""
    
    def test_full_verification_pipeline(self, face_verification_service):
        """Test complete verification pipeline"""
        # Step 1: Extract embeddings
        ref_embedding = face_verification_service.extract_face_embedding(
            "/path/to/ref.jpg", [10, 10, 100, 100], "facenet"
        )
        
        target_embedding = face_verification_service.extract_face_embedding(
            "/path/to/target.jpg", [15, 15, 105, 105], "facenet"
        )
        
        # Step 2: Verify faces
        verification = face_verification_service.verify_faces(
            ref_embedding, target_embedding, 0.6, "cosine"
        )
        
        # Step 3: Validate quality
        ref_quality = face_verification_service.validate_embedding_quality(ref_embedding)
        target_quality = face_verification_service.validate_embedding_quality(target_embedding)
        
        # Assertions
        assert ref_embedding.embedding_model == "facenet"
        assert target_embedding.embedding_model == "facenet"
        assert verification.verification_result == VerificationResult.MATCH
        assert ref_quality == True
        assert target_quality == True
    
    def test_gallery_search_pipeline(self, face_verification_service):
        """Test gallery search pipeline"""
        # Step 1: Extract target embedding
        target_embedding = face_verification_service.extract_face_embedding(
            "/path/to/target.jpg", [10, 10, 100, 100], "facenet"
        )
        
        # Step 2: Create gallery embeddings
        gallery_embeddings = [
            face_verification_service.extract_face_embedding(
                f"/path/to/gallery{i}.jpg", [10, 10, 100, 100], "facenet"
            )
            for i in range(3)
        ]
        
        # Step 3: Find similar faces
        matches = face_verification_service.find_similar_faces(
            target_embedding, gallery_embeddings, top_k=2, threshold=0.6
        )
        
        # Assertions
        assert isinstance(matches, list)
        assert len(matches) <= 2
        for match in matches:
            assert match.similarity_score >= 0.6
    
    def test_threshold_optimization_pipeline(self, face_verification_service, mock_verification_engine):
        """Test threshold optimization pipeline"""
        # Mock optimization
        mock_verification_engine.optimize_threshold.return_value = {
            'best_threshold': 0.65,
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.94,
            'f1_score': 0.92
        }
        
        # Step 1: Create training pairs
        positive_pairs = [
            (
                face_verification_service.extract_face_embedding(
                    "/path/to/pos_ref.jpg", [10, 10, 100, 100], "facenet"
                ),
                face_verification_service.extract_face_embedding(
                    "/path/to/pos_tar.jpg", [10, 10, 100, 100], "facenet"
                )
            )
        ]
        
        negative_pairs = [
            (
                face_verification_service.extract_face_embedding(
                    "/path/to/neg_ref.jpg", [10, 10, 100, 100], "facenet"
                ),
                face_verification_service.extract_face_embedding(
                    "/path/to/neg_tar.jpg", [10, 10, 100, 100], "facenet"
                )
            )
        ]
        
        # Step 2: Optimize threshold
        optimization = face_verification_service.optimize_threshold_for_pairs(
            positive_pairs, negative_pairs, "cosine"
        )
        
        # Step 3: Use optimized threshold for verification
        verification = face_verification_service.verify_faces(
            positive_pairs[0][0], positive_pairs[0][1], 
            optimization['best_threshold'], "cosine"
        )
        
        # Assertions
        assert optimization['best_threshold'] == 0.65
        assert verification.threshold_used == 0.65
        assert verification.verification_result == VerificationResult.MATCH
