from domain.entities.face_verification_result import FaceEmbedding, FaceVerificationResult, VerificationStatus, VerificationResult
from domain.repositories.face_verification_repository import FaceEmbeddingRepository, FaceVerificationRepository
from typing import Optional, List, Tuple
import logging
import time

class FaceVerificationService:
    """Service để xử lý face embedding và verification"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_face_embedding(self, image_path: str, face_bbox: List[int], 
                              model_type: str = "facenet") -> FaceEmbedding:
        """
        Extract face embedding từ ảnh
        """
        try:
            from infrastructure.ml_models.face_embedding_extractor import FaceEmbeddingExtractor
            
            extractor = FaceEmbeddingExtractor(model_type)
            
            # Extract embedding
            embedding_vector, confidence = extractor.extract_embedding(image_path, face_bbox)
            
            if embedding_vector is None:
                return FaceEmbedding(
                    image_path=image_path,
                    face_bbox=face_bbox,
                    embedding_model=model_type,
                    extraction_confidence=0.0,
                    feature_quality=0.0
                )
            
            # Calculate feature quality
            feature_quality = self._calculate_feature_quality(embedding_vector)
            
            # Calculate face alignment score (if landmarks available)
            face_alignment_score = self._estimate_alignment_from_bbox(face_bbox)
            
            return FaceEmbedding(
                image_path=image_path,
                face_bbox=face_bbox,
                embedding_vector=embedding_vector,
                embedding_model=model_type,
                feature_quality=feature_quality,
                extraction_confidence=confidence,
                face_alignment_score=face_alignment_score,
                preprocessing_applied=True
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting face embedding: {e}")
            return FaceEmbedding(
                image_path=image_path,
                face_bbox=face_bbox,
                embedding_model=model_type,
                extraction_confidence=0.0,
                feature_quality=0.0
            )
    
    def verify_faces(self, reference_embedding: FaceEmbedding, 
                    target_embedding: FaceEmbedding,
                    threshold: Optional[float] = None,
                    distance_metric: str = "cosine") -> FaceVerificationResult:
        """
        Verify hai face embeddings
        """
        start_time = time.time()
        
        try:
            from infrastructure.ml_models.face_verification_engine import FaceVerificationEngine
            
            # Validate embeddings
            if not reference_embedding.is_valid_embedding():
                return self._create_failed_verification(
                    reference_embedding.image_path,
                    target_embedding.image_path,
                    VerificationStatus.INSUFFICIENT_FEATURES,
                    "Reference embedding is invalid",
                    start_time
                )
            
            if not target_embedding.is_valid_embedding():
                return self._create_failed_verification(
                    reference_embedding.image_path,
                    target_embedding.image_path,
                    VerificationStatus.INSUFFICIENT_FEATURES,
                    "Target embedding is invalid",
                    start_time
                )
            
            # Check if embeddings are comparable
            if not self._are_embeddings_comparable(reference_embedding, target_embedding):
                return self._create_failed_verification(
                    reference_embedding.image_path,
                    target_embedding.image_path,
                    VerificationStatus.FACES_NOT_COMPARABLE,
                    "Embeddings are not comparable (different models or dimensions)",
                    start_time
                )
            
            # Initialize verification engine
            engine = FaceVerificationEngine(default_threshold=threshold or 0.6)
            
            # Perform verification
            verification_result = engine.verify_faces(
                reference_embedding.embedding_vector,
                target_embedding.embedding_vector,
                threshold=threshold,
                distance_metric=distance_metric
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Assess quality
            quality_assessment = self._assess_verification_quality(
                reference_embedding, target_embedding, verification_result
            )
            
            # Determine verification result
            if verification_result['is_match']:
                result_status = VerificationResult.MATCH
            elif verification_result['similarity_score'] > (threshold or 0.6) * 0.7:
                result_status = VerificationResult.UNCERTAIN
            else:
                result_status = VerificationResult.NO_MATCH
            
            return FaceVerificationResult(
                reference_image_path=reference_embedding.image_path,
                target_image_path=target_embedding.image_path,
                reference_embedding_id=reference_embedding.id,
                target_embedding_id=target_embedding.id,
                status=VerificationStatus.SUCCESS,
                verification_result=result_status,
                similarity_score=verification_result['similarity_score'],
                distance_metric=distance_metric,
                confidence=verification_result['confidence'],
                threshold_used=verification_result['threshold_used'],
                match_probability=verification_result['match_probability'],
                processing_time_ms=processing_time,
                model_used=reference_embedding.embedding_model,
                quality_assessment=quality_assessment
            )
            
        except Exception as e:
            self.logger.error(f"Error in face verification: {e}")
            return self._create_failed_verification(
                reference_embedding.image_path if reference_embedding else "unknown",
                target_embedding.image_path if target_embedding else "unknown",
                VerificationStatus.FAILED,
                str(e),
                start_time
            )
    
    def verify_face_against_gallery(self, target_embedding: FaceEmbedding,
                                   gallery_embeddings: List[FaceEmbedding],
                                   threshold: Optional[float] = None,
                                   distance_metric: str = "cosine") -> List[FaceVerificationResult]:
        """
        Verify một face embedding với một gallery các embeddings
        """
        results = []
        
        for reference_embedding in gallery_embeddings:
            verification_result = self.verify_faces(
                reference_embedding, target_embedding, threshold, distance_metric
            )
            results.append(verification_result)
        
        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    def find_best_matches(self, target_embedding: FaceEmbedding,
                         gallery_embeddings: List[FaceEmbedding],
                         top_k: int = 5,
                         threshold: Optional[float] = None,
                         distance_metric: str = "cosine") -> List[FaceVerificationResult]:
        """
        Tìm top-k best matches trong gallery
        """
        verification_results = self.verify_face_against_gallery(
            target_embedding, gallery_embeddings, threshold, distance_metric
        )
        
        # Filter by threshold if specified
        if threshold is not None:
            verification_results = [
                result for result in verification_results 
                if result.similarity_score >= threshold
            ]
        
        # Return top-k results
        return verification_results[:top_k]
    
    def batch_extract_embeddings(self, image_face_pairs: List[Tuple[str, List[int]]],
                                model_type: str = "facenet") -> List[FaceEmbedding]:
        """
        Extract embeddings cho nhiều faces
        """
        embeddings = []
        
        for image_path, face_bbox in image_face_pairs:
            embedding = self.extract_face_embedding(image_path, face_bbox, model_type)
            embeddings.append(embedding)
        
        return embeddings
    
    def _calculate_feature_quality(self, embedding_vector: List[float]) -> float:
        """Tính toán chất lượng features"""
        try:
            import numpy as np
            
            emb = np.array(embedding_vector)
            
            # Factor 1: Variance (good features should have variance)
            variance = np.var(emb)
            variance_score = min(variance * 10, 1.0)  # Scale factor
            
            # Factor 2: Norm (should be around 1.0 for normalized embeddings)
            norm = np.linalg.norm(emb)
            norm_score = 1.0 - abs(norm - 1.0)
            norm_score = max(0.0, min(norm_score, 1.0))
            
            # Factor 3: Non-zero elements
            non_zero_ratio = np.count_nonzero(emb) / len(emb)
            
            # Factor 4: Range of values
            value_range = np.max(emb) - np.min(emb)
            range_score = min(value_range / 2.0, 1.0)
            
            # Combine factors
            quality = (
                variance_score * 0.3 +
                norm_score * 0.3 +
                non_zero_ratio * 0.2 +
                range_score * 0.2
            )
            
            return max(0.0, min(quality, 1.0))
            
        except Exception:
            return 0.5  # Default quality
    
    def _estimate_alignment_from_bbox(self, face_bbox: List[int]) -> float:
        """Estimate alignment score từ face bounding box"""
        try:
            x1, y1, x2, y2 = face_bbox
            width = x2 - x1
            height = y2 - y1
            
            # Good face crops should have reasonable aspect ratio
            aspect_ratio = width / height if height > 0 else 0
            
            # Ideal aspect ratio for faces is around 0.8-1.2
            if 0.8 <= aspect_ratio <= 1.2:
                aspect_score = 1.0
            else:
                aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.0))
            
            # Face should be reasonably sized
            area = width * height
            size_score = min(area / 10000.0, 1.0)  # Normalize by typical face size
            
            return (aspect_score + size_score) / 2.0
            
        except Exception:
            return 0.5
    
    def _are_embeddings_comparable(self, emb1: FaceEmbedding, emb2: FaceEmbedding) -> bool:
        """Kiểm tra xem hai embeddings có thể so sánh được không"""
        # Same model
        if emb1.embedding_model != emb2.embedding_model:
            return False
        
        # Same dimensions
        if emb1.get_embedding_dimension() != emb2.get_embedding_dimension():
            return False
        
        # Both have valid embeddings
        if not emb1.is_valid_embedding() or not emb2.is_valid_embedding():
            return False
        
        return True
    
    def _assess_verification_quality(self, ref_emb: FaceEmbedding, 
                                   target_emb: FaceEmbedding,
                                   verification_result: dict) -> dict:
        """Assess overall quality của verification"""
        return {
            'reference_quality': ref_emb.feature_quality,
            'target_quality': target_emb.feature_quality,
            'reference_confidence': ref_emb.extraction_confidence,
            'target_confidence': target_emb.extraction_confidence,
            'average_quality': (ref_emb.feature_quality + target_emb.feature_quality) / 2,
            'average_confidence': (ref_emb.extraction_confidence + target_emb.extraction_confidence) / 2,
            'verification_confidence': verification_result.get('confidence', 0.0),
            'overall_quality_score': self._calculate_overall_quality_score(
                ref_emb, target_emb, verification_result
            )
        }
    
    def _calculate_overall_quality_score(self, ref_emb: FaceEmbedding,
                                       target_emb: FaceEmbedding,
                                       verification_result: dict) -> float:
        """Tính overall quality score"""
        try:
            # Factor 1: Average embedding quality
            avg_quality = (ref_emb.feature_quality + target_emb.feature_quality) / 2
            
            # Factor 2: Average extraction confidence
            avg_confidence = (ref_emb.extraction_confidence + target_emb.extraction_confidence) / 2
            
            # Factor 3: Verification confidence
            verification_confidence = verification_result.get('confidence', 0.0)
            
            # Factor 4: Face alignment
            avg_alignment = (ref_emb.face_alignment_score + target_emb.face_alignment_score) / 2
            
            # Combine factors
            overall_score = (
                avg_quality * 0.3 +
                avg_confidence * 0.3 +
                verification_confidence * 0.3 +
                avg_alignment * 0.1
            )
            
            return max(0.0, min(overall_score, 1.0))
            
        except Exception:
            return 0.5
    
    def _create_failed_verification(self, ref_image: str, target_image: str,
                                  status: VerificationStatus, error_message: str,
                                  start_time: float) -> FaceVerificationResult:
        """Create failed verification result"""
        processing_time = (time.time() - start_time) * 1000
        
        return FaceVerificationResult(
            reference_image_path=ref_image,
            target_image_path=target_image,
            status=status,
            verification_result=VerificationResult.NO_MATCH,
            similarity_score=0.0,
            confidence=0.0,
            processing_time_ms=processing_time,
            error_message=error_message
        )
    
    def get_embedding_statistics(self, embeddings: List[FaceEmbedding]) -> dict:
        """Lấy thống kê về các embeddings"""
        if not embeddings:
            return {
                'total_embeddings': 0,
                'valid_embeddings': 0,
                'average_quality': 0.0,
                'average_confidence': 0.0,
                'models_used': []
            }
        
        valid_embeddings = [emb for emb in embeddings if emb.is_valid_embedding()]
        
        # Calculate averages
        if valid_embeddings:
            avg_quality = sum(emb.feature_quality for emb in valid_embeddings) / len(valid_embeddings)
            avg_confidence = sum(emb.extraction_confidence for emb in valid_embeddings) / len(valid_embeddings)
        else:
            avg_quality = avg_confidence = 0.0
        
        # Models used
        models_used = list(set(emb.embedding_model for emb in embeddings))
        
        # Dimension distribution
        dimensions = [emb.get_embedding_dimension() for emb in valid_embeddings]
        
        return {
            'total_embeddings': len(embeddings),
            'valid_embeddings': len(valid_embeddings),
            'invalid_embeddings': len(embeddings) - len(valid_embeddings),
            'average_quality': round(avg_quality, 3),
            'average_confidence': round(avg_confidence, 3),
            'models_used': models_used,
            'dimensions': list(set(dimensions)),
            'quality_distribution': self._get_quality_distribution(valid_embeddings)
        }
    
    def _get_quality_distribution(self, embeddings: List[FaceEmbedding]) -> dict:
        """Get distribution của quality scores"""
        if not embeddings:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        high_quality = sum(1 for emb in embeddings if emb.feature_quality >= 0.7)
        medium_quality = sum(1 for emb in embeddings if 0.4 <= emb.feature_quality < 0.7)
        low_quality = sum(1 for emb in embeddings if emb.feature_quality < 0.4)
        
        return {
            'high': high_quality,
            'medium': medium_quality,
            'low': low_quality
        }
    
    def recommend_threshold_for_verification(self, verification_results: List[FaceVerificationResult]) -> dict:
        """Recommend optimal threshold dựa trên verification results"""
        if not verification_results:
            return {'recommended_threshold': 0.6, 'confidence': 0.0}
        
        # Analyze distribution of similarity scores
        similarities = [result.similarity_score for result in verification_results]
        
        import numpy as np
        
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        # Conservative threshold: mean - 0.5 * std
        conservative_threshold = max(0.3, mean_similarity - 0.5 * std_similarity)
        
        # Balanced threshold: mean
        balanced_threshold = mean_similarity
        
        # Liberal threshold: mean + 0.5 * std
        liberal_threshold = min(0.9, mean_similarity + 0.5 * std_similarity)
        
        return {
            'conservative_threshold': round(conservative_threshold, 3),
            'balanced_threshold': round(balanced_threshold, 3),
            'liberal_threshold': round(liberal_threshold, 3),
            'recommended_threshold': round(balanced_threshold, 3),
            'confidence': min(1.0 - std_similarity, 1.0),
            'analysis': {
                'mean_similarity': round(mean_similarity, 3),
                'std_similarity': round(std_similarity, 3),
                'min_similarity': round(min(similarities), 3),
                'max_similarity': round(max(similarities), 3)
            }
        }
