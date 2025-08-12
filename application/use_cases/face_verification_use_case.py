from domain.entities.face_verification_result import FaceEmbedding, FaceVerificationResult
from domain.repositories.face_verification_repository import FaceEmbeddingRepository, FaceVerificationRepository
from domain.services.face_verification_service import FaceVerificationService
from typing import Optional, List, Tuple
import logging

class FaceVerificationUseCase:
    """Use case cho face verification"""
    
    def __init__(self,
                 face_verification_service: FaceVerificationService,
                 embedding_repository: FaceEmbeddingRepository,
                 verification_repository: FaceVerificationRepository):
        self.face_verification_service = face_verification_service
        self.embedding_repository = embedding_repository
        self.verification_repository = verification_repository
        self.logger = logging.getLogger(__name__)
    
    async def extract_and_save_embedding(self, image_path: str, face_bbox: List[int],
                                        model_type: str = "facenet") -> FaceEmbedding:
        """
        Extract và lưu face embedding
        """
        try:
            self.logger.info(f"Extracting embedding for: {image_path}")
            
            # Extract embedding
            embedding = self.face_verification_service.extract_face_embedding(
                image_path, face_bbox, model_type
            )
            
            # Save to repository
            saved_embedding = await self.embedding_repository.save_embedding(embedding)
            
            self.logger.info(f"Embedding extracted and saved with ID: {saved_embedding.id}")
            
            return saved_embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting and saving embedding: {e}")
            raise
    
    async def verify_faces_by_embeddings(self, reference_embedding_id: str,
                                        target_embedding_id: str,
                                        threshold: Optional[float] = None,
                                        distance_metric: str = "cosine") -> FaceVerificationResult:
        """
        Verify faces sử dụng existing embeddings
        """
        try:
            # Retrieve embeddings
            reference_embedding = await self.embedding_repository.find_embedding_by_id(reference_embedding_id)
            target_embedding = await self.embedding_repository.find_embedding_by_id(target_embedding_id)
            
            if not reference_embedding:
                raise ValueError(f"Reference embedding not found: {reference_embedding_id}")
            
            if not target_embedding:
                raise ValueError(f"Target embedding not found: {target_embedding_id}")
            
            # Perform verification
            verification_result = self.face_verification_service.verify_faces(
                reference_embedding, target_embedding, threshold, distance_metric
            )
            
            # Save verification result
            saved_result = await self.verification_repository.save_verification(verification_result)
            
            self.logger.info(f"Face verification completed: {saved_result.verification_result}")
            
            return saved_result
            
        except Exception as e:
            self.logger.error(f"Error verifying faces by embeddings: {e}")
            raise
    
    async def verify_faces_by_images(self, reference_image_path: str, reference_face_bbox: List[int],
                                    target_image_path: str, target_face_bbox: List[int],
                                    threshold: Optional[float] = None,
                                    distance_metric: str = "cosine",
                                    model_type: str = "facenet") -> FaceVerificationResult:
        """
        Verify faces trực tiếp từ images
        """
        try:
            self.logger.info(f"Verifying faces: {reference_image_path} vs {target_image_path}")
            
            # Extract embeddings
            reference_embedding = await self.extract_and_save_embedding(
                reference_image_path, reference_face_bbox, model_type
            )
            
            target_embedding = await self.extract_and_save_embedding(
                target_image_path, target_face_bbox, model_type
            )
            
            # Perform verification
            verification_result = self.face_verification_service.verify_faces(
                reference_embedding, target_embedding, threshold, distance_metric
            )
            
            # Update with embedding IDs
            verification_result.reference_embedding_id = reference_embedding.id
            verification_result.target_embedding_id = target_embedding.id
            
            # Save verification result
            saved_result = await self.verification_repository.save_verification(verification_result)
            
            self.logger.info(f"Face verification completed: {saved_result.verification_result}")
            
            return saved_result
            
        except Exception as e:
            self.logger.error(f"Error verifying faces by images: {e}")
            raise
    
    async def find_matches_in_gallery(self, target_image_path: str, target_face_bbox: List[int],
                                     gallery_image_paths: List[str] = None,
                                     top_k: int = 5,
                                     threshold: Optional[float] = None,
                                     distance_metric: str = "cosine",
                                     model_type: str = "facenet") -> List[FaceVerificationResult]:
        """
        Tìm matches trong gallery
        """
        try:
            self.logger.info(f"Finding matches for: {target_image_path}")
            
            # Extract target embedding
            target_embedding = await self.extract_and_save_embedding(
                target_image_path, target_face_bbox, model_type
            )
            
            # Get gallery embeddings
            if gallery_image_paths:
                # Filter embeddings by image paths
                all_embeddings = await self.embedding_repository.get_all_embeddings()
                gallery_embeddings = [
                    emb for emb in all_embeddings 
                    if emb.image_path in gallery_image_paths
                ]
            else:
                # Use all embeddings as gallery
                gallery_embeddings = await self.embedding_repository.get_all_embeddings()
                # Remove target embedding from gallery
                gallery_embeddings = [
                    emb for emb in gallery_embeddings 
                    if emb.id != target_embedding.id
                ]
            
            if not gallery_embeddings:
                self.logger.warning("No gallery embeddings found")
                return []
            
            # Find best matches
            matches = self.face_verification_service.find_best_matches(
                target_embedding, gallery_embeddings, top_k, threshold, distance_metric
            )
            
            # Save verification results
            saved_matches = []
            for match in matches:
                match.target_embedding_id = target_embedding.id
                saved_match = await self.verification_repository.save_verification(match)
                saved_matches.append(saved_match)
            
            self.logger.info(f"Found {len(saved_matches)} matches")
            
            return saved_matches
            
        except Exception as e:
            self.logger.error(f"Error finding matches in gallery: {e}")
            raise
    
    async def batch_verify_faces(self, verification_pairs: List[Tuple[str, List[int], str, List[int]]],
                                threshold: Optional[float] = None,
                                distance_metric: str = "cosine",
                                model_type: str = "facenet") -> List[FaceVerificationResult]:
        """
        Batch verification cho nhiều cặp faces
        verification_pairs: List of (ref_image, ref_bbox, target_image, target_bbox)
        """
        try:
            self.logger.info(f"Batch verifying {len(verification_pairs)} face pairs")
            
            results = []
            
            for ref_image, ref_bbox, target_image, target_bbox in verification_pairs:
                try:
                    result = await self.verify_faces_by_images(
                        ref_image, ref_bbox, target_image, target_bbox,
                        threshold, distance_metric, model_type
                    )
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Error verifying pair {ref_image} vs {target_image}: {e}")
                    # Continue with other pairs
                    continue
            
            self.logger.info(f"Batch verification completed: {len(results)} successful")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch verification: {e}")
            raise
    
    async def get_verification_result(self, verification_id: str) -> Optional[FaceVerificationResult]:
        """Lấy verification result theo ID"""
        try:
            return await self.verification_repository.find_verification_by_id(verification_id)
        except Exception as e:
            self.logger.error(f"Error getting verification result: {e}")
            return None
    
    async def get_embedding(self, embedding_id: str) -> Optional[FaceEmbedding]:
        """Lấy embedding theo ID"""
        try:
            return await self.embedding_repository.find_embedding_by_id(embedding_id)
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            return None
    
    async def get_verifications_for_image_pair(self, ref_image: str, target_image: str) -> List[FaceVerificationResult]:
        """Lấy tất cả verification results cho một cặp ảnh"""
        try:
            return await self.verification_repository.find_verifications_by_images(ref_image, target_image)
        except Exception as e:
            self.logger.error(f"Error getting verifications for image pair: {e}")
            return []
    
    async def get_verification_statistics(self) -> dict:
        """Lấy thống kê verification"""
        try:
            # Get repository statistics
            repo_stats = await self.verification_repository.get_verification_statistics()
            
            # Get embedding statistics
            all_embeddings = await self.embedding_repository.get_all_embeddings()
            embedding_stats = self.face_verification_service.get_embedding_statistics(all_embeddings)
            
            # Get recent verifications for analysis
            recent_verifications = await self.verification_repository.get_recent_verifications(100)
            
            # Threshold recommendation
            threshold_analysis = self.face_verification_service.recommend_threshold_for_verification(
                recent_verifications
            )
            
            return {
                'verification_statistics': repo_stats,
                'embedding_statistics': embedding_stats,
                'threshold_analysis': threshold_analysis,
                'performance_metrics': self._calculate_performance_metrics(recent_verifications)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting verification statistics: {e}")
            return {
                'verification_statistics': {},
                'embedding_statistics': {},
                'threshold_analysis': {},
                'performance_metrics': {},
                'error': str(e)
            }
    
    async def optimize_threshold(self, positive_pairs: List[Tuple[str, str]], 
                                negative_pairs: List[Tuple[str, str]],
                                distance_metric: str = "cosine") -> dict:
        """
        Optimize threshold sử dụng positive và negative pairs
        """
        try:
            self.logger.info("Optimizing verification threshold")
            
            # Get embeddings for positive pairs
            positive_embedding_pairs = []
            for ref_image, target_image in positive_pairs:
                ref_embeddings = await self.embedding_repository.find_embeddings_by_image_path(ref_image)
                target_embeddings = await self.embedding_repository.find_embeddings_by_image_path(target_image)
                
                if ref_embeddings and target_embeddings:
                    positive_embedding_pairs.append((
                        ref_embeddings[0].embedding_vector,
                        target_embeddings[0].embedding_vector
                    ))
            
            # Get embeddings for negative pairs
            negative_embedding_pairs = []
            for ref_image, target_image in negative_pairs:
                ref_embeddings = await self.embedding_repository.find_embeddings_by_image_path(ref_image)
                target_embeddings = await self.embedding_repository.find_embeddings_by_image_path(target_image)
                
                if ref_embeddings and target_embeddings:
                    negative_embedding_pairs.append((
                        ref_embeddings[0].embedding_vector,
                        target_embeddings[0].embedding_vector
                    ))
            
            if not positive_embedding_pairs or not negative_embedding_pairs:
                raise ValueError("Insufficient embedding pairs for optimization")
            
            # Use verification engine to find best threshold
            from infrastructure.ml_models.face_verification_engine import FaceVerificationEngine
            engine = FaceVerificationEngine()
            
            optimization_result = engine.find_best_threshold(
                positive_embedding_pairs, negative_embedding_pairs, distance_metric
            )
            
            self.logger.info(f"Threshold optimization completed: {optimization_result['best_threshold']}")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing threshold: {e}")
            return {
                'best_threshold': 0.6,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'error': str(e)
            }
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> dict:
        """Clean up old embeddings và verification results"""
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Get old verification results
            all_verifications = await self.verification_repository.get_recent_verifications(1000)
            old_verifications = [
                v for v in all_verifications 
                if v.created_at and v.created_at < cutoff_date
            ]
            
            # Delete old verifications
            deleted_verifications = 0
            for verification in old_verifications:
                if await self.verification_repository.delete_verification(verification.id):
                    deleted_verifications += 1
            
            # Get old embeddings
            all_embeddings = await self.embedding_repository.get_all_embeddings()
            old_embeddings = [
                e for e in all_embeddings 
                if e.created_at and e.created_at < cutoff_date
            ]
            
            # Delete old embeddings
            deleted_embeddings = 0
            for embedding in old_embeddings:
                if await self.embedding_repository.delete_embedding(embedding.id):
                    deleted_embeddings += 1
            
            self.logger.info(f"Cleanup completed: {deleted_verifications} verifications, {deleted_embeddings} embeddings")
            
            return {
                'deleted_verifications': deleted_verifications,
                'deleted_embeddings': deleted_embeddings,
                'cutoff_date': cutoff_date.isoformat(),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {
                'deleted_verifications': 0,
                'deleted_embeddings': 0,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_performance_metrics(self, verifications: List[FaceVerificationResult]) -> dict:
        """Tính toán performance metrics"""
        if not verifications:
            return {
                'total_verifications': 0,
                'success_rate': 0.0,
                'average_processing_time': 0.0,
                'average_similarity_score': 0.0,
                'average_confidence': 0.0
            }
        
        successful_verifications = [
            v for v in verifications if v.status.value == "success"
        ]
        
        success_rate = len(successful_verifications) / len(verifications) * 100
        
        if successful_verifications:
            avg_processing_time = sum(v.processing_time_ms for v in successful_verifications) / len(successful_verifications)
            avg_similarity = sum(v.similarity_score for v in successful_verifications) / len(successful_verifications)
            avg_confidence = sum(v.confidence for v in successful_verifications) / len(successful_verifications)
        else:
            avg_processing_time = avg_similarity = avg_confidence = 0.0
        
        return {
            'total_verifications': len(verifications),
            'successful_verifications': len(successful_verifications),
            'success_rate': round(success_rate, 2),
            'average_processing_time': round(avg_processing_time, 2),
            'average_similarity_score': round(avg_similarity, 3),
            'average_confidence': round(avg_confidence, 3),
            'match_rate': round(
                len([v for v in successful_verifications if v.verification_result.value == "match"]) / 
                len(successful_verifications) * 100 if successful_verifications else 0, 2
            )
        }
