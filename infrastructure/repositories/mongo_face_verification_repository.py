from domain.entities.face_verification_result import FaceEmbedding, FaceVerificationResult
from domain.repositories.face_verification_repository import FaceEmbeddingRepository, FaceVerificationRepository
from infrastructure.database.mongodb_client import get_database
from pymongo.collection import Collection
from typing import List, Optional
from datetime import datetime
import logging

class MongoFaceEmbeddingRepository(FaceEmbeddingRepository):
    """MongoDB implementation cho FaceEmbeddingRepository"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db = get_database()
        self.collection: Collection = self.db.face_embeddings
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Tạo indexes cho collection"""
        try:
            # Index cho image_path
            self.collection.create_index("image_path")
            
            # Index cho embedding_model
            self.collection.create_index("embedding_model")
            
            # Index cho created_at
            self.collection.create_index("created_at")
            
            # Index cho feature_quality (descending)
            self.collection.create_index([("feature_quality", -1)])
            
            # Index cho extraction_confidence (descending)
            self.collection.create_index([("extraction_confidence", -1)])
            
            self.logger.info("Face embedding indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
    
    async def save_embedding(self, embedding: FaceEmbedding) -> FaceEmbedding:
        """Lưu face embedding"""
        try:
            # Convert to dict
            doc = self._to_document(embedding)
            
            if embedding.id:
                # Update existing
                result = await self.collection.replace_one(
                    {"_id": embedding.id},
                    doc
                )
                if result.matched_count == 0:
                    raise ValueError(f"Face embedding not found: {embedding.id}")
                
                self.logger.info(f"Updated face embedding: {embedding.id}")
                
            else:
                # Insert new
                result = await self.collection.insert_one(doc)
                embedding.id = str(result.inserted_id)
                
                self.logger.info(f"Saved new face embedding: {embedding.id}")
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error saving face embedding: {e}")
            raise
    
    async def find_embedding_by_id(self, embedding_id: str) -> Optional[FaceEmbedding]:
        """Tìm embedding theo ID"""
        try:
            from bson import ObjectId
            
            doc = await self.collection.find_one({"_id": ObjectId(embedding_id)})
            
            if doc:
                return self._from_document(doc)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding embedding by id: {e}")
            return None
    
    async def find_embeddings_by_image_path(self, image_path: str) -> List[FaceEmbedding]:
        """Tìm embeddings theo đường dẫn ảnh"""
        try:
            cursor = self.collection.find({"image_path": image_path})
            cursor = cursor.sort("created_at", -1)  # Newest first
            
            embeddings = []
            async for doc in cursor:
                embeddings.append(self._from_document(doc))
            
            self.logger.info(f"Found {len(embeddings)} embeddings for {image_path}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error finding embeddings by image path: {e}")
            return []
    
    async def find_embeddings_by_model(self, model_name: str) -> List[FaceEmbedding]:
        """Tìm embeddings theo model"""
        try:
            cursor = self.collection.find({"embedding_model": model_name})
            cursor = cursor.sort("created_at", -1)
            
            embeddings = []
            async for doc in cursor:
                embeddings.append(self._from_document(doc))
            
            self.logger.info(f"Found {len(embeddings)} embeddings for model: {model_name}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error finding embeddings by model: {e}")
            return []
    
    async def delete_embedding(self, embedding_id: str) -> bool:
        """Xóa embedding"""
        try:
            from bson import ObjectId
            
            result = await self.collection.delete_one({"_id": ObjectId(embedding_id)})
            
            if result.deleted_count > 0:
                self.logger.info(f"Deleted embedding: {embedding_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting embedding: {e}")
            return False
    
    async def get_all_embeddings(self) -> List[FaceEmbedding]:
        """Lấy tất cả embeddings"""
        try:
            cursor = self.collection.find({})
            cursor = cursor.sort("created_at", -1)
            
            embeddings = []
            async for doc in cursor:
                embeddings.append(self._from_document(doc))
            
            self.logger.info(f"Found {len(embeddings)} total embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error finding all embeddings: {e}")
            return []
    
    def _to_document(self, embedding: FaceEmbedding) -> dict:
        """Convert FaceEmbedding to MongoDB document"""
        doc = {
            "image_path": embedding.image_path,
            "face_bbox": embedding.face_bbox,
            "embedding_vector": embedding.embedding_vector,
            "embedding_model": embedding.embedding_model,
            "feature_quality": embedding.feature_quality,
            "extraction_confidence": embedding.extraction_confidence,
            "face_alignment_score": embedding.face_alignment_score,
            "preprocessing_applied": embedding.preprocessing_applied,
            "created_at": embedding.created_at,
            "updated_at": datetime.utcnow()
        }
        
        # Remove None values
        doc = {k: v for k, v in doc.items() if v is not None}
        
        return doc
    
    def _from_document(self, doc: dict) -> FaceEmbedding:
        """Convert MongoDB document to FaceEmbedding"""
        # Convert ObjectId to string
        embedding_id = str(doc["_id"]) if "_id" in doc else None
        
        return FaceEmbedding(
            id=embedding_id,
            image_path=doc.get("image_path", ""),
            face_bbox=doc.get("face_bbox"),
            embedding_vector=doc.get("embedding_vector"),
            embedding_model=doc.get("embedding_model", "unknown"),
            feature_quality=doc.get("feature_quality", 0.0),
            extraction_confidence=doc.get("extraction_confidence", 0.0),
            face_alignment_score=doc.get("face_alignment_score", 0.0),
            preprocessing_applied=doc.get("preprocessing_applied", False),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at")
        )

class MongoFaceVerificationRepository(FaceVerificationRepository):
    """MongoDB implementation cho FaceVerificationRepository"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db = get_database()
        self.collection: Collection = self.db.face_verifications
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Tạo indexes cho collection"""
        try:
            # Compound index cho image pair
            self.collection.create_index([
                ("reference_image_path", 1),
                ("target_image_path", 1)
            ])
            
            # Index cho status
            self.collection.create_index("status")
            
            # Index cho verification_result
            self.collection.create_index("verification_result")
            
            # Index cho created_at
            self.collection.create_index("created_at")
            
            # Index cho similarity_score (descending)
            self.collection.create_index([("similarity_score", -1)])
            
            # Index cho confidence (descending)
            self.collection.create_index([("confidence", -1)])
            
            # Index cho embedding IDs
            self.collection.create_index("reference_embedding_id")
            self.collection.create_index("target_embedding_id")
            
            self.logger.info("Face verification indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
    
    async def save_verification(self, verification: FaceVerificationResult) -> FaceVerificationResult:
        """Lưu verification result"""
        try:
            # Convert to dict
            doc = self._to_document(verification)
            
            if verification.id:
                # Update existing
                result = await self.collection.replace_one(
                    {"_id": verification.id},
                    doc
                )
                if result.matched_count == 0:
                    raise ValueError(f"Face verification not found: {verification.id}")
                
                self.logger.info(f"Updated face verification: {verification.id}")
                
            else:
                # Insert new
                result = await self.collection.insert_one(doc)
                verification.id = str(result.inserted_id)
                
                self.logger.info(f"Saved new face verification: {verification.id}")
            
            return verification
            
        except Exception as e:
            self.logger.error(f"Error saving face verification: {e}")
            raise
    
    async def find_verification_by_id(self, verification_id: str) -> Optional[FaceVerificationResult]:
        """Tìm verification result theo ID"""
        try:
            from bson import ObjectId
            
            doc = await self.collection.find_one({"_id": ObjectId(verification_id)})
            
            if doc:
                return self._from_document(doc)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding verification by id: {e}")
            return None
    
    async def find_verifications_by_images(self, ref_image: str, target_image: str) -> List[FaceVerificationResult]:
        """Tìm verification results theo cặp ảnh"""
        try:
            cursor = self.collection.find({
                "reference_image_path": ref_image,
                "target_image_path": target_image
            })
            cursor = cursor.sort("created_at", -1)
            
            verifications = []
            async for doc in cursor:
                verifications.append(self._from_document(doc))
            
            self.logger.info(f"Found {len(verifications)} verifications for image pair")
            return verifications
            
        except Exception as e:
            self.logger.error(f"Error finding verifications by images: {e}")
            return []
    
    async def find_verifications_by_status(self, status: str) -> List[FaceVerificationResult]:
        """Tìm verification results theo status"""
        try:
            cursor = self.collection.find({"status": status})
            cursor = cursor.sort("created_at", -1)
            
            verifications = []
            async for doc in cursor:
                verifications.append(self._from_document(doc))
            
            self.logger.info(f"Found {len(verifications)} verifications with status: {status}")
            return verifications
            
        except Exception as e:
            self.logger.error(f"Error finding verifications by status: {e}")
            return []
    
    async def find_verifications_by_result(self, result: str) -> List[FaceVerificationResult]:
        """Tìm verification results theo kết quả"""
        try:
            cursor = self.collection.find({"verification_result": result})
            cursor = cursor.sort("created_at", -1)
            
            verifications = []
            async for doc in cursor:
                verifications.append(self._from_document(doc))
            
            self.logger.info(f"Found {len(verifications)} verifications with result: {result}")
            return verifications
            
        except Exception as e:
            self.logger.error(f"Error finding verifications by result: {e}")
            return []
    
    async def get_verification_statistics(self) -> dict:
        """Lấy thống kê verification"""
        try:
            # Total count
            total_count = await self.collection.count_documents({})
            
            # Status distribution
            status_pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_cursor = self.collection.aggregate(status_pipeline)
            status_distribution = {}
            async for doc in status_cursor:
                status_distribution[doc["_id"]] = doc["count"]
            
            # Result distribution
            result_pipeline = [
                {"$group": {"_id": "$verification_result", "count": {"$sum": 1}}}
            ]
            result_cursor = self.collection.aggregate(result_pipeline)
            result_distribution = {}
            async for doc in result_cursor:
                result_distribution[doc["_id"]] = doc["count"]
            
            # Average scores
            avg_pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_similarity": {"$avg": "$similarity_score"},
                    "avg_confidence": {"$avg": "$confidence"},
                    "avg_processing_time": {"$avg": "$processing_time_ms"}
                }}
            ]
            avg_cursor = self.collection.aggregate(avg_pipeline)
            avg_stats = {}
            async for doc in avg_cursor:
                avg_stats = {
                    "average_similarity": round(doc.get("avg_similarity", 0.0), 3),
                    "average_confidence": round(doc.get("avg_confidence", 0.0), 3),
                    "average_processing_time": round(doc.get("avg_processing_time", 0.0), 2)
                }
            
            return {
                "total_verifications": total_count,
                "status_distribution": status_distribution,
                "result_distribution": result_distribution,
                **avg_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error getting verification statistics: {e}")
            return {}
    
    async def delete_verification(self, verification_id: str) -> bool:
        """Xóa verification result"""
        try:
            from bson import ObjectId
            
            result = await self.collection.delete_one({"_id": ObjectId(verification_id)})
            
            if result.deleted_count > 0:
                self.logger.info(f"Deleted verification: {verification_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting verification: {e}")
            return False
    
    async def get_recent_verifications(self, limit: int = 50) -> List[FaceVerificationResult]:
        """Lấy verification results gần đây"""
        try:
            cursor = self.collection.find({})
            cursor = cursor.sort("created_at", -1).limit(limit)
            
            verifications = []
            async for doc in cursor:
                verifications.append(self._from_document(doc))
            
            return verifications
            
        except Exception as e:
            self.logger.error(f"Error getting recent verifications: {e}")
            return []
    
    def _to_document(self, verification: FaceVerificationResult) -> dict:
        """Convert FaceVerificationResult to MongoDB document"""
        doc = {
            "reference_image_path": verification.reference_image_path,
            "target_image_path": verification.target_image_path,
            "reference_embedding_id": verification.reference_embedding_id,
            "target_embedding_id": verification.target_embedding_id,
            "status": verification.status.value if hasattr(verification.status, 'value') else verification.status,
            "verification_result": verification.verification_result.value if hasattr(verification.verification_result, 'value') else verification.verification_result,
            "similarity_score": verification.similarity_score,
            "distance_metric": verification.distance_metric,
            "confidence": verification.confidence,
            "threshold_used": verification.threshold_used,
            "match_probability": verification.match_probability,
            "processing_time_ms": verification.processing_time_ms,
            "model_used": verification.model_used,
            "quality_assessment": verification.quality_assessment,
            "error_message": verification.error_message,
            "created_at": verification.created_at,
            "updated_at": datetime.utcnow()
        }
        
        # Remove None values
        doc = {k: v for k, v in doc.items() if v is not None}
        
        return doc
    
    def _from_document(self, doc: dict) -> FaceVerificationResult:
        """Convert MongoDB document to FaceVerificationResult"""
        # Convert ObjectId to string
        verification_id = str(doc["_id"]) if "_id" in doc else None
        
        return FaceVerificationResult(
            id=verification_id,
            reference_image_path=doc.get("reference_image_path", ""),
            target_image_path=doc.get("target_image_path", ""),
            reference_embedding_id=doc.get("reference_embedding_id"),
            target_embedding_id=doc.get("target_embedding_id"),
            status=doc.get("status", "failed"),
            verification_result=doc.get("verification_result", "no_match"),
            similarity_score=doc.get("similarity_score", 0.0),
            distance_metric=doc.get("distance_metric", "cosine"),
            confidence=doc.get("confidence", 0.0),
            threshold_used=doc.get("threshold_used", 0.6),
            match_probability=doc.get("match_probability", 0.0),
            processing_time_ms=doc.get("processing_time_ms", 0.0),
            model_used=doc.get("model_used", "unknown"),
            quality_assessment=doc.get("quality_assessment", {}),
            error_message=doc.get("error_message"),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at")
        )
