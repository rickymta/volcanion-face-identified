from domain.entities.face_detection_result import FaceDetectionResult
from domain.repositories.face_detection_repository import FaceDetectionRepository
from infrastructure.database.mongodb_client import get_database
from pymongo.collection import Collection
from typing import List, Optional
from datetime import datetime
import logging

class MongoFaceDetectionRepository(FaceDetectionRepository):
    """MongoDB implementation của FaceDetectionRepository"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db = get_database()
        self.collection: Collection = self.db.face_detections
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Tạo indexes cho collection"""
        try:
            # Index cho image_path
            self.collection.create_index("image_path")
            
            # Index cho status
            self.collection.create_index("status")
            
            # Index cho created_at
            self.collection.create_index("created_at")
            
            # Index cho confidence (descending)
            self.collection.create_index([("confidence", -1)])
            
            self.logger.info("Face detection indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
    
    async def save(self, face_detection_result: FaceDetectionResult) -> FaceDetectionResult:
        """Lưu face detection result"""
        try:
            # Convert to dict
            doc = self._to_document(face_detection_result)
            
            if face_detection_result.id:
                # Update existing
                result = await self.collection.replace_one(
                    {"_id": face_detection_result.id},
                    doc
                )
                if result.matched_count == 0:
                    raise ValueError(f"Face detection result not found: {face_detection_result.id}")
                
                self.logger.info(f"Updated face detection result: {face_detection_result.id}")
                
            else:
                # Insert new
                result = await self.collection.insert_one(doc)
                face_detection_result.id = str(result.inserted_id)
                
                self.logger.info(f"Saved new face detection result: {face_detection_result.id}")
            
            return face_detection_result
            
        except Exception as e:
            self.logger.error(f"Error saving face detection result: {e}")
            raise
    
    async def find_by_id(self, face_detection_id: str) -> Optional[FaceDetectionResult]:
        """Tìm face detection result theo ID"""
        try:
            from bson import ObjectId
            
            doc = await self.collection.find_one({"_id": ObjectId(face_detection_id)})
            
            if doc:
                return self._from_document(doc)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding face detection result by id: {e}")
            return None
    
    async def find_by_image_path(self, image_path: str) -> List[FaceDetectionResult]:
        """Tìm tất cả face detection results của một ảnh"""
        try:
            cursor = self.collection.find({"image_path": image_path})
            cursor = cursor.sort("created_at", -1)  # Newest first
            
            results = []
            async for doc in cursor:
                results.append(self._from_document(doc))
            
            self.logger.info(f"Found {len(results)} face detection results for {image_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding face detections by image path: {e}")
            return []
    
    async def find_all(self) -> List[FaceDetectionResult]:
        """Lấy tất cả face detection results"""
        try:
            cursor = self.collection.find({})
            cursor = cursor.sort("created_at", -1)
            
            results = []
            async for doc in cursor:
                results.append(self._from_document(doc))
            
            self.logger.info(f"Found {len(results)} total face detection results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding all face detections: {e}")
            return []
    
    async def find_by_status(self, status: str) -> List[FaceDetectionResult]:
        """Tìm face detection results theo status"""
        try:
            cursor = self.collection.find({"status": status})
            cursor = cursor.sort("created_at", -1)
            
            results = []
            async for doc in cursor:
                results.append(self._from_document(doc))
            
            self.logger.info(f"Found {len(results)} face detection results with status: {status}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding face detections by status: {e}")
            return []
    
    async def delete_by_id(self, face_detection_id: str) -> bool:
        """Xóa face detection result theo ID"""
        try:
            from bson import ObjectId
            
            result = await self.collection.delete_one({"_id": ObjectId(face_detection_id)})
            
            if result.deleted_count > 0:
                self.logger.info(f"Deleted face detection result: {face_detection_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting face detection result: {e}")
            return False
    
    async def delete_by_image_path(self, image_path: str) -> int:
        """Xóa tất cả face detection results của một ảnh"""
        try:
            result = await self.collection.delete_many({"image_path": image_path})
            
            deleted_count = result.deleted_count
            self.logger.info(f"Deleted {deleted_count} face detection results for {image_path}")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error deleting face detections by image path: {e}")
            return 0
    
    async def count_by_status(self, status: str) -> int:
        """Đếm số face detection results theo status"""
        try:
            count = await self.collection.count_documents({"status": status})
            return count
            
        except Exception as e:
            self.logger.error(f"Error counting face detections by status: {e}")
            return 0
    
    async def get_recent_results(self, limit: int = 50) -> List[FaceDetectionResult]:
        """Lấy face detection results gần đây"""
        try:
            cursor = self.collection.find({})
            cursor = cursor.sort("created_at", -1).limit(limit)
            
            results = []
            async for doc in cursor:
                results.append(self._from_document(doc))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting recent face detections: {e}")
            return []
    
    def _to_document(self, face_detection_result: FaceDetectionResult) -> dict:
        """Convert FaceDetectionResult to MongoDB document"""
        doc = {
            "image_path": face_detection_result.image_path,
            "status": face_detection_result.status,
            "bbox": face_detection_result.bbox,
            "landmarks": face_detection_result.landmarks,
            "confidence": face_detection_result.confidence,
            "face_size": face_detection_result.face_size,
            "occlusion_detected": face_detection_result.occlusion_detected,
            "occlusion_type": face_detection_result.occlusion_type,
            "occlusion_confidence": face_detection_result.occlusion_confidence,
            "alignment_score": face_detection_result.alignment_score,
            "face_quality_score": face_detection_result.face_quality_score,
            "pose_angles": face_detection_result.pose_angles,
            "created_at": face_detection_result.created_at,
            "updated_at": datetime.utcnow()
        }
        
        # Remove None values
        doc = {k: v for k, v in doc.items() if v is not None}
        
        return doc
    
    def _from_document(self, doc: dict) -> FaceDetectionResult:
        """Convert MongoDB document to FaceDetectionResult"""
        # Convert ObjectId to string
        face_id = str(doc["_id"]) if "_id" in doc else None
        
        return FaceDetectionResult(
            id=face_id,
            image_path=doc.get("image_path"),
            status=doc.get("status"),
            bbox=doc.get("bbox"),
            landmarks=doc.get("landmarks"),
            confidence=doc.get("confidence", 0.0),
            face_size=doc.get("face_size"),
            occlusion_detected=doc.get("occlusion_detected", False),
            occlusion_type=doc.get("occlusion_type"),
            occlusion_confidence=doc.get("occlusion_confidence"),
            alignment_score=doc.get("alignment_score"),
            face_quality_score=doc.get("face_quality_score"),
            pose_angles=doc.get("pose_angles"),
            created_at=doc.get("created_at"),
            updated_at=doc.get("updated_at")
        )
