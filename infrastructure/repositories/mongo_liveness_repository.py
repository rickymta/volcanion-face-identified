from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import PyMongoError
import logging
import os
from domain.repositories.liveness_repository import LivenessRepository
from domain.entities.liveness_result import LivenessDetectionResult, LivenessStatus, LivenessResult, SpoofType

# Initialize logger
logger = logging.getLogger(__name__)

class MongoLivenessRepository(LivenessRepository):
    """MongoDB implementation của LivenessRepository"""
    
    def __init__(self, connection_string: str = None):
        try:
            # Default connection string
            if not connection_string:
                connection_string = os.getenv('MONGODB_CONNECTION_STRING', 'mongodb://localhost:27017/')
            
            self.client = MongoClient(connection_string)
            self.db = self.client.volcanion_face_db
            self.collection: Collection = self.db.liveness_results
            
            # Create indexes
            self._create_indexes()
            
            logger.info("MongoDB Liveness Repository initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing MongoDB Liveness Repository: {e}")
            raise e
    
    def _create_indexes(self):
        """Tạo indexes cho performance"""
        try:
            # Basic indexes
            self.collection.create_index("image_path")
            self.collection.create_index("status")
            self.collection.create_index("liveness_result")
            self.collection.create_index("created_at", direction=DESCENDING)
            self.collection.create_index("updated_at", direction=DESCENDING)
            
            # Compound indexes
            self.collection.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
            self.collection.create_index([("liveness_result", ASCENDING), ("confidence", DESCENDING)])
            self.collection.create_index([("detected_spoof_types", ASCENDING), ("created_at", DESCENDING)])
            
            # Spoof-specific indexes
            self.collection.create_index("primary_spoof_type")
            self.collection.create_index("spoof_probability", direction=DESCENDING)
            
            # Quality indexes
            self.collection.create_index("image_quality", direction=DESCENDING)
            self.collection.create_index("face_quality", direction=DESCENDING)
            self.collection.create_index("confidence", direction=DESCENDING)
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    async def save(self, result: LivenessDetectionResult) -> LivenessDetectionResult:
        """Lưu liveness detection result"""
        try:
            # Convert to dict
            result_dict = result.to_dict()
            
            # Insert to MongoDB
            insert_result = self.collection.insert_one(result_dict)
            
            # Update ID if new
            if insert_result.inserted_id:
                result.id = str(insert_result.inserted_id)
            
            logger.debug(f"Liveness result saved: {result.id}")
            return result
            
        except PyMongoError as e:
            logger.error(f"MongoDB error saving liveness result: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error saving liveness result: {e}")
            raise e
    
    async def find_by_id(self, result_id: str) -> Optional[LivenessDetectionResult]:
        """Tìm liveness result theo ID"""
        try:
            document = self.collection.find_one({"id": result_id})
            
            if document:
                return LivenessDetectionResult.from_dict(document)
            
            return None
            
        except PyMongoError as e:
            logger.error(f"MongoDB error finding liveness result by ID {result_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error finding liveness result by ID {result_id}: {e}")
            return None
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[LivenessDetectionResult]:
        """Lấy tất cả liveness results"""
        try:
            cursor = self.collection.find().sort("created_at", DESCENDING).skip(offset).limit(limit)
            
            results = []
            for document in cursor:
                try:
                    result = LivenessDetectionResult.from_dict(document)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error converting document to LivenessDetectionResult: {e}")
                    continue
            
            return results
            
        except PyMongoError as e:
            logger.error(f"MongoDB error finding all liveness results: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding all liveness results: {e}")
            return []
    
    async def find_by_image_path(self, image_path: str) -> List[LivenessDetectionResult]:
        """Tìm liveness results theo image path"""
        try:
            cursor = self.collection.find({"image_path": image_path}).sort("created_at", DESCENDING)
            
            results = []
            for document in cursor:
                try:
                    result = LivenessDetectionResult.from_dict(document)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error converting document: {e}")
                    continue
            
            return results
            
        except PyMongoError as e:
            logger.error(f"MongoDB error finding liveness results by image path: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding liveness results by image path: {e}")
            return []
    
    async def find_by_status(self, status: str, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm liveness results theo status"""
        try:
            cursor = self.collection.find({"status": status}).sort("created_at", DESCENDING).limit(limit)
            
            results = []
            for document in cursor:
                try:
                    result = LivenessDetectionResult.from_dict(document)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error converting document: {e}")
                    continue
            
            return results
            
        except PyMongoError as e:
            logger.error(f"MongoDB error finding liveness results by status: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding liveness results by status: {e}")
            return []
    
    async def find_by_result(self, liveness_result: str, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm liveness results theo kết quả"""
        try:
            cursor = self.collection.find({"liveness_result": liveness_result}).sort("confidence", DESCENDING).limit(limit)
            
            results = []
            for document in cursor:
                try:
                    result = LivenessDetectionResult.from_dict(document)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error converting document: {e}")
                    continue
            
            return results
            
        except PyMongoError as e:
            logger.error(f"MongoDB error finding liveness results by result: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding liveness results by result: {e}")
            return []
    
    async def find_real_faces(self, confidence_threshold: float = 0.8, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm các khuôn mặt thật với confidence cao"""
        try:
            query = {
                "liveness_result": "REAL",
                "confidence": {"$gte": confidence_threshold}
            }
            
            cursor = self.collection.find(query).sort("confidence", DESCENDING).limit(limit)
            
            results = []
            for document in cursor:
                try:
                    result = LivenessDetectionResult.from_dict(document)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error converting document: {e}")
                    continue
            
            return results
            
        except PyMongoError as e:
            logger.error(f"MongoDB error finding real faces: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding real faces: {e}")
            return []
    
    async def find_fake_faces(self, confidence_threshold: float = 0.8, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm các khuôn mặt giả được phát hiện"""
        try:
            query = {
                "liveness_result": "FAKE",
                "confidence": {"$gte": confidence_threshold}
            }
            
            cursor = self.collection.find(query).sort("confidence", DESCENDING).limit(limit)
            
            results = []
            for document in cursor:
                try:
                    result = LivenessDetectionResult.from_dict(document)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error converting document: {e}")
                    continue
            
            return results
            
        except PyMongoError as e:
            logger.error(f"MongoDB error finding fake faces: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding fake faces: {e}")
            return []
    
    async def find_by_spoof_type(self, spoof_type: str, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm liveness results theo loại spoof attack"""
        try:
            query = {
                "$or": [
                    {"detected_spoof_types": {"$in": [spoof_type]}},
                    {"primary_spoof_type": spoof_type}
                ]
            }
            
            cursor = self.collection.find(query).sort("created_at", DESCENDING).limit(limit)
            
            results = []
            for document in cursor:
                try:
                    result = LivenessDetectionResult.from_dict(document)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error converting document: {e}")
                    continue
            
            return results
            
        except PyMongoError as e:
            logger.error(f"MongoDB error finding results by spoof type: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding results by spoof type: {e}")
            return []
    
    async def find_recent_results(self, hours: int = 24, limit: int = 100) -> List[LivenessDetectionResult]:
        """Tìm liveness results gần đây"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            query = {
                "created_at": {"$gte": cutoff_time.isoformat()}
            }
            
            cursor = self.collection.find(query).sort("created_at", DESCENDING).limit(limit)
            
            results = []
            for document in cursor:
                try:
                    result = LivenessDetectionResult.from_dict(document)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error converting document: {e}")
                    continue
            
            return results
            
        except PyMongoError as e:
            logger.error(f"MongoDB error finding recent results: {e}")
            return []
        except Exception as e:
            logger.error(f"Error finding recent results: {e}")
            return []
    
    async def update(self, result: LivenessDetectionResult) -> LivenessDetectionResult:
        """Cập nhật liveness detection result"""
        try:
            # Set updated time
            result.updated_at = datetime.now()
            
            # Convert to dict
            result_dict = result.to_dict()
            
            # Update in MongoDB
            update_result = self.collection.replace_one(
                {"id": result.id},
                result_dict
            )
            
            if update_result.matched_count == 0:
                raise ValueError(f"Liveness result with ID {result.id} not found")
            
            logger.debug(f"Liveness result updated: {result.id}")
            return result
            
        except PyMongoError as e:
            logger.error(f"MongoDB error updating liveness result: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error updating liveness result: {e}")
            raise e
    
    async def delete_by_id(self, result_id: str) -> bool:
        """Xóa liveness result theo ID"""
        try:
            delete_result = self.collection.delete_one({"id": result_id})
            
            success = delete_result.deleted_count > 0
            
            if success:
                logger.debug(f"Liveness result deleted: {result_id}")
            
            return success
            
        except PyMongoError as e:
            logger.error(f"MongoDB error deleting liveness result: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting liveness result: {e}")
            return False
    
    async def delete_old_results(self, days: int = 30) -> int:
        """Xóa liveness results cũ"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            query = {
                "created_at": {"$lt": cutoff_time.isoformat()}
            }
            
            delete_result = self.collection.delete_many(query)
            deleted_count = delete_result.deleted_count
            
            logger.info(f"Deleted {deleted_count} old liveness results")
            return deleted_count
            
        except PyMongoError as e:
            logger.error(f"MongoDB error deleting old results: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error deleting old results: {e}")
            return 0
    
    async def count(self) -> int:
        """Đếm tổng số liveness results"""
        try:
            return self.collection.count_documents({})
        except PyMongoError as e:
            logger.error(f"MongoDB error counting liveness results: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error counting liveness results: {e}")
            return 0
    
    async def count_by_result(self, liveness_result: str) -> int:
        """Đếm số liveness results theo kết quả"""
        try:
            query = {"liveness_result": liveness_result}
            return self.collection.count_documents(query)
        except PyMongoError as e:
            logger.error(f"MongoDB error counting by result: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error counting by result: {e}")
            return 0
    
    async def count_by_spoof_type(self, spoof_type: str) -> int:
        """Đếm số liveness results theo loại spoof"""
        try:
            query = {
                "$or": [
                    {"detected_spoof_types": {"$in": [spoof_type]}},
                    {"primary_spoof_type": spoof_type}
                ]
            }
            return self.collection.count_documents(query)
        except PyMongoError as e:
            logger.error(f"MongoDB error counting by spoof type: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error counting by spoof type: {e}")
            return 0
    
    async def get_statistics(self) -> dict:
        """Lấy thống kê liveness detection"""
        try:
            pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "total_detections": {"$sum": 1},
                        "real_faces": {
                            "$sum": {"$cond": [{"$eq": ["$liveness_result", "REAL"]}, 1, 0]}
                        },
                        "fake_faces": {
                            "$sum": {"$cond": [{"$eq": ["$liveness_result", "FAKE"]}, 1, 0]}
                        },
                        "uncertain_faces": {
                            "$sum": {"$cond": [{"$eq": ["$liveness_result", "UNCERTAIN"]}, 1, 0]}
                        },
                        "avg_confidence": {"$avg": "$confidence"},
                        "avg_liveness_score": {"$avg": "$liveness_score"},
                        "avg_processing_time": {"$avg": "$processing_time_ms"},
                        "avg_image_quality": {"$avg": "$image_quality"},
                        "avg_face_quality": {"$avg": "$face_quality"}
                    }
                }
            ]
            
            result = list(self.collection.aggregate(pipeline))
            
            if result:
                stats = result[0]
                stats.pop("_id", None)
                
                # Calculate rates
                total = stats.get("total_detections", 1)
                stats["real_rate"] = stats.get("real_faces", 0) / total
                stats["fake_rate"] = stats.get("fake_faces", 0) / total
                stats["uncertain_rate"] = stats.get("uncertain_faces", 0) / total
                
                return stats
            
            return {
                "total_detections": 0,
                "real_faces": 0,
                "fake_faces": 0,
                "uncertain_faces": 0,
                "real_rate": 0.0,
                "fake_rate": 0.0,
                "uncertain_rate": 0.0,
                "avg_confidence": 0.0,
                "avg_liveness_score": 0.0,
                "avg_processing_time": 0.0,
                "avg_image_quality": 0.0,
                "avg_face_quality": 0.0
            }
            
        except PyMongoError as e:
            logger.error(f"MongoDB error getting statistics: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    async def get_performance_metrics(self, start_date: Optional[str] = None, 
                                    end_date: Optional[str] = None) -> dict:
        """Lấy performance metrics trong khoảng thời gian"""
        try:
            # Build query
            query = {}
            if start_date and end_date:
                query["created_at"] = {
                    "$gte": start_date,
                    "$lte": end_date
                }
            elif start_date:
                query["created_at"] = {"$gte": start_date}
            elif end_date:
                query["created_at"] = {"$lte": end_date}
            
            # Aggregation pipeline
            pipeline = [
                {"$match": query},
                {
                    "$group": {
                        "_id": None,
                        "total_processed": {"$sum": 1},
                        "successful_detections": {
                            "$sum": {"$cond": [{"$eq": ["$status", "COMPLETED"]}, 1, 0]}
                        },
                        "failed_detections": {
                            "$sum": {"$cond": [{"$eq": ["$status", "FAILED"]}, 1, 0]}
                        },
                        "high_confidence_detections": {
                            "$sum": {"$cond": [{"$gte": ["$confidence", 0.8]}, 1, 0]}
                        },
                        "avg_processing_time": {"$avg": "$processing_time_ms"},
                        "max_processing_time": {"$max": "$processing_time_ms"},
                        "min_processing_time": {"$min": "$processing_time_ms"}
                    }
                }
            ]
            
            result = list(self.collection.aggregate(pipeline))
            
            if result:
                metrics = result[0]
                metrics.pop("_id", None)
                
                # Calculate rates
                total = metrics.get("total_processed", 1)
                metrics["success_rate"] = metrics.get("successful_detections", 0) / total
                metrics["failure_rate"] = metrics.get("failed_detections", 0) / total
                metrics["high_confidence_rate"] = metrics.get("high_confidence_detections", 0) / total
                
                return metrics
            
            return {
                "total_processed": 0,
                "successful_detections": 0,
                "failed_detections": 0,
                "high_confidence_detections": 0,
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "high_confidence_rate": 0.0,
                "avg_processing_time": 0.0,
                "max_processing_time": 0.0,
                "min_processing_time": 0.0
            }
            
        except PyMongoError as e:
            logger.error(f"MongoDB error getting performance metrics: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    async def get_spoof_attack_trends(self, days: int = 30) -> dict:
        """Lấy xu hướng spoof attacks"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Spoof type distribution
            spoof_pipeline = [
                {
                    "$match": {
                        "created_at": {"$gte": cutoff_time.isoformat()},
                        "detected_spoof_types": {"$exists": True, "$ne": []}
                    }
                },
                {"$unwind": "$detected_spoof_types"},
                {
                    "$group": {
                        "_id": "$detected_spoof_types",
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            spoof_results = list(self.collection.aggregate(spoof_pipeline))
            spoof_distribution = {item["_id"]: item["count"] for item in spoof_results}
            
            # Daily trends
            daily_pipeline = [
                {
                    "$match": {
                        "created_at": {"$gte": cutoff_time.isoformat()}
                    }
                },
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": {"$dateFromString": {"dateString": "$created_at"}}
                            }
                        },
                        "total_detections": {"$sum": 1},
                        "fake_detections": {
                            "$sum": {"$cond": [{"$eq": ["$liveness_result", "FAKE"]}, 1, 0]}
                        }
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            daily_results = list(self.collection.aggregate(daily_pipeline))
            daily_trends = {
                item["_id"]: {
                    "total": item["total_detections"],
                    "fake": item["fake_detections"],
                    "fake_rate": item["fake_detections"] / item["total_detections"] if item["total_detections"] > 0 else 0
                }
                for item in daily_results
            }
            
            return {
                "analysis_period_days": days,
                "spoof_type_distribution": spoof_distribution,
                "daily_trends": daily_trends,
                "total_spoof_attacks": sum(spoof_distribution.values()),
                "most_common_attack": max(spoof_distribution.keys(), key=spoof_distribution.get) if spoof_distribution else None
            }
            
        except PyMongoError as e:
            logger.error(f"MongoDB error getting spoof attack trends: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting spoof attack trends: {e}")
            return {"error": str(e)}
