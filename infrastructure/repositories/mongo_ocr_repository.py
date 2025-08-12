from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
import os

from domain.entities.ocr_result import OCRResult, DocumentType, OCRStatus, FieldType
from domain.repositories.ocr_repository import OCRRepository

logger = logging.getLogger(__name__)

class MongoOCRRepository(OCRRepository):
    """MongoDB implementation của OCR repository"""
    
    def __init__(self):
        # MongoDB connection
        mongo_url = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/')
        database_name = os.getenv('DATABASE_NAME', 'volcanion_face_db')
        
        self.client = MongoClient(mongo_url)
        self.db = self.client[database_name]
        self.collection = self.db.ocr_results
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Tạo indexes cho performance"""
        try:
            # Basic indexes
            self.collection.create_index([("id", ASCENDING)], unique=True)
            self.collection.create_index([("image_path", ASCENDING)])
            self.collection.create_index([("document_type", ASCENDING)])
            self.collection.create_index([("status", ASCENDING)])
            self.collection.create_index([("created_at", DESCENDING)])
            
            # Compound indexes
            self.collection.create_index([
                ("document_type", ASCENDING),
                ("status", ASCENDING),
                ("created_at", DESCENDING)
            ])
            
            # Text search index
            self.collection.create_index([("full_text", TEXT)])
            
            # Performance indexes
            self.collection.create_index([("statistics.average_confidence", DESCENDING)])
            self.collection.create_index([("processing_time_ms", ASCENDING)])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {e}")
    
    async def save(self, ocr_result: OCRResult) -> str:
        """Lưu OCR result"""
        try:
            # Convert to dict
            doc = ocr_result.to_dict()
            
            # Insert or update
            filter_query = {"id": ocr_result.id}
            update_query = {"$set": doc}
            
            result = self.collection.update_one(
                filter_query, 
                update_query, 
                upsert=True
            )
            
            logger.info(f"OCR result saved: {ocr_result.id}")
            return ocr_result.id
            
        except Exception as e:
            logger.error(f"Error saving OCR result: {e}")
            raise
    
    async def find_by_id(self, ocr_id: str) -> Optional[OCRResult]:
        """Tìm OCR result theo ID"""
        try:
            doc = self.collection.find_one({"id": ocr_id})
            if doc:
                return self._doc_to_ocr_result(doc)
            return None
            
        except Exception as e:
            logger.error(f"Error finding OCR result by ID: {e}")
            return None
    
    async def find_by_image_path(self, image_path: str) -> List[OCRResult]:
        """Tìm OCR results theo image path"""
        try:
            cursor = self.collection.find({"image_path": image_path})
            results = []
            
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding OCR results by image path: {e}")
            return []
    
    async def find_by_document_type(self, document_type: DocumentType, limit: int = 100) -> List[OCRResult]:
        """Tìm OCR results theo document type"""
        try:
            cursor = self.collection.find({"document_type": document_type.value}).limit(limit)
            results = []
            
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding OCR results by document type: {e}")
            return []
    
    async def find_by_status(self, status: OCRStatus, limit: int = 100) -> List[OCRResult]:
        """Tìm OCR results theo status"""
        try:
            cursor = self.collection.find({"status": status.value}).limit(limit)
            results = []
            
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding OCR results by status: {e}")
            return []
    
    async def get_recent_results(self, hours: int = 24, limit: int = 100) -> List[OCRResult]:
        """Lấy OCR results gần đây"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            cursor = self.collection.find({
                "created_at": {"$gte": cutoff_time.isoformat()}
            }).sort("created_at", DESCENDING).limit(limit)
            
            results = []
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting recent results: {e}")
            return []
    
    async def get_successful_results(self, limit: int = 100) -> List[OCRResult]:
        """Lấy OCR results thành công"""
        try:
            cursor = self.collection.find({
                "status": OCRStatus.COMPLETED.value,
                "is_successful": True
            }).sort("created_at", DESCENDING).limit(limit)
            
            results = []
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting successful results: {e}")
            return []
    
    async def get_failed_results(self, limit: int = 100) -> List[OCRResult]:
        """Lấy OCR results thất bại"""
        try:
            cursor = self.collection.find({
                "$or": [
                    {"status": OCRStatus.FAILED.value},
                    {"is_successful": False}
                ]
            }).sort("created_at", DESCENDING).limit(limit)
            
            results = []
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting failed results: {e}")
            return []
    
    async def find_by_field_type(self, field_type: FieldType, limit: int = 100) -> List[OCRResult]:
        """Tìm OCR results có chứa field type cụ thể"""
        try:
            cursor = self.collection.find({
                "extracted_fields.field_type": field_type.value
            }).limit(limit)
            
            results = []
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding OCR results by field type: {e}")
            return []
    
    async def find_by_confidence_range(
        self, 
        min_confidence: float, 
        max_confidence: float = 1.0, 
        limit: int = 100
    ) -> List[OCRResult]:
        """Tìm OCR results theo confidence range"""
        try:
            cursor = self.collection.find({
                "overall_confidence": {
                    "$gte": min_confidence,
                    "$lte": max_confidence
                }
            }).limit(limit)
            
            results = []
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding OCR results by confidence range: {e}")
            return []
    
    async def search_by_text(self, search_text: str, limit: int = 100) -> List[OCRResult]:
        """Tìm kiếm OCR results theo text content"""
        try:
            cursor = self.collection.find({
                "$text": {"$search": search_text}
            }).limit(limit)
            
            results = []
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching by text: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Lấy statistics tổng quan về OCR"""
        try:
            # Basic counts
            total_count = self.collection.count_documents({})
            successful_count = self.collection.count_documents({"is_successful": True})
            failed_count = self.collection.count_documents({"is_successful": False})
            
            # Status distribution
            status_pipeline = [
                {"$group": {"_id": "$status", "count": {"$sum": 1}}}
            ]
            status_distribution = {}
            for result in self.collection.aggregate(status_pipeline):
                status_distribution[result["_id"]] = result["count"]
            
            # Document type distribution
            doc_type_pipeline = [
                {"$group": {"_id": "$document_type", "count": {"$sum": 1}}}
            ]
            doc_type_distribution = {}
            for result in self.collection.aggregate(doc_type_pipeline):
                doc_type_distribution[result["_id"]] = result["count"]
            
            # Confidence statistics
            confidence_pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "avg_confidence": {"$avg": "$overall_confidence"},
                        "min_confidence": {"$min": "$overall_confidence"},
                        "max_confidence": {"$max": "$overall_confidence"}
                    }
                }
            ]
            confidence_stats = list(self.collection.aggregate(confidence_pipeline))
            confidence_stats = confidence_stats[0] if confidence_stats else {}
            
            # Processing time statistics
            time_pipeline = [
                {
                    "$group": {
                        "_id": None,
                        "avg_processing_time": {"$avg": "$processing_time_ms"},
                        "min_processing_time": {"$min": "$processing_time_ms"},
                        "max_processing_time": {"$max": "$processing_time_ms"}
                    }
                }
            ]
            time_stats = list(self.collection.aggregate(time_pipeline))
            time_stats = time_stats[0] if time_stats else {}
            
            return {
                "basic_statistics": {
                    "total_results": total_count,
                    "successful_results": successful_count,
                    "failed_results": failed_count,
                    "success_rate": successful_count / total_count if total_count > 0 else 0.0
                },
                "status_distribution": status_distribution,
                "document_type_distribution": doc_type_distribution,
                "confidence_statistics": confidence_stats,
                "processing_time_statistics": time_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    async def get_statistics_by_document_type(self, document_type: DocumentType) -> Dict[str, Any]:
        """Lấy statistics theo document type"""
        try:
            filter_query = {"document_type": document_type.value}
            
            total_count = self.collection.count_documents(filter_query)
            successful_count = self.collection.count_documents({
                **filter_query,
                "is_successful": True
            })
            
            # Confidence statistics for this document type
            confidence_pipeline = [
                {"$match": filter_query},
                {
                    "$group": {
                        "_id": None,
                        "avg_confidence": {"$avg": "$overall_confidence"},
                        "min_confidence": {"$min": "$overall_confidence"},
                        "max_confidence": {"$max": "$overall_confidence"}
                    }
                }
            ]
            confidence_stats = list(self.collection.aggregate(confidence_pipeline))
            confidence_stats = confidence_stats[0] if confidence_stats else {}
            
            return {
                "document_type": document_type.value,
                "total_results": total_count,
                "successful_results": successful_count,
                "success_rate": successful_count / total_count if total_count > 0 else 0.0,
                "confidence_statistics": confidence_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics by document type: {e}")
            return {"error": str(e)}
    
    async def get_performance_metrics(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Lấy performance metrics trong khoảng thời gian"""
        try:
            filter_query = {
                "created_at": {
                    "$gte": start_date.isoformat(),
                    "$lte": end_date.isoformat()
                }
            }
            
            # Daily statistics
            daily_pipeline = [
                {"$match": filter_query},
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d",
                                "date": {"$dateFromString": {"dateString": "$created_at"}}
                            }
                        },
                        "total": {"$sum": 1},
                        "successful": {
                            "$sum": {"$cond": [{"$eq": ["$is_successful", True]}, 1, 0]}
                        },
                        "avg_confidence": {"$avg": "$overall_confidence"},
                        "avg_processing_time": {"$avg": "$processing_time_ms"}
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            daily_stats = list(self.collection.aggregate(daily_pipeline))
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "daily_statistics": daily_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e)}
    
    async def count_total(self) -> int:
        """Đếm tổng số OCR results"""
        try:
            return self.collection.count_documents({})
        except Exception as e:
            logger.error(f"Error counting total: {e}")
            return 0
    
    async def count_by_status(self, status: OCRStatus) -> int:
        """Đếm số OCR results theo status"""
        try:
            return self.collection.count_documents({"status": status.value})
        except Exception as e:
            logger.error(f"Error counting by status: {e}")
            return 0
    
    async def count_by_document_type(self, document_type: DocumentType) -> int:
        """Đếm số OCR results theo document type"""
        try:
            return self.collection.count_documents({"document_type": document_type.value})
        except Exception as e:
            logger.error(f"Error counting by document type: {e}")
            return 0
    
    async def update_status(self, ocr_id: str, status: OCRStatus, error_message: str = None) -> bool:
        """Cập nhật status của OCR result"""
        try:
            update_query = {"$set": {"status": status.value}}
            if error_message:
                update_query["$set"]["error_message"] = error_message
            
            result = self.collection.update_one(
                {"id": ocr_id},
                update_query
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating status: {e}")
            return False
    
    async def delete_by_id(self, ocr_id: str) -> bool:
        """Xóa OCR result theo ID"""
        try:
            result = self.collection.delete_one({"id": ocr_id})
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting OCR result: {e}")
            return False
    
    async def delete_old_results(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """Xóa OCR results cũ"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Count documents to be deleted
            count_to_delete = self.collection.count_documents({
                "created_at": {"$lt": cutoff_date.isoformat()}
            })
            
            # Delete old documents
            delete_result = self.collection.delete_many({
                "created_at": {"$lt": cutoff_date.isoformat()}
            })
            
            return {
                "deleted_count": delete_result.deleted_count,
                "cutoff_date": cutoff_date.isoformat(),
                "days_kept": days_to_keep,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error deleting old results: {e}")
            return {
                "deleted_count": 0,
                "success": False,
                "error": str(e)
            }
    
    async def get_duplicate_results(self, image_path: str) -> List[OCRResult]:
        """Tìm duplicate OCR results cho cùng một image"""
        try:
            cursor = self.collection.find({"image_path": image_path})
            results = []
            
            for doc in cursor:
                ocr_result = self._doc_to_ocr_result(doc)
                if ocr_result:
                    results.append(ocr_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting duplicate results: {e}")
            return []
    
    async def get_accuracy_trends(self, days: int = 30) -> Dict[str, Any]:
        """Lấy accuracy trends trong khoảng thời gian"""
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            pipeline = [
                {
                    "$match": {
                        "created_at": {"$gte": start_date.isoformat()}
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
                        "total": {"$sum": 1},
                        "successful": {
                            "$sum": {"$cond": [{"$eq": ["$is_successful", True]}, 1, 0]}
                        },
                        "avg_confidence": {"$avg": "$overall_confidence"}
                    }
                },
                {"$sort": {"_id": 1}}
            ]
            
            trends = list(self.collection.aggregate(pipeline))
            
            # Calculate success rate for each day
            for trend in trends:
                trend["success_rate"] = trend["successful"] / trend["total"] if trend["total"] > 0 else 0.0
            
            return {
                "period_days": days,
                "start_date": start_date.isoformat(),
                "trends": trends
            }
            
        except Exception as e:
            logger.error(f"Error getting accuracy trends: {e}")
            return {"error": str(e)}
    
    async def get_field_extraction_stats(self) -> Dict[str, Any]:
        """Lấy statistics về field extraction"""
        try:
            pipeline = [
                {"$unwind": "$extracted_fields"},
                {
                    "$group": {
                        "_id": "$extracted_fields.field_type",
                        "total_extractions": {"$sum": 1},
                        "successful_extractions": {
                            "$sum": {"$cond": [{"$eq": ["$extracted_fields.validation_status", True]}, 1, 0]}
                        },
                        "avg_confidence": {"$avg": "$extracted_fields.confidence"}
                    }
                }
            ]
            
            field_stats = {}
            for result in self.collection.aggregate(pipeline):
                field_type = result["_id"]
                field_stats[field_type] = {
                    "total_extractions": result["total_extractions"],
                    "successful_extractions": result["successful_extractions"],
                    "success_rate": result["successful_extractions"] / result["total_extractions"] if result["total_extractions"] > 0 else 0.0,
                    "average_confidence": result["avg_confidence"]
                }
            
            return {
                "field_statistics": field_stats,
                "total_field_types": len(field_stats)
            }
            
        except Exception as e:
            logger.error(f"Error getting field extraction stats: {e}")
            return {"error": str(e)}
    
    async def find_similar_results(
        self, 
        ocr_result: OCRResult, 
        similarity_threshold: float = 0.8, 
        limit: int = 10
    ) -> List[OCRResult]:
        """Tìm OCR results tương tự"""
        try:
            # Simple similarity based on document type and confidence range
            confidence_range = 0.1
            
            cursor = self.collection.find({
                "document_type": ocr_result.document_type.value,
                "overall_confidence": {
                    "$gte": ocr_result.get_overall_confidence() - confidence_range,
                    "$lte": ocr_result.get_overall_confidence() + confidence_range
                },
                "id": {"$ne": ocr_result.id}  # Exclude the input result
            }).limit(limit)
            
            results = []
            for doc in cursor:
                similar_result = self._doc_to_ocr_result(doc)
                if similar_result:
                    results.append(similar_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar results: {e}")
            return []
    
    def _doc_to_ocr_result(self, doc: Dict[str, Any]) -> Optional[OCRResult]:
        """Convert MongoDB document to OCRResult"""
        try:
            # Import here to avoid circular import
            from domain.entities.ocr_result import (
                OCRResult, TextRegion, ExtractedField, OCRStatistics,
                BoundingBox, DocumentType, OCRStatus, FieldType
            )
            
            # Convert text regions
            text_regions = []
            for region_data in doc.get("text_regions", []):
                bbox = BoundingBox(
                    x1=region_data["bbox"]["x1"],
                    y1=region_data["bbox"]["y1"],
                    x2=region_data["bbox"]["x2"],
                    y2=region_data["bbox"]["y2"]
                )
                
                field_type = None
                if region_data.get("field_type"):
                    field_type = FieldType(region_data["field_type"])
                
                region = TextRegion(
                    text=region_data["text"],
                    confidence=region_data["confidence"],
                    bbox=bbox,
                    field_type=field_type,
                    language=region_data.get("language"),
                    font_size=region_data.get("font_size"),
                    is_handwritten=region_data.get("is_handwritten", False)
                )
                text_regions.append(region)
            
            # Convert extracted fields
            extracted_fields = []
            for field_data in doc.get("extracted_fields", []):
                bbox = BoundingBox(
                    x1=field_data["bbox"]["x1"],
                    y1=field_data["bbox"]["y1"],
                    x2=field_data["bbox"]["x2"],
                    y2=field_data["bbox"]["y2"]
                )
                
                field = ExtractedField(
                    field_type=FieldType(field_data["field_type"]),
                    value=field_data["value"],
                    confidence=field_data["confidence"],
                    bbox=bbox,
                    raw_text=field_data["raw_text"],
                    normalized_value=field_data.get("normalized_value"),
                    validation_status=field_data.get("validation_status", True),
                    validation_errors=field_data.get("validation_errors", [])
                )
                extracted_fields.append(field)
            
            # Convert statistics
            stats_data = doc.get("statistics", {})
            statistics = OCRStatistics(
                total_text_regions=stats_data.get("total_text_regions", 0),
                reliable_regions=stats_data.get("reliable_regions", 0),
                unreliable_regions=stats_data.get("unreliable_regions", 0),
                average_confidence=stats_data.get("average_confidence", 0.0),
                total_characters=stats_data.get("total_characters", 0),
                processing_time_ms=stats_data.get("processing_time_ms", 0.0),
                languages_detected=stats_data.get("languages_detected", [])
            )
            
            # Parse created_at
            created_at = None
            if doc.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(doc["created_at"].replace('Z', '+00:00'))
                except:
                    created_at = datetime.now()
            
            # Create OCRResult
            ocr_result = OCRResult(
                id=doc["id"],
                image_path=doc["image_path"],
                document_type=DocumentType(doc["document_type"]),
                status=OCRStatus(doc["status"]),
                text_regions=text_regions,
                extracted_fields=extracted_fields,
                full_text=doc["full_text"],
                statistics=statistics,
                processing_time_ms=doc["processing_time_ms"],
                model_version=doc["model_version"],
                languages_detected=doc["languages_detected"],
                confidence_threshold=doc["confidence_threshold"],
                preprocessing_applied=doc["preprocessing_applied"],
                error_message=doc.get("error_message"),
                created_at=created_at
            )
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"Error converting document to OCRResult: {e}")
            return None
