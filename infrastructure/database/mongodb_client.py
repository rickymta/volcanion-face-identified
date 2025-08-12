"""
MongoDB database connection and client management
"""
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging
from typing import Optional

from infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

class MongoDBClient:
    """MongoDB client singleton"""
    
    _instance: Optional['MongoDBClient'] = None
    _client: Optional[MongoClient] = None
    _database: Optional[Database] = None
    
    def __new__(cls) -> 'MongoDBClient':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            self._connect()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            logger.info(f"Connecting to MongoDB: {settings.MONGODB_URL}")
            
            # Create MongoDB client with connection parameters
            self._client = MongoClient(
                settings.MONGODB_URL,
                serverSelectionTimeoutMS=5000,  # 5 seconds timeout
                connectTimeoutMS=10000,         # 10 seconds connect timeout
                socketTimeoutMS=20000,          # 20 seconds socket timeout
                maxPoolSize=10,                 # Maximum 10 connections in pool
                retryWrites=True,               # Enable retry writes
                retryReads=True                 # Enable retry reads
            )
            
            # Test the connection
            self._client.admin.command('ping')
            
            # Get the database
            self._database = self._client[settings.MONGODB_DATABASE]
            
            logger.info(f"Successfully connected to MongoDB database: {settings.MONGODB_DATABASE}")
            
            # Create indexes for better performance
            self._create_indexes()
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.warning("Continuing without database connection - some features may not work")
            # Don't raise exception to allow app to start without DB
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            logger.warning("Continuing without database connection")
    
    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            if self._database is None:
                return
            
            # Document detection indexes
            self._database.document_detections.create_index("detection_id", unique=True)
            self._database.document_detections.create_index("timestamp")
            self._database.document_detections.create_index("document_type")
            
            # Document quality indexes
            self._database.document_qualities.create_index("quality_id", unique=True)
            self._database.document_qualities.create_index("timestamp")
            self._database.document_qualities.create_index("overall_score")
            
            # Face detection indexes
            self._database.face_detections.create_index("detection_id", unique=True)
            self._database.face_detections.create_index("timestamp")
            self._database.face_detections.create_index("faces_count")
            
            # Face verification indexes
            self._database.face_verifications.create_index("verification_id", unique=True)
            self._database.face_verifications.create_index("timestamp")
            self._database.face_verifications.create_index("similarity_score")
            
            # Liveness detection indexes
            self._database.liveness_detections.create_index("liveness_id", unique=True)
            self._database.liveness_detections.create_index("timestamp")
            self._database.liveness_detections.create_index("is_live")
            
            # OCR extraction indexes
            self._database.ocr_results.create_index("result_id", unique=True)
            self._database.ocr_results.create_index("timestamp")
            self._database.ocr_results.create_index("text_fields.field_type")
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating database indexes: {e}")
    
    @property
    def client(self) -> Optional[MongoClient]:
        """Get MongoDB client"""
        return self._client
    
    @property
    def database(self) -> Optional[Database]:
        """Get MongoDB database"""
        return self._database
    
    def is_connected(self) -> bool:
        """Check if MongoDB is connected"""
        try:
            if self._client is None:
                return False
            self._client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            logger.info("MongoDB connection closed")
    
    def get_collection(self, collection_name: str):
        """Get a specific collection"""
        if self._database is None:
            logger.warning(f"Database not connected, cannot get collection: {collection_name}")
            return None
        return self._database[collection_name]
    
    def health_check(self) -> dict:
        """Perform database health check"""
        try:
            if not self.is_connected():
                return {
                    "status": "unhealthy",
                    "error": "Not connected to database"
                }
            
            # Check database stats
            stats = self._database.command("dbStats")
            
            return {
                "status": "healthy",
                "database": settings.MONGODB_DATABASE,
                "collections": stats.get("collections", 0),
                "dataSize": stats.get("dataSize", 0),
                "storageSize": stats.get("storageSize", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global MongoDB client instance
mongodb_client = MongoDBClient()

def get_database() -> Optional[Database]:
    """Get MongoDB database instance"""
    return mongodb_client.database

def get_collection(collection_name: str):
    """Get MongoDB collection"""
    return mongodb_client.get_collection(collection_name)

def is_database_connected() -> bool:
    """Check if database is connected"""
    return mongodb_client.is_connected()

def close_database():
    """Close database connection"""
    mongodb_client.close()
