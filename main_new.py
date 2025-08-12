"""
Main FastAPI application entry point with comprehensive monitoring and Swagger documentation
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Import configurations
from infrastructure.config.swagger_config import get_swagger_config, get_swagger_ui_parameters

# Import monitoring
from infrastructure.monitoring.middleware import PerformanceMiddleware
from infrastructure.monitoring.performance_monitor import performance_monitor

# Import API routers
from presentation.api.document_detection_api import router as document_detection_router
from presentation.api.document_quality_api import router as document_quality_router
from presentation.api.face_detection_api import router as face_detection_router
from presentation.api.face_verification_api import router as face_verification_router
from presentation.api.liveness_detection_api import router as liveness_detection_router
from presentation.api.ocr_api import router as ocr_router
from infrastructure.monitoring.monitoring_api import router as monitoring_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volcanion_face_api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Face Verification System...")
    
    # Start performance monitoring
    performance_monitor.start_monitoring()
    logger.info("📊 Performance monitoring started")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Face Verification System...")
    performance_monitor.stop_monitoring()
    logger.info("📊 Performance monitoring stopped")


# Get Swagger configuration
swagger_config = get_swagger_config()
swagger_ui_params = get_swagger_ui_parameters()

# Create FastAPI application with enhanced configuration
app = FastAPI(
    title=swagger_config["title"],
    description=swagger_config["description"],
    version=swagger_config["version"],
    contact=swagger_config["contact"],
    license_info=swagger_config["license_info"],
    tags_metadata=swagger_config["tags_metadata"],
    lifespan=lifespan,
    **swagger_ui_params
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add performance monitoring middleware
app.add_middleware(PerformanceMiddleware)

# Include API routers with proper tags
app.include_router(document_detection_router, tags=["Document Detection"])
app.include_router(document_quality_router, tags=["Document Quality"])
app.include_router(face_detection_router, tags=["Face Detection"])
app.include_router(face_verification_router, tags=["Face Verification"])
app.include_router(liveness_detection_router, tags=["Liveness Detection"])
app.include_router(ocr_router, tags=["OCR Extraction"])
app.include_router(monitoring_router, tags=["Monitoring"])


@app.get("/", summary="API Root", tags=["Health Check"])
async def root():
    """
    API Root endpoint providing system overview and available modules
    """
    return {
        "message": "Face Verification & ID Document Processing API",
        "version": "1.0.0",
        "status": "operational",
        "modules": [
            "Document Detection & Classification",
            "Document Quality & Tamper Check", 
            "Face Detection & Alignment",
            "Face Embedding & Verification",
            "Liveness & Anti-spoofing Detection",
            "OCR Text Extraction & Validation",
            "Performance Monitoring & Analytics"
        ],
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "monitoring": "/monitoring",
            "health": "/monitoring/health"
        },
        "features": [
            "Real-time face verification",
            "ID document processing",
            "OCR text extraction",
            "Anti-spoofing detection",
            "Performance monitoring",
            "Comprehensive API documentation"
        ]
    }


@app.get("/health", summary="Health Check", tags=["Health Check"])
async def health_check():
    """
    Detailed health check endpoint
    """
    return {
        "status": "healthy",
        "service": "Face Verification API",
        "version": "1.0.0",
        "timestamp": "2025-08-12T00:00:00Z",
        "modules_status": {
            "document_detection": "operational",
            "document_quality": "operational",
            "face_detection": "operational",
            "face_verification": "operational",
            "liveness_detection": "operational",
            "ocr_extraction": "operational",
            "monitoring": "operational"
        }
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Face Verification API server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
