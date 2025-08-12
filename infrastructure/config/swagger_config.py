"""
Swagger configuration for Face Verification API
"""
from typing import Dict, Any

def get_swagger_config() -> Dict[str, Any]:
    """Get Swagger/OpenAPI configuration"""
    return {
        "title": "Face Verification & ID Document Processing API",
        "description": """
        ## Face Verification System with ID Document Processing
        
        A comprehensive machine learning system for face verification with identity documents including OCR extraction.
        
        ### Features:
        - **Document Detection**: Automatic detection and classification of ID documents
        - **Document Quality Check**: Quality analysis and tamper detection
        - **Face Detection**: Advanced face detection and alignment
        - **Face Verification**: Face embedding and verification against ID photos
        - **Liveness Detection**: Anti-spoofing and liveness verification
        - **OCR Extraction**: Text extraction and validation from documents
        - **Monitoring**: Real-time system monitoring and analytics
        
        ### Architecture:
        Built using Domain-Driven Design (DDD) with clean architecture principles.
        
        ### Security:
        All endpoints support file upload validation and comprehensive error handling.
        """,
        "version": "1.0.0",
        "contact": {
            "name": "Face Verification API Support",
            "email": "support@faceverification.com"
        },
        "license_info": {
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        "tags_metadata": [
            {
                "name": "Document Detection",
                "description": "Document detection and classification endpoints"
            },
            {
                "name": "Document Quality",
                "description": "Document quality analysis and tamper detection"
            },
            {
                "name": "Face Detection",
                "description": "Face detection and alignment operations"
            },
            {
                "name": "Face Verification",
                "description": "Face embedding and verification services"
            },
            {
                "name": "Liveness Detection",
                "description": "Anti-spoofing and liveness verification"
            },
            {
                "name": "OCR Extraction",
                "description": "Text extraction and validation from documents"
            },
            {
                "name": "Monitoring",
                "description": "System monitoring and analytics endpoints"
            },
            {
                "name": "Health Check",
                "description": "Application health and status monitoring"
            }
        ]
    }

def get_swagger_ui_parameters() -> Dict[str, Any]:
    """Get Swagger UI configuration parameters"""
    return {
        "swagger_ui_parameters": {
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "none",
            "operationsSorter": "method",
            "filter": True,
            "tryItOutEnabled": True
        }
    }
