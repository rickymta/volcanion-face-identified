"""
Swagger/OpenAPI Configuration
"""
from typing import Dict, Any

SWAGGER_CONFIG: Dict[str, Any] = {
    "title": "Volcanion Face Verification System",
    "description": """
## Hệ thống xác thực khuôn mặt với giấy tờ tùy thân

Hệ thống ML/AI hoàn chỉnh cho xác thực danh tính sử dụng kiến trúc Domain-Driven Design (DDD).

### Các module chính:

#### 📄 Document Detection & Classification
- Phát hiện và phân loại các loại giấy tờ tùy thân
- Hỗ trợ: CCCD, Hộ chiếu, GPLX, v.v.
- API: `/document/`

#### 🔍 Document Quality & Tamper Detection
- Kiểm tra chất lượng ảnh tài liệu
- Phát hiện giấy tờ giả mạo/chỉnh sửa
- API: `/quality/`

#### 👤 Face Detection & Alignment
- Phát hiện khuôn mặt trong ảnh
- Căn chỉnh và chuẩn hóa khuôn mặt
- API: `/face-detection/`

#### 🎯 Face Embedding & Verification
- Tạo vector đặc trưng khuôn mặt
- So sánh và xác thực danh tính
- API: `/face-verification/`

#### 🔴 Liveness & Anti-spoofing Detection
- Phát hiện khuôn mặt thật/giả
- Chống lừa đảo bằng ảnh/video
- API: `/liveness/`

#### 📝 OCR Text Extraction
- Trích xuất thông tin văn bản từ tài liệu
- Validation và xác thực dữ liệu
- API: `/ocr/`

### Tính năng nổi bật:
- 🏗️ **Kiến trúc DDD**: Tách biệt rõ ràng các layer
- 🚀 **High Performance**: Xử lý real-time
- 🔒 **Security**: Bảo mật cao
- 📊 **Analytics**: Thống kê và phân tích
- 🔄 **Scalability**: Dễ mở rộng
- ✅ **Testing**: Coverage > 90%

### Cách sử dụng:
1. Upload ảnh giấy tờ tùy thân
2. Upload ảnh khuôn mặt người dùng
3. Hệ thống sẽ xử lý và trả về kết quả xác thực

### Liên hệ:
- Developer: Volcanion Team
- Email: support@volcanion.ai
- Version: 1.0.0
    """,
    "version": "1.0.0",
    "contact": {
        "name": "Volcanion Team",
        "email": "support@volcanion.ai",
        "url": "https://volcanion.ai"
    },
    "license_info": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    "tags_metadata": [
        {
            "name": "Document Detection",
            "description": "Phát hiện và phân loại tài liệu",
            "externalDocs": {
                "description": "Tài liệu chi tiết",
                "url": "https://docs.volcanion.ai/document-detection"
            }
        },
        {
            "name": "Document Quality",
            "description": "Kiểm tra chất lượng và phát hiện giả mạo",
            "externalDocs": {
                "description": "Tài liệu chi tiết", 
                "url": "https://docs.volcanion.ai/document-quality"
            }
        },
        {
            "name": "Face Detection",
            "description": "Phát hiện và căn chỉnh khuôn mặt",
            "externalDocs": {
                "description": "Tài liệu chi tiết",
                "url": "https://docs.volcanion.ai/face-detection"
            }
        },
        {
            "name": "Face Verification",
            "description": "Xác thực khuôn mặt và so sánh",
            "externalDocs": {
                "description": "Tài liệu chi tiết",
                "url": "https://docs.volcanion.ai/face-verification"
            }
        },
        {
            "name": "Liveness Detection",
            "description": "Phát hiện khuôn mặt thật và chống giả mạo",
            "externalDocs": {
                "description": "Tài liệu chi tiết",
                "url": "https://docs.volcanion.ai/liveness-detection"
            }
        },
        {
            "name": "OCR Extraction",
            "description": "Trích xuất và xác thực văn bản",
            "externalDocs": {
                "description": "Tài liệu chi tiết",
                "url": "https://docs.volcanion.ai/ocr-extraction"
            }
        },
        {
            "name": "System Monitoring",
            "description": "Giám sát hệ thống và phân tích",
            "externalDocs": {
                "description": "Tài liệu chi tiết",
                "url": "https://docs.volcanion.ai/monitoring"
            }
        }
    ],
    "servers": [
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.volcanion.ai",
            "description": "Production server"
        }
    ]
}

# Custom CSS for Swagger UI
SWAGGER_UI_PARAMETERS = {
    "dom_id": "#swagger-ui",
    "layout": "BaseLayout",
    "deepLinking": True,
    "showExtensions": True,
    "showCommonExtensions": True,
    "syntaxHighlight.theme": "arta",
    "tryItOutEnabled": True,
    "displayRequestDuration": True,
    "filter": True,
    "persistAuthorization": True
}

CUSTOM_CSS = """
.swagger-ui .topbar { display: none }
.swagger-ui .info .title { color: #1890ff; font-size: 36px; }
.swagger-ui .info .description { font-size: 16px; line-height: 1.6; }
.swagger-ui .scheme-container { background: #fafafa; padding: 20px; border-radius: 8px; }
.swagger-ui .opblock-tag { font-size: 18px; font-weight: bold; }
.swagger-ui .opblock.opblock-post { border-color: #49cc90; }
.swagger-ui .opblock.opblock-get { border-color: #61affe; }
.swagger-ui .opblock.opblock-put { border-color: #fca130; }
.swagger-ui .opblock.opblock-delete { border-color: #f93e3e; }
"""
