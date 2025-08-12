"""
Environment configuration loader
"""
import os
from typing import Optional
from pathlib import Path

class Settings:
    """Application settings loaded from environment variables and .env file"""
    
    def __init__(self):
        # Load .env file if it exists
        self._load_env_file()
        
        # MongoDB settings
        self.MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "face_verification_db")
        
        # API settings
        self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT: int = int(os.getenv("API_PORT", "8000"))
        self.API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
        
        # Security settings
        self.SECRET_KEY: str = os.getenv("SECRET_KEY", "default-secret-key-change-in-production")
        self.API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")
        
        # Logging settings
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE: str = os.getenv("LOG_FILE", "volcanion_face_api.log")
        
        # ML Model paths
        self.FACE_DETECTION_MODEL_PATH: str = os.getenv("FACE_DETECTION_MODEL_PATH", "models/face_detection")
        self.FACE_VERIFICATION_MODEL_PATH: str = os.getenv("FACE_VERIFICATION_MODEL_PATH", "models/face_verification")
        self.LIVENESS_MODEL_PATH: str = os.getenv("LIVENESS_MODEL_PATH", "models/liveness_detection")
        
        # File upload settings
        self.MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
        self.ALLOWED_EXTENSIONS: list = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,bmp,tiff").split(",")
        
        # Performance settings
        self.WORKERS: int = int(os.getenv("WORKERS", "1"))
        self.TIMEOUT: int = int(os.getenv("TIMEOUT", "30"))
        
        # Debug mode
        self.DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    def _load_env_file(self):
        """Load environment variables from .env file"""
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Remove inline comments
                            if '#' in value:
                                value = value.split('#')[0].strip()
                            
                            # Remove quotes if present
                            if value.startswith('"') and value.endswith('"'):
                                value = value[1:-1]
                            elif value.startswith("'") and value.endswith("'"):
                                value = value[1:-1]
                            os.environ[key] = value
    
    @property
    def mongodb_connection_string(self) -> str:
        """Get MongoDB connection string"""
        return f"{self.MONGODB_URL}/{self.MONGODB_DATABASE}"
    
    def get_model_path(self, model_type: str) -> str:
        """Get path for specific model type"""
        model_paths = {
            "face_detection": self.FACE_DETECTION_MODEL_PATH,
            "face_verification": self.FACE_VERIFICATION_MODEL_PATH,
            "liveness": self.LIVENESS_MODEL_PATH
        }
        return model_paths.get(model_type, "models/default")
    
    def is_valid_file_extension(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        if not filename:
            return False
        extension = filename.split('.')[-1].lower()
        return extension in [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS]


# Global settings instance
settings = Settings()
