import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2
import logging
from pathlib import Path

class FaceEmbeddingExtractor:
    """
    Face embedding extractor sử dụng FaceNet-like architecture
    Trong production, có thể thay thế bằng pre-trained models như:
    - FaceNet
    - ArcFace
    - VGGFace
    - Dlib face recognition
    """
    
    def __init__(self, model_type: str = "facenet_keras"):
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.model = None
        self.embedding_size = 128  # Standard FaceNet embedding size
        self._load_model()
    
    def _load_model(self):
        """Load face embedding model"""
        try:
            if self.model_type == "facenet_keras":
                self._load_facenet_keras_model()
            elif self.model_type == "dlib":
                self._load_dlib_model()
            else:
                self.logger.warning(f"Model type {self.model_type} not supported, using fallback")
                self._load_fallback_model()
                
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self._load_fallback_model()
    
    def _load_facenet_keras_model(self):
        """Load FaceNet Keras model"""
        try:
            # Trong thực tế, load pre-trained FaceNet model
            # from keras_facenet import FaceNet
            # self.model = FaceNet()
            
            # Tạm thời sử dụng mock model cho demo
            self.model = MockFaceNetModel(self.embedding_size)
            self.logger.info("FaceNet Keras model loaded (mock version)")
            
        except ImportError:
            self.logger.warning("FaceNet Keras not available, using fallback")
            self._load_fallback_model()
    
    def _load_dlib_model(self):
        """Load Dlib face recognition model"""
        try:
            import dlib
            
            # Load dlib's face recognition model
            model_path = Path(__file__).parent.parent.parent / "models" / "dlib_face_recognition_resnet_model_v1.dat"
            
            if model_path.exists():
                self.model = dlib.face_recognition_model_v1(str(model_path))
                self.embedding_size = 128
                self.logger.info("Dlib face recognition model loaded")
            else:
                self.logger.warning("Dlib model file not found, using fallback")
                self._load_fallback_model()
                
        except ImportError:
            self.logger.warning("Dlib not available, using fallback")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback model (CNN-based feature extractor)"""
        self.model = FallbackEmbeddingModel(self.embedding_size)
        self.logger.info("Fallback embedding model loaded")
    
    def extract_embedding(self, image_path: str, face_bbox: List[int]) -> Tuple[Optional[List[float]], float]:
        """
        Extract face embedding từ ảnh
        Returns: (embedding_vector, confidence)
        """
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return None, 0.0
            
            # Crop face region
            x1, y1, x2, y2 = face_bbox
            face_image = image[y1:y2, x1:x2]
            
            if face_image.size == 0:
                return None, 0.0
            
            # Preprocess face for embedding extraction
            preprocessed_face = self._preprocess_face(face_image)
            
            # Extract embedding
            embedding = self._extract_features(preprocessed_face)
            
            # Calculate confidence based on feature quality
            confidence = self._calculate_extraction_confidence(preprocessed_face, embedding)
            
            return embedding.tolist() if embedding is not None else None, confidence
            
        except Exception as e:
            self.logger.error(f"Error extracting embedding: {e}")
            return None, 0.0
    
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for embedding extraction"""
        try:
            # Resize to model input size (typically 160x160 for FaceNet)
            target_size = (160, 160)
            resized_face = cv2.resize(face_image, target_size)
            
            # Convert BGR to RGB
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            normalized_face = rgb_face.astype(np.float32) / 255.0
            
            # Apply additional preprocessing if needed
            # (whitening, histogram equalization, etc.)
            enhanced_face = self._enhance_face_quality(normalized_face)
            
            # Add batch dimension
            preprocessed = np.expand_dims(enhanced_face, axis=0)
            
            return preprocessed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing face: {e}")
            return None
    
    def _enhance_face_quality(self, face: np.ndarray) -> np.ndarray:
        """Enhance face quality for better embedding extraction"""
        try:
            # Convert to grayscale for some operations
            gray = cv2.cvtColor((face * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
            
            # Convert back to RGB
            enhanced_rgb = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
            enhanced_face = enhanced_rgb.astype(np.float32) / 255.0
            
            # Blend with original (50-50)
            final_face = 0.5 * face + 0.5 * enhanced_face
            
            return final_face
            
        except Exception:
            # Return original if enhancement fails
            return face
    
    def _extract_features(self, preprocessed_face: np.ndarray) -> Optional[np.ndarray]:
        """Extract features using the loaded model"""
        try:
            if self.model is None:
                return None
            
            # Extract embedding using the model
            if isinstance(self.model, MockFaceNetModel):
                embedding = self.model.predict(preprocessed_face)
            elif isinstance(self.model, FallbackEmbeddingModel):
                embedding = self.model.extract_features(preprocessed_face)
            else:
                # Real model prediction
                embedding = self.model.predict(preprocessed_face)
            
            # Normalize embedding (L2 normalization)
            if embedding is not None:
                embedding = embedding.flatten()
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _calculate_extraction_confidence(self, preprocessed_face: np.ndarray, embedding: np.ndarray) -> float:
        """Calculate confidence score for embedding extraction"""
        try:
            if embedding is None or preprocessed_face is None:
                return 0.0
            
            # Factor 1: Face quality (sharpness)
            gray = cv2.cvtColor((preprocessed_face[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)
            
            # Factor 2: Embedding magnitude (stronger features = higher confidence)
            embedding_magnitude = np.linalg.norm(embedding)
            magnitude_score = min(embedding_magnitude / 10.0, 1.0)
            
            # Factor 3: Feature distribution (avoid uniform distributions)
            feature_std = np.std(embedding)
            distribution_score = min(feature_std * 5.0, 1.0)
            
            # Factor 4: Face size (larger faces generally give better embeddings)
            face_area = preprocessed_face.shape[1] * preprocessed_face.shape[2]
            size_score = min(face_area / 25600.0, 1.0)  # 160x160 = 25600
            
            # Combine factors
            confidence = (
                sharpness_score * 0.3 +
                magnitude_score * 0.3 +
                distribution_score * 0.2 +
                size_score * 0.2
            )
            
            return min(max(confidence, 0.0), 1.0)
            
        except Exception:
            return 0.5  # Default confidence
    
    def batch_extract_embeddings(self, image_face_pairs: List[Tuple[str, List[int]]]) -> List[Tuple[Optional[List[float]], float]]:
        """Extract embeddings for multiple faces"""
        results = []
        
        for image_path, face_bbox in image_face_pairs:
            embedding, confidence = self.extract_embedding(image_path, face_bbox)
            results.append((embedding, confidence))
        
        return results
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_type": self.model_type,
            "embedding_size": self.embedding_size,
            "model_loaded": self.model is not None,
            "model_class": type(self.model).__name__ if self.model else None
        }

class MockFaceNetModel:
    """Mock FaceNet model for testing/demo purposes"""
    
    def __init__(self, embedding_size: int = 128):
        self.embedding_size = embedding_size
        # Initialize random weights for demo
        np.random.seed(42)  # For reproducible results
        self.weights = np.random.randn(160, 160, 3, embedding_size) * 0.1
    
    def predict(self, face_batch: np.ndarray) -> np.ndarray:
        """Generate mock embedding"""
        if face_batch.shape[0] == 0:
            return np.zeros((0, self.embedding_size))
        
        # Simple feature extraction simulation
        # In reality, this would be a deep CNN
        face = face_batch[0]  # Take first face from batch
        
        # Simulate CNN layers
        features = []
        
        # Global average pooling simulation
        avg_r = np.mean(face[:, :, 0])
        avg_g = np.mean(face[:, :, 1])
        avg_b = np.mean(face[:, :, 2])
        
        # Texture features (simplified)
        gray = np.mean(face, axis=2)
        texture_features = [
            np.std(gray),
            np.mean(np.gradient(gray)[0]),
            np.mean(np.gradient(gray)[1])
        ]
        
        # Edge features
        edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine features
        basic_features = [avg_r, avg_g, avg_b] + texture_features + [edge_density]
        
        # Expand to embedding size with some randomness for uniqueness
        embedding = np.zeros(self.embedding_size)
        for i in range(min(len(basic_features), self.embedding_size)):
            embedding[i] = basic_features[i]
        
        # Fill remaining with derived features
        for i in range(len(basic_features), self.embedding_size):
            # Create derived features based on face content
            seed_val = int((np.sum(face) * 1000) % 1000) + i
            np.random.seed(seed_val)
            embedding[i] = np.random.normal(0, 0.1)
        
        return embedding.reshape(1, -1)

class FallbackEmbeddingModel:
    """Fallback embedding model using traditional CV features"""
    
    def __init__(self, embedding_size: int = 128):
        self.embedding_size = embedding_size
    
    def extract_features(self, face_batch: np.ndarray) -> np.ndarray:
        """Extract features using traditional computer vision methods"""
        if face_batch.shape[0] == 0:
            return np.zeros((0, self.embedding_size))
        
        face = face_batch[0]
        gray = cv2.cvtColor((face * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # 1. LBP features
        lbp_features = self._extract_lbp_features(gray)
        features.extend(lbp_features[:32])
        
        # 2. HOG features
        hog_features = self._extract_hog_features(gray)
        features.extend(hog_features[:32])
        
        # 3. Gabor filter responses
        gabor_features = self._extract_gabor_features(gray)
        features.extend(gabor_features[:32])
        
        # 4. Statistical features
        stat_features = self._extract_statistical_features(gray)
        features.extend(stat_features[:32])
        
        # Pad or truncate to desired size
        if len(features) < self.embedding_size:
            features.extend([0.0] * (self.embedding_size - len(features)))
        else:
            features = features[:self.embedding_size]
        
        return np.array(features).reshape(1, -1)
    
    def _extract_lbp_features(self, gray: np.ndarray) -> List[float]:
        """Extract Local Binary Pattern features"""
        try:
            from skimage.feature import local_binary_pattern
            
            # LBP parameters
            radius = 3
            n_points = 8 * radius
            
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate histogram
            hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)  # Normalize
            
            return hist.tolist()
            
        except ImportError:
            # Fallback LBP implementation
            return self._simple_lbp_features(gray)
    
    def _simple_lbp_features(self, gray: np.ndarray) -> List[float]:
        """Simple LBP implementation"""
        height, width = gray.shape
        features = []
        
        # Sample points in a grid
        for i in range(4, height-4, 8):
            for j in range(4, width-4, 8):
                center = gray[i, j]
                
                # 8-neighborhood
                neighbors = [
                    gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                    gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                    gray[i+1, j-1], gray[i, j-1]
                ]
                
                # Binary pattern
                pattern = sum((n >= center) * (2**idx) for idx, n in enumerate(neighbors))
                features.append(pattern / 255.0)
        
        return features[:32]
    
    def _extract_hog_features(self, gray: np.ndarray) -> List[float]:
        """Extract HOG features"""
        try:
            from skimage.feature import hog
            
            features = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys'
            )
            
            return features.tolist()
            
        except ImportError:
            # Simple gradient-based features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Simple histogram of gradients
            hist, _ = np.histogram(magnitude.ravel(), bins=32, range=(0, 255))
            hist = hist.astype(float)
            hist /= (hist.sum() + 1e-7)
            
            return hist.tolist()
    
    def _extract_gabor_features(self, gray: np.ndarray) -> List[float]:
        """Extract Gabor filter features"""
        features = []
        
        # Multiple Gabor filters with different orientations and frequencies
        for theta in [0, 45, 90, 135]:
            for frequency in [0.1, 0.3]:
                kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                
                # Statistical features from filtered image
                features.extend([
                    np.mean(filtered),
                    np.std(filtered),
                    np.max(filtered),
                    np.min(filtered)
                ])
        
        return features
    
    def _extract_statistical_features(self, gray: np.ndarray) -> List[float]:
        """Extract statistical features"""
        features = []
        
        # Global statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.max(gray),
            np.min(gray),
            np.median(gray)
        ])
        
        # Histogram features
        hist, _ = np.histogram(gray.ravel(), bins=16, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)
        features.extend(hist.tolist())
        
        # Moments
        moments = cv2.moments(gray)
        features.extend([
            moments['m00'], moments['m10'], moments['m01'],
            moments['m20'], moments['m11'], moments['m02']
        ])
        
        return features
