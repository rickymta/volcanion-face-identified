import numpy as np
from typing import List, Tuple, Optional, Dict
import logging
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

class FaceVerificationEngine:
    """
    Engine để thực hiện face verification
    So sánh face embeddings và quyết định match/no-match
    """
    
    def __init__(self, default_threshold: float = 0.6):
        self.logger = logging.getLogger(__name__)
        self.default_threshold = default_threshold
        self.distance_metrics = {
            'cosine': self._cosine_distance,
            'euclidean': self._euclidean_distance,
            'manhattan': self._manhattan_distance,
            'cosine_similarity': self._cosine_similarity_score
        }
    
    def verify_faces(self, embedding1: List[float], embedding2: List[float], 
                    threshold: Optional[float] = None, 
                    distance_metric: str = 'cosine') -> Dict:
        """
        Verify hai face embeddings
        Returns: {
            'is_match': bool,
            'similarity_score': float,
            'distance': float,
            'confidence': float,
            'threshold_used': float,
            'metric_used': str
        }
        """
        try:
            if threshold is None:
                threshold = self.default_threshold
            
            # Validate embeddings
            if not self._validate_embeddings(embedding1, embedding2):
                return self._create_failed_result("Invalid embeddings", threshold, distance_metric)
            
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate distance/similarity
            if distance_metric not in self.distance_metrics:
                distance_metric = 'cosine'
                self.logger.warning(f"Unknown distance metric, using cosine")
            
            distance_score = self.distance_metrics[distance_metric](emb1, emb2)
            
            # Convert distance to similarity score (0-1, higher = more similar)
            similarity_score = self._distance_to_similarity(distance_score, distance_metric)
            
            # Determine match
            is_match = similarity_score >= threshold
            
            # Calculate confidence
            confidence = self._calculate_verification_confidence(
                similarity_score, threshold, emb1, emb2, distance_metric
            )
            
            return {
                'is_match': is_match,
                'similarity_score': similarity_score,
                'distance': distance_score,
                'confidence': confidence,
                'threshold_used': threshold,
                'metric_used': distance_metric,
                'match_probability': self._calculate_match_probability(similarity_score, threshold)
            }
            
        except Exception as e:
            self.logger.error(f"Error in face verification: {e}")
            return self._create_failed_result(str(e), threshold or self.default_threshold, distance_metric)
    
    def _validate_embeddings(self, embedding1: List[float], embedding2: List[float]) -> bool:
        """Validate embedding inputs"""
        if not embedding1 or not embedding2:
            return False
        
        if len(embedding1) != len(embedding2):
            self.logger.error(f"Embedding dimension mismatch: {len(embedding1)} vs {len(embedding2)}")
            return False
        
        if len(embedding1) < 64:  # Minimum reasonable embedding size
            self.logger.error(f"Embedding too small: {len(embedding1)}")
            return False
        
        # Check for valid numeric values
        try:
            np.array(embedding1, dtype=float)
            np.array(embedding2, dtype=float)
        except (ValueError, TypeError):
            self.logger.error("Invalid numeric values in embeddings")
            return False
        
        return True
    
    def _cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine distance"""
        try:
            return float(cosine(emb1, emb2))
        except:
            # Fallback calculation
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 1.0  # Maximum distance
            
            cosine_sim = dot_product / (norm1 * norm2)
            return 1.0 - cosine_sim
    
    def _euclidean_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate Euclidean distance"""
        try:
            return float(euclidean(emb1, emb2))
        except:
            # Fallback calculation
            return float(np.sqrt(np.sum((emb1 - emb2) ** 2)))
    
    def _manhattan_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate Manhattan distance"""
        return float(np.sum(np.abs(emb1 - emb2)))
    
    def _cosine_similarity_score(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity (returns similarity, not distance)"""
        try:
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            return float(similarity)
        except:
            # Fallback calculation
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
    
    def _distance_to_similarity(self, distance_score: float, metric: str) -> float:
        """Convert distance to similarity score (0-1)"""
        if metric == 'cosine':
            # Cosine distance: 0 = identical, 2 = opposite
            return max(0.0, 1.0 - distance_score)
        
        elif metric == 'euclidean':
            # Euclidean distance: normalize based on typical embedding ranges
            # For normalized embeddings, typical max distance is around sqrt(2)
            max_distance = np.sqrt(2.0)
            return max(0.0, 1.0 - (distance_score / max_distance))
        
        elif metric == 'manhattan':
            # Manhattan distance: normalize based on embedding dimension
            # For normalized embeddings, max Manhattan distance ≈ 2.0
            max_distance = 2.0
            return max(0.0, 1.0 - (distance_score / max_distance))
        
        elif metric == 'cosine_similarity':
            # Already a similarity score
            return max(0.0, min(1.0, distance_score))
        
        else:
            # Default normalization
            return max(0.0, 1.0 - distance_score)
    
    def _calculate_verification_confidence(self, similarity_score: float, threshold: float,
                                         emb1: np.ndarray, emb2: np.ndarray, metric: str) -> float:
        """Calculate confidence in verification result"""
        try:
            # Factor 1: Distance from threshold
            distance_from_threshold = abs(similarity_score - threshold)
            threshold_confidence = min(distance_from_threshold * 2.0, 1.0)
            
            # Factor 2: Embedding quality (based on norm and distribution)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            avg_norm = (norm1 + norm2) / 2.0
            
            # Good embeddings should have reasonable norms (around 1.0 for normalized)
            norm_confidence = 1.0 - abs(avg_norm - 1.0)
            norm_confidence = max(0.0, min(norm_confidence, 1.0))
            
            # Factor 3: Embedding variance (avoid uniform distributions)
            var1 = np.var(emb1)
            var2 = np.var(emb2)
            avg_variance = (var1 + var2) / 2.0
            variance_confidence = min(avg_variance * 10.0, 1.0)  # Scale factor
            
            # Factor 4: Metric-specific confidence
            metric_confidence = self._get_metric_confidence(similarity_score, metric)
            
            # Combine factors
            confidence = (
                threshold_confidence * 0.4 +
                norm_confidence * 0.2 +
                variance_confidence * 0.2 +
                metric_confidence * 0.2
            )
            
            return max(0.0, min(confidence, 1.0))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default moderate confidence
    
    def _get_metric_confidence(self, similarity_score: float, metric: str) -> float:
        """Get metric-specific confidence adjustments"""
        if metric == 'cosine' or metric == 'cosine_similarity':
            # Cosine similarity is generally reliable
            return 0.9 if similarity_score > 0.1 else 0.5
        
        elif metric == 'euclidean':
            # Euclidean distance can be affected by dimensionality
            return 0.8 if similarity_score > 0.2 else 0.6
        
        elif metric == 'manhattan':
            # Manhattan distance is less reliable for high-dimensional data
            return 0.7 if similarity_score > 0.3 else 0.5
        
        else:
            return 0.7  # Default
    
    def _calculate_match_probability(self, similarity_score: float, threshold: float) -> float:
        """Calculate probability that this is a true match"""
        # Sigmoid function to convert similarity to probability
        # More confident as we get further from threshold
        
        distance_from_threshold = similarity_score - threshold
        
        # Sigmoid parameters (adjust for desired sensitivity)
        scale = 5.0  # How steep the sigmoid is
        
        # Sigmoid function: 1 / (1 + exp(-scale * distance))
        probability = 1.0 / (1.0 + np.exp(-scale * distance_from_threshold))
        
        return float(probability)
    
    def _create_failed_result(self, error_message: str, threshold: float, metric: str) -> Dict:
        """Create failed verification result"""
        return {
            'is_match': False,
            'similarity_score': 0.0,
            'distance': float('inf'),
            'confidence': 0.0,
            'threshold_used': threshold,
            'metric_used': metric,
            'match_probability': 0.0,
            'error': error_message
        }
    
    def batch_verify_faces(self, embedding_pairs: List[Tuple[List[float], List[float]]], 
                          threshold: Optional[float] = None,
                          distance_metric: str = 'cosine') -> List[Dict]:
        """Verify multiple face pairs"""
        results = []
        
        for emb1, emb2 in embedding_pairs:
            result = self.verify_faces(emb1, emb2, threshold, distance_metric)
            results.append(result)
        
        return results
    
    def find_best_threshold(self, positive_pairs: List[Tuple[List[float], List[float]]],
                           negative_pairs: List[Tuple[List[float], List[float]]],
                           distance_metric: str = 'cosine') -> Dict:
        """
        Find optimal threshold using positive and negative pairs
        Returns: {
            'best_threshold': float,
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1_score': float
        }
        """
        try:
            # Calculate similarities for all pairs
            positive_similarities = []
            for emb1, emb2 in positive_pairs:
                result = self.verify_faces(emb1, emb2, threshold=0.0, distance_metric=distance_metric)
                positive_similarities.append(result['similarity_score'])
            
            negative_similarities = []
            for emb1, emb2 in negative_pairs:
                result = self.verify_faces(emb1, emb2, threshold=0.0, distance_metric=distance_metric)
                negative_similarities.append(result['similarity_score'])
            
            # Try different thresholds
            all_similarities = positive_similarities + negative_similarities
            min_sim = min(all_similarities)
            max_sim = max(all_similarities)
            
            best_threshold = 0.5
            best_f1 = 0.0
            best_metrics = {}
            
            # Test thresholds from min to max
            for threshold in np.linspace(min_sim, max_sim, 100):
                # Calculate metrics
                tp = sum(1 for sim in positive_similarities if sim >= threshold)
                fp = sum(1 for sim in negative_similarities if sim >= threshold)
                tn = sum(1 for sim in negative_similarities if sim < threshold)
                fn = sum(1 for sim in positive_similarities if sim < threshold)
                
                if tp + fp == 0:
                    precision = 0.0
                else:
                    precision = tp / (tp + fp)
                
                if tp + fn == 0:
                    recall = 0.0
                else:
                    recall = tp / (tp + fn)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                
                accuracy = (tp + tn) / (tp + fp + tn + fn)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_metrics = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
                    }
            
            return {
                'best_threshold': float(best_threshold),
                **best_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error finding best threshold: {e}")
            return {
                'best_threshold': self.default_threshold,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'error': str(e)
            }
    
    def calibrate_threshold_for_use_case(self, use_case: str) -> float:
        """Get recommended threshold for specific use cases"""
        thresholds = {
            'high_security': 0.8,      # Banking, government - minimize false positives
            'access_control': 0.7,     # Building access - balanced
            'user_verification': 0.6,  # App login - user-friendly
            'photo_tagging': 0.5,      # Social media - more permissive
            'loose_matching': 0.4      # Photo organization - very permissive
        }
        
        return thresholds.get(use_case, self.default_threshold)
    
    def analyze_embedding_quality(self, embedding: List[float]) -> Dict:
        """Analyze quality of a single embedding"""
        try:
            emb = np.array(embedding)
            
            return {
                'dimension': len(embedding),
                'norm': float(np.linalg.norm(emb)),
                'mean': float(np.mean(emb)),
                'std': float(np.std(emb)),
                'min': float(np.min(emb)),
                'max': float(np.max(emb)),
                'zero_ratio': float(np.sum(emb == 0) / len(emb)),
                'is_normalized': abs(np.linalg.norm(emb) - 1.0) < 0.1,
                'has_good_variance': np.std(emb) > 0.01,
                'quality_score': self._calculate_embedding_quality_score(emb)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing embedding quality: {e}")
            return {'error': str(e)}
    
    def _calculate_embedding_quality_score(self, embedding: np.ndarray) -> float:
        """Calculate overall quality score for an embedding"""
        score = 0.0
        
        # Factor 1: Norm should be around 1.0 for normalized embeddings
        norm = np.linalg.norm(embedding)
        norm_score = 1.0 - abs(norm - 1.0)
        norm_score = max(0.0, min(norm_score, 1.0))
        score += norm_score * 0.3
        
        # Factor 2: Good variance (not all values the same)
        variance = np.var(embedding)
        variance_score = min(variance * 100, 1.0)  # Scale factor
        score += variance_score * 0.3
        
        # Factor 3: Not too many zeros
        zero_ratio = np.sum(embedding == 0) / len(embedding)
        zero_score = 1.0 - zero_ratio
        score += zero_score * 0.2
        
        # Factor 4: Reasonable range of values
        value_range = np.max(embedding) - np.min(embedding)
        range_score = min(value_range / 2.0, 1.0)  # Expect range around 2.0
        score += range_score * 0.2
        
        return score
