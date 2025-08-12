"""
System Monitoring và Metrics
"""
import time
import psutil
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Metrics hệ thống"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int

@dataclass
class APIMetrics:
    """Metrics API"""
    endpoint: str
    method: str
    response_time: float
    status_code: int
    timestamp: datetime
    user_agent: str = ""
    ip_address: str = ""
    request_size: int = 0
    response_size: int = 0

@dataclass
class MLMetrics:
    """Metrics Machine Learning"""
    model_name: str
    operation: str
    processing_time: float
    accuracy: float
    confidence: float
    timestamp: datetime
    input_size: int = 0
    success: bool = True
    error_message: str = ""

class MetricsCollector:
    """Thu thập metrics hệ thống"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics: deque = deque(maxlen=max_history)
        self.api_metrics: deque = deque(maxlen=max_history)
        self.ml_metrics: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=2)
        
    def start_collection(self, interval: int = 30):
        """Bắt đầu thu thập metrics"""
        self._running = True
        self._executor.submit(self._collect_system_metrics, interval)
        logger.info(f"Started metrics collection with {interval}s interval")
        
    def stop_collection(self):
        """Dừng thu thập metrics"""
        self._running = False
        self._executor.shutdown(wait=True)
        logger.info("Stopped metrics collection")
        
    def _collect_system_metrics(self, interval: int):
        """Thu thập system metrics"""
        while self._running:
            try:
                # CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # Memory
                memory = psutil.virtual_memory()
                
                # Disk
                disk = psutil.disk_usage('/')
                
                # Network
                network = psutil.net_io_counters()
                
                # Connections
                connections = len(psutil.net_connections())
                
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / 1024 / 1024,
                    memory_available_mb=memory.available / 1024 / 1024,
                    disk_usage_percent=disk.percent,
                    disk_free_gb=disk.free / 1024 / 1024 / 1024,
                    network_bytes_sent=network.bytes_sent,
                    network_bytes_recv=network.bytes_recv,
                    active_connections=connections
                )
                
                with self._lock:
                    self.system_metrics.append(metrics)
                    
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                
            time.sleep(interval)
            
    def add_api_metric(self, metric: APIMetrics):
        """Thêm API metric"""
        with self._lock:
            self.api_metrics.append(metric)
            
    def add_ml_metric(self, metric: MLMetrics):
        """Thêm ML metric"""
        with self._lock:
            self.ml_metrics.append(metric)
            
    def get_system_summary(self) -> Dict[str, Any]:
        """Lấy tổng quan system metrics"""
        with self._lock:
            if not self.system_metrics:
                return {}
                
            latest = self.system_metrics[-1]
            recent_metrics = list(self.system_metrics)[-10:]  # 10 metrics gần nhất
            
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            
            return {
                "current": asdict(latest),
                "averages": {
                    "cpu_percent": round(avg_cpu, 2),
                    "memory_percent": round(avg_memory, 2)
                },
                "alerts": self._check_system_alerts(latest)
            }
            
    def get_api_summary(self) -> Dict[str, Any]:
        """Lấy tổng quan API metrics"""
        with self._lock:
            if not self.api_metrics:
                return {}
                
            recent_metrics = list(self.api_metrics)[-100:]  # 100 requests gần nhất
            
            # Phân tích theo endpoint
            endpoint_stats = defaultdict(list)
            status_codes = defaultdict(int)
            
            for metric in recent_metrics:
                endpoint_stats[metric.endpoint].append(metric.response_time)
                status_codes[metric.status_code] += 1
                
            # Tính toán thống kê
            endpoint_summary = {}
            for endpoint, times in endpoint_stats.items():
                endpoint_summary[endpoint] = {
                    "avg_response_time": round(sum(times) / len(times), 3),
                    "min_response_time": round(min(times), 3),
                    "max_response_time": round(max(times), 3),
                    "request_count": len(times)
                }
                
            return {
                "total_requests": len(recent_metrics),
                "status_codes": dict(status_codes),
                "endpoints": endpoint_summary,
                "alerts": self._check_api_alerts(recent_metrics)
            }
            
    def get_ml_summary(self) -> Dict[str, Any]:
        """Lấy tổng quan ML metrics"""
        with self._lock:
            if not self.ml_metrics:
                return {}
                
            recent_metrics = list(self.ml_metrics)[-100:]
            
            # Phân tích theo model
            model_stats = defaultdict(list)
            operation_stats = defaultdict(list)
            
            for metric in recent_metrics:
                model_stats[metric.model_name].append(metric)
                operation_stats[metric.operation].append(metric)
                
            # Model summary
            model_summary = {}
            for model, metrics_list in model_stats.items():
                processing_times = [m.processing_time for m in metrics_list]
                accuracies = [m.accuracy for m in metrics_list if m.accuracy > 0]
                success_rate = sum(1 for m in metrics_list if m.success) / len(metrics_list)
                
                model_summary[model] = {
                    "avg_processing_time": round(sum(processing_times) / len(processing_times), 3),
                    "avg_accuracy": round(sum(accuracies) / len(accuracies), 3) if accuracies else 0,
                    "success_rate": round(success_rate, 3),
                    "total_operations": len(metrics_list)
                }
                
            return {
                "total_operations": len(recent_metrics),
                "models": model_summary,
                "alerts": self._check_ml_alerts(recent_metrics)
            }
            
    def _check_system_alerts(self, metrics: SystemMetrics) -> List[str]:
        """Kiểm tra system alerts"""
        alerts = []
        
        if metrics.cpu_percent > 80:
            alerts.append(f"High CPU usage: {metrics.cpu_percent}%")
            
        if metrics.memory_percent > 85:
            alerts.append(f"High memory usage: {metrics.memory_percent}%")
            
        if metrics.disk_usage_percent > 90:
            alerts.append(f"Low disk space: {metrics.disk_usage_percent}% used")
            
        return alerts
        
    def _check_api_alerts(self, metrics: List[APIMetrics]) -> List[str]:
        """Kiểm tra API alerts"""
        alerts = []
        
        # Tính error rate
        error_count = sum(1 for m in metrics if m.status_code >= 400)
        error_rate = error_count / len(metrics) if metrics else 0
        
        if error_rate > 0.1:  # > 10% error rate
            alerts.append(f"High error rate: {error_rate:.1%}")
            
        # Kiểm tra response time
        slow_requests = [m for m in metrics if m.response_time > 5.0]
        if len(slow_requests) > len(metrics) * 0.05:  # > 5% slow requests
            alerts.append(f"High response time: {len(slow_requests)} slow requests")
            
        return alerts
        
    def _check_ml_alerts(self, metrics: List[MLMetrics]) -> List[str]:
        """Kiểm tra ML alerts"""
        alerts = []
        
        # Tính success rate
        success_rate = sum(1 for m in metrics if m.success) / len(metrics) if metrics else 0
        
        if success_rate < 0.95:  # < 95% success rate
            alerts.append(f"Low ML success rate: {success_rate:.1%}")
            
        # Kiểm tra processing time
        processing_times = [m.processing_time for m in metrics]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        if avg_processing_time > 10.0:  # > 10 seconds
            alerts.append(f"High ML processing time: {avg_processing_time:.1f}s")
            
        return alerts

# Global metrics collector instance
metrics_collector = MetricsCollector()

def start_monitoring():
    """Khởi động monitoring"""
    metrics_collector.start_collection()
    
def stop_monitoring():
    """Dừng monitoring"""
    metrics_collector.stop_collection()
