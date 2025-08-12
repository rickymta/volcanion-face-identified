"""
Performance monitoring system for tracking API metrics
"""
import time
import psutil
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json


@dataclass
class APIMetric:
    """Individual API call metric"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: datetime
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None


@dataclass
class SystemMetric:
    """System resource metric"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int


@dataclass
class PerformanceStats:
    """Performance statistics for an endpoint"""
    total_requests: int = 0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    error_count: int = 0
    success_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class PerformanceMonitor:
    """Performance monitoring system"""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.api_metrics: deque = deque(maxlen=max_metrics)
        self.system_metrics: deque = deque(maxlen=1000)
        self.endpoint_stats: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)
        self.lock = threading.Lock()
        self._monitoring_active = False
        self._system_monitor_task = None
        
    def start_monitoring(self):
        """Start system monitoring"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._system_monitor_task = asyncio.create_task(self._system_monitor_loop())
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self._monitoring_active = False
        if self._system_monitor_task:
            self._system_monitor_task.cancel()
    
    async def _system_monitor_loop(self):
        """Background system monitoring loop"""
        while self._monitoring_active:
            try:
                metric = SystemMetric(
                    timestamp=datetime.now(),
                    cpu_percent=psutil.cpu_percent(interval=1),
                    memory_percent=psutil.virtual_memory().percent,
                    disk_usage=psutil.disk_usage('/').percent,
                    network_io=dict(psutil.net_io_counters()._asdict()),
                    process_count=len(psutil.pids())
                )
                
                with self.lock:
                    self.system_metrics.append(metric)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                print(f"Error in system monitoring: {e}")
                await asyncio.sleep(60)
    
    def record_api_call(self, metric: APIMetric):
        """Record an API call metric"""
        with self.lock:
            self.api_metrics.append(metric)
            
            # Update endpoint statistics
            key = f"{metric.method}:{metric.endpoint}"
            stats = self.endpoint_stats[key]
            
            stats.total_requests += 1
            stats.last_updated = datetime.now()
            
            # Update response time statistics
            if stats.total_requests == 1:
                stats.avg_response_time = metric.response_time
                stats.min_response_time = metric.response_time
                stats.max_response_time = metric.response_time
            else:
                # Running average
                stats.avg_response_time = (
                    (stats.avg_response_time * (stats.total_requests - 1) + metric.response_time) 
                    / stats.total_requests
                )
                stats.min_response_time = min(stats.min_response_time, metric.response_time)
                stats.max_response_time = max(stats.max_response_time, metric.response_time)
            
            # Update success/error counts
            if 200 <= metric.status_code < 400:
                stats.success_count += 1
            else:
                stats.error_count += 1
    
    def get_api_metrics(self, 
                       endpoint: Optional[str] = None,
                       hours: int = 24) -> List[APIMetric]:
        """Get API metrics for analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            metrics = [m for m in self.api_metrics if m.timestamp >= cutoff_time]
            
            if endpoint:
                metrics = [m for m in metrics if m.endpoint == endpoint]
            
            return list(metrics)
    
    def get_system_metrics(self, hours: int = 24) -> List[SystemMetric]:
        """Get system metrics for analysis"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.lock:
            return [m for m in self.system_metrics if m.timestamp >= cutoff_time]
    
    def get_endpoint_stats(self) -> Dict[str, PerformanceStats]:
        """Get current endpoint statistics"""
        with self.lock:
            return dict(self.endpoint_stats)
    
    def get_top_endpoints(self, limit: int = 10) -> List[tuple]:
        """Get top endpoints by request count"""
        with self.lock:
            sorted_endpoints = sorted(
                self.endpoint_stats.items(),
                key=lambda x: x[1].total_requests,
                reverse=True
            )
            return sorted_endpoints[:limit]
    
    def get_slow_endpoints(self, limit: int = 10) -> List[tuple]:
        """Get slowest endpoints by average response time"""
        with self.lock:
            sorted_endpoints = sorted(
                self.endpoint_stats.items(),
                key=lambda x: x[1].avg_response_time,
                reverse=True
            )
            return sorted_endpoints[:limit]
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, int]:
        """Get error summary by status code"""
        metrics = self.get_api_metrics(hours=hours)
        error_counts = defaultdict(int)
        
        for metric in metrics:
            if metric.status_code >= 400:
                error_counts[str(metric.status_code)] += 1
        
        return dict(error_counts)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            total_requests = sum(stats.total_requests for stats in self.endpoint_stats.values())
            total_errors = sum(stats.error_count for stats in self.endpoint_stats.values())
            
            if total_requests > 0:
                error_rate = (total_errors / total_requests) * 100
                avg_response_time = sum(
                    stats.avg_response_time * stats.total_requests 
                    for stats in self.endpoint_stats.values()
                ) / total_requests
            else:
                error_rate = 0
                avg_response_time = 0
            
            # Latest system metrics
            latest_system = self.system_metrics[-1] if self.system_metrics else None
            
            return {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate_percent": round(error_rate, 2),
                "average_response_time_ms": round(avg_response_time * 1000, 2),
                "unique_endpoints": len(self.endpoint_stats),
                "system_metrics": {
                    "cpu_percent": latest_system.cpu_percent if latest_system else 0,
                    "memory_percent": latest_system.memory_percent if latest_system else 0,
                    "disk_usage": latest_system.disk_usage if latest_system else 0,
                } if latest_system else None,
                "monitoring_window_hours": 24,
                "last_updated": datetime.now().isoformat()
            }
    
    def export_metrics(self, filepath: str, hours: int = 24):
        """Export metrics to JSON file"""
        data = {
            "api_metrics": [
                {
                    "endpoint": m.endpoint,
                    "method": m.method,
                    "status_code": m.status_code,
                    "response_time": m.response_time,
                    "timestamp": m.timestamp.isoformat(),
                    "request_size": m.request_size,
                    "response_size": m.response_size,
                    "user_agent": m.user_agent,
                    "ip_address": m.ip_address
                }
                for m in self.get_api_metrics(hours=hours)
            ],
            "system_metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "disk_usage": m.disk_usage,
                    "network_io": m.network_io,
                    "process_count": m.process_count
                }
                for m in self.get_system_metrics(hours=hours)
            ],
            "endpoint_stats": {
                k: {
                    "total_requests": v.total_requests,
                    "avg_response_time": v.avg_response_time,
                    "min_response_time": v.min_response_time if v.min_response_time != float('inf') else 0,
                    "max_response_time": v.max_response_time,
                    "error_count": v.error_count,
                    "success_count": v.success_count,
                    "last_updated": v.last_updated.isoformat()
                }
                for k, v in self.get_endpoint_stats().items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
