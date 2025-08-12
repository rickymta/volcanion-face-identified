"""
Comprehensive test suite for monitoring system
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import json
import os

from infrastructure.monitoring.performance_monitor import (
    PerformanceMonitor, APIMetric, SystemMetric, PerformanceStats
)
from infrastructure.monitoring.middleware import PerformanceMiddleware
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.monitor = PerformanceMonitor(max_metrics=100)
    
    def test_record_api_call(self):
        """Test recording API calls"""
        metric = APIMetric(
            endpoint="/test",
            method="GET",
            status_code=200,
            response_time=0.1,
            timestamp=datetime.now()
        )
        
        self.monitor.record_api_call(metric)
        
        # Check metric was recorded
        assert len(self.monitor.api_metrics) == 1
        assert self.monitor.api_metrics[0] == metric
        
        # Check endpoint stats were updated
        stats = self.monitor.get_endpoint_stats()
        key = "GET:/test"
        assert key in stats
        assert stats[key].total_requests == 1
        assert stats[key].success_count == 1
        assert stats[key].error_count == 0
        assert stats[key].avg_response_time == 0.1
    
    def test_endpoint_statistics(self):
        """Test endpoint statistics calculation"""
        # Add multiple metrics for same endpoint
        for i in range(5):
            metric = APIMetric(
                endpoint="/test",
                method="GET",
                status_code=200 if i < 4 else 500,
                response_time=0.1 + (i * 0.05),
                timestamp=datetime.now()
            )
            self.monitor.record_api_call(metric)
        
        stats = self.monitor.get_endpoint_stats()
        key = "GET:/test"
        
        assert stats[key].total_requests == 5
        assert stats[key].success_count == 4
        assert stats[key].error_count == 1
        assert stats[key].min_response_time == 0.1
        assert stats[key].max_response_time == 0.3
        # Average should be (0.1 + 0.15 + 0.2 + 0.25 + 0.3) / 5 = 0.2
        assert abs(stats[key].avg_response_time - 0.2) < 0.01
    
    def test_get_api_metrics_filtering(self):
        """Test API metrics filtering by endpoint and time"""
        now = datetime.now()
        old_time = now - timedelta(hours=25)
        
        # Add metrics with different times and endpoints
        metrics = [
            APIMetric("/test1", "GET", 200, 0.1, now),
            APIMetric("/test2", "POST", 201, 0.2, now),
            APIMetric("/test1", "GET", 200, 0.15, old_time),
        ]
        
        for metric in metrics:
            self.monitor.record_api_call(metric)
        
        # Test time filtering (last 24 hours)
        recent_metrics = self.monitor.get_api_metrics(hours=24)
        assert len(recent_metrics) == 2
        
        # Test endpoint filtering
        test1_metrics = self.monitor.get_api_metrics(endpoint="/test1")
        assert len(test1_metrics) == 2
        
        # Test combined filtering
        recent_test1 = self.monitor.get_api_metrics(endpoint="/test1", hours=24)
        assert len(recent_test1) == 1
    
    def test_top_endpoints(self):
        """Test getting top endpoints by request count"""
        # Add metrics for different endpoints
        endpoints = ["/api/v1/users", "/api/v1/posts", "/api/v1/comments"]
        request_counts = [10, 5, 15]
        
        for endpoint, count in zip(endpoints, request_counts):
            for i in range(count):
                metric = APIMetric(endpoint, "GET", 200, 0.1, datetime.now())
                self.monitor.record_api_call(metric)
        
        top_endpoints = self.monitor.get_top_endpoints(limit=3)
        
        # Should be sorted by request count (descending)
        assert len(top_endpoints) == 3
        assert top_endpoints[0][0] == "GET:/api/v1/comments"  # 15 requests
        assert top_endpoints[1][0] == "GET:/api/v1/users"     # 10 requests
        assert top_endpoints[2][0] == "GET:/api/v1/posts"     # 5 requests
    
    def test_slow_endpoints(self):
        """Test getting slowest endpoints"""
        endpoints_times = [
            ("/fast", 0.1),
            ("/medium", 0.5),
            ("/slow", 1.0)
        ]
        
        for endpoint, response_time in endpoints_times:
            metric = APIMetric(endpoint, "GET", 200, response_time, datetime.now())
            self.monitor.record_api_call(metric)
        
        slow_endpoints = self.monitor.get_slow_endpoints(limit=3)
        
        # Should be sorted by response time (descending)
        assert len(slow_endpoints) == 3
        assert slow_endpoints[0][0] == "GET:/slow"    # 1.0s
        assert slow_endpoints[1][0] == "GET:/medium"  # 0.5s
        assert slow_endpoints[2][0] == "GET:/fast"    # 0.1s
    
    def test_error_summary(self):
        """Test error summary generation"""
        # Add metrics with different status codes
        status_codes = [200, 200, 400, 404, 500, 500, 500]
        
        for status_code in status_codes:
            metric = APIMetric("/test", "GET", status_code, 0.1, datetime.now())
            self.monitor.record_api_call(metric)
        
        error_summary = self.monitor.get_error_summary()
        
        assert error_summary["400"] == 1
        assert error_summary["404"] == 1
        assert error_summary["500"] == 3
        assert "200" not in error_summary  # Success codes not included
    
    def test_performance_summary(self):
        """Test comprehensive performance summary"""
        # Add some test metrics
        metrics = [
            APIMetric("/test1", "GET", 200, 0.1, datetime.now()),
            APIMetric("/test2", "POST", 201, 0.2, datetime.now()),
            APIMetric("/test1", "GET", 500, 0.15, datetime.now()),
        ]
        
        for metric in metrics:
            self.monitor.record_api_call(metric)
        
        summary = self.monitor.get_performance_summary()
        
        assert summary["total_requests"] == 3
        assert summary["total_errors"] == 1
        assert summary["error_rate_percent"] == 33.33
        assert summary["unique_endpoints"] == 2
        assert "average_response_time_ms" in summary
        assert "last_updated" in summary
    
    def test_export_metrics(self):
        """Test metrics export functionality"""
        # Add test metrics
        metric = APIMetric("/test", "GET", 200, 0.1, datetime.now())
        self.monitor.record_api_call(metric)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            self.monitor.export_metrics(temp_path, hours=24)
            
            # Verify file was created and contains data
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "api_metrics" in data
            assert "endpoint_stats" in data
            assert len(data["api_metrics"]) == 1
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestPerformanceMiddleware:
    """Test performance middleware functionality"""
    
    def test_middleware_basic_functionality(self):
        """Test basic middleware functionality"""
        app = FastAPI()
        app.add_middleware(PerformanceMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        
        # Clear any existing metrics
        from infrastructure.monitoring.performance_monitor import performance_monitor
        performance_monitor.api_metrics.clear()
        performance_monitor.endpoint_stats.clear()
        
        # Make request
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-Response-Time" in response.headers
        
        # Check metric was recorded
        stats = performance_monitor.get_endpoint_stats()
        assert "GET:/test" in stats
        assert stats["GET:/test"].total_requests == 1
    
    def test_middleware_error_handling(self):
        """Test middleware handles errors properly"""
        app = FastAPI()
        app.add_middleware(PerformanceMiddleware)
        
        @app.get("/error")
        async def error_endpoint():
            raise Exception("Test error")
        
        client = TestClient(app)
        
        # Clear existing metrics
        from infrastructure.monitoring.performance_monitor import performance_monitor
        performance_monitor.api_metrics.clear()
        performance_monitor.endpoint_stats.clear()
        
        # Make request that will cause error
        response = client.get("/error")
        
        assert response.status_code == 500
        
        # Check error was recorded
        stats = performance_monitor.get_endpoint_stats()
        assert "GET:/error" in stats
        assert stats["GET:/error"].error_count == 1


@pytest.mark.asyncio
class TestAsyncMonitoring:
    """Test async monitoring functionality"""
    
    async def test_system_monitoring_start_stop(self):
        """Test starting and stopping system monitoring"""
        monitor = PerformanceMonitor()
        
        # Test start monitoring
        monitor.start_monitoring()
        assert monitor._monitoring_active is True
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Test stop monitoring
        monitor.stop_monitoring()
        assert monitor._monitoring_active is False
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    @patch('psutil.pids')
    async def test_system_metrics_collection(self, mock_pids, mock_net_io, 
                                           mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection"""
        # Mock system information
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=70.0)
        mock_net_io.return_value = Mock(_asdict=lambda: {'bytes_sent': 1000, 'bytes_recv': 2000})
        mock_pids.return_value = list(range(100))
        
        monitor = PerformanceMonitor()
        
        # Manually trigger system metric collection
        await monitor._system_monitor_loop.__wrapped__(monitor)
        
        # Should have collected one metric (but loop will exit due to _monitoring_active = False)
        system_metrics = monitor.get_system_metrics(hours=1)
        # Note: This test might need adjustment based on actual implementation


class TestMonitoringIntegration:
    """Integration tests for monitoring system"""
    
    def test_full_monitoring_flow(self):
        """Test complete monitoring flow"""
        # Create FastAPI app with monitoring
        app = FastAPI()
        app.add_middleware(PerformanceMiddleware)
        
        @app.get("/api/users")
        async def get_users():
            await asyncio.sleep(0.1)  # Simulate some processing time
            return {"users": []}
        
        @app.post("/api/users")
        async def create_user():
            await asyncio.sleep(0.05)
            return {"id": 1, "name": "Test User"}
        
        client = TestClient(app)
        
        # Clear existing metrics
        from infrastructure.monitoring.performance_monitor import performance_monitor
        performance_monitor.api_metrics.clear()
        performance_monitor.endpoint_stats.clear()
        
        # Make multiple requests
        client.get("/api/users")
        client.post("/api/users")
        client.get("/api/users")
        
        # Verify metrics were collected
        stats = performance_monitor.get_endpoint_stats()
        assert len(stats) == 2
        assert stats["GET:/api/users"].total_requests == 2
        assert stats["POST:/api/users"].total_requests == 1
        
        # Verify performance summary
        summary = performance_monitor.get_performance_summary()
        assert summary["total_requests"] == 3
        assert summary["total_errors"] == 0
        assert summary["error_rate_percent"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
