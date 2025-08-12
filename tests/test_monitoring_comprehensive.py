"""
Comprehensive Test Suite cho Monitoring System
"""
import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

# Test cho MetricsCollector
class TestMetricsCollector:
    
    @pytest.fixture
    def metrics_collector(self):
        from infrastructure.monitoring.metrics_collector import MetricsCollector
        return MetricsCollector(max_history=100)
    
    @pytest.fixture
    def sample_system_metric(self):
        from infrastructure.monitoring.metrics_collector import SystemMetrics
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=50.5,
            memory_percent=60.2,
            memory_used_mb=1024.5,
            memory_available_mb=2048.0,
            disk_usage_percent=70.1,
            disk_free_gb=100.5,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            active_connections=10
        )
    
    @pytest.fixture
    def sample_api_metric(self):
        from infrastructure.monitoring.metrics_collector import APIMetrics
        return APIMetrics(
            endpoint="/api/face-detection",
            method="POST",
            response_time=1.5,
            status_code=200,
            timestamp=datetime.now(),
            user_agent="test-agent",
            ip_address="127.0.0.1",
            request_size=1024,
            response_size=512
        )
    
    @pytest.fixture
    def sample_ml_metric(self):
        from infrastructure.monitoring.metrics_collector import MLMetrics
        return MLMetrics(
            model_name="face_detector",
            operation="detect_faces",
            processing_time=2.5,
            accuracy=0.95,
            confidence=0.87,
            timestamp=datetime.now(),
            input_size=2048,
            success=True,
            error_message=""
        )
    
    def test_add_system_metric(self, metrics_collector, sample_system_metric):
        """Test thêm system metric"""
        # Add metric to the deque manually for testing
        metrics_collector.system_metrics.append(sample_system_metric)
        
        assert len(metrics_collector.system_metrics) == 1
        assert metrics_collector.system_metrics[0].cpu_percent == 50.5
    
    def test_add_api_metric(self, metrics_collector, sample_api_metric):
        """Test thêm API metric"""
        metrics_collector.add_api_metric(sample_api_metric)
        
        assert len(metrics_collector.api_metrics) == 1
        assert metrics_collector.api_metrics[0].endpoint == "/api/face-detection"
    
    def test_add_ml_metric(self, metrics_collector, sample_ml_metric):
        """Test thêm ML metric"""
        metrics_collector.add_ml_metric(sample_ml_metric)
        
        assert len(metrics_collector.ml_metrics) == 1
        assert metrics_collector.ml_metrics[0].model_name == "face_detector"
    
    def test_max_history_limit(self, sample_api_metric):
        """Test giới hạn max_history"""
        from infrastructure.monitoring.metrics_collector import MetricsCollector
        collector = MetricsCollector(max_history=5)
        
        # Add 10 metrics
        for i in range(10):
            metric = sample_api_metric
            metric.timestamp = datetime.now()
            collector.add_api_metric(metric)
        
        # Should only keep 5 most recent
        assert len(collector.api_metrics) == 5
    
    def test_get_system_summary_empty(self, metrics_collector):
        """Test system summary khi không có data"""
        summary = metrics_collector.get_system_summary()
        assert summary == {}
    
    def test_get_api_summary_empty(self, metrics_collector):
        """Test API summary khi không có data"""
        summary = metrics_collector.get_api_summary()
        assert summary == {}
    
    def test_get_ml_summary_empty(self, metrics_collector):
        """Test ML summary khi không có data"""
        summary = metrics_collector.get_ml_summary()
        assert summary == {}
    
    def test_get_system_summary_with_data(self, metrics_collector, sample_system_metric):
        """Test system summary với data"""
        metrics_collector.system_metrics.append(sample_system_metric)
        
        summary = metrics_collector.get_system_summary()
        
        assert "current" in summary
        assert "averages" in summary
        assert "alerts" in summary
        assert summary["current"]["cpu_percent"] == 50.5
    
    def test_get_api_summary_with_data(self, metrics_collector, sample_api_metric):
        """Test API summary với data"""
        metrics_collector.add_api_metric(sample_api_metric)
        
        summary = metrics_collector.get_api_summary()
        
        assert "total_requests" in summary
        assert "status_codes" in summary
        assert "endpoints" in summary
        assert summary["total_requests"] == 1
    
    def test_get_ml_summary_with_data(self, metrics_collector, sample_ml_metric):
        """Test ML summary với data"""
        metrics_collector.add_ml_metric(sample_ml_metric)
        
        summary = metrics_collector.get_ml_summary()
        
        assert "total_operations" in summary
        assert "models" in summary
        assert summary["total_operations"] == 1
    
    def test_system_alerts_high_cpu(self, metrics_collector):
        """Test alert khi CPU cao"""
        from infrastructure.monitoring.metrics_collector import SystemMetrics
        
        high_cpu_metric = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # High CPU
            memory_percent=50.0,
            memory_used_mb=1024.0,
            memory_available_mb=2048.0,
            disk_usage_percent=50.0,
            disk_free_gb=100.0,
            network_bytes_sent=1000000,
            network_bytes_recv=2000000,
            active_connections=10
        )
        
        alerts = metrics_collector._check_system_alerts(high_cpu_metric)
        assert len(alerts) > 0
        assert any("CPU" in alert for alert in alerts)
    
    def test_api_alerts_high_error_rate(self, metrics_collector):
        """Test alert khi error rate cao"""
        # Add metrics with high error rate
        from infrastructure.monitoring.metrics_collector import APIMetrics
        
        metrics = []
        for i in range(10):
            # 50% error rate
            status_code = 500 if i % 2 == 0 else 200
            metric = APIMetrics(
                endpoint="/test",
                method="POST",
                response_time=1.0,
                status_code=status_code,
                timestamp=datetime.now(),
                user_agent="test",
                ip_address="127.0.0.1"
            )
            metrics.append(metric)
        
        alerts = metrics_collector._check_api_alerts(metrics)
        assert len(alerts) > 0

# Test cho PerformanceAnalyzer
class TestPerformanceAnalyzer:
    
    @pytest.fixture
    def mock_metrics_collector(self):
        mock = Mock()
        mock._lock = Mock()
        mock.system_metrics = []
        mock.api_metrics = []
        mock.ml_metrics = []
        return mock
    
    @pytest.fixture
    def performance_analyzer(self, mock_metrics_collector):
        from infrastructure.monitoring.performance_analyzer import PerformanceAnalyzer
        return PerformanceAnalyzer(mock_metrics_collector)
    
    def test_generate_system_report_empty(self, performance_analyzer):
        """Test generate report khi không có data"""
        report = performance_analyzer.generate_system_report(hours=24)
        
        assert "generated_at" in report
        assert "period_hours" in report
        assert report["period_hours"] == 24
    
    @patch('pandas.DataFrame')
    def test_analyze_system_metrics(self, mock_df, performance_analyzer):
        """Test phân tích system metrics"""
        # Mock pandas DataFrame
        mock_df_instance = Mock()
        mock_df_instance.mean.return_value = 50.0
        mock_df_instance.max.return_value = 80.0
        mock_df_instance.min.return_value = 20.0
        mock_df_instance.std.return_value = 10.0
        mock_df_instance.__getitem__.return_value = mock_df_instance
        mock_df_instance.iloc = Mock()
        mock_df_instance.iloc.__getitem__.return_value = 1000
        mock_df.return_value = mock_df_instance
        
        # Create sample metrics
        from infrastructure.monitoring.metrics_collector import SystemMetrics
        metrics = [
            SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_used_mb=1024.0,
                memory_available_mb=2048.0,
                disk_usage_percent=70.0,
                disk_free_gb=100.0,
                network_bytes_sent=1000000,
                network_bytes_recv=2000000,
                active_connections=10
            )
        ]
        
        result = performance_analyzer._analyze_system_metrics(metrics)
        
        assert "total_samples" in result
        assert "cpu" in result
        assert "memory" in result

# Test cho Monitoring API
class TestMonitoringAPI:
    
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from presentation.api.monitoring_api import router
        
        app = FastAPI()
        app.include_router(router)
        
        return TestClient(app)
    
    @patch('presentation.api.monitoring_api.metrics_collector')
    def test_health_check_healthy(self, mock_collector, client):
        """Test health check khi hệ thống healthy"""
        mock_collector.get_system_summary.return_value = {"alerts": []}
        mock_collector.get_api_summary.return_value = {"alerts": []}
        mock_collector.get_ml_summary.return_value = {"alerts": []}
        
        response = client.get("/monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    @patch('presentation.api.monitoring_api.metrics_collector')
    def test_health_check_warning(self, mock_collector, client):
        """Test health check khi có warnings"""
        mock_collector.get_system_summary.return_value = {"alerts": ["High CPU"]}
        mock_collector.get_api_summary.return_value = {"alerts": []}
        mock_collector.get_ml_summary.return_value = {"alerts": ["Low accuracy"]}
        
        response = client.get("/monitoring/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "warning"
        assert len(data["alerts"]) == 2
    
    def test_get_system_metrics(self, client):
        """Test lấy system metrics"""
        with patch('presentation.api.monitoring_api.metrics_collector') as mock_collector:
            mock_collector.get_system_summary.return_value = {"cpu_percent": 50.0}
            
            response = client.get("/monitoring/metrics/system")
            
            assert response.status_code == 200
            data = response.json()
            assert "cpu_percent" in data
    
    def test_get_api_metrics(self, client):
        """Test lấy API metrics"""
        with patch('presentation.api.monitoring_api.metrics_collector') as mock_collector:
            mock_collector.get_api_summary.return_value = {"total_requests": 100}
            
            response = client.get("/monitoring/metrics/api")
            
            assert response.status_code == 200
            data = response.json()
            assert "total_requests" in data
    
    def test_get_ml_metrics(self, client):
        """Test lấy ML metrics"""
        with patch('presentation.api.monitoring_api.metrics_collector') as mock_collector:
            mock_collector.get_ml_summary.return_value = {"total_operations": 50}
            
            response = client.get("/monitoring/metrics/ml")
            
            assert response.status_code == 200
            data = response.json()
            assert "total_operations" in data
    
    def test_record_api_metric(self, client):
        """Test ghi API metric"""
        payload = {
            "endpoint": "/test",
            "method": "POST",
            "response_time": 1.5,
            "status_code": 200
        }
        
        with patch('presentation.api.monitoring_api.metrics_collector') as mock_collector:
            response = client.post("/monitoring/metrics/api/record", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "recorded"
            mock_collector.add_api_metric.assert_called_once()
    
    def test_record_ml_metric(self, client):
        """Test ghi ML metric"""
        payload = {
            "model_name": "face_detector",
            "operation": "detect",
            "processing_time": 2.5,
            "accuracy": 0.95,
            "confidence": 0.87
        }
        
        with patch('presentation.api.monitoring_api.metrics_collector') as mock_collector:
            response = client.post("/monitoring/metrics/ml/record", json=payload)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "recorded"
            mock_collector.add_ml_metric.assert_called_once()

# Test cho Middleware
class TestMetricsMiddleware:
    
    @pytest.fixture
    def mock_app(self):
        return Mock()
    
    @pytest.fixture
    def middleware(self, mock_app):
        from infrastructure.monitoring.middleware import MetricsMiddleware
        return MetricsMiddleware(mock_app)
    
    @pytest.fixture
    def mock_request(self):
        request = Mock()
        request.url.path = "/api/test"
        request.method = "POST"
        request.headers = {"user-agent": "test-agent", "content-length": "1024"}
        request.client.host = "127.0.0.1"
        return request
    
    @pytest.fixture
    def mock_response(self):
        response = Mock()
        response.status_code = 200
        response.headers = {"content-length": "512"}
        return response
    
    @pytest.mark.asyncio
    async def test_dispatch_successful_request(self, middleware, mock_request, mock_response):
        """Test middleware với request thành công"""
        async def mock_call_next(request):
            return mock_response
        
        with patch('infrastructure.monitoring.middleware.metrics_collector') as mock_collector:
            result = await middleware.dispatch(mock_request, mock_call_next)
            
            assert result == mock_response
            mock_collector.add_api_metric.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dispatch_excluded_path(self, middleware, mock_response):
        """Test middleware với excluded path"""
        request = Mock()
        request.url.path = "/docs"
        
        async def mock_call_next(request):
            return mock_response
        
        with patch('infrastructure.monitoring.middleware.metrics_collector') as mock_collector:
            result = await middleware.dispatch(request, mock_call_next)
            
            assert result == mock_response
            mock_collector.add_api_metric.assert_not_called()
    
    def test_get_client_ip_forwarded(self, middleware):
        """Test lấy client IP từ forwarded header"""
        request = Mock()
        request.headers = {"x-forwarded-for": "192.168.1.1, 10.0.0.1"}
        
        ip = middleware._get_client_ip(request)
        assert ip == "192.168.1.1"
    
    def test_get_client_ip_real_ip(self, middleware):
        """Test lấy client IP từ real-ip header"""
        request = Mock()
        request.headers = {"x-real-ip": "192.168.1.2"}
        
        ip = middleware._get_client_ip(request)
        assert ip == "192.168.1.2"
    
    def test_get_client_ip_direct(self, middleware):
        """Test lấy client IP từ direct connection"""
        request = Mock()
        request.headers = {}
        request.client.host = "192.168.1.3"
        
        ip = middleware._get_client_ip(request)
        assert ip == "192.168.1.3"

# Test cho ML Metrics Helper
class TestMLMetricsHelper:
    
    def test_record_ml_operation(self):
        """Test ghi ML operation"""
        from infrastructure.monitoring.middleware import MLMetricsHelper
        
        start_time = time.time() - 1.0  # 1 second ago
        
        with patch('infrastructure.monitoring.middleware.metrics_collector') as mock_collector:
            MLMetricsHelper.record_ml_operation(
                model_name="test_model",
                operation="test_op",
                start_time=start_time,
                success=True,
                accuracy=0.95,
                confidence=0.87
            )
            
            mock_collector.add_ml_metric.assert_called_once()
            
            # Verify the metric
            call_args = mock_collector.add_ml_metric.call_args[0][0]
            assert call_args.model_name == "test_model"
            assert call_args.operation == "test_op"
            assert call_args.success == True
            assert call_args.accuracy == 0.95
            assert call_args.confidence == 0.87
    
    def test_track_ml_operation_decorator(self):
        """Test ML operation decorator"""
        from infrastructure.monitoring.middleware import track_ml_operation
        
        @track_ml_operation("test_model", "test_operation")
        def test_function(x):
            return {"result": x * 2, "accuracy": 0.9}
        
        with patch('infrastructure.monitoring.middleware.MLMetricsHelper.record_ml_operation') as mock_record:
            result = test_function(5)
            
            assert result["result"] == 10
            mock_record.assert_called_once()
            
            # Check the call arguments
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs["model_name"] == "test_model"
            assert call_kwargs["operation"] == "test_operation"
            assert call_kwargs["success"] == True
            assert call_kwargs["accuracy"] == 0.9
    
    def test_track_ml_operation_decorator_error(self):
        """Test ML operation decorator với error"""
        from infrastructure.monitoring.middleware import track_ml_operation
        
        @track_ml_operation("test_model", "test_operation")
        def failing_function():
            raise ValueError("Test error")
        
        with patch('infrastructure.monitoring.middleware.MLMetricsHelper.record_ml_operation') as mock_record:
            with pytest.raises(ValueError):
                failing_function()
            
            mock_record.assert_called_once()
            
            # Check error was recorded
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs["success"] == False
            assert "Test error" in call_kwargs["error_message"]
    
    def test_ml_operation_tracker_context_manager(self):
        """Test ML operation tracker context manager"""
        from infrastructure.monitoring.middleware import MLOperationTracker
        
        with patch('infrastructure.monitoring.middleware.MLMetricsHelper.record_ml_operation') as mock_record:
            with MLOperationTracker("test_model", "test_op", input_size=1024) as tracker:
                tracker.set_metrics(accuracy=0.95, confidence=0.87)
            
            mock_record.assert_called_once()
            
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs["model_name"] == "test_model"
            assert call_kwargs["operation"] == "test_op"
            assert call_kwargs["input_size"] == 1024
            assert call_kwargs["accuracy"] == 0.95
            assert call_kwargs["confidence"] == 0.87
    
    def test_ml_operation_tracker_with_exception(self):
        """Test ML operation tracker với exception"""
        from infrastructure.monitoring.middleware import MLOperationTracker
        
        with patch('infrastructure.monitoring.middleware.MLMetricsHelper.record_ml_operation') as mock_record:
            with pytest.raises(ValueError):
                with MLOperationTracker("test_model", "test_op") as tracker:
                    raise ValueError("Test error")
            
            mock_record.assert_called_once()
            
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs["success"] == False
            assert "Test error" in call_kwargs["error_message"]

# Integration Tests
class TestMonitoringIntegration:
    
    @pytest.mark.asyncio
    async def test_full_monitoring_flow(self):
        """Test full monitoring flow end-to-end"""
        from infrastructure.monitoring.metrics_collector import MetricsCollector, APIMetrics, MLMetrics
        from infrastructure.monitoring.performance_analyzer import PerformanceAnalyzer
        
        # Initialize components
        collector = MetricsCollector(max_history=100)
        analyzer = PerformanceAnalyzer(collector)
        
        # Add sample metrics
        api_metric = APIMetrics(
            endpoint="/test",
            method="POST",
            response_time=1.5,
            status_code=200,
            timestamp=datetime.now(),
            user_agent="test",
            ip_address="127.0.0.1"
        )
        
        ml_metric = MLMetrics(
            model_name="test_model",
            operation="test_op",
            processing_time=2.0,
            accuracy=0.95,
            confidence=0.87,
            timestamp=datetime.now(),
            success=True
        )
        
        collector.add_api_metric(api_metric)
        collector.add_ml_metric(ml_metric)
        
        # Test summaries
        api_summary = collector.get_api_summary()
        ml_summary = collector.get_ml_summary()
        
        assert api_summary["total_requests"] == 1
        assert ml_summary["total_operations"] == 1
        
        # Test performance analysis
        report = analyzer.generate_system_report(hours=1)
        
        assert "generated_at" in report
        assert "api_analysis" in report
        assert "ml_analysis" in report

if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__ + "::TestMetricsCollector::test_add_api_metric", "-v"])
