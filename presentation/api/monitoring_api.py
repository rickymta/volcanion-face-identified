"""
Monitoring API for System Analytics
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import io
import json

from infrastructure.monitoring.metrics_collector import metrics_collector, APIMetrics, MLMetrics
from infrastructure.monitoring.performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["System Monitoring"])

# Initialize performance analyzer
performance_analyzer = PerformanceAnalyzer(metrics_collector)

@router.get("/health", summary="System Health Check")
async def health_check():
    """
    Kiểm tra tình trạng sức khỏe hệ thống
    
    Returns:
    - status: healthy/warning/critical
    - metrics: Các chỉ số quan trọng
    - alerts: Cảnh báo (nếu có)
    """
    try:
        system_summary = metrics_collector.get_system_summary()
        api_summary = metrics_collector.get_api_summary()
        ml_summary = metrics_collector.get_ml_summary()
        
        # Xác định trạng thái tổng thể
        all_alerts = []
        if system_summary.get('alerts'):
            all_alerts.extend(system_summary['alerts'])
        if api_summary.get('alerts'):
            all_alerts.extend(api_summary['alerts'])
        if ml_summary.get('alerts'):
            all_alerts.extend(ml_summary['alerts'])
            
        if len(all_alerts) == 0:
            status = "healthy"
        elif len(all_alerts) <= 2:
            status = "warning"
        else:
            status = "critical"
            
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "system": system_summary,
            "api": api_summary,
            "ml": ml_summary,
            "alerts": all_alerts
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/system", summary="System Metrics")
async def get_system_metrics():
    """
    Lấy metrics hệ thống (CPU, Memory, Disk, Network)
    """
    try:
        return metrics_collector.get_system_summary()
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/api", summary="API Metrics") 
async def get_api_metrics():
    """
    Lấy metrics API (Response time, Status codes, Endpoints)
    """
    try:
        return metrics_collector.get_api_summary()
    except Exception as e:
        logger.error(f"Error getting API metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/ml", summary="ML Metrics")
async def get_ml_metrics():
    """
    Lấy metrics Machine Learning (Processing time, Accuracy, Success rate)
    """
    try:
        return metrics_collector.get_ml_summary()
    except Exception as e:
        logger.error(f"Error getting ML metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/performance", summary="Performance Analysis")
async def get_performance_analysis(
    hours: int = Query(24, description="Số giờ để phân tích", ge=1, le=168)
):
    """
    Phân tích hiệu suất hệ thống trong khoảng thời gian
    
    Args:
    - hours: Số giờ để phân tích (1-168 giờ)
    
    Returns:
    - Báo cáo phân tích chi tiết
    - Khuyến nghị cải thiện
    - Xu hướng hệ thống
    """
    try:
        report = performance_analyzer.generate_system_report(hours=hours)
        return report
    except Exception as e:
        logger.error(f"Error generating performance analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/charts", summary="Performance Charts")
async def get_performance_charts(
    hours: int = Query(24, description="Số giờ để tạo chart", ge=1, le=168)
):
    """
    Tạo charts hiệu suất hệ thống (base64 encoded images)
    
    Args:
    - hours: Số giờ để tạo chart
    
    Returns:
    - Charts dưới dạng base64 strings
    """
    try:
        charts = performance_analyzer.generate_charts(hours=hours)
        return {
            "charts": charts,
            "timestamp": datetime.now().isoformat(),
            "period_hours": hours
        }
    except Exception as e:
        logger.error(f"Error generating charts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/data", summary="Export Metrics Data")
async def export_metrics_data(
    format: str = Query("json", description="Format xuất dữ liệu (json/csv)"),
    hours: int = Query(24, description="Số giờ để xuất", ge=1, le=168)
):
    """
    Xuất dữ liệu metrics
    
    Args:
    - format: json hoặc csv
    - hours: Số giờ để xuất dữ liệu
    
    Returns:
    - File dữ liệu metrics
    """
    try:
        data = performance_analyzer.export_data(format=format, hours=hours)
        
        if format.lower() == "csv":
            return PlainTextResponse(
                content=data,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        else:
            return PlainTextResponse(
                content=data,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
            )
            
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", summary="System Alerts")
async def get_system_alerts():
    """
    Lấy tất cả alerts hiện tại của hệ thống
    """
    try:
        system_summary = metrics_collector.get_system_summary()
        api_summary = metrics_collector.get_api_summary()
        ml_summary = metrics_collector.get_ml_summary()
        
        alerts = {
            "timestamp": datetime.now().isoformat(),
            "system_alerts": system_summary.get('alerts', []),
            "api_alerts": api_summary.get('alerts', []),
            "ml_alerts": ml_summary.get('alerts', []),
        }
        
        # Tổng hợp alerts
        all_alerts = []
        all_alerts.extend([{"type": "system", "message": alert} for alert in alerts["system_alerts"]])
        all_alerts.extend([{"type": "api", "message": alert} for alert in alerts["api_alerts"]])
        all_alerts.extend([{"type": "ml", "message": alert} for alert in alerts["ml_alerts"]])
        
        alerts["total_alerts"] = len(all_alerts)
        alerts["all_alerts"] = all_alerts
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics/api/record", summary="Record API Metric")
async def record_api_metric(
    endpoint: str,
    method: str,
    response_time: float,
    status_code: int,
    user_agent: str = "",
    ip_address: str = "",
    request_size: int = 0,
    response_size: int = 0
):
    """
    Ghi lại metric cho API call (internal use)
    """
    try:
        metric = APIMetrics(
            endpoint=endpoint,
            method=method,
            response_time=response_time,
            status_code=status_code,
            timestamp=datetime.now(),
            user_agent=user_agent,
            ip_address=ip_address,
            request_size=request_size,
            response_size=response_size
        )
        
        metrics_collector.add_api_metric(metric)
        
        return {"status": "recorded", "timestamp": metric.timestamp.isoformat()}
        
    except Exception as e:
        logger.error(f"Error recording API metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/metrics/ml/record", summary="Record ML Metric")
async def record_ml_metric(
    model_name: str,
    operation: str,
    processing_time: float,
    accuracy: float = 0.0,
    confidence: float = 0.0,
    input_size: int = 0,
    success: bool = True,
    error_message: str = ""
):
    """
    Ghi lại metric cho ML operation (internal use)
    """
    try:
        metric = MLMetrics(
            model_name=model_name,
            operation=operation,
            processing_time=processing_time,
            accuracy=accuracy,
            confidence=confidence,
            timestamp=datetime.now(),
            input_size=input_size,
            success=success,
            error_message=error_message
        )
        
        metrics_collector.add_ml_metric(metric)
        
        return {"status": "recorded", "timestamp": metric.timestamp.isoformat()}
        
    except Exception as e:
        logger.error(f"Error recording ML metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dashboard", summary="Monitoring Dashboard Data")
async def get_dashboard_data(
    hours: int = Query(6, description="Số giờ hiển thị", ge=1, le=48)
):
    """
    Lấy dữ liệu cho monitoring dashboard
    
    Returns:
    - Tổng quan metrics realtime
    - Charts data
    - Alerts
    - Top statistics
    """
    try:
        # Basic metrics
        health_data = await health_check()
        
        # Performance analysis
        performance_data = performance_analyzer.generate_system_report(hours=hours)
        
        # Dashboard specific aggregations
        dashboard_data = {
            "overview": {
                "status": health_data["status"],
                "total_alerts": len(health_data["alerts"]),
                "uptime_hours": hours,  # Simplified - trong thực tế cần tính từ start time
                "last_updated": datetime.now().isoformat()
            },
            "metrics": {
                "system": health_data.get("system", {}),
                "api": health_data.get("api", {}),
                "ml": health_data.get("ml", {})
            },
            "performance": performance_data,
            "alerts": health_data["alerts"],
            "quick_stats": {
                "cpu_usage": health_data.get("system", {}).get("current", {}).get("cpu_percent", 0),
                "memory_usage": health_data.get("system", {}).get("current", {}).get("memory_percent", 0),
                "total_requests": health_data.get("api", {}).get("total_requests", 0),
                "avg_response_time": health_data.get("api", {}).get("overall_stats", {}).get("avg_response_time", 0),
                "ml_success_rate": health_data.get("ml", {}).get("overall_stats", {}).get("success_rate", 0)
            }
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/control/start", summary="Start Monitoring")
async def start_monitoring_endpoint(background_tasks: BackgroundTasks):
    """
    Bắt đầu thu thập metrics
    """
    try:
        from infrastructure.monitoring.metrics_collector import start_monitoring
        background_tasks.add_task(start_monitoring)
        
        return {
            "status": "started",
            "message": "Monitoring started successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/control/stop", summary="Stop Monitoring")
async def stop_monitoring_endpoint():
    """
    Dừng thu thập metrics
    """
    try:
        from infrastructure.monitoring.metrics_collector import stop_monitoring
        stop_monitoring()
        
        return {
            "status": "stopped",
            "message": "Monitoring stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))
