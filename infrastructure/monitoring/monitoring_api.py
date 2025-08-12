"""
Monitoring and analytics API endpoints
"""
from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import JSONResponse, FileResponse
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import tempfile

from .performance_monitor import performance_monitor, PerformanceStats
from ..database.mongodb_client import mongodb_client

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/health", summary="Health Check")
async def health_check():
    """
    Comprehensive health check endpoint including database status
    """
    # Get database health
    db_health = mongodb_client.health_check()
    
    # Get performance summary
    perf_summary = performance_monitor.get_performance_summary()
    
    return {
        "status": "healthy" if db_health["status"] == "healthy" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "service": "Face Verification API",
        "version": "1.0.0",
        "database": db_health,
        "performance": {
            "total_requests": perf_summary.get("total_requests", 0),
            "error_rate_percent": perf_summary.get("error_rate_percent", 0),
            "average_response_time_ms": perf_summary.get("average_response_time_ms", 0)
        },
        "monitoring": {
            "active": True,
            "metrics_collected": len(performance_monitor.api_metrics),
            "endpoints_tracked": len(performance_monitor.endpoint_stats)
        }
    }


@router.get("/performance/summary", summary="Performance Summary")
async def get_performance_summary():
    """
    Get comprehensive performance summary including:
    - Total requests and errors
    - Average response time
    - Error rate
    - System metrics
    """
    try:
        summary = performance_monitor.get_performance_summary()
        return JSONResponse(content=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")


@router.get("/performance/endpoints", summary="Endpoint Statistics")
async def get_endpoint_stats():
    """
    Get detailed statistics for all endpoints
    """
    try:
        stats = performance_monitor.get_endpoint_stats()
        
        # Convert to serializable format
        result = {}
        for endpoint, stat in stats.items():
            result[endpoint] = {
                "total_requests": stat.total_requests,
                "avg_response_time_ms": round(stat.avg_response_time * 1000, 2),
                "min_response_time_ms": round(stat.min_response_time * 1000, 2) if stat.min_response_time != float('inf') else 0,
                "max_response_time_ms": round(stat.max_response_time * 1000, 2),
                "error_count": stat.error_count,
                "success_count": stat.success_count,
                "error_rate_percent": round((stat.error_count / stat.total_requests) * 100, 2) if stat.total_requests > 0 else 0,
                "last_updated": stat.last_updated.isoformat()
            }
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get endpoint stats: {str(e)}")


@router.get("/performance/top-endpoints", summary="Top Endpoints by Request Count")
async def get_top_endpoints(limit: int = Query(10, ge=1, le=50)):
    """
    Get top endpoints by request count
    """
    try:
        top_endpoints = performance_monitor.get_top_endpoints(limit)
        
        result = []
        for endpoint, stats in top_endpoints:
            result.append({
                "endpoint": endpoint,
                "total_requests": stats.total_requests,
                "avg_response_time_ms": round(stats.avg_response_time * 1000, 2),
                "error_rate_percent": round((stats.error_count / stats.total_requests) * 100, 2) if stats.total_requests > 0 else 0
            })
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get top endpoints: {str(e)}")


@router.get("/performance/slow-endpoints", summary="Slowest Endpoints")
async def get_slow_endpoints(limit: int = Query(10, ge=1, le=50)):
    """
    Get slowest endpoints by average response time
    """
    try:
        slow_endpoints = performance_monitor.get_slow_endpoints(limit)
        
        result = []
        for endpoint, stats in slow_endpoints:
            result.append({
                "endpoint": endpoint,
                "avg_response_time_ms": round(stats.avg_response_time * 1000, 2),
                "total_requests": stats.total_requests,
                "error_rate_percent": round((stats.error_count / stats.total_requests) * 100, 2) if stats.total_requests > 0 else 0
            })
        
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get slow endpoints: {str(e)}")


@router.get("/performance/errors", summary="Error Summary")
async def get_error_summary(hours: int = Query(24, ge=1, le=168)):
    """
    Get error summary by status code for the last N hours
    """
    try:
        error_summary = performance_monitor.get_error_summary(hours)
        
        # Add total error count
        total_errors = sum(error_summary.values())
        
        return JSONResponse(content={
            "total_errors": total_errors,
            "errors_by_status_code": error_summary,
            "time_window_hours": hours,
            "generated_at": datetime.now().isoformat()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get error summary: {str(e)}")


@router.get("/performance/api-metrics", summary="API Metrics")
async def get_api_metrics(
    endpoint: Optional[str] = Query(None, description="Filter by specific endpoint"),
    hours: int = Query(24, ge=1, le=168, description="Hours of data to retrieve")
):
    """
    Get detailed API metrics for analysis
    """
    try:
        metrics = performance_monitor.get_api_metrics(endpoint=endpoint, hours=hours)
        
        result = []
        for metric in metrics[-1000:]:  # Limit to last 1000 records
            result.append({
                "endpoint": metric.endpoint,
                "method": metric.method,
                "status_code": metric.status_code,
                "response_time_ms": round(metric.response_time * 1000, 2),
                "timestamp": metric.timestamp.isoformat(),
                "request_size": metric.request_size,
                "response_size": metric.response_size
            })
        
        return JSONResponse(content={
            "metrics": result,
            "total_count": len(metrics),
            "returned_count": len(result),
            "time_window_hours": hours,
            "endpoint_filter": endpoint
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get API metrics: {str(e)}")


@router.get("/performance/system-metrics", summary="System Metrics")
async def get_system_metrics(hours: int = Query(24, ge=1, le=168)):
    """
    Get system resource metrics (CPU, memory, disk)
    """
    try:
        metrics = performance_monitor.get_system_metrics(hours)
        
        result = []
        for metric in metrics:
            result.append({
                "timestamp": metric.timestamp.isoformat(),
                "cpu_percent": metric.cpu_percent,
                "memory_percent": metric.memory_percent,
                "disk_usage": metric.disk_usage,
                "process_count": metric.process_count,
                "network_bytes_sent": metric.network_io.get('bytes_sent', 0),
                "network_bytes_recv": metric.network_io.get('bytes_recv', 0)
            })
        
        return JSONResponse(content={
            "metrics": result,
            "count": len(result),
            "time_window_hours": hours
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")


@router.get("/analytics/trends", summary="Performance Trends")
async def get_performance_trends(hours: int = Query(24, ge=1, le=168)):
    """
    Get performance trends and analytics
    """
    try:
        # Get API metrics
        metrics = performance_monitor.get_api_metrics(hours=hours)
        
        if not metrics:
            return JSONResponse(content={
                "message": "No metrics available for the specified time period",
                "time_window_hours": hours
            })
        
        # Calculate hourly aggregations
        hourly_data = {}
        for metric in metrics:
            hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {
                    "requests": 0,
                    "errors": 0,
                    "total_response_time": 0,
                    "response_times": []
                }
            
            hourly_data[hour_key]["requests"] += 1
            hourly_data[hour_key]["total_response_time"] += metric.response_time
            hourly_data[hour_key]["response_times"].append(metric.response_time)
            
            if metric.status_code >= 400:
                hourly_data[hour_key]["errors"] += 1
        
        # Create trend data
        trends = []
        for hour, data in sorted(hourly_data.items()):
            avg_response_time = data["total_response_time"] / data["requests"]
            error_rate = (data["errors"] / data["requests"]) * 100
            
            # Calculate percentiles
            response_times = sorted(data["response_times"])
            count = len(response_times)
            p50 = response_times[int(count * 0.5)] if count > 0 else 0
            p95 = response_times[int(count * 0.95)] if count > 0 else 0
            p99 = response_times[int(count * 0.99)] if count > 0 else 0
            
            trends.append({
                "hour": hour.isoformat(),
                "requests": data["requests"],
                "errors": data["errors"],
                "error_rate_percent": round(error_rate, 2),
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "p50_response_time_ms": round(p50 * 1000, 2),
                "p95_response_time_ms": round(p95 * 1000, 2),
                "p99_response_time_ms": round(p99 * 1000, 2)
            })
        
        return JSONResponse(content={
            "trends": trends,
            "summary": {
                "total_requests": len(metrics),
                "total_errors": sum(1 for m in metrics if m.status_code >= 400),
                "time_window_hours": hours,
                "data_points": len(trends)
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance trends: {str(e)}")


@router.get("/export/metrics", summary="Export Metrics")
async def export_metrics(
    hours: int = Query(24, ge=1, le=168),
    format: str = Query("json", regex="^(json|csv)$")
):
    """
    Export performance metrics to file
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{format}', delete=False) as temp_file:
            temp_path = temp_file.name
        
        if format == "json":
            performance_monitor.export_metrics(temp_path, hours)
            media_type = "application/json"
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            # For CSV, we'd need to implement CSV export
            raise HTTPException(status_code=400, detail="CSV export not implemented yet")
        
        return FileResponse(
            path=temp_path,
            media_type=media_type,
            filename=filename,
            background=lambda: os.unlink(temp_path)  # Clean up temp file
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export metrics: {str(e)}")


@router.post("/control/start-monitoring", summary="Start System Monitoring")
async def start_monitoring():
    """
    Start system resource monitoring
    """
    try:
        performance_monitor.start_monitoring()
        return {"message": "System monitoring started", "status": "active"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")


@router.post("/control/stop-monitoring", summary="Stop System Monitoring")
async def stop_monitoring():
    """
    Stop system resource monitoring
    """
    try:
        performance_monitor.stop_monitoring()
        return {"message": "System monitoring stopped", "status": "inactive"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")
