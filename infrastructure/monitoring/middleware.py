"""
Performance monitoring middleware for FastAPI
"""
import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
from typing import Callable

from .performance_monitor import performance_monitor, APIMetric


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware to collect API performance metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timing
        start_time = time.time()
        
        # Get request info
        method = request.method
        url_path = request.url.path
        user_agent = request.headers.get("user-agent")
        client_host = request.client.host if request.client else None
        
        # Get request size if available
        request_size = None
        if "content-length" in request.headers:
            try:
                request_size = int(request.headers["content-length"])
            except (ValueError, TypeError):
                pass
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Get response size if available
        response_size = None
        if hasattr(response, 'headers') and "content-length" in response.headers:
            try:
                response_size = int(response.headers["content-length"])
            except (ValueError, TypeError):
                pass
        
        # Create metric
        metric = APIMetric(
            endpoint=url_path,
            method=method,
            status_code=response.status_code,
            response_time=response_time,
            timestamp=datetime.now(),
            request_size=request_size,
            response_size=response_size,
            user_agent=user_agent,
            ip_address=client_host
        )
        
        # Record metric
        performance_monitor.record_api_call(metric)
        
        # Add response headers for debugging
        response.headers["X-Response-Time"] = str(round(response_time * 1000, 2))
        
        return response
