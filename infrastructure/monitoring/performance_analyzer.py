"""
Performance Analysis và Report System
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import asdict
import logging
import io
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Phân tích hiệu suất hệ thống"""
    
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
        
    def generate_system_report(self, hours: int = 24) -> Dict[str, Any]:
        """Tạo báo cáo hiệu suất hệ thống"""
        try:
            # Lấy dữ liệu metrics
            with self.metrics_collector._lock:
                system_metrics = list(self.metrics_collector.system_metrics)
                api_metrics = list(self.metrics_collector.api_metrics)
                ml_metrics = list(self.metrics_collector.ml_metrics)
            
            # Filter theo thời gian
            cutoff_time = datetime.now() - timedelta(hours=hours)
            system_metrics = [m for m in system_metrics if m.timestamp >= cutoff_time]
            api_metrics = [m for m in api_metrics if m.timestamp >= cutoff_time]
            ml_metrics = [m for m in ml_metrics if m.timestamp >= cutoff_time]
            
            report = {
                "generated_at": datetime.now().isoformat(),
                "period_hours": hours,
                "system_analysis": self._analyze_system_metrics(system_metrics),
                "api_analysis": self._analyze_api_metrics(api_metrics),
                "ml_analysis": self._analyze_ml_metrics(ml_metrics),
                "recommendations": self._generate_recommendations(system_metrics, api_metrics, ml_metrics)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating system report: {e}")
            return {"error": str(e)}
            
    def _analyze_system_metrics(self, metrics: List) -> Dict[str, Any]:
        """Phân tích system metrics"""
        if not metrics:
            return {"error": "No system metrics available"}
            
        df = pd.DataFrame([asdict(m) for m in metrics])
        
        analysis = {
            "total_samples": len(metrics),
            "time_range": {
                "start": metrics[0].timestamp.isoformat(),
                "end": metrics[-1].timestamp.isoformat()
            },
            "cpu": {
                "avg": float(df['cpu_percent'].mean()),
                "max": float(df['cpu_percent'].max()),
                "min": float(df['cpu_percent'].min()),
                "std": float(df['cpu_percent'].std())
            },
            "memory": {
                "avg_percent": float(df['memory_percent'].mean()),
                "max_percent": float(df['memory_percent'].max()),
                "avg_used_mb": float(df['memory_used_mb'].mean()),
                "max_used_mb": float(df['memory_used_mb'].max())
            },
            "disk": {
                "avg_usage_percent": float(df['disk_usage_percent'].mean()),
                "min_free_gb": float(df['disk_free_gb'].min())
            },
            "network": {
                "total_bytes_sent": int(df['network_bytes_sent'].iloc[-1] - df['network_bytes_sent'].iloc[0]),
                "total_bytes_recv": int(df['network_bytes_recv'].iloc[-1] - df['network_bytes_recv'].iloc[0]),
                "avg_connections": float(df['active_connections'].mean())
            }
        }
        
        # Thêm trends
        analysis["trends"] = self._calculate_trends(df)
        
        return analysis
        
    def _analyze_api_metrics(self, metrics: List) -> Dict[str, Any]:
        """Phân tích API metrics"""
        if not metrics:
            return {"error": "No API metrics available"}
            
        df = pd.DataFrame([asdict(m) for m in metrics])
        
        # Phân tích theo endpoint
        endpoint_analysis = {}
        for endpoint in df['endpoint'].unique():
            endpoint_df = df[df['endpoint'] == endpoint]
            endpoint_analysis[endpoint] = {
                "total_requests": len(endpoint_df),
                "avg_response_time": float(endpoint_df['response_time'].mean()),
                "p95_response_time": float(endpoint_df['response_time'].quantile(0.95)),
                "p99_response_time": float(endpoint_df['response_time'].quantile(0.99)),
                "error_rate": float((endpoint_df['status_code'] >= 400).mean()),
                "success_rate": float((endpoint_df['status_code'] < 400).mean())
            }
            
        # Phân tích theo status code
        status_analysis = df['status_code'].value_counts().to_dict()
        
        # Phân tích theo thời gian
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        hourly_requests = df.groupby('hour').size().to_dict()
        
        analysis = {
            "total_requests": len(metrics),
            "unique_endpoints": len(df['endpoint'].unique()),
            "overall_stats": {
                "avg_response_time": float(df['response_time'].mean()),
                "p95_response_time": float(df['response_time'].quantile(0.95)),
                "p99_response_time": float(df['response_time'].quantile(0.99)),
                "error_rate": float((df['status_code'] >= 400).mean())
            },
            "endpoints": endpoint_analysis,
            "status_codes": {str(k): int(v) for k, v in status_analysis.items()},
            "hourly_distribution": {str(k): int(v) for k, v in hourly_requests.items()}
        }
        
        return analysis
        
    def _analyze_ml_metrics(self, metrics: List) -> Dict[str, Any]:
        """Phân tích ML metrics"""
        if not metrics:
            return {"error": "No ML metrics available"}
            
        df = pd.DataFrame([asdict(m) for m in metrics])
        
        # Phân tích theo model
        model_analysis = {}
        for model in df['model_name'].unique():
            model_df = df[df['model_name'] == model]
            model_analysis[model] = {
                "total_operations": len(model_df),
                "success_rate": float(model_df['success'].mean()),
                "avg_processing_time": float(model_df['processing_time'].mean()),
                "p95_processing_time": float(model_df['processing_time'].quantile(0.95)),
                "avg_accuracy": float(model_df[model_df['accuracy'] > 0]['accuracy'].mean()) if len(model_df[model_df['accuracy'] > 0]) > 0 else 0,
                "avg_confidence": float(model_df[model_df['confidence'] > 0]['confidence'].mean()) if len(model_df[model_df['confidence'] > 0]) > 0 else 0
            }
            
        # Phân tích theo operation
        operation_analysis = {}
        for operation in df['operation'].unique():
            operation_df = df[df['operation'] == operation]
            operation_analysis[operation] = {
                "total_count": len(operation_df),
                "success_rate": float(operation_df['success'].mean()),
                "avg_processing_time": float(operation_df['processing_time'].mean())
            }
            
        analysis = {
            "total_operations": len(metrics),
            "unique_models": len(df['model_name'].unique()),
            "overall_stats": {
                "success_rate": float(df['success'].mean()),
                "avg_processing_time": float(df['processing_time'].mean()),
                "avg_accuracy": float(df[df['accuracy'] > 0]['accuracy'].mean()) if len(df[df['accuracy'] > 0]) > 0 else 0,
                "avg_confidence": float(df[df['confidence'] > 0]['confidence'].mean()) if len(df[df['confidence'] > 0]) > 0 else 0
            },
            "models": model_analysis,
            "operations": operation_analysis
        }
        
        return analysis
        
    def _calculate_trends(self, df: pd.DataFrame) -> Dict[str, str]:
        """Tính toán xu hướng"""
        trends = {}
        
        # CPU trend
        cpu_trend = np.polyfit(range(len(df)), df['cpu_percent'], 1)[0]
        trends['cpu'] = "increasing" if cpu_trend > 0.1 else "decreasing" if cpu_trend < -0.1 else "stable"
        
        # Memory trend
        memory_trend = np.polyfit(range(len(df)), df['memory_percent'], 1)[0]
        trends['memory'] = "increasing" if memory_trend > 0.1 else "decreasing" if memory_trend < -0.1 else "stable"
        
        return trends
        
    def _generate_recommendations(self, system_metrics: List, api_metrics: List, ml_metrics: List) -> List[str]:
        """Tạo khuyến nghị cải thiện"""
        recommendations = []
        
        # System recommendations
        if system_metrics:
            df_system = pd.DataFrame([asdict(m) for m in system_metrics])
            
            if df_system['cpu_percent'].mean() > 70:
                recommendations.append("CPU usage cao - Cân nhắc scale up hoặc tối ưu code")
                
            if df_system['memory_percent'].mean() > 80:
                recommendations.append("Memory usage cao - Kiểm tra memory leaks hoặc tăng RAM")
                
            if df_system['disk_usage_percent'].mean() > 85:
                recommendations.append("Disk space thấp - Dọn dẹp files hoặc mở rộng storage")
                
        # API recommendations
        if api_metrics:
            df_api = pd.DataFrame([asdict(m) for m in api_metrics])
            
            if df_api['response_time'].mean() > 2.0:
                recommendations.append("API response time chậm - Tối ưu database queries và caching")
                
            error_rate = (df_api['status_code'] >= 400).mean()
            if error_rate > 0.05:
                recommendations.append(f"Error rate cao ({error_rate:.1%}) - Kiểm tra logs và fix bugs")
                
        # ML recommendations
        if ml_metrics:
            df_ml = pd.DataFrame([asdict(m) for m in ml_metrics])
            
            if df_ml['processing_time'].mean() > 5.0:
                recommendations.append("ML processing time chậm - Cân nhắc model optimization hoặc GPU acceleration")
                
            success_rate = df_ml['success'].mean()
            if success_rate < 0.95:
                recommendations.append(f"ML success rate thấp ({success_rate:.1%}) - Kiểm tra model stability")
                
        if not recommendations:
            recommendations.append("Hệ thống hoạt động tốt - Không có khuyến nghị đặc biệt")
            
        return recommendations
        
    def generate_charts(self, hours: int = 24) -> Dict[str, str]:
        """Tạo charts base64 encoded"""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            charts = {}
            
            # System metrics chart
            with self.metrics_collector._lock:
                system_metrics = list(self.metrics_collector.system_metrics)
                
            if system_metrics:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                system_metrics = [m for m in system_metrics if m.timestamp >= cutoff_time]
                
                if system_metrics:
                    charts['system_metrics'] = self._create_system_chart(system_metrics)
                    
            # API metrics chart
            with self.metrics_collector._lock:
                api_metrics = list(self.metrics_collector.api_metrics)
                
            if api_metrics:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                api_metrics = [m for m in api_metrics if m.timestamp >= cutoff_time]
                
                if api_metrics:
                    charts['api_metrics'] = self._create_api_chart(api_metrics)
                    
            return charts
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            return {}
            
    def _create_system_chart(self, metrics: List) -> str:
        """Tạo system metrics chart"""
        df = pd.DataFrame([asdict(m) for m in metrics])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Metrics', fontsize=16)
        
        # CPU
        axes[0, 0].plot(df['timestamp'], df['cpu_percent'])
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('Percent')
        
        # Memory
        axes[0, 1].plot(df['timestamp'], df['memory_percent'])
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_ylabel('Percent')
        
        # Disk
        axes[1, 0].plot(df['timestamp'], df['disk_usage_percent'])
        axes[1, 0].set_title('Disk Usage (%)')
        axes[1, 0].set_ylabel('Percent')
        
        # Network
        axes[1, 1].plot(df['timestamp'], df['network_bytes_sent'], label='Sent')
        axes[1, 1].plot(df['timestamp'], df['network_bytes_recv'], label='Received')
        axes[1, 1].set_title('Network Traffic')
        axes[1, 1].set_ylabel('Bytes')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
        
    def _create_api_chart(self, metrics: List) -> str:
        """Tạo API metrics chart"""
        df = pd.DataFrame([asdict(m) for m in metrics])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('API Metrics', fontsize=16)
        
        # Response time
        axes[0, 0].plot(df['timestamp'], df['response_time'])
        axes[0, 0].set_title('Response Time')
        axes[0, 0].set_ylabel('Seconds')
        
        # Status codes
        status_counts = df['status_code'].value_counts()
        axes[0, 1].bar(status_counts.index.astype(str), status_counts.values)
        axes[0, 1].set_title('Status Codes')
        axes[0, 1].set_ylabel('Count')
        
        # Requests per hour
        df['hour'] = df['timestamp'].dt.hour
        hourly_counts = df.groupby('hour').size()
        axes[1, 0].bar(hourly_counts.index, hourly_counts.values)
        axes[1, 0].set_title('Requests per Hour')
        axes[1, 0].set_ylabel('Count')
        
        # Top endpoints
        endpoint_counts = df['endpoint'].value_counts().head(10)
        axes[1, 1].barh(range(len(endpoint_counts)), endpoint_counts.values)
        axes[1, 1].set_yticks(range(len(endpoint_counts)))
        axes[1, 1].set_yticklabels(endpoint_counts.index)
        axes[1, 1].set_title('Top 10 Endpoints')
        axes[1, 1].set_xlabel('Requests')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return chart_base64
        
    def export_data(self, format: str = "json", hours: int = 24) -> str:
        """Export dữ liệu metrics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.metrics_collector._lock:
                system_metrics = [asdict(m) for m in self.metrics_collector.system_metrics if m.timestamp >= cutoff_time]
                api_metrics = [asdict(m) for m in self.metrics_collector.api_metrics if m.timestamp >= cutoff_time]
                ml_metrics = [asdict(m) for m in self.metrics_collector.ml_metrics if m.timestamp >= cutoff_time]
            
            data = {
                "exported_at": datetime.now().isoformat(),
                "period_hours": hours,
                "system_metrics": system_metrics,
                "api_metrics": api_metrics,
                "ml_metrics": ml_metrics
            }
            
            if format.lower() == "json":
                return json.dumps(data, indent=2, default=str)
            elif format.lower() == "csv":
                # Combine all metrics and save as CSV
                all_metrics = []
                for metric in system_metrics:
                    metric['type'] = 'system'
                    all_metrics.append(metric)
                for metric in api_metrics:
                    metric['type'] = 'api'
                    all_metrics.append(metric)
                for metric in ml_metrics:
                    metric['type'] = 'ml'
                    all_metrics.append(metric)
                    
                df = pd.DataFrame(all_metrics)
                return df.to_csv(index=False)
            else:
                return json.dumps(data, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return json.dumps({"error": str(e)})
