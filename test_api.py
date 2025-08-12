"""
API Testing Script
"""
import requests
import json
import time
from typing import Dict, Any

class APITester:
    """Simple API testing utility"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_endpoint(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Test a single endpoint"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data)
            else:
                return {"error": f"Unsupported method: {method}"}
            
            return {
                "endpoint": endpoint,
                "method": method,
                "status_code": response.status_code,
                "success": 200 <= response.status_code < 300,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "method": method,
                "error": str(e),
                "success": False
            }
    
    def run_health_tests(self):
        """Run health and monitoring tests"""
        print("🔍 Running API Health Tests...")
        print("=" * 50)
        
        endpoints = [
            "/",
            "/health", 
            "/monitoring/health",
            "/monitoring/performance/summary",
            "/docs",
            "/openapi.json"
        ]
        
        results = []
        for endpoint in endpoints:
            print(f"Testing {endpoint}...", end=" ")
            result = self.test_endpoint(endpoint)
            results.append(result)
            
            if result.get("success"):
                print(f"✅ {result['status_code']} ({result.get('response_time_ms', 0):.1f}ms)")
            else:
                print(f"❌ {result.get('status_code', 'ERROR')} - {result.get('error', 'Unknown error')}")
        
        print("\n📊 Summary:")
        print("=" * 50)
        
        successful = sum(1 for r in results if r.get("success"))
        total = len(results)
        print(f"Successful tests: {successful}/{total}")
        print(f"Success rate: {(successful/total)*100:.1f}%")
        
        if successful > 0:
            avg_response_time = sum(r.get("response_time_ms", 0) for r in results if r.get("success")) / successful
            print(f"Average response time: {avg_response_time:.1f}ms")
        
        return results
    
    def test_monitoring_features(self):
        """Test monitoring-specific features"""
        print("\n📊 Testing Monitoring Features...")
        print("=" * 50)
        
        monitoring_endpoints = [
            "/monitoring/performance/endpoints",
            "/monitoring/performance/system-metrics",
            "/monitoring/analytics/trends"
        ]
        
        for endpoint in monitoring_endpoints:
            print(f"Testing {endpoint}...", end=" ")
            result = self.test_endpoint(endpoint)
            
            if result.get("success"):
                print(f"✅ {result['status_code']}")
                # Print some data info
                data = result.get("data", {})
                if isinstance(data, dict):
                    keys = list(data.keys())[:3]
                    print(f"    Data keys: {keys}")
            else:
                print(f"❌ {result.get('status_code', 'ERROR')}")


def main():
    """Main test function"""
    print("🚀 Face Verification API Test Suite")
    print("=" * 50)
    
    tester = APITester()
    
    # Basic health tests
    health_results = tester.run_health_tests()
    
    # Test monitoring features if health is good
    if any(r.get("success") for r in health_results):
        tester.test_monitoring_features()
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    main()
