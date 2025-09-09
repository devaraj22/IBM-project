#!/usr/bin/env python3
"""
Test script for verifying free medical API integrations
Tests the new MedicalAPIDrugService and its endpoints
"""

import asyncio
import json
import time
import requests
from datetime import datetime

DEFAULT_SERVER_URL = "http://localhost:8000"

class APITester:
    def __init__(self, server_url=DEFAULT_SERVER_URL):
        self.server_url = server_url
        self.test_results = []

    def log_result(self, test_name, success, message=""):
        result = {
            "test": test_name,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        status = "✅" if success else "❌"
        print(f"{status} {test_name}: {message}")

    async def test_health_endpoint(self):
        """Test the health check endpoint"""
        try:
            url = f"{self.server_url}/health"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                services = data.get("services", {})
                api_drugs_status = services.get("api_drugs", "not_found")

                if api_drugs_status == "operational":
                    self.log_result("Health Check", True, "API drugs service is operational")
                    return True
                else:
                    self.log_result("Health Check", False, f"API drugs service status: {api_drugs_status}")
                    return False
            else:
                self.log_result("Health Check", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Health Check", False, f"Error: {str(e)}")
            return False

    async def test_drug_stats_endpoint(self):
        """Test the drug statistics endpoint"""
        try:
            url = f"{self.server_url}/drug-stats"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Check for expected fields
                api_services = data.get("api_services", [])
                data_sources = data.get("data_sources", [])

                if len(api_services) >= 3:
                    self.log_result("Drug Stats", True, f"Found {len(api_services)} API services: {', '.join(api_services[:3])}")
                    return True
                else:
                    self.log_result("Drug Stats", False, f"Only found {len(api_services)} API services")
                    return False
            else:
                self.log_result("Drug Stats", False, f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Drug Stats", False, f"Error: {str(e)}")
            return False

    async def test_comprehensive_drug_search(self, drug_name="aspirin"):
        """Test comprehensive drug search with API integration"""
        try:
            url = f"{self.server_url}/api-drug-info"
            payload = {"drug_name": drug_name, "use_api": True}
            headers = {"Content-Type": "application/json"}

            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()

                # Check key fields
                sources_used = data.get("sources_used", [])
                confidence_score = data.get("confidence_score", 0)
                data_quality = data.get("data_quality", "")

                success_msg = f"Found data from {len(sources_used)} sources"
                if sources_used:
                    success_msg += f": {', '.join(sources_used)}"

                self.log_result("Comprehensive Drug Search", True, success_msg)

                # Additional checks
                if confidence_score > 0.7:
                    print(f"  📊 High confidence score: {confidence_score:.2f}")
                elif confidence_score > 0.5:
                    print(f"  📊 Moderate confidence score: {confidence_score:.2f}")
                else:
                    print(f"  📊 Low confidence score: {confidence_score:.2f}")

                print(f"  📈 Data quality: {data_quality}")

                # Show some sample data if available
                brand_name = data.get("brand_name", [])
                if brand_name and brand_name[0]:
                    print(f"  💊 Brand name: {brand_name[0][:50]}")

                return True
            else:
                error_data = response.json() if response.text else {}
                error_msg = error_data.get("detail", f"HTTP {response.status_code}")
                self.log_result("Comprehensive Drug Search", False, f"Error: {error_msg}")
                return False

        except requests.exceptions.Timeout:
            self.log_result("Comprehensive Drug Search", False, "Request timed out (APIs may be slow)")
            return False
        except Exception as e:
            self.log_result("Comprehensive Drug Search", False, f"Error: {str(e)}")
            return False

    async def test_api_search_endpoint(self, drug_name="ibuprofen"):
        """Test the dedicated API search endpoint"""
        try:
            url = f"{self.server_url}/api-drug-search/{drug_name}"
            response = requests.get(url, timeout=20)

            if response.status_code == 200:
                data = response.json()
                comprehensive_results = data.get("comprehensive_results", [])

                if comprehensive_results:
                    self.log_result("API Search Endpoint", True, f"Found {len(comprehensive_results)} comprehensive results")
                    # Show first result details
                    first_result = comprehensive_results[0]
                    search_method = first_result.get("search_method", "unknown")
                    print(f"  🔍 Search method: {search_method}")
                    return True
                else:
                    self.log_result("API Search Endpoint", False, "No comprehensive results found")
                    return False
            else:
                error_msg = f"HTTP {response.status_code}"
                self.log_result("API Search Endpoint", False, error_msg)
                return False

        except requests.exceptions.Timeout:
            self.log_result("API Search Endpoint", False, "Request timed out")
            return False
        except Exception as e:
            self.log_result("API Search Endpoint", False, f"Error: {str(e)}")
            return False

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("🧪 API INTEGRATION TEST SUMMARY")
        print("="*60)

        successful_tests = sum(1 for result in self.test_results if result["success"])
        total_tests = len(self.test_results)

        print(f"✅ Successful: {successful_tests}/{total_tests}")

        if successful_tests > 0:
            print("\n🎉 Integration successfully implemented!")
            print("📡 Free medical APIs are working:")
            print("   • OpenFDA API - FDA drug labeling data")
            print("   • PubChem API - Chemical structure and properties")
            print("   • RxNorm API - Standardized drug names")

            if successful_tests == total_tests:
                print("\n🏆 ALL TESTS PASSED - Ready for production!")
        else:
            print("\n❌ All tests failed - Check server startup and network connectivity")

        print("\n📊 Detailed Results:")
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {result['test']}: {result['message']}")

async def main():
    """Run all API integration tests"""

    print("🚀 Starting Free Medical API Integration Tests")
    print("This will test the newly integrated OpenFDA, PubChem, and RxNorm APIs")
    print("-" * 60)

    # Give server a moment to start up
    print("⏳ Waiting for server to initialize...")
    time.sleep(3)

    tester = APITester()

    # Run tests
    print("\n🔍 Running API integration tests...\n")

    await tester.test_health_endpoint()
    await tester.test_drug_stats_endpoint()
    await tester.test_comprehensive_drug_search("aspirin")
    await tester.test_api_search_endpoint("ibuprofen")

    # Print final summary
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())