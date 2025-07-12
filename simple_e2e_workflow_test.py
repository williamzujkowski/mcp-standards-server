#!/usr/bin/env python3
"""
Simple End-to-End Workflow Tests using actual API endpoints.

Tests realistic user workflows with the available HTTP API.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class WorkflowStep:
    name: str
    description: str
    endpoint: str
    method: str = "GET"
    payload: Dict = None
    expected_status: int = 200

@dataclass
class WorkflowResult:
    workflow_name: str
    success: bool
    steps_completed: int
    total_steps: int
    execution_time: float
    step_results: List[Dict]

class SimpleE2EClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    async def execute_step(self, session: aiohttp.ClientSession, step: WorkflowStep) -> Dict[str, Any]:
        """Execute a single workflow step."""
        start_time = time.time()
        url = f"{self.base_url}{step.endpoint}"
        
        try:
            if step.method == "GET":
                async with session.get(url) as response:
                    content = await response.text()
                    execution_time = (time.time() - start_time) * 1000
                    
                    return {
                        "step_name": step.name,
                        "url": url,
                        "status": response.status,
                        "success": response.status == step.expected_status,
                        "execution_time_ms": round(execution_time, 2),
                        "content_length": len(content),
                        "error": None if response.status == step.expected_status else f"Expected {step.expected_status}, got {response.status}"
                    }
            else:
                # POST/PUT requests (for future use)
                async with session.request(step.method, url, json=step.payload) as response:
                    content = await response.text()
                    execution_time = (time.time() - start_time) * 1000
                    
                    return {
                        "step_name": step.name,
                        "url": url,
                        "status": response.status,
                        "success": response.status == step.expected_status,
                        "execution_time_ms": round(execution_time, 2),
                        "content_length": len(content),
                        "error": None if response.status == step.expected_status else f"Expected {step.expected_status}, got {response.status}"
                    }
                    
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return {
                "step_name": step.name,
                "url": url,
                "status": 0,
                "success": False,
                "execution_time_ms": round(execution_time, 2),
                "content_length": 0,
                "error": str(e)
            }

    async def execute_workflow(self, workflow_name: str, steps: List[WorkflowStep]) -> WorkflowResult:
        """Execute a complete workflow."""
        print(f"\n🔄 Executing Workflow: {workflow_name}")
        print("=" * 60)
        
        start_time = time.time()
        step_results = []
        steps_completed = 0
        
        async with aiohttp.ClientSession() as session:
            for i, step in enumerate(steps, 1):
                print(f"Step {i}/{len(steps)}: {step.name}")
                print(f"  {step.description}")
                
                result = await self.execute_step(session, step)
                step_results.append(result)
                
                if result["success"]:
                    steps_completed += 1
                    print(f"  ✅ Success ({result['execution_time_ms']}ms)")
                else:
                    print(f"  ❌ Failed: {result['error']}")
                    # For E2E tests, we'll continue with other steps even if one fails
                
                # Brief pause between steps
                await asyncio.sleep(0.1)
        
        total_time = time.time() - start_time
        success = steps_completed == len(steps)
        
        result = WorkflowResult(
            workflow_name=workflow_name,
            success=success,
            steps_completed=steps_completed,
            total_steps=len(steps),
            execution_time=round(total_time, 2),
            step_results=step_results
        )
        
        print(f"\n📊 Workflow Summary:")
        print(f"   Steps: {steps_completed}/{len(steps)} completed")
        print(f"   Time: {total_time:.2f}s")
        print(f"   Status: {'✅ SUCCESS' if success else '❌ PARTIAL/FAILED'}")
        
        return result

async def main():
    """Run simplified E2E workflow tests."""
    print("🚀 Simple End-to-End Workflow Tests")
    print("=" * 70)
    print("Testing realistic user workflows with actual API endpoints...")
    
    client = SimpleE2EClient()
    all_results = []
    
    # Workflow 1: Developer Getting Started
    developer_workflow = [
        WorkflowStep(
            name="check_server_health",
            description="Verify server is healthy and responsive",
            endpoint="/health"
        ),
        WorkflowStep(
            name="get_server_info", 
            description="Get server version and available endpoints",
            endpoint="/info"
        ),
        WorkflowStep(
            name="list_all_standards",
            description="Get complete list of available standards",
            endpoint="/api/standards"
        ),
        WorkflowStep(
            name="check_metrics",
            description="View server metrics and performance data",
            endpoint="/metrics"
        )
    ]
    
    result1 = await client.execute_workflow("Developer Getting Started", developer_workflow)
    all_results.append(result1)
    
    # Workflow 2: Standards Explorer
    explorer_workflow = [
        WorkflowStep(
            name="browse_standards",
            description="Browse available standards catalog", 
            endpoint="/api/standards"
        ),
        WorkflowStep(
            name="health_check",
            description="Verify system health during browsing",
            endpoint="/health"
        ),
        WorkflowStep(
            name="get_system_status",
            description="Check overall system status",
            endpoint="/status"
        )
    ]
    
    result2 = await client.execute_workflow("Standards Explorer", explorer_workflow)
    all_results.append(result2)
    
    # Workflow 3: System Administrator
    admin_workflow = [
        WorkflowStep(
            name="system_health_check",
            description="Comprehensive system health verification",
            endpoint="/health"
        ),
        WorkflowStep(
            name="liveness_probe",
            description="Kubernetes-style liveness probe",
            endpoint="/health/live"
        ),
        WorkflowStep(
            name="readiness_probe", 
            description="Kubernetes-style readiness probe",
            endpoint="/health/ready"
        ),
        WorkflowStep(
            name="performance_metrics",
            description="Collect performance and operational metrics",
            endpoint="/metrics"
        ),
        WorkflowStep(
            name="service_status",
            description="Get detailed service status information",
            endpoint="/status"
        )
    ]
    
    result3 = await client.execute_workflow("System Administrator", admin_workflow)
    all_results.append(result3)
    
    # Workflow 4: API Integration Test
    integration_workflow = [
        WorkflowStep(
            name="root_endpoint",
            description="Test root API endpoint discovery",
            endpoint="/"
        ),
        WorkflowStep(
            name="standards_api",
            description="Test main standards API endpoint",
            endpoint="/api/standards"
        ),
        WorkflowStep(
            name="info_endpoint",
            description="Test service information endpoint",
            endpoint="/info"
        )
    ]
    
    result4 = await client.execute_workflow("API Integration Test", integration_workflow)
    all_results.append(result4)
    
    # Generate comprehensive summary
    print(f"\n" + "=" * 80)
    print("📊 END-TO-END WORKFLOW TEST SUMMARY")
    print("=" * 80)
    
    total_workflows = len(all_results)
    successful_workflows = sum(1 for r in all_results if r.success)
    total_steps = sum(r.total_steps for r in all_results)
    completed_steps = sum(r.steps_completed for r in all_results)
    total_time = sum(r.execution_time for r in all_results)
    
    print(f"\n🎯 Overall Results:")
    print(f"   Workflows: {successful_workflows}/{total_workflows} successful ({(successful_workflows/total_workflows)*100:.1f}%)")
    print(f"   Steps: {completed_steps}/{total_steps} completed ({(completed_steps/total_steps)*100:.1f}%)")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Average Time per Workflow: {total_time/total_workflows:.2f}s")
    
    print(f"\n📋 Workflow Details:")
    for result in all_results:
        status_icon = "✅" if result.success else "❌"
        success_rate = (result.steps_completed / result.total_steps) * 100
        print(f"   {status_icon} {result.workflow_name}: {result.steps_completed}/{result.total_steps} steps ({success_rate:.0f}%) in {result.execution_time:.2f}s")
    
    # Analyze step performance
    all_step_results = []
    for result in all_results:
        all_step_results.extend(result.step_results)
    
    successful_steps = [s for s in all_step_results if s["success"]]
    failed_steps = [s for s in all_step_results if not s["success"]]
    
    if successful_steps:
        avg_response_time = sum(s["execution_time_ms"] for s in successful_steps) / len(successful_steps)
        max_response_time = max(s["execution_time_ms"] for s in successful_steps)
        min_response_time = min(s["execution_time_ms"] for s in successful_steps)
        
        print(f"\n⚡ Performance Analysis:")
        print(f"   Successful Steps: {len(successful_steps)}")
        print(f"   Average Response Time: {avg_response_time:.1f}ms")
        print(f"   Fastest Response: {min_response_time:.1f}ms")
        print(f"   Slowest Response: {max_response_time:.1f}ms")
    
    if failed_steps:
        print(f"\n❌ Failed Steps Analysis:")
        print(f"   Total Failures: {len(failed_steps)}")
        
        # Group failures by error type
        error_types = {}
        for step in failed_steps:
            error = step.get("error", "Unknown")
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in error_types.items():
            print(f"   {error}: {count} occurrences")
    
    # Overall assessment
    print(f"\n🏁 E2E Test Assessment:")
    
    if successful_workflows == total_workflows:
        assessment = "✅ EXCELLENT - All workflows completed successfully"
    elif successful_workflows >= total_workflows * 0.8:
        assessment = "⚠️  GOOD - Most workflows successful, minor issues detected"
    elif successful_workflows >= total_workflows * 0.5:
        assessment = "⚠️  CONCERNING - Significant workflow failures"
    else:
        assessment = "❌ CRITICAL - Major workflow failures affecting user experience"
    
    print(f"   {assessment}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if completed_steps == total_steps:
        print(f"   ✅ All workflows function correctly")
        print(f"   📈 Consider testing more complex workflows")
    elif completed_steps >= total_steps * 0.9:
        print(f"   ⚠️  Minor issues detected in some workflows")
        print(f"   🔧 Review and fix failing steps")
    else:
        print(f"   🚨 Significant workflow issues detected")
        print(f"   🛠️  Immediate attention required for failed workflows")
        print(f"   📋 Review API endpoint availability and error handling")
    
    # Save results
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_workflows": total_workflows,
            "successful_workflows": successful_workflows,
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "success_rate": (completed_steps / total_steps) * 100,
            "total_execution_time": total_time,
            "assessment": assessment
        },
        "workflows": [
            {
                "name": r.workflow_name,
                "success": r.success,
                "steps_completed": r.steps_completed,
                "total_steps": r.total_steps,
                "execution_time": r.execution_time,
                "step_results": r.step_results
            }
            for r in all_results
        ]
    }
    
    with open("simple_e2e_workflow_results.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Results saved to: simple_e2e_workflow_results.json")
    print(f"✅ End-to-end workflow testing completed!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ E2E test interrupted by user")
    except Exception as e:
        print(f"\n❌ E2E test failed: {e}")
        import traceback
        traceback.print_exc()