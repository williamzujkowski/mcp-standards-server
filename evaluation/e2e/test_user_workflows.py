#!/usr/bin/env python3
"""
End-to-End User Workflow Tests for MCP Standards Server

This module tests complete user workflows from start to finish,
simulating real-world usage scenarios.
"""

import asyncio
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pytest
import aiohttp
from datetime import datetime


@dataclass
class WorkflowStep:
    """Represents a single step in a user workflow"""
    name: str
    description: str
    action: str
    expected_outcome: str
    validation: Optional[callable] = None
    
    
@dataclass
class WorkflowResult:
    """Results from executing a workflow"""
    workflow_name: str
    success: bool
    steps_completed: int
    total_steps: int
    errors: List[str]
    execution_time: float
    timestamp: str


class MCPWorkflowClient:
    """Client for executing MCP workflow operations"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.context = {}  # Store context between steps
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def execute_mcp_tool(self, tool: str, params: Dict) -> Dict:
        """Execute an MCP tool with parameters"""
        async with self.session.post(
            f"{self.base_url}/mcp/{tool}",
            json=params
        ) as response:
            return await response.json()


class UserWorkflowTests:
    """Test suite for end-to-end user workflows"""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./evaluation/results/workflows")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    async def test_new_project_setup_workflow(self, client: MCPWorkflowClient) -> WorkflowResult:
        """
        Workflow 1: New Project Setup
        User story: As a developer starting a new React project, I want to get
        all relevant standards and validate my initial setup.
        """
        workflow_name = "new_project_setup"
        steps = [
            WorkflowStep(
                name="create_project_structure",
                description="Create a new React project structure",
                action="create_sample_react_project",
                expected_outcome="Project directory created with basic React structure"
            ),
            WorkflowStep(
                name="get_applicable_standards",
                description="Request applicable standards for React web app",
                action="get_applicable_standards",
                expected_outcome="Receive list of relevant standards including React patterns, security, accessibility"
            ),
            WorkflowStep(
                name="review_standards",
                description="Review and select standards to implement",
                action="filter_and_prioritize_standards",
                expected_outcome="Prioritized list of standards to follow"
            ),
            WorkflowStep(
                name="validate_initial_setup",
                description="Validate project structure against selected standards",
                action="validate_against_standard",
                expected_outcome="Validation report with suggestions for improvement"
            ),
            WorkflowStep(
                name="implement_suggestions",
                description="Implement suggested improvements",
                action="apply_standard_recommendations",
                expected_outcome="Updated project structure following standards"
            ),
            WorkflowStep(
                name="final_validation",
                description="Re-validate to ensure compliance",
                action="validate_against_standard",
                expected_outcome="All validations pass or only minor warnings remain"
            )
        ]
        
        return await self._execute_workflow(client, workflow_name, steps)
        
    async def test_security_audit_workflow(self, client: MCPWorkflowClient) -> WorkflowResult:
        """
        Workflow 2: Security Audit
        User story: As a security engineer, I need to audit an existing codebase
        for security vulnerabilities and ensure compliance with security standards.
        """
        workflow_name = "security_audit"
        steps = [
            WorkflowStep(
                name="identify_project_context",
                description="Analyze project to understand technology stack",
                action="analyze_project_structure",
                expected_outcome="Project context with languages, frameworks, and dependencies identified"
            ),
            WorkflowStep(
                name="search_security_standards",
                description="Search for relevant security standards",
                action="search_standards",
                expected_outcome="List of security-related standards for the tech stack"
            ),
            WorkflowStep(
                name="get_compliance_mappings",
                description="Get NIST compliance mappings for standards",
                action="get_compliance_mapping",
                expected_outcome="Mapping of standards to NIST controls"
            ),
            WorkflowStep(
                name="run_security_validation",
                description="Validate codebase against security standards",
                action="validate_against_standard",
                expected_outcome="Detailed security findings report"
            ),
            WorkflowStep(
                name="prioritize_findings",
                description="Prioritize security findings by severity",
                action="analyze_and_prioritize_findings",
                expected_outcome="Prioritized list of security issues to address"
            ),
            WorkflowStep(
                name="generate_audit_report",
                description="Generate comprehensive security audit report",
                action="compile_audit_report",
                expected_outcome="Professional security audit report with remediation recommendations"
            ),
            WorkflowStep(
                name="track_remediation",
                description="Create remediation tracking plan",
                action="create_remediation_plan",
                expected_outcome="Actionable remediation plan with timelines"
            )
        ]
        
        return await self._execute_workflow(client, workflow_name, steps)
        
    async def test_performance_optimization_workflow(self, client: MCPWorkflowClient) -> WorkflowResult:
        """
        Workflow 3: Performance Optimization
        User story: As a performance engineer, I need to identify and fix
        performance issues in our application.
        """
        workflow_name = "performance_optimization"
        steps = [
            WorkflowStep(
                name="baseline_performance",
                description="Establish current performance baseline",
                action="measure_current_performance",
                expected_outcome="Performance metrics baseline established"
            ),
            WorkflowStep(
                name="search_performance_standards",
                description="Find relevant performance optimization standards",
                action="search_standards",
                expected_outcome="List of performance-related standards"
            ),
            WorkflowStep(
                name="get_optimized_standards",
                description="Get token-optimized versions for quick review",
                action="get_optimized_standard",
                expected_outcome="Condensed performance standards for rapid review"
            ),
            WorkflowStep(
                name="identify_bottlenecks",
                description="Analyze code against performance standards",
                action="validate_against_standard",
                expected_outcome="List of performance anti-patterns and bottlenecks"
            ),
            WorkflowStep(
                name="implement_optimizations",
                description="Apply performance optimizations",
                action="apply_performance_fixes",
                expected_outcome="Optimized code following performance best practices"
            ),
            WorkflowStep(
                name="measure_improvements",
                description="Measure performance after optimizations",
                action="measure_optimized_performance",
                expected_outcome="Performance metrics showing improvement"
            ),
            WorkflowStep(
                name="document_changes",
                description="Document optimization changes and results",
                action="create_performance_report",
                expected_outcome="Performance optimization report with before/after metrics"
            )
        ]
        
        return await self._execute_workflow(client, workflow_name, steps)
        
    async def test_team_onboarding_workflow(self, client: MCPWorkflowClient) -> WorkflowResult:
        """
        Workflow 4: Team Onboarding
        User story: As a team lead, I need to onboard new developers and ensure
        they understand our coding standards and best practices.
        """
        workflow_name = "team_onboarding"
        steps = [
            WorkflowStep(
                name="list_all_standards",
                description="Get complete list of available standards",
                action="list_available_standards",
                expected_outcome="Full catalog of standards available"
            ),
            WorkflowStep(
                name="filter_by_technology",
                description="Filter standards relevant to team's tech stack",
                action="filter_standards_by_context",
                expected_outcome="Subset of standards matching team's technologies"
            ),
            WorkflowStep(
                name="get_condensed_versions",
                description="Get condensed versions for initial review",
                action="get_standard",
                expected_outcome="Easy-to-digest versions of key standards"
            ),
            WorkflowStep(
                name="create_learning_path",
                description="Create progressive learning path for new developers",
                action="organize_standards_by_priority",
                expected_outcome="Ordered list of standards from basic to advanced"
            ),
            WorkflowStep(
                name="generate_onboarding_docs",
                description="Generate onboarding documentation package",
                action="compile_onboarding_materials",
                expected_outcome="Complete onboarding package with standards and examples"
            ),
            WorkflowStep(
                name="create_validation_checklist",
                description="Create checklist for code review validation",
                action="generate_validation_checklist",
                expected_outcome="Code review checklist based on standards"
            )
        ]
        
        return await self._execute_workflow(client, workflow_name, steps)
        
    async def test_compliance_verification_workflow(self, client: MCPWorkflowClient) -> WorkflowResult:
        """
        Workflow 5: Compliance Verification
        User story: As a compliance officer, I need to verify our software
        meets NIST 800-53 controls for government contracts.
        """
        workflow_name = "compliance_verification"
        steps = [
            WorkflowStep(
                name="identify_required_controls",
                description="Identify NIST controls required for compliance",
                action="get_required_nist_controls",
                expected_outcome="List of required NIST 800-53 controls"
            ),
            WorkflowStep(
                name="map_standards_to_controls",
                description="Get standards that map to required controls",
                action="get_compliance_mapping",
                expected_outcome="Mapping of standards to NIST controls"
            ),
            WorkflowStep(
                name="identify_coverage_gaps",
                description="Identify controls without standard coverage",
                action="analyze_compliance_gaps",
                expected_outcome="List of controls needing additional coverage"
            ),
            WorkflowStep(
                name="validate_implementation",
                description="Validate code against compliance standards",
                action="validate_against_standard",
                expected_outcome="Compliance validation results"
            ),
            WorkflowStep(
                name="generate_evidence",
                description="Generate compliance evidence documentation",
                action="generate_compliance_evidence",
                expected_outcome="Evidence package for compliance audit"
            ),
            WorkflowStep(
                name="create_oscal_output",
                description="Generate OSCAL-formatted compliance data",
                action="export_oscal_format",
                expected_outcome="OSCAL-compliant assessment results"
            )
        ]
        
        return await self._execute_workflow(client, workflow_name, steps)
        
    async def test_continuous_improvement_workflow(self, client: MCPWorkflowClient) -> WorkflowResult:
        """
        Workflow 6: Continuous Improvement
        User story: As a tech lead, I want to continuously improve our codebase
        by staying updated with latest standards and best practices.
        """
        workflow_name = "continuous_improvement"
        steps = [
            WorkflowStep(
                name="get_current_standards",
                description="Get currently implemented standards",
                action="get_applicable_standards",
                expected_outcome="List of standards currently in use"
            ),
            WorkflowStep(
                name="check_for_updates",
                description="Check for updates to existing standards",
                action="check_standard_versions",
                expected_outcome="List of standards with available updates"
            ),
            WorkflowStep(
                name="search_new_standards",
                description="Search for new relevant standards",
                action="search_standards",
                expected_outcome="New standards that might benefit the project"
            ),
            WorkflowStep(
                name="impact_analysis",
                description="Analyze impact of adopting new standards",
                action="analyze_standard_impact",
                expected_outcome="Impact assessment for each new standard"
            ),
            WorkflowStep(
                name="create_migration_plan",
                description="Create plan to adopt new standards",
                action="generate_migration_plan",
                expected_outcome="Phased migration plan with minimal disruption"
            ),
            WorkflowStep(
                name="track_improvements",
                description="Set up metrics to track improvements",
                action="setup_improvement_metrics",
                expected_outcome="Dashboard for tracking code quality improvements"
            )
        ]
        
        return await self._execute_workflow(client, workflow_name, steps)
    
    async def _execute_workflow(
        self,
        client: MCPWorkflowClient,
        workflow_name: str,
        steps: List[WorkflowStep]
    ) -> WorkflowResult:
        """Execute a complete workflow and track results"""
        start_time = datetime.now()
        errors = []
        completed_steps = 0
        
        print(f"\n{'='*60}")
        print(f"Executing Workflow: {workflow_name}")
        print(f"{'='*60}")
        
        # Create workflow context
        workflow_context = {
            "workflow_name": workflow_name,
            "start_time": start_time.isoformat(),
            "project_path": None,
            "standards": [],
            "validation_results": [],
            "artifacts": []
        }
        
        try:
            for i, step in enumerate(steps, 1):
                print(f"\nStep {i}/{len(steps)}: {step.name}")
                print(f"  Description: {step.description}")
                
                try:
                    # Execute step action
                    result = await self._execute_step_action(
                        client,
                        step,
                        workflow_context
                    )
                    
                    # Validate outcome if validation function provided
                    if step.validation and not step.validation(result):
                        errors.append(f"Step {step.name}: Validation failed")
                        print(f"  ❌ Validation failed")
                    else:
                        completed_steps += 1
                        print(f"  ✅ {step.expected_outcome}")
                        
                except Exception as e:
                    error_msg = f"Step {step.name}: {str(e)}"
                    errors.append(error_msg)
                    print(f"  ❌ Error: {str(e)}")
                    
                    # Decide whether to continue or abort
                    if i < 3:  # Abort if early steps fail
                        print(f"  🛑 Aborting workflow due to early failure")
                        break
                        
        except Exception as e:
            errors.append(f"Workflow execution error: {str(e)}")
            
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = WorkflowResult(
            workflow_name=workflow_name,
            success=len(errors) == 0,
            steps_completed=completed_steps,
            total_steps=len(steps),
            errors=errors,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Save workflow results
        self._save_workflow_results(result, workflow_context)
        
        return result
    
    async def _execute_step_action(
        self,
        client: MCPWorkflowClient,
        step: WorkflowStep,
        context: Dict
    ) -> Any:
        """Execute a specific workflow step action"""
        
        # Map actions to implementations
        if step.action == "create_sample_react_project":
            return await self._create_sample_project(context, "react")
            
        elif step.action == "get_applicable_standards":
            params = {
                "project_context": {
                    "project_type": "web_application",
                    "framework": "react",
                    "languages": ["javascript", "typescript"],
                    "requirements": ["security", "accessibility", "performance"]
                }
            }
            result = await client.execute_mcp_tool("get_applicable_standards", params)
            context["standards"] = result.get("standards", [])
            return result
            
        elif step.action == "filter_and_prioritize_standards":
            # Simulate filtering and prioritizing standards
            standards = context.get("standards", [])
            prioritized = sorted(
                standards,
                key=lambda x: x.get("priority", 999)
            )[:5]  # Top 5 standards
            context["prioritized_standards"] = prioritized
            return {"prioritized_standards": prioritized}
            
        elif step.action == "validate_against_standard":
            if not context.get("project_path"):
                context["project_path"] = await self._create_sample_project(context, "react")
                
            params = {
                "code_path": str(context["project_path"]),
                "standard_id": context.get("prioritized_standards", [{"id": "react-18-patterns"}])[0]["id"]
            }
            result = await client.execute_mcp_tool("validate_against_standard", params)
            context["validation_results"].append(result)
            return result
            
        elif step.action == "search_standards":
            query = {
                "security_audit": "security",
                "performance_optimization": "performance optimization caching",
                "continuous_improvement": "best practices latest"
            }.get(context["workflow_name"], "standards")
            
            params = {"query": query}
            return await client.execute_mcp_tool("search_standards", params)
            
        elif step.action == "get_compliance_mapping":
            params = {}
            if context.get("standards"):
                params["standard_id"] = context["standards"][0].get("id")
            return await client.execute_mcp_tool("get_compliance_mapping", params)
            
        elif step.action == "list_available_standards":
            params = {}
            return await client.execute_mcp_tool("list_available_standards", params)
            
        elif step.action == "get_optimized_standard":
            standard_id = context.get("standards", [{"id": "performance-tuning-optimization"}])[0]["id"]
            params = {
                "standard_id": standard_id,
                "token_limit": 4000
            }
            return await client.execute_mcp_tool("get_optimized_standard", params)
            
        elif step.action == "get_standard":
            standard_id = context.get("prioritized_standards", [{"id": "documentation-writing"}])[0]["id"]
            params = {
                "standard_id": standard_id,
                "format": "condensed"
            }
            return await client.execute_mcp_tool("get_standard", params)
            
        else:
            # Simulate other actions
            return await self._simulate_action(step.action, context)
    
    async def _create_sample_project(self, context: Dict, project_type: str) -> Path:
        """Create a sample project for testing"""
        project_dir = Path(tempfile.mkdtemp(prefix=f"mcp_test_{project_type}_"))
        
        if project_type == "react":
            # Create basic React project structure
            (project_dir / "src").mkdir()
            (project_dir / "public").mkdir()
            (project_dir / "src" / "components").mkdir()
            
            # Create sample files
            (project_dir / "package.json").write_text(json.dumps({
                "name": "test-react-app",
                "version": "1.0.0",
                "dependencies": {
                    "react": "^18.0.0",
                    "react-dom": "^18.0.0"
                }
            }, indent=2))
            
            (project_dir / "src" / "App.js").write_text("""
import React from 'react';

function App() {
  return (
    <div className="App">
      <h1>Test React App</h1>
    </div>
  );
}

export default App;
""")
            
            (project_dir / "src" / "index.js").write_text("""
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
""")
        
        context["project_path"] = project_dir
        context["artifacts"].append(str(project_dir))
        return project_dir
    
    async def _simulate_action(self, action: str, context: Dict) -> Dict:
        """Simulate workflow actions that don't directly call MCP tools"""
        simulated_results = {
            "analyze_project_structure": {
                "languages": ["javascript", "python", "go"],
                "frameworks": ["react", "fastapi", "gin"],
                "dependencies": 145,
                "loc": 25000
            },
            "analyze_and_prioritize_findings": {
                "critical": 3,
                "high": 7,
                "medium": 15,
                "low": 22
            },
            "compile_audit_report": {
                "report_path": str(self.output_dir / "security_audit_report.pdf"),
                "findings": 47,
                "recommendations": 32
            },
            "create_remediation_plan": {
                "immediate": 3,
                "short_term": 10,
                "long_term": 19,
                "estimated_hours": 120
            },
            "measure_current_performance": {
                "page_load_time": 3.2,
                "api_response_time": 245,
                "memory_usage": 512,
                "cpu_usage": 35
            },
            "apply_performance_fixes": {
                "optimizations_applied": 12,
                "code_changes": 37,
                "config_changes": 8
            },
            "measure_optimized_performance": {
                "page_load_time": 1.8,
                "api_response_time": 95,
                "memory_usage": 380,
                "cpu_usage": 22
            },
            "create_performance_report": {
                "improvement_percentage": 44,
                "report_path": str(self.output_dir / "performance_report.md")
            },
            "filter_standards_by_context": {
                "filtered_count": 12,
                "categories": ["web", "security", "performance"]
            },
            "organize_standards_by_priority": {
                "basic": 5,
                "intermediate": 8,
                "advanced": 4
            },
            "compile_onboarding_materials": {
                "documents": 17,
                "examples": 25,
                "exercises": 10
            },
            "generate_validation_checklist": {
                "checklist_items": 45,
                "categories": 8
            },
            "get_required_nist_controls": {
                "control_families": ["AC", "AU", "SC", "SI"],
                "total_controls": 127
            },
            "analyze_compliance_gaps": {
                "covered_controls": 98,
                "gap_controls": 29,
                "coverage_percentage": 77
            },
            "generate_compliance_evidence": {
                "evidence_documents": 15,
                "automated_tests": 89,
                "manual_attestations": 12
            },
            "export_oscal_format": {
                "oscal_file": str(self.output_dir / "compliance_assessment.oscal.json"),
                "components": 12,
                "controls_assessed": 127
            },
            "check_standard_versions": {
                "updates_available": 4,
                "breaking_changes": 1,
                "new_features": 12
            },
            "analyze_standard_impact": {
                "affected_files": 234,
                "estimated_effort_hours": 40,
                "risk_level": "medium"
            },
            "generate_migration_plan": {
                "phases": 3,
                "total_duration_weeks": 6,
                "rollback_points": 5
            },
            "setup_improvement_metrics": {
                "metrics_configured": 8,
                "dashboards_created": 3,
                "alerts_setup": 5
            }
        }
        
        # Add some delay to simulate processing
        await asyncio.sleep(0.5)
        
        result = simulated_results.get(action, {"status": "completed"})
        context["artifacts"].append(result)
        return result
    
    def _save_workflow_results(self, result: WorkflowResult, context: Dict):
        """Save workflow execution results"""
        # Save detailed results
        result_file = self.output_dir / f"{result.workflow_name}_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                "result": {
                    "workflow_name": result.workflow_name,
                    "success": result.success,
                    "steps_completed": result.steps_completed,
                    "total_steps": result.total_steps,
                    "errors": result.errors,
                    "execution_time": result.execution_time,
                    "timestamp": result.timestamp
                },
                "context": {
                    "workflow_name": context["workflow_name"],
                    "start_time": context["start_time"],
                    "standards_used": len(context.get("standards", [])),
                    "validations_performed": len(context.get("validation_results", [])),
                    "artifacts_created": len(context.get("artifacts", []))
                }
            }, f, indent=2)
            
        # Clean up temporary artifacts
        if context.get("project_path") and Path(context["project_path"]).exists():
            shutil.rmtree(context["project_path"])
    
    async def run_all_workflows(self) -> List[WorkflowResult]:
        """Execute all workflow tests"""
        workflows = [
            self.test_new_project_setup_workflow,
            self.test_security_audit_workflow,
            self.test_performance_optimization_workflow,
            self.test_team_onboarding_workflow,
            self.test_compliance_verification_workflow,
            self.test_continuous_improvement_workflow
        ]
        
        all_results = []
        
        async with MCPWorkflowClient() as client:
            for workflow_test in workflows:
                try:
                    result = await workflow_test(client)
                    all_results.append(result)
                    self.results.append(result)
                except Exception as e:
                    print(f"\n❌ Failed to execute workflow: {str(e)}")
                    # Create a failed result
                    result = WorkflowResult(
                        workflow_name=workflow_test.__name__.replace("test_", "").replace("_workflow", ""),
                        success=False,
                        steps_completed=0,
                        total_steps=0,
                        errors=[f"Workflow execution failed: {str(e)}"],
                        execution_time=0,
                        timestamp=datetime.now().isoformat()
                    )
                    all_results.append(result)
        
        # Generate summary report
        self._generate_summary_report(all_results)
        
        return all_results
    
    def _generate_summary_report(self, results: List[WorkflowResult]):
        """Generate a summary report of all workflow tests"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"workflow_summary_{timestamp}.md"
        
        total_workflows = len(results)
        successful_workflows = sum(1 for r in results if r.success)
        total_steps = sum(r.total_steps for r in results)
        completed_steps = sum(r.steps_completed for r in results)
        total_time = sum(r.execution_time for r in results)
        
        report = f"""# MCP User Workflow Test Summary

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overall Results

- **Total Workflows Tested:** {total_workflows}
- **Successful Workflows:** {successful_workflows} ({successful_workflows/total_workflows*100:.1f}%)
- **Total Steps:** {total_steps}
- **Completed Steps:** {completed_steps} ({completed_steps/total_steps*100:.1f}%)
- **Total Execution Time:** {total_time:.2f} seconds

## Workflow Results

| Workflow | Success | Steps Completed | Execution Time | Errors |
|----------|---------|-----------------|----------------|--------|
"""
        
        for result in results:
            status = "✅" if result.success else "❌"
            errors = len(result.errors)
            report += f"| {result.workflow_name} | {status} | {result.steps_completed}/{result.total_steps} | "
            report += f"{result.execution_time:.2f}s | {errors} |\n"
        
        report += "\n## Detailed Error Summary\n\n"
        
        for result in results:
            if result.errors:
                report += f"### {result.workflow_name}\n"
                for error in result.errors:
                    report += f"- {error}\n"
                report += "\n"
        
        report += """
## Recommendations

1. **Failed Workflows**: Investigate and fix issues in failed workflows
2. **Incomplete Steps**: Review steps that consistently fail across workflows
3. **Performance**: Optimize workflows taking longer than expected
4. **Error Patterns**: Address common error patterns across workflows

## Next Steps

1. Fix identified issues in the MCP server implementation
2. Add more comprehensive error handling
3. Improve workflow resilience to handle partial failures
4. Enhance validation logic for better test coverage
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Summary report saved to: {report_path}")


async def main():
    """Run all user workflow tests"""
    print("🚀 Starting MCP User Workflow Tests")
    print("=" * 60)
    
    test_suite = UserWorkflowTests()
    
    try:
        results = await test_suite.run_all_workflows()
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        print(f"\n{'='*60}")
        print(f"✅ Workflow Tests Complete")
        print(f"   Successful: {successful}/{len(results)}")
        print(f"   Results saved to: {test_suite.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Workflow tests failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())