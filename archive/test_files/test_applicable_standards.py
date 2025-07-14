#!/usr/bin/env python3
"""
Test script for evaluating get_applicable_standards functionality.
This script tests the intelligent standard selection capabilities of the MCP server.
"""

import asyncio
import json
import time
from typing import Any

from src.core.standards.engine import StandardsEngine
from src.core.standards.models import StandardMetadata


class ApplicableStandardsTester:
    """Test harness for get_applicable_standards functionality."""

    def __init__(self):
        self.engine = None
        self.test_results = []

    async def setup(self):
        """Initialize the standards engine."""
        print("🔧 Setting up standards engine...")

        # Monkey patch StandardMetadata to handle 'author' field
        original_init = StandardMetadata.__init__
        def patched_init(self, **kwargs):
            # Convert 'author' to 'authors' if present
            if 'author' in kwargs and 'authors' not in kwargs:
                kwargs['authors'] = [kwargs['author']]
                del kwargs['author']

            # Filter out unknown fields
            known_fields = {
                'version', 'last_updated', 'authors', 'source', 'compliance_frameworks',
                'nist_controls', 'tags', 'dependencies', 'language', 'scope', 'applicability'
            }
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
            original_init(self, **filtered_kwargs)

        StandardMetadata.__init__ = patched_init

        self.engine = StandardsEngine(
            data_dir="./data/standards",
            enable_semantic_search=True,
            enable_rule_engine=True,
            enable_token_optimization=True,
            enable_caching=True
        )

        await self.engine.initialize()
        print(f"✅ Engine initialized with {len(self.engine._standards_cache)} standards")

        # Load rules from the enhanced rules file AFTER initialization
        if self.engine.rule_engine:
            from pathlib import Path
            rules_path = Path("./data/standards/meta/enhanced-selection-rules.json")
            if rules_path.exists():
                try:
                    self.engine.rule_engine.load_rules(rules_path)
                    print(f"✅ Loaded {len(self.engine.rule_engine.rules)} rules")
                except Exception as e:
                    print(f"⚠️ Error loading rules: {e}")
            else:
                print("⚠️ No rules file found")

    def define_test_cases(self) -> list[dict[str, Any]]:
        """Define the four test cases."""
        return [
            {
                "name": "React Web App",
                "project_context": {
                    "project_type": "web_application",
                    "technologies": ["react", "javascript", "npm"],
                    "requirements": ["accessibility", "security", "performance"],
                    "framework": "react",
                    "language": "javascript"
                },
                "expected_domains": ["frontend", "web", "javascript", "react"]
            },
            {
                "name": "Python API",
                "project_context": {
                    "project_type": "api",
                    "technologies": ["python", "fastapi", "postgresql"],
                    "requirements": ["security", "database", "authentication"],
                    "language": "python",
                    "framework": "fastapi"
                },
                "expected_domains": ["backend", "api", "python", "database"]
            },
            {
                "name": "Mobile IoT",
                "project_context": {
                    "project_type": "mobile_app",
                    "technologies": ["react-native", "iot", "bluetooth"],
                    "requirements": ["privacy", "iot", "edge_computing"],
                    "framework": "react-native",
                    "language": "javascript"
                },
                "expected_domains": ["mobile", "iot", "privacy", "edge"]
            },
            {
                "name": "AI/ML Project",
                "project_context": {
                    "project_type": "machine_learning",
                    "technologies": ["python", "tensorflow", "docker"],
                    "requirements": ["mlops", "ethics", "monitoring"],
                    "language": "python",
                    "domain": "ai_ml"
                },
                "expected_domains": ["ai", "ml", "mlops", "ethics", "python"]
            }
        ]

    async def test_case(self, test_case: dict[str, Any]) -> dict[str, Any]:
        """Test a single case and return results."""
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"Context: {json.dumps(test_case['project_context'], indent=2)}")

        # Measure response time
        start_time = time.time()

        try:
            # Call get_applicable_standards
            results = await self.engine.get_applicable_standards(test_case['project_context'])
            response_time = time.time() - start_time

            print(f"⏱️ Response time: {response_time:.3f}s")
            print(f"📦 Found {len(results)} applicable standards")

            # Analyze results
            standards_found = []
            for result in results:
                standard = result.get('standard')
                if standard:
                    standards_found.append({
                        'id': standard.id,
                        'title': standard.title,
                        'category': standard.category,
                        'tags': list(standard.tags),
                        'confidence': result.get('confidence', 0.0),
                        'reasoning': result.get('reasoning', ''),
                        'priority': result.get('priority', 99)
                    })
                    print(f"  📋 {standard.id}: {standard.title} (confidence: {result.get('confidence', 0.0)})")

            # Calculate relevance score
            relevance_score = self._calculate_relevance(standards_found, test_case)

            # Check if requirements are addressed
            requirements_addressed = self._check_requirements_coverage(
                standards_found, test_case['project_context'].get('requirements', [])
            )

            test_result = {
                'test_case': test_case['name'],
                'context': test_case['project_context'],
                'response_time': response_time,
                'standards_count': len(standards_found),
                'standards_found': standards_found,
                'relevance_score': relevance_score,
                'requirements_addressed': requirements_addressed,
                'success': len(standards_found) > 0,
                'issues': [],
                'recommendations': []
            }

            # Add issues and recommendations
            if len(standards_found) == 0:
                test_result['issues'].append("No standards returned")
            if relevance_score < 5:
                test_result['issues'].append(f"Low relevance score: {relevance_score}/10")
            if response_time > 1.0:
                test_result['issues'].append(f"Slow response time: {response_time:.3f}s")

            return test_result

        except Exception as e:
            response_time = time.time() - start_time
            print(f"❌ Error: {e}")
            return {
                'test_case': test_case['name'],
                'context': test_case['project_context'],
                'response_time': response_time,
                'standards_count': 0,
                'standards_found': [],
                'relevance_score': 0,
                'requirements_addressed': {},
                'success': False,
                'error': str(e),
                'issues': [f"Exception occurred: {e}"],
                'recommendations': ["Fix the underlying error"]
            }

    def _calculate_relevance(self, standards: list[dict], test_case: dict) -> float:
        """Calculate relevance score (1-10) based on how well standards match the context."""
        if not standards:
            return 0.0

        total_score = 0.0
        expected_domains = set(test_case.get('expected_domains', []))

        for standard in standards:
            score = 0.0

            # Check category match
            category_words = standard['category'].lower().split()
            if any(word in expected_domains for word in category_words):
                score += 3.0

            # Check tags match
            tags = [tag.lower() for tag in standard['tags']]
            matching_tags = len(set(tags) & expected_domains)
            score += min(matching_tags * 2.0, 4.0)

            # Use confidence if available
            confidence = standard.get('confidence', 0.5)
            score *= confidence

            total_score += min(score, 10.0)

        return min(total_score / len(standards), 10.0)

    def _check_requirements_coverage(self, standards: list[dict], requirements: list[str]) -> dict:
        """Check how well the standards address the stated requirements."""
        coverage = {}

        for requirement in requirements:
            covered = False
            covering_standards = []

            for standard in standards:
                # Check if requirement appears in standard info
                standard_text = f"{standard['title']} {standard['category']} {' '.join(standard['tags'])}".lower()
                if requirement.lower() in standard_text:
                    covered = True
                    covering_standards.append(standard['id'])

            coverage[requirement] = {
                'covered': covered,
                'standards': covering_standards
            }

        return coverage

    async def run_all_tests(self):
        """Run all test cases and generate summary."""
        await self.setup()

        test_cases = self.define_test_cases()

        print("🚀 Starting applicable standards tests...")
        print("=" * 60)

        for test_case in test_cases:
            result = await self.test_case(test_case)
            self.test_results.append(result)

        # Generate summary
        self.generate_summary()

    def generate_summary(self):
        """Generate test summary and recommendations."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        successful_tests = [r for r in self.test_results if r['success']]
        avg_response_time = sum(r['response_time'] for r in self.test_results) / len(self.test_results)
        avg_relevance = sum(r['relevance_score'] for r in successful_tests) / len(successful_tests) if successful_tests else 0

        print(f"✅ Successful tests: {len(successful_tests)}/{len(self.test_results)}")
        print(f"⏱️ Average response time: {avg_response_time:.3f}s")
        print(f"🎯 Average relevance score: {avg_relevance:.1f}/10")

        # Individual test results
        print("\n📋 Individual Test Results:")
        for result in self.test_results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['test_case']}: {result['standards_count']} standards, relevance {result['relevance_score']:.1f}/10")

            if result['issues']:
                for issue in result['issues']:
                    print(f"    ⚠️ {issue}")

        # Requirements coverage analysis
        print("\n🎯 Requirements Coverage Analysis:")
        all_requirements = set()
        covered_requirements = set()

        for result in successful_tests:
            for req, coverage in result['requirements_addressed'].items():
                all_requirements.add(req)
                if coverage['covered']:
                    covered_requirements.add(req)

        if all_requirements:
            coverage_percentage = len(covered_requirements) / len(all_requirements) * 100
            print(f"Overall coverage: {coverage_percentage:.1f}% ({len(covered_requirements)}/{len(all_requirements)})")

            uncovered = all_requirements - covered_requirements
            if uncovered:
                print(f"Uncovered requirements: {', '.join(uncovered)}")

        # Overall recommendations
        print("\n💡 Recommendations:")

        if avg_response_time > 0.5:
            print("  • Optimize response time (target < 500ms)")

        if avg_relevance < 7:
            print("  • Improve standard relevance matching")
            print("  • Consider adding more specific rules")

        if len(successful_tests) < len(self.test_results):
            print("  • Fix errors preventing standard selection")

        # Look for patterns in issues
        all_issues = []
        for result in self.test_results:
            all_issues.extend(result['issues'])

        if "No standards returned" in str(all_issues):
            print("  • Investigate why some contexts return no standards")

        print("\n🔧 Technical Analysis:")
        print(f"  • Rule engine loaded: {self.engine.rule_engine is not None}")
        print(f"  • Semantic search enabled: {self.engine.semantic_search is not None}")
        print(f"  • Standards in cache: {len(self.engine._standards_cache)}")

        if self.engine.rule_engine:
            print(f"  • Rules loaded: {len(self.engine.rule_engine.rules)}")

        # Show sample of available standards
        print("\n📚 Sample of Available Standards:")
        for i, (std_id, standard) in enumerate(list(self.engine._standards_cache.items())[:10]):
            print(f"  {i+1}. {std_id}: {standard.title} [{standard.category}]")
        if len(self.engine._standards_cache) > 10:
            print(f"  ... and {len(self.engine._standards_cache) - 10} more")

        # Show rules that were evaluated
        print("\n⚙️ Available Rules:")
        if self.engine.rule_engine:
            for rule in self.engine.rule_engine.rules:
                print(f"  • {rule.name} (priority {rule.priority}) -> {rule.standards}")

        # Test rule engine directly with sample context
        print("\n🔍 Rule Engine Debug - Sample Evaluation:")
        if self.engine.rule_engine:
            sample_context = {
                "project_type": "web_application",
                "framework": "react",
                "language": "javascript"
            }
            evaluation = self.engine.rule_engine.evaluate(sample_context)
            print(f"  Sample context: {sample_context}")
            print(f"  Matched rules: {len(evaluation.get('matched_rules', []))}")
            for matched in evaluation.get('matched_rules', []):
                print(f"    - {matched.get('rule_name')} -> {matched.get('standards')}")


async def main():
    """Main test execution."""
    tester = ApplicableStandardsTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
