"""
Benchmark script for token optimization.

Demonstrates token savings across different formats and scenarios.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Optional matplotlib imports - gracefully handle missing dependency
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Create dummy classes/functions for when matplotlib is not available
    plt = None



from src.core.standards.token_optimizer import (
    ModelType,
    StandardFormat,
    TokenBudget,
    create_token_optimizer,
)


class TokenOptimizationBenchmark:
    """Benchmark suite for token optimization."""

    def __init__(self):
        self.optimizers = {
            'gpt4': create_token_optimizer(ModelType.GPT4),
            'gpt35': create_token_optimizer(ModelType.GPT35_TURBO),
            'claude': create_token_optimizer(ModelType.CLAUDE),
        }
        self.results = {}

    def generate_test_standards(self) -> list[dict[str, Any]]:
        """Generate test standards of varying sizes."""
        standards = []

        # Small standard (< 1000 tokens)
        small_content = """# Small Standard

## Overview
This is a small standard for testing token optimization.

## Requirements
- Follow best practices
- Implement security measures
- Write tests

## Implementation
Basic implementation guidelines.

## Examples
```python
def example():
    return "Hello"
```
"""
        standards.append({
            'id': 'small-standard',
            'size': 'small',
            'content': small_content
        })

        # Medium standard (1000-5000 tokens)
        medium_sections = []
        for section in ['Overview', 'Requirements', 'Architecture', 'Implementation',
                       'Security', 'Testing', 'Deployment', 'Monitoring']:
            medium_sections.append(f"""## {section}

This section covers {section.lower()} aspects of the standard.

### Key Points
- Important point 1 about {section.lower()}
- Critical consideration 2 for {section.lower()}
- Best practice 3 regarding {section.lower()}
- Additional guideline 4 for {section.lower()}

### Detailed Information
{section} requires careful attention to multiple factors. First, you must consider
the implications on the overall system architecture. Second, ensure compatibility
with existing systems. Third, maintain security throughout the process.

```python
# Example for {section.lower()}
def handle_{section.lower()}():
    # Implementation details
    config = load_config()
    validate_input()
    process_data()
    return result
```

### Common Pitfalls
1. Not considering edge cases
2. Insufficient error handling
3. Poor documentation
4. Lack of testing

### Best Practices
Always follow established patterns and conventions. Document your decisions
and rationale. Test thoroughly before deployment.
""")

        standards.append({
            'id': 'medium-standard',
            'size': 'medium',
            'content': '\n\n'.join(medium_sections)
        })

        # Large standard (> 5000 tokens)
        large_sections = []
        for i in range(15):
            large_sections.append(f"""## Section {i}: Comprehensive Guidelines

### Introduction to Section {i}
This section provides extensive documentation about topic {i}. It includes
detailed explanations, multiple examples, and comprehensive guidelines that
cover all aspects of implementation and usage.

### Background and Context
Understanding the historical context and evolution of these practices is
crucial for proper implementation. Over the years, the industry has developed
numerous approaches to handling these challenges, each with its own advantages
and trade-offs.

### Detailed Requirements
1. **Requirement {i}.1**: Comprehensive description of the first requirement
   - Sub-requirement A: Detailed explanation with examples
   - Sub-requirement B: Additional context and considerations
   - Sub-requirement C: Edge cases and special scenarios

2. **Requirement {i}.2**: Second major requirement with extensive details
   - Implementation guidelines
   - Performance considerations
   - Security implications
   - Compatibility requirements

3. **Requirement {i}.3**: Third requirement focusing on best practices
   - Industry standards compliance
   - Regulatory considerations
   - Future-proofing strategies

### Implementation Guide
```python
# Comprehensive implementation example
class Section{i}Handler:
    def __init__(self, config):
        self.config = config
        self.validator = Validator()
        self.processor = Processor()

    def process(self, data):
        # Validate input
        if not self.validator.validate(data):
            raise ValidationError("Invalid input data")

        # Process data with error handling
        try:
            result = self.processor.process(data)
            self.log_success(result)
            return result
        except ProcessingError as e:
            self.handle_error(e)
            raise

    def handle_error(self, error):
        # Comprehensive error handling
        logger.error(f"Processing failed: {{error}}")
        self.notify_admins(error)
        self.rollback_changes()
```

### Testing Strategy
Comprehensive testing is essential for ensuring reliability and correctness.
The testing strategy should include:

1. Unit tests for individual components
2. Integration tests for component interactions
3. End-to-end tests for complete workflows
4. Performance tests under various load conditions
5. Security tests for vulnerability assessment

### Performance Optimization
Performance considerations are crucial for scalability:
- Optimize database queries
- Implement caching strategies
- Use asynchronous processing where appropriate
- Monitor and profile regularly
- Set up alerts for performance degradation

### Security Considerations
Security must be built-in from the start:
- Input validation and sanitization
- Authentication and authorization
- Encryption of sensitive data
- Regular security audits
- Incident response procedures

### Monitoring and Observability
Proper monitoring ensures system health:
- Set up comprehensive logging
- Implement distributed tracing
- Create meaningful dashboards
- Configure alerting rules
- Establish SLOs and SLIs
""")

        standards.append({
            'id': 'large-standard',
            'size': 'large',
            'content': '\n\n'.join(large_sections)
        })

        return standards

    def benchmark_compression_formats(self, standards: list[dict[str, Any]]):
        """Benchmark different compression formats."""
        print("\n=== Compression Format Benchmark ===\n")

        results = {}

        for standard in standards:
            print(f"\nTesting {standard['id']} ({standard['size']} size)...")
            standard_results = {}

            for format_type in StandardFormat:
                if format_type == StandardFormat.CUSTOM:
                    continue  # Skip custom format

                # Use GPT-4 optimizer for this benchmark
                optimizer = self.optimizers['gpt4']

                # Measure compression
                start_time = time.time()
                content, result = optimizer.optimize_standard(
                    standard,
                    format_type=format_type,
                    budget=TokenBudget(total=8000)
                )
                duration = time.time() - start_time

                standard_results[format_type.value] = {
                    'original_tokens': result.original_tokens,
                    'compressed_tokens': result.compressed_tokens,
                    'compression_ratio': result.compression_ratio,
                    'sections_included': len(result.sections_included),
                    'sections_excluded': len(result.sections_excluded),
                    'processing_time': duration
                }

                print(f"  {format_type.value}: {result.original_tokens} -> {result.compressed_tokens} tokens "
                      f"({result.compression_ratio:.2%} ratio) in {duration:.3f}s")

            results[standard['id']] = standard_results

        self.results['compression_formats'] = results
        return results

    def benchmark_model_differences(self, standards: list[dict[str, Any]]):
        """Benchmark token counting differences between models."""
        print("\n=== Model Token Counting Benchmark ===\n")

        results = {}

        # Use medium standard for this test
        medium_standard = next(s for s in standards if s['size'] == 'medium')

        for model_name, optimizer in self.optimizers.items():
            print(f"\nTesting {model_name}...")

            token_counts = {}
            for format_type in [StandardFormat.FULL, StandardFormat.CONDENSED, StandardFormat.REFERENCE]:
                content, result = optimizer.optimize_standard(
                    medium_standard,
                    format_type=format_type
                )

                token_counts[format_type.value] = {
                    'tokens': result.compressed_tokens,
                    'compression': 1 - result.compression_ratio
                }

            results[model_name] = token_counts

            print(f"  Token counts: {json.dumps(token_counts, indent=2)}")

        self.results['model_differences'] = results
        return results

    def benchmark_budget_constraints(self, standards: list[dict[str, Any]]):
        """Benchmark behavior under different token budgets."""
        print("\n=== Token Budget Constraint Benchmark ===\n")

        results = {}
        budgets = [500, 1000, 2000, 4000, 8000, 16000]

        # Use large standard for this test
        large_standard = next(s for s in standards if s['size'] == 'large')
        optimizer = self.optimizers['gpt4']

        for budget_size in budgets:
            budget = TokenBudget(total=budget_size)

            # Let optimizer auto-select format
            selected_format = optimizer.auto_select_format(large_standard, budget)

            # Optimize with selected format
            content, result = optimizer.optimize_standard(
                large_standard,
                format_type=selected_format,
                budget=budget
            )

            results[budget_size] = {
                'selected_format': selected_format.value,
                'original_tokens': result.original_tokens,
                'compressed_tokens': result.compressed_tokens,
                'compression_ratio': result.compression_ratio,
                'sections_included': len(result.sections_included),
                'sections_excluded': len(result.sections_excluded),
                'warnings': result.warnings
            }

            print(f"\nBudget {budget_size}: Selected {selected_format.value}")
            print(f"  Compressed to {result.compressed_tokens} tokens ({result.compression_ratio:.2%})")
            print(f"  Sections: {len(result.sections_included)} included, {len(result.sections_excluded)} excluded")

        self.results['budget_constraints'] = results
        return results

    def benchmark_progressive_loading(self, standards: list[dict[str, Any]]):
        """Benchmark progressive loading efficiency."""
        print("\n=== Progressive Loading Benchmark ===\n")

        results = {}

        # Use large standard
        large_standard = next(s for s in standards if s['size'] == 'large')
        optimizer = self.optimizers['gpt4']

        # Test different initial section counts
        initial_section_counts = [1, 2, 3, 5]

        for initial_count in initial_section_counts:
            # Get all sections
            sections = optimizer._parse_standard_sections(large_standard)
            initial_sections = [s.id for s in sections[:initial_count]]

            # Generate loading plan
            loading_plan = optimizer.progressive_load(
                large_standard,
                initial_sections=initial_sections,
                max_depth=3
            )

            # Calculate cumulative tokens
            cumulative_tokens = []
            total = 0

            for batch in loading_plan:
                batch_tokens = sum(tokens for _, tokens in batch)
                total += batch_tokens
                cumulative_tokens.append(total)

            results[f'initial_{initial_count}'] = {
                'initial_sections': initial_sections,
                'total_batches': len(loading_plan),
                'total_sections': sum(len(batch) for batch in loading_plan),
                'cumulative_tokens': cumulative_tokens,
                'final_tokens': total
            }

            print(f"\nStarting with {initial_count} sections:")
            print(f"  Total batches: {len(loading_plan)}")
            print(f"  Total sections loaded: {sum(len(batch) for batch in loading_plan)}")
            print(f"  Final token count: {total}")

        self.results['progressive_loading'] = results
        return results

    def benchmark_compression_techniques(self, text: str):
        """Benchmark individual compression techniques."""
        print("\n=== Compression Techniques Benchmark ===\n")

        from src.core.standards.token_optimizer import CompressionTechniques

        techniques = CompressionTechniques()
        optimizer = self.optimizers['gpt4']

        original_tokens = optimizer.token_counter.count_tokens(text)
        print(f"Original text: {original_tokens} tokens\n")

        results = {
            'original': {
                'tokens': original_tokens,
                'length': len(text)
            }
        }

        # Test each technique
        techniques_to_test = [
            ('remove_redundancy', techniques.remove_redundancy),
            ('use_abbreviations', techniques.use_abbreviations),
            ('compress_code_examples', techniques.compress_code_examples),
            ('extract_essential_only', techniques.extract_essential_only),
        ]

        for name, technique in techniques_to_test:
            compressed = technique(text)
            tokens = optimizer.token_counter.count_tokens(compressed)

            results[name] = {
                'tokens': tokens,
                'length': len(compressed),
                'reduction': original_tokens - tokens,
                'reduction_percent': ((original_tokens - tokens) / original_tokens * 100) if original_tokens > 0 else 0
            }

            print(f"{name}:")
            print(f"  Tokens: {tokens} (reduced by {results[name]['reduction']} / {results[name]['reduction_percent']:.1f}%)")
            print(f"  Length: {len(compressed)} chars (from {len(text)} chars)\n")

        # Test combined techniques
        combined = text
        for _, technique in techniques_to_test:
            combined = technique(combined)

        combined_tokens = optimizer.token_counter.count_tokens(combined)
        results['combined_all'] = {
            'tokens': combined_tokens,
            'length': len(combined),
            'reduction': original_tokens - combined_tokens,
            'reduction_percent': ((original_tokens - combined_tokens) / original_tokens * 100) if original_tokens > 0 else 0
        }

        print("Combined all techniques:")
        print(f"  Tokens: {combined_tokens} (reduced by {results['combined_all']['reduction']} / {results['combined_all']['reduction_percent']:.1f}%)")

        self.results['compression_techniques'] = results
        return results

    def generate_report(self):
        """Generate comprehensive benchmark report."""
        print("\n\n=== BENCHMARK REPORT ===\n")

        # Compression format summary
        if 'compression_formats' in self.results:
            print("## Compression Format Performance\n")
            for standard_id, formats in self.results['compression_formats'].items():
                print(f"### {standard_id}")
                print("| Format | Original | Compressed | Ratio | Time |")
                print("|--------|----------|------------|-------|------|")
                for format_name, data in formats.items():
                    print(f"| {format_name} | {data['original_tokens']} | {data['compressed_tokens']} | "
                          f"{data['compression_ratio']:.2%} | {data['processing_time']:.3f}s |")
                print()

        # Budget constraint summary
        if 'budget_constraints' in self.results:
            print("\n## Token Budget Auto-Selection\n")
            print("| Budget | Format Selected | Compression | Sections |")
            print("|--------|----------------|-------------|----------|")
            for budget, data in self.results['budget_constraints'].items():
                print(f"| {budget} | {data['selected_format']} | "
                      f"{data['compression_ratio']:.2%} | "
                      f"{data['sections_included']}/{data['sections_included'] + data['sections_excluded']} |")

        # Save results to file
        output_path = Path('benchmarks/token_optimization_results.json')
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n\nDetailed results saved to: {output_path}")

    def plot_results(self):
        """Generate visualization plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available, skipping plots")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Token Optimization Benchmark Results')

        # Plot 1: Compression ratios by format
        if 'compression_formats' in self.results:
            ax = axes[0, 0]
            formats = list(StandardFormat)
            formats = [f for f in formats if f != StandardFormat.CUSTOM]

            for standard_id, data in self.results['compression_formats'].items():
                ratios = [data.get(f.value, {}).get('compression_ratio', 0) for f in formats]
                ax.plot([f.value for f in formats], ratios, marker='o', label=standard_id)

            ax.set_xlabel('Format')
            ax.set_ylabel('Compression Ratio')
            ax.set_title('Compression Ratio by Format')
            ax.legend()
            ax.grid(True)

        # Plot 2: Token usage by budget
        if 'budget_constraints' in self.results:
            ax = axes[0, 1]
            budgets = list(self.results['budget_constraints'].keys())
            tokens_used = [data['compressed_tokens'] for data in self.results['budget_constraints'].values()]

            ax.plot(budgets, tokens_used, marker='s', color='blue', label='Tokens Used')
            ax.plot(budgets, budgets, '--', color='red', label='Budget Limit')

            ax.set_xlabel('Token Budget')
            ax.set_ylabel('Tokens')
            ax.set_title('Token Usage vs Budget')
            ax.legend()
            ax.grid(True)

        # Plot 3: Progressive loading
        if 'progressive_loading' in self.results:
            ax = axes[1, 0]

            for config, data in self.results['progressive_loading'].items():
                cumulative = data['cumulative_tokens']
                ax.plot(range(1, len(cumulative) + 1), cumulative, marker='o', label=config)

            ax.set_xlabel('Loading Batch')
            ax.set_ylabel('Cumulative Tokens')
            ax.set_title('Progressive Loading Token Accumulation')
            ax.legend()
            ax.grid(True)

        # Plot 4: Compression technique effectiveness
        if 'compression_techniques' in self.results:
            ax = axes[1, 1]
            techniques = [k for k in self.results['compression_techniques'].keys() if k != 'original']
            reductions = [self.results['compression_techniques'][k]['reduction_percent']
                         for k in techniques]

            ax.bar(techniques, reductions, color='green')
            ax.set_xlabel('Technique')
            ax.set_ylabel('Token Reduction %')
            ax.set_title('Compression Technique Effectiveness')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, axis='y')

        plt.tight_layout()

        # Save plot
        plot_path = Path('benchmarks/token_optimization_plots.png')
        plt.savefig(plot_path)
        print(f"\nPlots saved to: {plot_path}")

        plt.close()


def main():
    """Run the benchmark suite."""
    print("Token Optimization Benchmark Suite")
    print("==================================")

    benchmark = TokenOptimizationBenchmark()

    # Generate test standards
    standards = benchmark.generate_test_standards()

    # Run benchmarks
    benchmark.benchmark_compression_formats(standards)
    benchmark.benchmark_model_differences(standards)
    benchmark.benchmark_budget_constraints(standards)
    benchmark.benchmark_progressive_loading(standards)

    # Test compression techniques on sample text
    sample_text = standards[1]['content']  # Use medium standard
    benchmark.benchmark_compression_techniques(sample_text)

    # Generate report
    benchmark.generate_report()

    # Generate plots
    benchmark.plot_results()

    print("\n\nBenchmark complete!")


if __name__ == '__main__':
    main()
