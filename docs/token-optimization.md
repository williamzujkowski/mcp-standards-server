# Token Optimization Guide

## Overview

The MCP Standards Server includes a comprehensive token optimization system designed to significantly reduce token usage while maintaining information quality. This feature is crucial for:

- Working within LLM context limits
- Reducing API costs
- Improving response times
- Enabling more standards to be loaded simultaneously

## Key Features

### 1. Multiple Format Variants

The system provides four predefined formats, each optimized for different use cases:

#### Full Format
- **Token Usage**: ~90% of original
- **Use Case**: When complete documentation with examples is needed
- **Features**: Minimal compression, preserves all content

#### Condensed Format
- **Token Usage**: ~50% of original
- **Use Case**: Standard development workflow
- **Features**: 
  - Removes redundancy
  - Uses abbreviations
  - Compresses code examples
  - Extracts essential information

#### Reference Format
- **Token Usage**: ~20% of original
- **Use Case**: Quick lookup and reference
- **Features**:
  - Headers and key points only
  - Bullet points and numbered lists
  - Truncated to fit budget

#### Summary Format
- **Token Usage**: ~5% of original
- **Use Case**: Executive overview
- **Features**:
  - One-paragraph summary
  - Most important sections only
  - Metadata about coverage

### 2. Dynamic Loading

Progressive disclosure system that loads content based on query depth:

```python
# Example: Progressive loading plan
loading_plan = optimizer.progressive_load(
    standard,
    initial_sections=['overview', 'requirements'],
    max_depth=3
)
```

Features:
- Dependency-aware loading
- Priority-based section selection
- Batch loading for efficiency
- Context-aware suggestions

### 3. Token Counting and Budgeting

Accurate token counting for different models:

- **GPT-4**: Uses tiktoken for exact counts
- **GPT-3.5**: Uses tiktoken with appropriate encoding
- **Claude**: Approximation based on character count
- **Custom**: Configurable counting method

Budget management:
```python
budget = TokenBudget(
    total=8000,
    reserved_for_context=1000,
    reserved_for_response=2000,
    warning_threshold=0.8
)
```

### 4. Compression Techniques

Multiple compression strategies applied intelligently:

1. **Redundancy Removal**
   - Multiple spaces → single space
   - Excessive newlines → normalized
   - Trailing whitespace → removed

2. **Abbreviations**
   - Common terms replaced with standard abbreviations
   - Domain-specific abbreviations
   - Configurable abbreviation dictionary

3. **Code Compression**
   - Remove comments from examples
   - Minimize indentation
   - Remove empty lines
   - Preserve language identifiers

4. **Essential Extraction**
   - Identifies critical keywords
   - Preserves security warnings
   - Keeps best practices
   - Maintains requirements

5. **Lookup Tables**
   - Creates references for repeated phrases
   - Replaces with short codes
   - Provides lookup dictionary

## API Usage

### Get Optimized Standard

```python
result = await mcp.get_optimized_standard(
    standard_id="react-18-patterns",
    format_type="condensed",
    token_budget=2000,
    required_sections=["security", "testing"],
    context={
        "user_expertise": "intermediate",
        "focus_areas": ["performance", "security"]
    }
)
```

### Auto-Optimize Multiple Standards

```python
result = await mcp.auto_optimize_standards(
    standard_ids=["react-18-patterns", "python-pep8", "rest-api-design"],
    total_token_budget=5000,
    context={
        "query_type": "implementation_guide"
    }
)
```

### Progressive Loading

```python
plan = await mcp.progressive_load_standard(
    standard_id="large-standard",
    initial_sections=["overview"],
    max_depth=3
)

# Load sections progressively based on plan
for batch in plan['loading_plan']:
    # Load batch sections as needed
    pass
```

### Estimate Token Usage

```python
estimates = await mcp.estimate_token_usage(
    standard_ids=["standard1", "standard2"],
    format_types=["full", "condensed", "summary"]
)
```

## Configuration

### Server Configuration

```json
{
    "token_model": "gpt-4",
    "default_token_budget": 8000,
    "token_optimization": {
        "cache_ttl": 3600,
        "abbreviations": {
            "custom_term": "ct"
        }
    }
}
```

### Format Selection Strategy

The system automatically selects the best format based on:

1. **Token Budget Ratio**: Available tokens / Original tokens
   - ≥ 80%: Full format
   - ≥ 40%: Condensed format
   - ≥ 15%: Reference format
   - < 15%: Summary format

2. **Context Hints**:
   - `query_type: "quick_lookup"` → Reference format
   - `need_examples: true` → Prefer full/condensed
   - `user_expertise: "beginner"` → Include examples

3. **Section Priorities**:
   - Security: Priority 9
   - Requirements: Priority 9
   - Implementation: Priority 7
   - Examples: Priority 5 (boosted for beginners)

## Performance Benchmarks

Based on testing with various standard sizes:

### Small Standards (<1000 tokens)
- Full format: 100% retention
- Condensed: ~60% compression
- Reference: ~25% compression
- Summary: ~8% compression

### Medium Standards (1000-5000 tokens)
- Full format: ~95% retention
- Condensed: ~50% compression
- Reference: ~20% compression
- Summary: ~5% compression

### Large Standards (>5000 tokens)
- Full format: ~90% retention
- Condensed: ~45% compression
- Reference: ~18% compression
- Summary: ~4% compression

### Processing Time
- Small standards: <100ms
- Medium standards: 100-300ms
- Large standards: 300-800ms
- With caching: <50ms

## Best Practices

### 1. Choose the Right Format
- Use `summary` for overviews and planning
- Use `reference` for quick lookups during coding
- Use `condensed` for standard implementation work
- Use `full` only when complete details are essential

### 2. Leverage Progressive Loading
- Start with overview sections
- Load additional sections based on user interaction
- Use context to predict needed sections

### 3. Budget Management
- Reserve tokens for context and response
- Set warning thresholds
- Monitor token usage with estimates

### 4. Context-Aware Optimization
- Provide user expertise level
- Specify focus areas
- Indicate query type

### 5. Caching Strategy
- Results are cached for 1 hour by default
- Cache key includes format and required sections
- Clear cache when standards update

## Example Scenarios

### Scenario 1: Limited Context Window
```python
# Working with 4K context limit
budget = TokenBudget(total=4000, reserved_for_context=500, reserved_for_response=500)

# Auto-optimize multiple standards
result = await mcp.auto_optimize_standards(
    standard_ids=["std1", "std2", "std3"],
    total_token_budget=3000
)
```

### Scenario 2: Beginner Tutorial
```python
# Optimize for beginners with examples
result = await mcp.get_optimized_standard(
    standard_id="react-patterns",
    format_type="condensed",
    context={
        "user_expertise": "beginner",
        "focus_areas": ["examples", "implementation"]
    }
)
```

### Scenario 3: Expert Quick Reference
```python
# Quick reference for experts
result = await mcp.get_optimized_standard(
    standard_id="advanced-patterns",
    format_type="reference",
    required_sections=["api", "configuration"],
    context={
        "user_expertise": "expert",
        "query_type": "quick_lookup"
    }
)
```

## Troubleshooting

### Common Issues

1. **Token count exceeds budget**
   - Solution: Use more aggressive format or increase budget
   - Check warnings in response

2. **Important sections excluded**
   - Solution: Use `required_sections` parameter
   - Adjust section priorities in configuration

3. **Slow performance**
   - Solution: Enable caching
   - Use batch operations
   - Consider pre-warming cache

4. **Inconsistent formatting**
   - Solution: Specify format explicitly
   - Provide consistent context

## Future Enhancements

- Semantic compression using embeddings
- Learning from user interactions
- Custom compression strategies
- Multi-language support
- Streaming progressive loading