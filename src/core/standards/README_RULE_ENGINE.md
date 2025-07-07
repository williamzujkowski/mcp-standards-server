# Rule Engine for Automatic Standard Selection

The Rule Engine is a flexible system for automatically selecting appropriate development standards based on project context. It evaluates a set of configurable rules against project metadata to determine which standards should be applied.

## Key Features

- **Flexible Condition System**: Support for various operators (equals, contains, regex, etc.)
- **Complex Logic**: Combine conditions with AND, OR, and NOT logic
- **Priority-Based Resolution**: Resolve conflicts between competing standards
- **Extensible Design**: Easy to add new rules and conditions
- **Decision Tree Generation**: Visualize rule relationships and decision paths
- **Tag-Based Filtering**: Filter rules by categories
- **Performance Optimized**: Efficient evaluation of large rule sets

## Architecture

### Core Components

1. **RuleCondition**: Represents a single condition to evaluate
   - Field path (supports nested fields with dot notation)
   - Operator (equals, contains, in, exists, etc.)
   - Value to compare against
   - Case sensitivity option

2. **RuleGroup**: Groups multiple conditions with logical operators
   - AND: All conditions must match
   - OR: At least one condition must match
   - NOT: Negates the grouped conditions

3. **StandardRule**: Complete rule definition
   - Unique identifier and metadata
   - Priority for conflict resolution
   - Conditions (single or grouped)
   - List of standards to apply when matched
   - Tags for categorization

4. **RuleEngine**: Main engine for rule evaluation
   - Loads rules from JSON/YAML files
   - Evaluates rules against context
   - Resolves conflicts based on priority
   - Generates decision trees

## Rule Configuration

Rules are defined in JSON format with the following structure:

```json
{
  "rules": [
    {
      "id": "unique-rule-id",
      "name": "Human-readable name",
      "description": "Detailed description",
      "priority": 10,  // Lower number = higher priority
      "conditions": {
        // Simple condition
        "field": "project_type",
        "operator": "equals",
        "value": "web_application"
      },
      // OR complex conditions with logic
      "conditions": {
        "logic": "AND",
        "conditions": [
          {
            "field": "language",
            "operator": "equals",
            "value": "python"
          },
          {
            "field": "framework",
            "operator": "in",
            "value": ["django", "fastapi", "flask"]
          }
        ]
      },
      "standards": [
        "standard-id-1",
        "standard-id-2"
      ],
      "tags": ["category1", "category2"],
      "metadata": {
        // Additional metadata
      }
    }
  ]
}
```

## Supported Operators

- **equals**: Exact match (with optional case sensitivity)
- **not_equals**: Not equal to value
- **contains**: Value is contained in field (for lists/strings)
- **not_contains**: Value is not contained in field
- **in**: Field value is in the provided list
- **not_in**: Field value is not in the provided list
- **matches_regex**: Field matches regular expression
- **exists**: Field exists in context
- **not_exists**: Field does not exist in context
- **greater_than**: Numeric greater than comparison
- **less_than**: Numeric less than comparison
- **greater_equal**: Numeric greater than or equal
- **less_equal**: Numeric less than or equal

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from src.core.standards.rule_engine import RuleEngine

# Load rules from file
rules_path = Path("data/standards/meta/standard-selection-rules.json")
engine = RuleEngine(rules_path)

# Define project context
context = {
    "project_type": "web_application",
    "framework": "react",
    "language": "javascript",
    "requirements": ["accessibility", "performance"]
}

# Evaluate rules
result = engine.evaluate(context)

# Access results
print(f"Matched rules: {len(result['matched_rules'])}")
print(f"Selected standards: {result['resolved_standards']}")
```

### Advanced Usage

```python
# Evaluate with filters
result = engine.evaluate(
    context,
    max_priority=20,  # Only rules with priority <= 20
    tags=["frontend", "security"]  # Only rules with these tags
)

# Add rules programmatically
from src.core.standards.rule_engine import StandardRule, RuleCondition, RuleOperator

rule = StandardRule(
    id="custom-rule",
    name="Custom Rule",
    description="My custom rule",
    priority=15,
    conditions=RuleCondition(
        field="custom_field",
        operator=RuleOperator.EQUALS,
        value="custom_value"
    ),
    standards=["custom-standard-1", "custom-standard-2"],
    tags=["custom"]
)

engine.add_rule(rule)
```

### Complex Conditions

```python
from src.core.standards.rule_engine import RuleGroup, ConditionLogic

# Create complex conditions: (A AND B) OR (C AND D)
complex_condition = RuleGroup(
    logic=ConditionLogic.OR,
    conditions=[
        RuleGroup(
            logic=ConditionLogic.AND,
            conditions=[
                RuleCondition("language", RuleOperator.EQUALS, "python"),
                RuleCondition("framework", RuleOperator.EQUALS, "django")
            ]
        ),
        RuleGroup(
            logic=ConditionLogic.AND,
            conditions=[
                RuleCondition("language", RuleOperator.EQUALS, "javascript"),
                RuleCondition("framework", RuleOperator.EQUALS, "express")
            ]
        )
    ]
)
```

## Decision Tree Generation

The rule engine can generate a decision tree to visualize rule relationships:

```python
tree = engine.get_decision_tree()
print(json.dumps(tree, indent=2))
```

This produces a hierarchical structure showing:
- Total number of rules
- Branches grouped by decision fields
- Rules within each branch sorted by priority

## Conflict Resolution

When multiple rules provide the same standard, conflicts are resolved based on:

1. **Priority**: Rules with lower priority numbers win
2. **Specificity**: More specific rules (more conditions) typically have higher priority
3. **Order**: If priorities are equal, first rule wins

## Best Practices

### Rule Design

1. **Use Clear IDs**: Rule IDs should be descriptive and unique
2. **Set Appropriate Priorities**: 
   - 1-10: Critical rules (security, compliance)
   - 11-20: Framework-specific rules
   - 21-30: General language rules
   - 31-40: Optional enhancements
   - 41+: Low priority suggestions

3. **Avoid Over-Specificity**: Balance between specific and general rules
4. **Use Tags**: Tag rules for easier filtering and organization
5. **Document Rules**: Provide clear descriptions for maintenance

### Performance Considerations

1. **Rule Order**: Place frequently matched rules first
2. **Condition Complexity**: Simple conditions evaluate faster
3. **Nested Fields**: Minimize deep nesting in field paths
4. **Regex Usage**: Use simple patterns when possible

### Testing Rules

1. **Unit Test Rules**: Test individual rules with various contexts
2. **Integration Testing**: Test rule combinations and conflicts
3. **Coverage Analysis**: Ensure all project types have applicable rules
4. **Conflict Testing**: Verify priority resolution works correctly

## Extending the Rule Engine

### Adding New Operators

To add a new operator:

1. Add to `RuleOperator` enum
2. Implement evaluation logic in `RuleCondition.evaluate()`
3. Add unit tests for the new operator
4. Update documentation

### Custom Rule Loaders

Implement custom loaders for different rule sources:

```python
class CustomRuleLoader:
    def load_rules(self) -> List[StandardRule]:
        # Load rules from database, API, etc.
        pass

# Use with engine
engine = RuleEngine()
for rule in custom_loader.load_rules():
    engine.add_rule(rule)
```

## Integration with Standards Engine

The Rule Engine integrates with the broader standards system:

1. **Standard Resolution**: Rule engine determines which standards apply
2. **Standard Loading**: Standards engine loads the selected standards
3. **Token Optimization**: Only load necessary standards based on rules
4. **Caching**: Cache rule evaluation results for performance

## Troubleshooting

### Common Issues

1. **Rules Not Matching**:
   - Check field names and paths
   - Verify operator logic
   - Ensure values match expected types

2. **Unexpected Conflicts**:
   - Review rule priorities
   - Check for overlapping conditions
   - Use decision tree to visualize

3. **Performance Issues**:
   - Profile rule evaluation
   - Optimize complex conditions
   - Consider caching results

### Debug Mode

Enable debug logging to trace rule evaluation:

```python
import logging

logging.getLogger('src.core.standards.rule_engine').setLevel(logging.DEBUG)
```

This will show detailed information about condition evaluation and rule matching.