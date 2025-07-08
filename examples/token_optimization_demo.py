#!/usr/bin/env python3
"""
Token Optimization Demo

This script demonstrates the token optimization capabilities of the MCP Standards Server.
"""

import asyncio
import json
from typing import Dict, Any, List

from src.core.standards.token_optimizer import (
    TokenOptimizer,
    TokenBudget,
    StandardFormat,
    ModelType,
    create_token_optimizer,
    DynamicLoader
)


class TokenOptimizationDemo:
    """Demonstration of token optimization features."""
    
    def __init__(self):
        self.optimizer = create_token_optimizer(ModelType.GPT4, default_budget=8000)
        self.dynamic_loader = DynamicLoader(self.optimizer)
    
    def create_sample_standard(self) -> Dict[str, Any]:
        """Create a sample standard for demonstration."""
        return {
            'id': 'demo-standard',
            'name': 'Demonstration Standard',
            'content': """# React Best Practices Standard

## Overview

This comprehensive standard covers best practices for React application development,
including component design, state management, performance optimization, and testing
strategies. Following these guidelines ensures maintainable, scalable, and performant
React applications.

## Requirements

### Functional Components
- **Must** use functional components with hooks instead of class components
- **Should** follow the single responsibility principle
- **Must** use proper TypeScript types for all props and state
- **Should** implement proper error boundaries for production apps

### State Management
- **Must** lift state up to the appropriate level
- **Should** use Context API for cross-cutting concerns
- **Must** avoid prop drilling beyond 2-3 levels
- **Should** consider state management libraries for complex apps

### Performance Requirements
- **Must** implement React.memo for expensive components
- **Should** use useMemo and useCallback appropriately
- **Must** lazy load routes and heavy components
- **Should** implement virtualization for long lists

## Implementation

### Component Structure

```typescript
// Preferred functional component structure
interface UserProfileProps {
  userId: string;
  onUpdate?: (user: User) => void;
}

export const UserProfile: React.FC<UserProfileProps> = ({ userId, onUpdate }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    fetchUser(userId)
      .then(setUser)
      .catch(setError)
      .finally(() => setLoading(false));
  }, [userId]);

  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!user) return <NotFound />;

  return (
    <div className="user-profile">
      <h1>{user.name}</h1>
      <UserDetails user={user} />
      {onUpdate && <EditButton onClick={() => onUpdate(user)} />}
    </div>
  );
};
```

### Custom Hooks

```typescript
// Custom hook for data fetching
function useApiData<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const abortController = new AbortController();

    async function fetchData() {
      try {
        setLoading(true);
        const response = await fetch(url, {
          signal: abortController.signal
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        setData(data);
      } catch (err) {
        if (err.name !== 'AbortError') {
          setError(err as Error);
        }
      } finally {
        setLoading(false);
      }
    }

    fetchData();

    return () => abortController.abort();
  }, [url]);

  return { data, loading, error };
}
```

## Security Considerations

### Input Validation
- Always validate and sanitize user inputs
- Use proper escaping for dynamic content
- Implement Content Security Policy (CSP)
- Avoid using dangerouslySetInnerHTML

### Authentication & Authorization
- Implement proper JWT handling
- Use secure storage for tokens
- Implement proper CORS policies
- Regular security audits

## Performance Optimization

### Code Splitting
```typescript
// Route-based code splitting
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings = lazy(() => import('./pages/Settings'));

function App() {
  return (
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Suspense>
  );
}
```

### Memoization Strategies
```typescript
// Expensive computation memoization
const ExpensiveComponent = ({ data, filter }) => {
  const processedData = useMemo(() => {
    return data
      .filter(item => item.category === filter)
      .map(item => ({
        ...item,
        computed: expensiveComputation(item)
      }));
  }, [data, filter]);

  return <DataGrid data={processedData} />;
};
```

## Testing Strategy

### Unit Tests
- Test components in isolation
- Mock external dependencies
- Test edge cases and error states
- Maintain high code coverage

### Integration Tests
- Test component interactions
- Test data flow
- Test routing behavior
- Test API integrations

### Example Test
```typescript
describe('UserProfile', () => {
  it('should display user information', async () => {
    const mockUser = { id: '1', name: 'John Doe' };
    jest.spyOn(api, 'fetchUser').mockResolvedValue(mockUser);

    render(<UserProfile userId="1" />);

    expect(screen.getByText('Loading...')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('John Doe')).toBeInTheDocument();
    });
  });

  it('should handle errors gracefully', async () => {
    jest.spyOn(api, 'fetchUser').mockRejectedValue(new Error('Network error'));

    render(<UserProfile userId="1" />);

    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument();
    });
  });
});
```

## Best Practices Summary

1. **Component Design**
   - Keep components small and focused
   - Use composition over inheritance
   - Implement proper prop validation
   - Document complex components

2. **State Management**
   - Keep state as local as possible
   - Use proper state update patterns
   - Avoid unnecessary re-renders
   - Implement proper cleanup

3. **Performance**
   - Profile before optimizing
   - Use React DevTools
   - Implement proper caching
   - Monitor bundle size

4. **Code Quality**
   - Follow consistent naming conventions
   - Write self-documenting code
   - Implement proper error handling
   - Regular code reviews

## References

- [React Documentation](https://react.dev)
- [React TypeScript Cheatsheet](https://react-typescript-cheatsheet.netlify.app)
- [React Performance Optimization](https://react.dev/learn/render-and-commit)
- [Testing React Applications](https://testing-library.com/docs/react-testing-library/intro)
"""
        }
    
    async def demo_format_comparison(self):
        """Demonstrate different format outputs."""
        print("=== Format Comparison Demo ===\n")
        
        standard = self.create_sample_standard()
        formats = [StandardFormat.FULL, StandardFormat.CONDENSED, 
                  StandardFormat.REFERENCE, StandardFormat.SUMMARY]
        
        for format_type in formats:
            print(f"\n--- {format_type.value.upper()} FORMAT ---")
            
            content, result = self.optimizer.optimize_standard(
                standard,
                format_type=format_type,
                budget=TokenBudget(total=8000)
            )
            
            print(f"Original tokens: {result.original_tokens}")
            print(f"Compressed tokens: {result.compressed_tokens}")
            print(f"Compression ratio: {result.compression_ratio:.2%}")
            print(f"Sections included: {', '.join(result.sections_included)}")
            
            # Show preview
            preview = content[:500] + "..." if len(content) > 500 else content
            print(f"\nPreview:\n{preview}\n")
            print("-" * 80)
    
    async def demo_budget_constraints(self):
        """Demonstrate behavior under different budgets."""
        print("\n\n=== Budget Constraints Demo ===\n")
        
        standard = self.create_sample_standard()
        budgets = [500, 1000, 2000, 5000]
        
        for budget_size in budgets:
            print(f"\n--- Budget: {budget_size} tokens ---")
            
            budget = TokenBudget(total=budget_size)
            selected_format = self.optimizer.auto_select_format(standard, budget)
            
            content, result = self.optimizer.optimize_standard(
                standard,
                format_type=selected_format,
                budget=budget
            )
            
            print(f"Auto-selected format: {selected_format.value}")
            print(f"Tokens used: {result.compressed_tokens}")
            print(f"Sections included: {len(result.sections_included)}")
            print(f"Sections excluded: {len(result.sections_excluded)}")
            
            if result.warnings:
                print(f"Warnings: {', '.join(result.warnings)}")
    
    async def demo_context_aware_optimization(self):
        """Demonstrate context-aware optimization."""
        print("\n\n=== Context-Aware Optimization Demo ===\n")
        
        standard = self.create_sample_standard()
        
        contexts = [
            {
                'name': 'Beginner Developer',
                'context': {
                    'user_expertise': 'beginner',
                    'focus_areas': ['examples', 'implementation'],
                    'query_type': 'learning'
                }
            },
            {
                'name': 'Expert Quick Lookup',
                'context': {
                    'user_expertise': 'expert',
                    'focus_areas': ['performance', 'security'],
                    'query_type': 'quick_lookup'
                }
            },
            {
                'name': 'Security Audit',
                'context': {
                    'user_expertise': 'intermediate',
                    'focus_areas': ['security'],
                    'query_type': 'detailed_explanation'
                }
            }
        ]
        
        for ctx in contexts:
            print(f"\n--- Context: {ctx['name']} ---")
            
            content, result = self.optimizer.optimize_standard(
                standard,
                format_type=StandardFormat.CUSTOM,
                budget=TokenBudget(total=3000),
                context=ctx['context']
            )
            
            print(f"Format used: {result.format_used.value}")
            print(f"Tokens: {result.compressed_tokens}")
            print(f"Focus sections: {[s for s in result.sections_included if any(f in s for f in ctx['context'].get('focus_areas', []))]}")
    
    async def demo_progressive_loading(self):
        """Demonstrate progressive loading."""
        print("\n\n=== Progressive Loading Demo ===\n")
        
        standard = self.create_sample_standard()
        
        # Generate loading plan
        loading_plan = self.optimizer.progressive_load(
            standard,
            initial_sections=['overview'],
            max_depth=3
        )
        
        print("Loading Plan:")
        total_tokens = 0
        
        for i, batch in enumerate(loading_plan):
            batch_tokens = sum(tokens for _, tokens in batch)
            total_tokens += batch_tokens
            
            print(f"\nBatch {i + 1}:")
            for section_id, tokens in batch:
                print(f"  - {section_id}: {tokens} tokens")
            print(f"  Batch total: {batch_tokens} tokens")
            print(f"  Cumulative: {total_tokens} tokens")
        
        # Simulate dynamic loading
        print("\n\nSimulating Dynamic Loading:")
        budget = TokenBudget(total=2000)
        loaded_tokens = 0
        
        for section_id in ['overview', 'requirements', 'security']:
            content, tokens = self.dynamic_loader.load_section(
                'demo-standard',
                section_id,
                budget
            )
            
            if loaded_tokens + tokens <= budget.available:
                loaded_tokens += tokens
                print(f"Loaded {section_id}: {tokens} tokens (total: {loaded_tokens})")
            else:
                print(f"Cannot load {section_id}: would exceed budget")
                break
        
        # Get loading suggestions
        context = {
            'recent_queries': ['How to implement security?', 'Performance tips'],
            'user_expertise': 'intermediate'
        }
        
        suggestions = self.dynamic_loader.get_loading_suggestions(
            'demo-standard',
            context
        )
        
        print(f"\nSuggested sections to load next: {suggestions}")
    
    async def demo_compression_techniques(self):
        """Demonstrate individual compression techniques."""
        print("\n\n=== Compression Techniques Demo ===\n")
        
        from src.core.standards.token_optimizer import CompressionTechniques
        
        sample_text = """
        The   application    configuration     requires    careful    attention.
        
        
        
        This documentation provides comprehensive implementation guidelines.
        
        You must always validate user input for security.
        Never store passwords in plain text.
        
        ```python
        # This is a comment
        def configure_application():
            # Another comment
            
            
            config = load_configuration()
            
            return config
        ```
        """
        
        techniques = CompressionTechniques()
        
        print("Original text:")
        print(f"Length: {len(sample_text)} chars")
        print(f"Tokens: {self.optimizer.token_counter.count_tokens(sample_text)}")
        
        # Test each technique
        print("\n1. Remove Redundancy:")
        cleaned = techniques.remove_redundancy(sample_text)
        print(f"Length: {len(cleaned)} chars")
        print(f"Tokens: {self.optimizer.token_counter.count_tokens(cleaned)}")
        
        print("\n2. Use Abbreviations:")
        abbreviated = techniques.use_abbreviations(cleaned)
        print(f"Length: {len(abbreviated)} chars")
        print(f"Tokens: {self.optimizer.token_counter.count_tokens(abbreviated)}")
        print(f"Sample: {abbreviated[:100]}...")
        
        print("\n3. Compress Code:")
        code_compressed = techniques.compress_code_examples(sample_text)
        print(f"Length: {len(code_compressed)} chars")
        print(f"Tokens: {self.optimizer.token_counter.count_tokens(code_compressed)}")
        
        print("\n4. Extract Essential:")
        essential = techniques.extract_essential_only(sample_text)
        print(f"Length: {len(essential)} chars")
        print(f"Tokens: {self.optimizer.token_counter.count_tokens(essential)}")
        print(f"Content:\n{essential}")
    
    async def run_all_demos(self):
        """Run all demonstration scenarios."""
        print("MCP Standards Server - Token Optimization Demo")
        print("=" * 80)
        
        await self.demo_format_comparison()
        await self.demo_budget_constraints()
        await self.demo_context_aware_optimization()
        await self.demo_progressive_loading()
        await self.demo_compression_techniques()
        
        print("\n\nDemo completed!")
        
        # Show cache statistics
        stats = self.optimizer.get_compression_stats()
        print(f"\nCache Statistics:")
        print(f"- Cache size: {stats['cache_size']} entries")
        if 'average_compression_ratio' in stats:
            print(f"- Average compression: {stats['average_compression_ratio']:.2%}")
        if 'format_usage' in stats:
            print(f"- Format usage: {json.dumps(stats['format_usage'], indent=2)}")


async def main():
    """Run the demonstration."""
    demo = TokenOptimizationDemo()
    await demo.run_all_demos()


if __name__ == '__main__':
    asyncio.run(main())