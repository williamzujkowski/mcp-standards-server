# Enhanced Semantic Search for MCP Standards

This module provides advanced semantic search capabilities for the MCP Standards Server, enabling intelligent and efficient searching through standards documentation.

## Features

### 1. Query Preprocessing
- **Tokenization and Stemming**: Reduces words to their root forms for better matching
- **Synonym Expansion**: Automatically expands queries with domain-specific synonyms
- **Stopword Removal**: Filters out common words to focus on meaningful terms
- **Boolean Operators**: Supports AND, OR, NOT operators for complex queries

### 2. Embedding Generation with Caching
- **Multi-layer Caching**: Memory, Redis, and file-based caching for embeddings
- **Batch Processing**: Efficient embedding generation for multiple documents
- **TTL Management**: Automatic cache expiration and refresh
- **Model Flexibility**: Supports various sentence transformer models

### 3. Fuzzy Matching
- **Typo Tolerance**: Automatically corrects common spelling mistakes
- **Configurable Threshold**: Adjustable matching sensitivity
- **Known Terms Database**: Learns from indexed documents

### 4. Advanced Search Features
- **Metadata Filtering**: Filter results by document metadata
- **Re-ranking**: Multi-factor scoring including semantic similarity, term frequency, and recency
- **Result Highlighting**: Automatic extraction of relevant snippets
- **Query Expansion**: Enhances queries with related terms

### 5. Performance Optimization
- **Result Caching**: Caches search results with configurable TTL
- **Parallel Processing**: Uses thread pools for concurrent operations
- **Async Support**: Full async/await support for non-blocking operations
- **Batch Indexing**: Efficient indexing of multiple documents

### 6. Search Analytics
- **Query Tracking**: Records all searches for analysis
- **Performance Metrics**: Tracks latency and cache hit rates
- **Popular Queries**: Identifies frequently searched terms
- **Click-through Tracking**: Monitors which results users select

## Usage

### Basic Usage

```python
from src.core.standards.semantic_search import create_search_engine

# Create search engine
search = create_search_engine()

# Index documents
documents = [
    ("doc1", "React component testing best practices", {"category": "testing"}),
    ("doc2", "API security guidelines", {"category": "security"}),
]
search.index_documents_batch(documents)

# Search
results = search.search("React testing", top_k=5)
for result in results:
    print(f"{result.id}: {result.score:.3f}")
```

### Advanced Usage

```python
# Fuzzy search with typos
results = search.search("Reakt tesing", use_fuzzy=True)

# Boolean operators
results = search.search("security AND api NOT react")

# Metadata filtering
results = search.search("guidelines", filters={"category": "security"})

# With re-ranking
results = search.search("best practices", rerank=True)
```

### Async Usage

```python
import asyncio

# Create async search engine
search = create_search_engine(async_mode=True)

async def search_async():
    # Index documents
    await search.index_documents_batch_async(documents)
    
    # Perform search
    results = await search.search_async("testing")
    return results

# Run
results = asyncio.run(search_async())
```

## Configuration

### Search Engine Options

```python
search = create_search_engine(
    embedding_model='all-MiniLM-L6-v2',  # Sentence transformer model
    enable_analytics=True,               # Enable search analytics
    cache_dir=Path('/custom/cache'),     # Custom cache directory
    async_mode=False                     # Enable async mode
)
```

### Query Preprocessor Customization

The `QueryPreprocessor` class can be extended to add custom synonyms:

```python
preprocessor = QueryPreprocessor()
preprocessor.synonyms['mcp'] = ['model context protocol', 'context protocol']
```

### Fuzzy Matcher Configuration

```python
fuzzy_matcher = FuzzyMatcher(threshold=80)  # 80% similarity required
fuzzy_matcher.add_known_terms(['react', 'angular', 'vue'])
```

## Performance Considerations

### Caching Strategy

The system uses a three-tier caching approach:
1. **Memory Cache**: Fastest, limited by RAM
2. **Redis Cache**: Distributed, good for multi-instance deployments
3. **File Cache**: Persistent, survives restarts

### Embedding Model Selection

- **all-MiniLM-L6-v2**: Default, good balance of speed and quality
- **all-mpnet-base-v2**: Higher quality, slower
- **all-MiniLM-L12-v2**: Faster, slightly lower quality

### Optimization Tips

1. **Batch Operations**: Always use batch methods when indexing multiple documents
2. **Result Limits**: Use appropriate `top_k` values to limit result set size
3. **Caching**: Enable caching for production deployments
4. **Async Mode**: Use async mode for web applications

## Analytics and Monitoring

### Getting Analytics Report

```python
report = search.get_analytics_report()
print(f"Total queries: {report['total_queries']}")
print(f"Average latency: {report['average_latency_ms']} ms")
print(f"Cache hit rate: {report['cache_hit_rate']:.2%}")
print(f"Top queries: {report['top_queries']}")
```

### Click-through Tracking

```python
# Track when user clicks on a result
search.track_click(query="React testing", result_id="doc1")
```

## Integration with MCP Standards Server

The semantic search module is designed to integrate seamlessly with the MCP Standards Server:

1. **Standards Indexing**: Automatically indexes all standards documents
2. **Metadata Support**: Uses standard categories, tags, and versions
3. **Query Enhancement**: Leverages domain knowledge for better results
4. **Performance**: Optimized for the scale of standards repositories

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**: Run `nltk.download('punkt', 'stopwords', 'wordnet')`
2. **Redis Connection Failed**: Redis is optional; the system falls back to file cache
3. **Memory Issues**: Reduce batch sizes or use a smaller embedding model
4. **Slow Searches**: Enable caching and consider using async mode

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('src.core.standards.semantic_search').setLevel(logging.DEBUG)
```

## Future Enhancements

Planned improvements include:
- Vector database integration (Pinecone, Weaviate, Qdrant)
- Custom embedding fine-tuning for standards domain
- Multi-language support
- Query suggestion and auto-complete
- Federated search across multiple repositories