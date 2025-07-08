"""
Enhanced semantic search implementation for MCP standards.

This module provides advanced search capabilities including:
- Query preprocessing with synonyms and stemming
- Embedding generation with caching
- Query expansion techniques
- Re-ranking based on relevance scores
- Boolean operator support
- Fuzzy matching for typos
- Search analytics and performance tracking
"""

import json
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from fuzzywuzzy import fuzz, process
import redis
from functools import lru_cache
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    highlights: List[str] = field(default_factory=list)
    explanation: Optional[str] = None


@dataclass
class SearchQuery:
    """Represents a parsed search query."""
    original: str
    preprocessed: str
    tokens: List[str]
    stems: List[str]
    expanded_terms: List[str] = field(default_factory=list)
    boolean_operators: Dict[str, List[str]] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchAnalytics:
    """Tracks search analytics and metrics."""
    query_count: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    popular_queries: Counter = field(default_factory=Counter)
    failed_queries: List[Tuple[str, str]] = field(default_factory=list)
    average_results_per_query: float = 0.0
    click_through_data: Dict[str, List[str]] = field(default_factory=dict)


class QueryPreprocessor:
    """Handles query preprocessing including synonyms, stemming, and expansion."""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
        # Domain-specific synonyms for standards/MCP context
        self.synonyms = {
            'api': ['interface', 'endpoint', 'service'],
            'test': ['testing', 'validation', 'verification', 'check'],
            'security': ['secure', 'safety', 'protection', 'auth', 'authentication'],
            'standard': ['convention', 'guideline', 'best practice', 'rule'],
            'web': ['website', 'webapp', 'frontend', 'browser'],
            'react': ['reactjs', 'react.js'],
            'vue': ['vuejs', 'vue.js'],
            'angular': ['angularjs', 'angular.js'],
            'mcp': ['model context protocol', 'context protocol'],
            'llm': ['language model', 'ai model', 'gpt', 'claude'],
            'performance': ['speed', 'optimization', 'efficiency', 'fast'],
            'accessibility': ['a11y', 'wcag', 'aria'],
            'database': ['db', 'data store', 'storage'],
            'deploy': ['deployment', 'release', 'publish'],
            'config': ['configuration', 'settings', 'setup'],
        }
        
        # Build reverse synonym mapping
        self.reverse_synonyms = {}
        for key, values in self.synonyms.items():
            for value in values:
                if value not in self.reverse_synonyms:
                    self.reverse_synonyms[value] = []
                self.reverse_synonyms[value].append(key)
    
    def preprocess(self, query: str) -> SearchQuery:
        """Preprocess a query with all enhancement techniques."""
        # Parse boolean operators
        boolean_ops = self._extract_boolean_operators(query)
        clean_query = self._remove_boolean_operators(query)
        
        # Tokenize and clean
        tokens = word_tokenize(clean_query.lower())
        tokens = [t for t in tokens if t.isalnum() and t not in self.stopwords]
        
        # Generate stems
        stems = [self.stemmer.stem(token) for token in tokens]
        
        # Expand query with synonyms
        expanded_terms = self._expand_with_synonyms(tokens)
        
        # Create search query object
        search_query = SearchQuery(
            original=query,
            preprocessed=clean_query.lower(),
            tokens=tokens,
            stems=stems,
            expanded_terms=expanded_terms,
            boolean_operators=boolean_ops
        )
        
        return search_query
    
    def _extract_boolean_operators(self, query: str) -> Dict[str, List[str]]:
        """Extract AND, OR, NOT operators from query."""
        operators = {
            'AND': [],
            'OR': [],
            'NOT': []
        }
        
        # Match patterns like "term1 AND term2"
        and_pattern = r'(\w+)\s+AND\s+(\w+)'
        or_pattern = r'(\w+)\s+OR\s+(\w+)'
        not_pattern = r'NOT\s+(\w+)'
        
        for match in re.finditer(and_pattern, query):
            operators['AND'].append((match.group(1).lower(), match.group(2).lower()))
        
        for match in re.finditer(or_pattern, query):
            operators['OR'].append((match.group(1).lower(), match.group(2).lower()))
        
        for match in re.finditer(not_pattern, query):
            operators['NOT'].append(match.group(1).lower())
        
        return operators
    
    def _remove_boolean_operators(self, query: str) -> str:
        """Remove boolean operators from query."""
        # Remove boolean operators but keep the terms
        query = re.sub(r'\s+AND\s+', ' ', query)
        query = re.sub(r'\s+OR\s+', ' ', query)
        query = re.sub(r'\s+NOT\s+', ' ', query)
        return query
    
    def _expand_with_synonyms(self, tokens: List[str]) -> List[str]:
        """Expand tokens with synonyms."""
        expanded = set()
        
        for token in tokens:
            # Add original token
            expanded.add(token)
            
            # Add direct synonyms
            if token in self.synonyms:
                expanded.update(self.synonyms[token])
            
            # Check if token is a synonym of something else
            if token in self.reverse_synonyms:
                expanded.update(self.reverse_synonyms[token])
        
        return list(expanded)


class EmbeddingCache:
    """Manages embedding generation with caching."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: Optional[Path] = None):
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir or Path.home() / '.mcp_search_cache'
        self.cache_dir.mkdir(exist_ok=True)
        
        # In-memory cache with TTL
        self.memory_cache = {}
        self.cache_ttl = timedelta(hours=24)
        
        # Redis cache for distributed systems (optional)
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
        except:
            logger.info("Redis not available, using file-based cache only")
    
    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Get embedding for text with caching."""
        # Generate cache key
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        
        if use_cache:
            # Check memory cache
            if cache_key in self.memory_cache:
                cached_time, embedding = self.memory_cache[cache_key]
                if datetime.now() - cached_time < self.cache_ttl:
                    return embedding
            
            # Check Redis cache
            if self.redis_client:
                try:
                    cached = self.redis_client.get(f"emb:{cache_key}")
                    if cached:
                        return pickle.loads(cached.encode('latin-1'))
                except:
                    pass
            
            # Check file cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        # Cache the result
        if use_cache:
            self._cache_embedding(cache_key, embedding)
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """Get embeddings for multiple texts with batching."""
        if not use_cache:
            return self.model.encode(texts, convert_to_numpy=True)
        
        # Separate cached and uncached texts
        embeddings = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cache_key = hashlib.sha256(text.encode()).hexdigest()
            
            # Try to get from cache
            cached_emb = self._get_cached_embedding(cache_key)
            if cached_emb is not None:
                embeddings[i] = cached_emb
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            
            # Cache and assign results
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                cache_key = hashlib.sha256(text.encode()).hexdigest()
                self._cache_embedding(cache_key, emb)
                embeddings[idx] = emb
        
        return np.vstack(embeddings)
    
    def _get_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        """Try to get embedding from various cache layers."""
        # Memory cache
        if cache_key in self.memory_cache:
            cached_time, embedding = self.memory_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return embedding
        
        # Redis cache
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"emb:{cache_key}")
                if cached:
                    return pickle.loads(cached.encode('latin-1'))
            except:
                pass
        
        # File cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        
        return None
    
    def _cache_embedding(self, cache_key: str, embedding: np.ndarray):
        """Cache embedding in multiple layers."""
        # Memory cache
        self.memory_cache[cache_key] = (datetime.now(), embedding)
        
        # Redis cache
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"emb:{cache_key}",
                    int(self.cache_ttl.total_seconds()),
                    pickle.dumps(embedding).decode('latin-1')
                )
            except:
                pass
        
        # File cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except:
            pass
    
    def clear_cache(self):
        """Clear all caches."""
        self.memory_cache.clear()
        
        if self.redis_client:
            try:
                for key in self.redis_client.scan_iter("emb:*"):
                    self.redis_client.delete(key)
            except:
                pass
        
        # Clear file cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except:
                pass


class FuzzyMatcher:
    """Handles fuzzy matching for typo tolerance."""
    
    def __init__(self, threshold: int = 80):
        self.threshold = threshold
        self.known_terms = set()
    
    def add_known_terms(self, terms: List[str]):
        """Add terms to the known terms set."""
        self.known_terms.update(terms)
    
    def find_matches(self, query: str, candidates: List[str] = None) -> List[Tuple[str, int]]:
        """Find fuzzy matches for a query."""
        if candidates is None:
            candidates = list(self.known_terms)
        
        # Use token set ratio for better matching of partial terms
        matches = process.extract(
            query,
            candidates,
            scorer=fuzz.token_set_ratio,
            limit=5
        )
        
        # Filter by threshold
        return [(match, score) for match, score in matches if score >= self.threshold]
    
    def correct_query(self, query: str) -> Tuple[str, List[str]]:
        """Attempt to correct typos in query."""
        tokens = query.lower().split()
        corrections = []
        corrected_tokens = []
        
        for token in tokens:
            matches = self.find_matches(token)
            if matches and matches[0][1] < 100:  # Not exact match
                best_match = matches[0][0]
                corrections.append(f"{token} -> {best_match}")
                corrected_tokens.append(best_match)
            else:
                corrected_tokens.append(token)
        
        corrected_query = ' '.join(corrected_tokens)
        return corrected_query, corrections


class SemanticSearch:
    """Main semantic search engine with all enhanced features."""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 enable_analytics: bool = True,
                 cache_dir: Optional[Path] = None):
        self.preprocessor = QueryPreprocessor()
        self.embedding_cache = EmbeddingCache(embedding_model, cache_dir)
        self.fuzzy_matcher = FuzzyMatcher()
        self.analytics = SearchAnalytics() if enable_analytics else None
        
        # Document store (in production, this would be a vector database)
        self.documents = {}
        self.document_embeddings = {}
        self.document_metadata = {}
        
        # Query result cache
        self.result_cache = {}
        self.result_cache_ttl = timedelta(minutes=30)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def index_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        """Index a document for searching."""
        # Store document
        self.documents[doc_id] = content
        self.document_metadata[doc_id] = metadata or {}
        
        # Generate and cache embedding
        embedding = self.embedding_cache.get_embedding(content)
        self.document_embeddings[doc_id] = embedding
        
        # Update fuzzy matcher with document terms
        tokens = word_tokenize(content.lower())
        self.fuzzy_matcher.add_known_terms(tokens)
    
    def index_documents_batch(self, documents: List[Tuple[str, str, Dict[str, Any]]]):
        """Index multiple documents efficiently."""
        # Extract components
        doc_ids = [doc[0] for doc in documents]
        contents = [doc[1] for doc in documents]
        metadatas = [doc[2] if len(doc) > 2 else {} for doc in documents]
        
        # Store documents
        for doc_id, content, metadata in zip(doc_ids, contents, metadatas):
            self.documents[doc_id] = content
            self.document_metadata[doc_id] = metadata
        
        # Generate embeddings in batch
        embeddings = self.embedding_cache.get_embeddings_batch(contents)
        
        # Store embeddings
        for doc_id, embedding in zip(doc_ids, embeddings):
            self.document_embeddings[doc_id] = embedding
        
        # Update fuzzy matcher
        all_tokens = []
        for content in contents:
            tokens = word_tokenize(content.lower())
            all_tokens.extend(tokens)
        self.fuzzy_matcher.add_known_terms(list(set(all_tokens)))
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None,
               rerank: bool = True,
               use_fuzzy: bool = True,
               use_cache: bool = True) -> List[SearchResult]:
        """Perform semantic search with all enhancements."""
        start_time = time.time()
        
        # Check result cache
        cache_key = self._get_result_cache_key(query, top_k, filters)
        if use_cache and cache_key in self.result_cache:
            cached_time, cached_results = self.result_cache[cache_key]
            if datetime.now() - cached_time < self.result_cache_ttl:
                if self.analytics:
                    self.analytics.cache_hits += 1
                return cached_results
        
        if self.analytics:
            self.analytics.cache_misses += 1
        
        try:
            # Preprocess query
            search_query = self.preprocessor.preprocess(query)
            
            # Apply fuzzy matching if enabled
            if use_fuzzy:
                corrected_query, corrections = self.fuzzy_matcher.correct_query(
                    search_query.preprocessed
                )
                if corrections:
                    logger.info(f"Applied corrections: {corrections}")
                    # Re-preprocess with corrected query
                    search_query = self.preprocessor.preprocess(corrected_query)
            
            # Generate query embedding with expanded terms
            expanded_query = ' '.join(search_query.tokens + search_query.expanded_terms)
            query_embedding = self.embedding_cache.get_embedding(expanded_query)
            
            # Calculate similarities
            similarities = self._calculate_similarities(
                query_embedding,
                search_query,
                filters
            )
            
            # Get top results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Create initial results
            results = []
            for idx in top_indices:
                doc_id = list(self.documents.keys())[idx]
                score = float(similarities[idx])
                
                if score > 0:  # Only include positive scores
                    result = SearchResult(
                        id=doc_id,
                        content=self.documents[doc_id],
                        score=score,
                        metadata=self.document_metadata.get(doc_id, {})
                    )
                    results.append(result)
            
            # Apply re-ranking if enabled
            if rerank and len(results) > 1:
                results = self._rerank_results(results, search_query)
            
            # Generate highlights
            for result in results:
                result.highlights = self._generate_highlights(
                    result.content,
                    search_query.tokens
                )
            
            # Cache results
            if use_cache:
                self.result_cache[cache_key] = (datetime.now(), results)
            
            # Update analytics
            if self.analytics:
                elapsed = time.time() - start_time
                self.analytics.query_count += 1
                self.analytics.total_latency += elapsed
                self.analytics.popular_queries[query] += 1
                self.analytics.average_results_per_query = (
                    (self.analytics.average_results_per_query * (self.analytics.query_count - 1) + len(results))
                    / self.analytics.query_count
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            if self.analytics:
                self.analytics.failed_queries.append((query, str(e)))
            return []
    
    def _calculate_similarities(self, 
                              query_embedding: np.ndarray,
                              search_query: SearchQuery,
                              filters: Optional[Dict[str, Any]]) -> np.ndarray:
        """Calculate similarities with boolean operators and filters."""
        # Get all document embeddings as matrix
        doc_ids = list(self.documents.keys())
        doc_embeddings = np.vstack([self.document_embeddings[doc_id] for doc_id in doc_ids])
        
        # Calculate base similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Apply boolean operators
        if search_query.boolean_operators:
            similarities = self._apply_boolean_operators(
                similarities,
                doc_ids,
                search_query.boolean_operators
            )
        
        # Apply filters
        if filters:
            similarities = self._apply_filters(similarities, doc_ids, filters)
        
        return similarities
    
    def _apply_boolean_operators(self,
                                similarities: np.ndarray,
                                doc_ids: List[str],
                                operators: Dict[str, List]) -> np.ndarray:
        """Apply AND, OR, NOT boolean operators."""
        modified_similarities = similarities.copy()
        
        for i, doc_id in enumerate(doc_ids):
            content = self.documents[doc_id].lower()
            
            # Apply NOT operators
            for not_term in operators.get('NOT', []):
                if not_term in content:
                    modified_similarities[i] = 0
            
            # Apply AND operators
            for term1, term2 in operators.get('AND', []):
                if not (term1 in content and term2 in content):
                    modified_similarities[i] *= 0.5  # Penalize but don't eliminate
            
            # Apply OR operators (boost if either term present)
            for term1, term2 in operators.get('OR', []):
                if term1 in content or term2 in content:
                    modified_similarities[i] *= 1.2  # Boost score
        
        return modified_similarities
    
    def _apply_filters(self,
                      similarities: np.ndarray,
                      doc_ids: List[str],
                      filters: Dict[str, Any]) -> np.ndarray:
        """Apply metadata filters."""
        modified_similarities = similarities.copy()
        
        for i, doc_id in enumerate(doc_ids):
            metadata = self.document_metadata.get(doc_id, {})
            
            # Check each filter
            for key, value in filters.items():
                if key not in metadata:
                    modified_similarities[i] = 0
                elif isinstance(value, list):
                    # Filter value is a list - check if metadata value is in list
                    if metadata[key] not in value:
                        modified_similarities[i] = 0
                else:
                    # Direct comparison
                    if metadata[key] != value:
                        modified_similarities[i] = 0
        
        return modified_similarities
    
    def _rerank_results(self,
                       results: List[SearchResult],
                       search_query: SearchQuery) -> List[SearchResult]:
        """Re-rank results based on additional factors."""
        # Calculate additional scoring factors
        for result in results:
            # Term frequency boost
            term_freq_score = self._calculate_term_frequency_score(
                result.content,
                search_query.tokens
            )
            
            # Recency boost (if timestamp in metadata)
            recency_score = self._calculate_recency_score(result.metadata)
            
            # Combine scores
            result.score = (
                result.score * 0.7 +  # Original semantic similarity
                term_freq_score * 0.2 +  # Term frequency
                recency_score * 0.1  # Recency
            )
            
            # Add explanation
            result.explanation = (
                f"Semantic: {result.score:.3f}, "
                f"Term Freq: {term_freq_score:.3f}, "
                f"Recency: {recency_score:.3f}"
            )
        
        # Re-sort by new scores
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    def _calculate_term_frequency_score(self, content: str, terms: List[str]) -> float:
        """Calculate term frequency score."""
        content_lower = content.lower()
        total_occurrences = sum(content_lower.count(term) for term in terms)
        
        # Normalize by content length
        return min(total_occurrences / (len(content_lower.split()) + 1), 1.0)
    
    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score based on timestamp."""
        if 'timestamp' not in metadata:
            return 0.5  # Neutral score if no timestamp
        
        try:
            timestamp = datetime.fromisoformat(metadata['timestamp'])
            days_old = (datetime.now() - timestamp).days
            
            # Exponential decay - newer documents get higher scores
            return np.exp(-days_old / 30)  # Half-life of 30 days
        except:
            return 0.5
    
    def _generate_highlights(self, content: str, terms: List[str]) -> List[str]:
        """Generate highlighted snippets."""
        highlights = []
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in terms):
                # Highlight matching terms
                highlighted = sentence
                for term in terms:
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    highlighted = pattern.sub(f"**{term}**", highlighted)
                
                highlights.append(highlighted.strip())
                
                if len(highlights) >= 3:  # Limit to 3 highlights
                    break
        
        return highlights
    
    def _get_result_cache_key(self, query: str, top_k: int, filters: Optional[Dict]) -> str:
        """Generate cache key for results."""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        return hashlib.sha256(f"{query}:{top_k}:{filter_str}".encode()).hexdigest()
    
    def get_analytics_report(self) -> Dict[str, Any]:
        """Generate analytics report."""
        if not self.analytics:
            return {"error": "Analytics not enabled"}
        
        avg_latency = (
            self.analytics.total_latency / self.analytics.query_count
            if self.analytics.query_count > 0
            else 0
        )
        
        cache_hit_rate = (
            self.analytics.cache_hits / (self.analytics.cache_hits + self.analytics.cache_misses)
            if (self.analytics.cache_hits + self.analytics.cache_misses) > 0
            else 0
        )
        
        return {
            "total_queries": self.analytics.query_count,
            "average_latency_ms": avg_latency * 1000,
            "cache_hit_rate": cache_hit_rate,
            "average_results_per_query": self.analytics.average_results_per_query,
            "top_queries": self.analytics.popular_queries.most_common(10),
            "failed_queries_count": len(self.analytics.failed_queries),
            "recent_failures": self.analytics.failed_queries[-5:],
        }
    
    def track_click(self, query: str, result_id: str):
        """Track click-through data for analytics."""
        if self.analytics:
            if query not in self.analytics.click_through_data:
                self.analytics.click_through_data[query] = []
            self.analytics.click_through_data[query].append(result_id)
    
    def close(self):
        """Clean up resources."""
        self.executor.shutdown()


# Async wrapper for non-blocking search
class AsyncSemanticSearch:
    """Async wrapper for SemanticSearch."""
    
    def __init__(self, semantic_search: SemanticSearch):
        self.search_engine = semantic_search
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
    
    def _run_loop(self):
        """Run the async event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    async def search_async(self, query: str, **kwargs) -> List[SearchResult]:
        """Perform search asynchronously."""
        future = self.loop.run_in_executor(
            None,
            self.search_engine.search,
            query,
            *kwargs
        )
        return await future
    
    async def index_documents_batch_async(self, documents: List[Tuple[str, str, Dict[str, Any]]]):
        """Index documents asynchronously."""
        future = self.loop.run_in_executor(
            None,
            self.search_engine.index_documents_batch,
            documents
        )
        await future
    
    def close(self):
        """Clean up async resources."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.search_engine.close()


# Convenience function for creating search engine
def create_search_engine(
    embedding_model: str = 'all-MiniLM-L6-v2',
    enable_analytics: bool = True,
    cache_dir: Optional[Path] = None,
    async_mode: bool = False
) -> Union[SemanticSearch, AsyncSemanticSearch]:
    """Create a semantic search engine instance."""
    search_engine = SemanticSearch(
        embedding_model=embedding_model,
        enable_analytics=enable_analytics,
        cache_dir=cache_dir
    )
    
    if async_mode:
        return AsyncSemanticSearch(search_engine)
    
    return search_engine