"""
Semantic search engine for standards using embeddings
@nist-controls: SI-10, AC-4
@evidence: ML-based query understanding and content matching
@oscal-component: standards-engine
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from semantic search"""
    content: str
    score: float
    metadata: dict[str, Any]
    chunk_id: str | None = None
    standard_id: str | None = None
    section_id: str | None = None


class EmbeddingModel:
    """
    Wrapper for embedding model
    @nist-controls: SI-10
    @evidence: Secure model loading and inference
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Initialized embedding model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise ImportError("sentence-transformers is required for semantic search")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")

        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.encode([text])[0]


class VectorIndex:
    """
    Vector index for similarity search
    @nist-controls: AC-4, SC-8
    @evidence: Secure vector storage and retrieval
    """

    def __init__(self, dimension: int | None = None):
        self.dimension = dimension
        self.index: Any = None  # FAISS index or None
        self.embeddings: np.ndarray | None = None  # Numpy embeddings for fallback
        self.metadata: list[dict[str, Any]] = []
        self.use_faiss = self._try_import_faiss()
        self.faiss: Any = None  # FAISS module

    def _try_import_faiss(self) -> bool:
        """Try to import FAISS for optimized search"""
        try:
            import faiss
            self.faiss = faiss
            logger.info("Using FAISS for optimized vector search")
            return True
        except ImportError:
            logger.warning("FAISS not available, using numpy for search (slower)")
            return False

    def build(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]):
        """Build index from embeddings"""
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")

        self.dimension = embeddings.shape[1]
        self.metadata = metadata

        if self.use_faiss:
            # Use FAISS for fast similarity search
            self.index = self.faiss.IndexFlatIP(self.dimension)  # Inner product
            # Normalize embeddings for cosine similarity
            normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index.add(normalized.astype('float32'))
        else:
            # Store embeddings directly for numpy search
            self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[tuple[int, float]]:
        """Search for k nearest neighbors"""
        if self.index is None and not hasattr(self, 'embeddings'):
            raise RuntimeError("Index not built")

        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        if self.use_faiss and self.index is not None:
            # FAISS search
            scores, indices = self.index.search(
                query_norm.reshape(1, -1).astype('float32'), k
            )
            return list(zip(indices[0], scores[0], strict=False))
        else:
            # Numpy cosine similarity
            if self.embeddings is None:
                return []
            similarities = np.dot(self.embeddings, query_norm)
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            return [(int(idx), float(similarities[idx])) for idx in top_k_indices]

    def save(self, path: Path):
        """Save index to disk"""
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f)

        # Save index
        if self.use_faiss:
            self.faiss.write_index(self.index, str(path / "index.faiss"))
        else:
            if self.embeddings is not None:
                np.save(path / "embeddings.npy", self.embeddings)

    def load(self, path: Path):
        """Load index from disk"""
        # Load metadata
        with open(path / "metadata.json") as f:
            self.metadata = json.load(f)

        # Load index
        if self.use_faiss and (path / "index.faiss").exists():
            self.index = self.faiss.read_index(str(path / "index.faiss"))
            if hasattr(self.index, 'd'):
                self.dimension = self.index.d
            else:
                self.dimension = None
        elif (path / "embeddings.npy").exists():
            self.embeddings = np.load(path / "embeddings.npy")
            self.dimension = self.embeddings.shape[1]
        else:
            raise ValueError("No valid index found")


class SemanticSearchEngine:
    """
    Main semantic search engine
    @nist-controls: SI-10, AC-4, AU-2
    @evidence: ML-based search with audit logging
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        index_path: Path | None = None
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index = VectorIndex()
        self.index_path = index_path
        self._is_indexed = False

        # Try to load existing index
        if index_path and index_path.exists():
            try:
                self.load_index()
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")

    def index_standards(self, standards: list[dict[str, Any]], chunk_size: int = 500):
        """
        Index standards content for semantic search

        Args:
            standards: List of standards with content
            chunk_size: Size of text chunks to index
        """
        logger.info(f"Indexing {len(standards)} standards")

        # Prepare texts and metadata for indexing
        texts = []
        metadata = []

        for standard in standards:
            # Index different parts of the standard
            standard_id = standard.get('id', 'unknown')

            # Index title and description
            if 'title' in standard:
                texts.append(standard['title'])
                metadata.append({
                    'standard_id': standard_id,
                    'type': 'title',
                    'content': standard['title']
                })

            if 'description' in standard:
                texts.append(standard['description'])
                metadata.append({
                    'standard_id': standard_id,
                    'type': 'description',
                    'content': standard['description']
                })

            # Index sections
            for section in standard.get('sections', []):
                section_text = f"{section.get('title', '')}\n{section.get('content', '')}"

                # Split into chunks if too large
                if len(section_text) > chunk_size * 4:  # Approximate chars
                    chunks = self._chunk_text(section_text, chunk_size)
                    for i, chunk in enumerate(chunks):
                        texts.append(chunk)
                        metadata.append({
                            'standard_id': standard_id,
                            'section_id': section.get('id'),
                            'type': 'section',
                            'chunk': i,
                            'content': chunk
                        })
                else:
                    texts.append(section_text)
                    metadata.append({
                        'standard_id': standard_id,
                        'section_id': section.get('id'),
                        'type': 'section',
                        'content': section_text
                    })

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} text chunks")
        embeddings = self.embedding_model.encode(texts)

        # Build index
        self.index.build(embeddings, metadata)
        self._is_indexed = True

        # Save index if path specified
        if self.index_path:
            self.save_index()

        logger.info("Indexing complete")

    def search(
        self,
        query: str,
        k: int = 10,
        filters: dict[str, Any] | None = None,
        min_score: float = 0.3
    ) -> list[SearchResult]:
        """
        Search for relevant content

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional filters (e.g., standard_id, type)
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        if not self._is_indexed:
            raise RuntimeError("No content indexed yet")

        # Encode query
        query_embedding = self.embedding_model.encode_single(query)

        # Search index
        results = self.index.search(query_embedding, k * 2)  # Get extra for filtering

        # Process results
        search_results = []
        for idx, score in results:
            if score < min_score:
                continue

            metadata = self.index.metadata[idx]

            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue

            result = SearchResult(
                content=metadata.get('content', ''),
                score=float(score),
                metadata=metadata,
                chunk_id=metadata.get('chunk_id'),
                standard_id=metadata.get('standard_id'),
                section_id=metadata.get('section_id')
            )
            search_results.append(result)

            if len(search_results) >= k:
                break

        return search_results

    def find_similar(self, text: str, k: int = 5) -> list[SearchResult]:
        """Find content similar to given text"""
        if not self._is_indexed:
            raise RuntimeError("No content indexed yet")

        # Encode text
        text_embedding = self.embedding_model.encode_single(text)

        # Search
        results = self.index.search(text_embedding, k)

        search_results = []
        for idx, score in results:
            metadata = self.index.metadata[idx]
            result = SearchResult(
                content=metadata.get('content', ''),
                score=float(score),
                metadata=metadata,
                standard_id=metadata.get('standard_id'),
                section_id=metadata.get('section_id')
            )
            search_results.append(result)

        return search_results

    def rerank_results(
        self,
        query: str,
        results: list[SearchResult],
        context: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        Rerank results based on additional context

        Args:
            query: Original query
            results: Initial search results
            context: Additional context (e.g., project type, language)

        Returns:
            Reranked results
        """
        if not context:
            return results

        # Apply context-based boosting
        for result in results:
            boost = 1.0

            # Boost based on project type match
            if 'project_type' in context:
                project_type = context['project_type'].lower()
                if project_type in result.content.lower():
                    boost *= 1.2

            # Boost based on language match
            if 'language' in context:
                language = context['language'].lower()
                if language in result.content.lower():
                    boost *= 1.1

            # Boost based on compliance level
            if 'compliance_level' in context:
                level = context['compliance_level'].lower()
                if level in ['high', 'strict'] and any(
                    keyword in result.content.lower()
                    for keyword in ['must', 'shall', 'required', 'mandatory']
                ):
                    boost *= 1.15

            result.score *= boost

        # Re-sort by score
        return sorted(results, key=lambda r: r.score, reverse=True)

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """Split text into chunks"""
        # Simple paragraph-based chunking
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk: list[str] = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para.split())

            if current_size + para_size > chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def save_index(self):
        """Save index to disk"""
        if not self.index_path:
            raise ValueError("No index path specified")

        logger.info(f"Saving index to {self.index_path}")
        self.index.save(self.index_path)

    def load_index(self):
        """Load index from disk"""
        if not self.index_path:
            raise ValueError("No index path specified")

        logger.info(f"Loading index from {self.index_path}")
        self.index.load(self.index_path)
        self._is_indexed = True

    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the index"""
        if not self._is_indexed:
            return {"indexed": False}

        # Count different types
        type_counts: dict[str, int] = {}
        for metadata in self.index.metadata:
            doc_type = metadata.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        return {
            "indexed": True,
            "total_documents": len(self.index.metadata),
            "dimension": self.index.dimension,
            "types": type_counts,
            "has_faiss": self.index.use_faiss
        }


class QueryExpander:
    """
    Expands queries with synonyms and related terms
    @nist-controls: SI-10
    @evidence: Query enhancement for better search
    """

    def __init__(self):
        self.expansions = {
            # Security terms
            "authentication": ["auth", "login", "identity", "credentials", "authn"],
            "authorization": ["authz", "access control", "permissions", "rbac", "acl"],
            "encryption": ["crypto", "cryptography", "cipher", "encrypt", "tls", "ssl"],
            "security": ["secure", "protection", "safeguard", "defense"],
            "compliance": ["compliant", "conform", "adherence", "regulatory"],

            # Development terms
            "api": ["interface", "endpoint", "rest", "graphql", "service"],
            "database": ["db", "storage", "persistence", "sql", "nosql"],
            "testing": ["test", "qa", "quality", "validation", "verification"],
            "logging": ["log", "audit", "monitoring", "telemetry", "observability"],

            # NIST specific
            "nist": ["800-53", "controls", "compliance framework"],
            "control": ["requirement", "safeguard", "measure", "countermeasure"],
            "risk": ["threat", "vulnerability", "exposure", "impact"],
        }

    def expand_query(self, query: str) -> str:
        """Expand query with related terms"""
        query_lower = query.lower()
        expanded_terms = [query]

        # Check each term in expansions
        for key, synonyms in self.expansions.items():
            if key in query_lower:
                # Add synonyms that aren't already in query
                for synonym in synonyms:
                    if synonym not in query_lower:
                        expanded_terms.append(synonym)

        # Also check if query matches any synonym
        for key, synonyms in self.expansions.items():
            for synonym in synonyms:
                if synonym in query_lower and key not in query_lower:
                    expanded_terms.append(key)

        return " ".join(expanded_terms)


# Convenience functions
def create_semantic_search_engine(
    model_name: str = "all-MiniLM-L6-v2",
    index_path: Path | None = None
) -> SemanticSearchEngine:
    """Create a semantic search engine"""
    embedding_model = EmbeddingModel(model_name)
    return SemanticSearchEngine(embedding_model, index_path)


def search_standards(
    query: str,
    standards: list[dict[str, Any]],
    k: int = 10,
    context: dict[str, Any] | None = None
) -> list[SearchResult]:
    """
    Quick search function for standards

    Args:
        query: Search query
        standards: Standards to search
        k: Number of results
        context: Optional context for reranking

    Returns:
        Search results
    """
    engine = create_semantic_search_engine()
    engine.index_standards(standards)

    # Expand query
    expander = QueryExpander()
    expanded_query = expander.expand_query(query)

    # Search
    results = engine.search(expanded_query, k)

    # Rerank if context provided
    if context:
        results = engine.rerank_results(query, results, context)

    return results
