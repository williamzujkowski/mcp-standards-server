"""
ChromaDB tier for persistent vector storage with rich metadata filtering.

@nist-controls: SC-28, SI-12, AU-12
@evidence: Persistent vector storage with metadata filtering and audit trails
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..logging import get_logger

logger = get_logger(__name__)

# Import from hybrid_vector_store - avoiding circular import
from .hybrid_vector_store import (
    HybridConfig,
    SearchResult,
    TierMetrics,
    VectorStoreTier,
)


class ChromaDBTier(VectorStoreTier):
    """ChromaDB-based persistent storage for full corpus with metadata."""
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.metrics = TierMetrics()
        self._client = None
        self._collection = None
        self._init_chromadb()
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create persistent client
            self._client = chromadb.PersistentClient(
                path=str(Path(self.config.chroma_path).absolute()),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.config.chroma_collection,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Initialized ChromaDB collection '{self.config.chroma_collection}' "
                       f"with {self._collection.count()} documents")
            
        except ImportError:
            logger.warning("ChromaDB not available, persistent storage disabled")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
    
    async def search(self, query_embedding: np.ndarray, k: int = 10,
                    filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search in ChromaDB with metadata filtering."""
        if not self._collection:
            self.metrics.record_miss()
            return []
        
        start_time = time.time()
        
        try:
            # Prepare where clause for metadata filtering
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)
            
            # Query ChromaDB
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
            
            # Convert to SearchResult format
            search_results = []
            if results['ids'][0]:  # Check if we got results
                for i, (id, doc, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (1 - distance for cosine)
                    score = 1.0 - distance
                    
                    search_results.append(SearchResult(
                        id=id,
                        score=float(score),
                        content=doc,
                        metadata=metadata or {},
                        source_tier="chromadb"
                    ))
            
            latency = time.time() - start_time
            
            if search_results:
                self.metrics.record_hit(latency)
            else:
                self.metrics.record_miss()
            
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            self.metrics.record_miss()
            return []
    
    async def add(self, id: str, embedding: np.ndarray,
                 content: str, metadata: Dict[str, Any]) -> bool:
        """Add document to ChromaDB with metadata."""
        if not self._collection:
            return False
        
        try:
            # Ensure metadata is serializable
            clean_metadata = self._clean_metadata(metadata)
            
            # Add to collection
            self._collection.add(
                ids=[id],
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[clean_metadata]
            )
            
            logger.debug(f"Added document {id} to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB add error: {e}")
            return False
    
    async def update(self, id: str, embedding: Optional[np.ndarray] = None,
                    content: Optional[str] = None, 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update existing document in ChromaDB."""
        if not self._collection:
            return False
        
        try:
            update_kwargs = {"ids": [id]}
            
            if embedding is not None:
                update_kwargs["embeddings"] = [embedding.tolist()]
            
            if content is not None:
                update_kwargs["documents"] = [content]
            
            if metadata is not None:
                update_kwargs["metadatas"] = [self._clean_metadata(metadata)]
            
            self._collection.update(**update_kwargs)
            logger.debug(f"Updated document {id} in ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB update error: {e}")
            return False
    
    async def remove(self, id: str) -> bool:
        """Remove document from ChromaDB."""
        if not self._collection:
            return False
        
        try:
            self._collection.delete(ids=[id])
            logger.debug(f"Removed document {id} from ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"ChromaDB remove error: {e}")
            return False
    
    async def get_by_ids(self, ids: List[str]) -> List[SearchResult]:
        """Retrieve documents by IDs."""
        if not self._collection:
            return []
        
        try:
            results = self._collection.get(
                ids=ids,
                include=["metadatas", "documents", "embeddings"]
            )
            
            search_results = []
            for i, id in enumerate(results['ids']):
                if results['documents'][i]:  # Check if document exists
                    search_results.append(SearchResult(
                        id=id,
                        score=1.0,  # Direct retrieval, max score
                        content=results['documents'][i],
                        metadata=results['metadatas'][i] or {},
                        source_tier="chromadb"
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"ChromaDB get error: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics."""
        stats = self.metrics.get_stats()
        
        if self._collection:
            try:
                stats.update({
                    "total_documents": self._collection.count(),
                    "collection_name": self.config.chroma_collection,
                    "persistent_path": self.config.chroma_path,
                    "status": "connected"
                })
            except Exception as e:
                stats["status"] = f"error: {e}"
        else:
            stats["status"] = "not_initialized"
        
        return stats
    
    def _build_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Use $in operator for list values
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                # Pass through complex operators
                where_clause[key] = value
            else:
                # Simple equality
                where_clause[key] = {"$eq": value}
        
        return where_clause
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Clean metadata to ensure it's serializable for ChromaDB."""
        clean = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                # Convert list items to strings if necessary
                clean[key] = [str(v) for v in value]
            else:
                # Convert other types to string
                clean[key] = str(value)
        
        return clean
    
    async def create_index(self, standards_data: List[Dict[str, Any]],
                          embedding_model: Any) -> int:
        """
        Create or update index with standards data.
        
        @nist-controls: SI-12, AU-12
        @evidence: Batch indexing with audit logging
        """
        if not self._collection:
            logger.error("ChromaDB not initialized")
            return 0
        
        indexed_count = 0
        batch_size = 100
        
        for i in range(0, len(standards_data), batch_size):
            batch = standards_data[i:i + batch_size]
            
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for item in batch:
                # Generate embedding
                embedding = await embedding_model.encode(item['content'])
                
                ids.append(item['id'])
                embeddings.append(embedding.tolist())
                documents.append(item['content'])
                metadatas.append(self._clean_metadata(item.get('metadata', {})))
            
            try:
                # Add batch to ChromaDB
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                indexed_count += len(ids)
                logger.info(f"Indexed batch {i//batch_size + 1}: {len(ids)} documents")
                
            except Exception as e:
                logger.error(f"Failed to index batch: {e}")
        
        logger.info(f"Successfully indexed {indexed_count} documents to ChromaDB")
        return indexed_count
    
    async def search_with_rerank(self, query_embedding: np.ndarray, 
                                query_text: str, k: int = 10,
                                filters: Optional[Dict[str, Any]] = None,
                                rerank_top_n: int = 20) -> List[SearchResult]:
        """
        Search with reranking for better relevance.
        
        @nist-controls: SI-10
        @evidence: Advanced search with relevance optimization
        """
        # Get more candidates for reranking
        candidates = await self.search(query_embedding, rerank_top_n, filters)
        
        if not candidates or len(candidates) <= k:
            return candidates
        
        # Simple reranking based on keyword matching
        # In production, use a cross-encoder model
        query_terms = set(query_text.lower().split())
        
        for candidate in candidates:
            content_terms = set(candidate.content.lower().split())
            keyword_score = len(query_terms & content_terms) / len(query_terms)
            
            # Combine embedding score with keyword score
            candidate.score = 0.7 * candidate.score + 0.3 * keyword_score
        
        # Re-sort by combined score
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        return candidates[:k]