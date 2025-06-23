"""
Unit tests for ChromaDB tier implementation.

@nist-controls: SA-11, CA-7
@evidence: Comprehensive testing of ChromaDB persistence tier
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.core.standards.chromadb_tier import ChromaDBTier
from src.core.standards.hybrid_vector_store import HybridConfig, SearchResult

# Check if ChromaDB is available
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

skipif_no_chromadb = pytest.mark.skipif(not CHROMADB_AVAILABLE, reason="ChromaDB not installed")


class TestChromaDBTier:
    """Test ChromaDB tier implementation"""

    @pytest.fixture
    def config(self):
        """Create test config"""
        return HybridConfig(
            chroma_path=".test_chroma",
            chroma_collection="test_collection",
            faiss_dimension=384
        )

    @pytest.fixture
    def mock_chromadb_client(self):
        """Create mock ChromaDB client"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_or_create_collection.return_value = mock_collection
        return mock_client, mock_collection

    @pytest.fixture
    def chromadb_tier(self, config):
        """Create ChromaDB tier with mocked client"""
        with patch('src.core.standards.chromadb_tier.chromadb') as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            
            tier = ChromaDBTier(config)
            tier._client = mock_client
            tier._collection = mock_collection
            
            return tier

    def test_initialization(self, config):
        """Test ChromaDB tier initialization"""
        with patch('src.core.standards.chromadb_tier.chromadb') as mock_chromadb:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 10
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            
            tier = ChromaDBTier(config)
            
            assert tier.config == config
            assert tier._client is not None
            assert tier._collection is not None
            
            # Verify client creation
            mock_chromadb.PersistentClient.assert_called_once()
            call_args = mock_chromadb.PersistentClient.call_args
            assert str(Path(config.chroma_path).absolute()) in str(call_args)
            
            # Verify collection creation
            mock_client.get_or_create_collection.assert_called_once_with(
                name=config.chroma_collection,
                metadata={"hnsw:space": "cosine"}
            )

    def test_initialization_import_error(self, config):
        """Test initialization when ChromaDB is not available"""
        with patch.dict('sys.modules', {'chromadb': None}):
            tier = ChromaDBTier(config)
            assert tier._client is None
            assert tier._collection is None

    def test_initialization_exception(self, config):
        """Test initialization with exception"""
        with patch('src.core.standards.chromadb_tier.chromadb') as mock_chromadb:
            mock_chromadb.PersistentClient.side_effect = Exception("Connection failed")
            
            tier = ChromaDBTier(config)
            assert tier._client is None
            assert tier._collection is None

    @pytest.mark.asyncio
    async def test_search_success(self, chromadb_tier):
        """Test successful search"""
        query_embedding = np.random.rand(384).astype(np.float32)
        
        # Mock ChromaDB query results
        chromadb_tier._collection.query.return_value = {
            'ids': [['doc1', 'doc2', 'doc3']],
            'documents': [['Content 1', 'Content 2', 'Content 3']],
            'metadatas': [[
                {'type': 'standard', 'category': 'security'},
                {'type': 'standard', 'category': 'development'},
                {'type': 'guide', 'category': 'security'}
            ]],
            'distances': [[0.1, 0.2, 0.3]]  # Lower distance = higher similarity
        }
        
        results = await chromadb_tier.search(query_embedding, k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        
        # Check first result
        assert results[0].id == 'doc1'
        assert results[0].content == 'Content 1'
        assert results[0].score == pytest.approx(0.9, rel=0.01)  # 1 - 0.1
        assert results[0].metadata['type'] == 'standard'
        assert results[0].source_tier == 'chromadb'
        
        # Verify query call
        chromadb_tier._collection.query.assert_called_once()
        query_args = chromadb_tier._collection.query.call_args
        assert query_args.kwargs['n_results'] == 3
        assert query_args.kwargs['where'] is None

    @pytest.mark.asyncio
    async def test_search_with_filters(self, chromadb_tier):
        """Test search with metadata filters"""
        query_embedding = np.random.rand(384).astype(np.float32)
        filters = {
            'type': 'standard',
            'category': ['security', 'compliance'],
            'version': {'$gte': '2.0'}
        }
        
        chromadb_tier._collection.query.return_value = {
            'ids': [['doc1']],
            'documents': [['Filtered content']],
            'metadatas': [[{'type': 'standard', 'category': 'security'}]],
            'distances': [[0.05]]
        }
        
        results = await chromadb_tier.search(query_embedding, k=5, filters=filters)
        
        assert len(results) == 1
        
        # Check where clause was built correctly
        query_args = chromadb_tier._collection.query.call_args
        where_clause = query_args.kwargs['where']
        assert where_clause is not None
        assert where_clause['type'] == {'$eq': 'standard'}
        assert where_clause['category'] == {'$in': ['security', 'compliance']}
        assert where_clause['version'] == {'$gte': '2.0'}

    @pytest.mark.asyncio
    async def test_search_no_results(self, chromadb_tier):
        """Test search with no results"""
        query_embedding = np.random.rand(384).astype(np.float32)
        
        chromadb_tier._collection.query.return_value = {
            'ids': [[]],
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = await chromadb_tier.search(query_embedding, k=5)
        
        assert results == []
        assert chromadb_tier.metrics.misses == 1

    @pytest.mark.asyncio
    async def test_search_no_collection(self, chromadb_tier):
        """Test search when collection is not initialized"""
        chromadb_tier._collection = None
        query_embedding = np.random.rand(384).astype(np.float32)
        
        results = await chromadb_tier.search(query_embedding, k=5)
        
        assert results == []
        assert chromadb_tier.metrics.misses == 1

    @pytest.mark.asyncio
    async def test_search_exception(self, chromadb_tier):
        """Test search with exception"""
        query_embedding = np.random.rand(384).astype(np.float32)
        chromadb_tier._collection.query.side_effect = Exception("Query failed")
        
        results = await chromadb_tier.search(query_embedding, k=5)
        
        assert results == []
        assert chromadb_tier.metrics.misses == 1

    @pytest.mark.asyncio
    async def test_add_success(self, chromadb_tier):
        """Test adding document successfully"""
        doc_id = "test_doc"
        embedding = np.random.rand(384).astype(np.float32)
        content = "Test document content"
        metadata = {
            'type': 'standard',
            'version': '1.0',
            'tags': ['security', 'compliance']
        }
        
        success = await chromadb_tier.add(doc_id, embedding, content, metadata)
        
        assert success is True
        
        # Verify add was called correctly
        chromadb_tier._collection.add.assert_called_once()
        add_args = chromadb_tier._collection.add.call_args
        assert add_args.kwargs['ids'] == [doc_id]
        assert len(add_args.kwargs['embeddings'][0]) == 384
        assert add_args.kwargs['documents'] == [content]
        assert add_args.kwargs['metadatas'][0]['type'] == 'standard'

    @pytest.mark.asyncio
    async def test_add_no_collection(self, chromadb_tier):
        """Test add when collection is not initialized"""
        chromadb_tier._collection = None
        
        success = await chromadb_tier.add(
            "doc1",
            np.random.rand(384),
            "content",
            {}
        )
        
        assert success is False

    @pytest.mark.asyncio
    async def test_add_exception(self, chromadb_tier):
        """Test add with exception"""
        chromadb_tier._collection.add.side_effect = Exception("Add failed")
        
        success = await chromadb_tier.add(
            "doc1",
            np.random.rand(384),
            "content",
            {}
        )
        
        assert success is False

    @pytest.mark.asyncio
    async def test_update_all_fields(self, chromadb_tier):
        """Test updating all fields of a document"""
        doc_id = "update_doc"
        new_embedding = np.random.rand(384).astype(np.float32)
        new_content = "Updated content"
        new_metadata = {'version': '2.0', 'updated': True}
        
        success = await chromadb_tier.update(
            doc_id,
            embedding=new_embedding,
            content=new_content,
            metadata=new_metadata
        )
        
        assert success is True
        
        # Verify update was called correctly
        chromadb_tier._collection.update.assert_called_once()
        update_args = chromadb_tier._collection.update.call_args
        assert update_args.kwargs['ids'] == [doc_id]
        assert 'embeddings' in update_args.kwargs
        assert update_args.kwargs['documents'] == [new_content]
        assert update_args.kwargs['metadatas'][0]['version'] == '2.0'

    @pytest.mark.asyncio
    async def test_update_partial_fields(self, chromadb_tier):
        """Test updating only some fields"""
        doc_id = "partial_update"
        new_metadata = {'status': 'reviewed'}
        
        success = await chromadb_tier.update(
            doc_id,
            metadata=new_metadata
        )
        
        assert success is True
        
        # Verify only metadata was updated
        update_args = chromadb_tier._collection.update.call_args
        assert update_args.kwargs['ids'] == [doc_id]
        assert 'embeddings' not in update_args.kwargs
        assert 'documents' not in update_args.kwargs
        assert 'metadatas' in update_args.kwargs

    @pytest.mark.asyncio
    async def test_update_exception(self, chromadb_tier):
        """Test update with exception"""
        chromadb_tier._collection.update.side_effect = Exception("Update failed")
        
        success = await chromadb_tier.update("doc1", content="new content")
        
        assert success is False

    @pytest.mark.asyncio
    async def test_remove_success(self, chromadb_tier):
        """Test removing document successfully"""
        doc_id = "remove_doc"
        
        success = await chromadb_tier.remove(doc_id)
        
        assert success is True
        chromadb_tier._collection.delete.assert_called_once_with(ids=[doc_id])

    @pytest.mark.asyncio
    async def test_remove_exception(self, chromadb_tier):
        """Test remove with exception"""
        chromadb_tier._collection.delete.side_effect = Exception("Delete failed")
        
        success = await chromadb_tier.remove("doc1")
        
        assert success is False

    @pytest.mark.asyncio
    async def test_get_by_ids_success(self, chromadb_tier):
        """Test retrieving documents by IDs"""
        ids = ['doc1', 'doc2', 'doc3']
        
        chromadb_tier._collection.get.return_value = {
            'ids': ['doc1', 'doc2'],  # doc3 not found
            'documents': ['Content 1', 'Content 2'],
            'metadatas': [
                {'type': 'standard'},
                {'type': 'guide'}
            ],
            'embeddings': [
                [0.1] * 384,
                [0.2] * 384
            ]
        }
        
        results = await chromadb_tier.get_by_ids(ids)
        
        assert len(results) == 2
        assert results[0].id == 'doc1'
        assert results[0].score == 1.0  # Direct retrieval
        assert results[1].id == 'doc2'
        assert results[1].source_tier == 'chromadb'

    @pytest.mark.asyncio
    async def test_get_by_ids_empty(self, chromadb_tier):
        """Test get_by_ids with no results"""
        chromadb_tier._collection.get.return_value = {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'embeddings': []
        }
        
        results = await chromadb_tier.get_by_ids(['nonexistent'])
        
        assert results == []

    @pytest.mark.asyncio
    async def test_get_stats(self, chromadb_tier):
        """Test getting statistics"""
        chromadb_tier._collection.count.return_value = 1000
        
        stats = await chromadb_tier.get_stats()
        
        assert stats['total_documents'] == 1000
        assert stats['collection_name'] == chromadb_tier.config.chroma_collection
        assert stats['persistent_path'] == chromadb_tier.config.chroma_path
        assert stats['status'] == 'connected'
        assert 'hits' in stats
        assert 'misses' in stats

    @pytest.mark.asyncio
    async def test_get_stats_no_collection(self, chromadb_tier):
        """Test get_stats when collection is not initialized"""
        chromadb_tier._collection = None
        
        stats = await chromadb_tier.get_stats()
        
        assert stats['status'] == 'not_initialized'

    @pytest.mark.asyncio
    async def test_get_stats_exception(self, chromadb_tier):
        """Test get_stats with exception"""
        chromadb_tier._collection.count.side_effect = Exception("Count failed")
        
        stats = await chromadb_tier.get_stats()
        
        assert 'error' in stats['status']

    def test_build_where_clause(self, chromadb_tier):
        """Test building ChromaDB where clause"""
        filters = {
            'type': 'standard',
            'categories': ['security', 'compliance'],
            'version': {'$gte': '2.0'},
            'active': True,
            'score': 0.9
        }
        
        where_clause = chromadb_tier._build_where_clause(filters)
        
        assert where_clause['type'] == {'$eq': 'standard'}
        assert where_clause['categories'] == {'$in': ['security', 'compliance']}
        assert where_clause['version'] == {'$gte': '2.0'}
        assert where_clause['active'] == {'$eq': True}
        assert where_clause['score'] == {'$eq': 0.9}

    def test_clean_metadata(self, chromadb_tier):
        """Test metadata cleaning"""
        metadata = {
            'string': 'value',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': ['a', 'b', 'c'],
            'none': None,
            'dict': {'nested': 'value'},
            'date': '2024-01-01',
            'complex_list': [1, 2.5, {'key': 'val'}]
        }
        
        clean = chromadb_tier._clean_metadata(metadata)
        
        assert clean['string'] == 'value'
        assert clean['int'] == 42
        assert clean['float'] == 3.14
        assert clean['bool'] is True
        assert clean['list'] == ['a', 'b', 'c']
        assert 'none' not in clean
        assert clean['dict'] == "{'nested': 'value'}"
        assert clean['date'] == '2024-01-01'
        assert clean['complex_list'] == ['1', '2.5', "{'key': 'val'}"]

    @pytest.mark.asyncio
    async def test_create_index(self, chromadb_tier):
        """Test creating index from standards data"""
        # Mock embedding model
        mock_embedding_model = AsyncMock()
        mock_embedding_model.encode.return_value = np.random.rand(384)
        
        standards_data = [
            {
                'id': f'std_{i}',
                'content': f'Standard content {i}',
                'metadata': {'type': 'standard', 'index': i}
            }
            for i in range(250)  # Test batching
        ]
        
        count = await chromadb_tier.create_index(standards_data, mock_embedding_model)
        
        assert count == 250
        
        # Should be called 3 times (batch size 100)
        assert chromadb_tier._collection.add.call_count == 3
        
        # Check first batch
        first_batch = chromadb_tier._collection.add.call_args_list[0]
        assert len(first_batch.kwargs['ids']) == 100
        assert len(first_batch.kwargs['embeddings']) == 100
        assert len(first_batch.kwargs['documents']) == 100

    @pytest.mark.asyncio
    async def test_create_index_no_collection(self, chromadb_tier):
        """Test create_index when collection is not initialized"""
        chromadb_tier._collection = None
        mock_embedding_model = AsyncMock()
        
        count = await chromadb_tier.create_index([], mock_embedding_model)
        
        assert count == 0

    @pytest.mark.asyncio
    async def test_create_index_batch_failure(self, chromadb_tier):
        """Test create_index with batch failure"""
        mock_embedding_model = AsyncMock()
        mock_embedding_model.encode.return_value = np.random.rand(384)
        
        # Fail on second batch
        chromadb_tier._collection.add.side_effect = [None, Exception("Batch failed"), None]
        
        standards_data = [
            {'id': f'std_{i}', 'content': f'Content {i}'}
            for i in range(250)
        ]
        
        count = await chromadb_tier.create_index(standards_data, mock_embedding_model)
        
        # Only first and third batch should succeed
        assert count == 200

    @pytest.mark.asyncio
    async def test_search_with_rerank(self, chromadb_tier):
        """Test search with reranking"""
        query_embedding = np.random.rand(384).astype(np.float32)
        query_text = "security authentication access control"
        
        # Mock initial search results
        initial_results = [
            SearchResult(
                id=f'doc{i}',
                score=0.9 - i * 0.1,
                content=content,
                metadata={},
                source_tier='chromadb'
            )
            for i, content in enumerate([
                "This is about database optimization",
                "Security and access control mechanisms",
                "Authentication methods for web apps",
                "General programming concepts",
                "Access control and authentication security"
            ])
        ]
        
        # Mock the search method
        with patch.object(chromadb_tier, 'search', return_value=initial_results):
            results = await chromadb_tier.search_with_rerank(
                query_embedding,
                query_text,
                k=3,
                rerank_top_n=5
            )
        
        assert len(results) == 3
        
        # Results should be reranked based on keyword matching
        # doc4 should rank higher due to all three keywords
        assert results[0].id == 'doc4'  # Has all keywords
        assert results[0].score > initial_results[4].score  # Score improved
        
        # Verify search was called with rerank_top_n
        chromadb_tier.search.assert_called_once_with(
            query_embedding, 5, None
        )

    @pytest.mark.asyncio
    async def test_search_with_rerank_few_results(self, chromadb_tier):
        """Test reranking when results are less than k"""
        query_embedding = np.random.rand(384).astype(np.float32)
        query_text = "test query"
        
        # Only 2 results, but k=5
        initial_results = [
            SearchResult('doc1', 0.9, 'Content 1', {}, 'chromadb'),
            SearchResult('doc2', 0.8, 'Content 2', {}, 'chromadb')
        ]
        
        with patch.object(chromadb_tier, 'search', return_value=initial_results):
            results = await chromadb_tier.search_with_rerank(
                query_embedding,
                query_text,
                k=5
            )
        
        # Should return original results without reranking
        assert results == initial_results


@skipif_no_chromadb
class TestChromaDBTierIntegration:
    """Integration tests with real ChromaDB (if available)"""

    @pytest.fixture
    def temp_chroma_path(self, tmp_path):
        """Create temporary ChromaDB path"""
        return tmp_path / "test_chroma"

    @pytest.fixture
    async def real_chromadb_tier(self, temp_chroma_path):
        """Create real ChromaDB tier"""
        config = HybridConfig(
            chroma_path=str(temp_chroma_path),
            chroma_collection="test_integration"
        )
        tier = ChromaDBTier(config)
        yield tier
        # Cleanup
        if tier._client:
            try:
                tier._client.delete_collection("test_integration")
            except:
                pass

    @pytest.mark.asyncio
    async def test_real_add_and_search(self, real_chromadb_tier):
        """Test real add and search operations"""
        # Add documents
        docs = [
            ("doc1", "Python programming security best practices", {"lang": "python", "type": "security"}),
            ("doc2", "JavaScript authentication patterns", {"lang": "javascript", "type": "auth"}),
            ("doc3", "Python authentication and authorization", {"lang": "python", "type": "auth"})
        ]
        
        for doc_id, content, metadata in docs:
            embedding = np.random.rand(384).astype(np.float32)
            success = await real_chromadb_tier.add(doc_id, embedding, content, metadata)
            assert success is True
        
        # Search without filters
        query_embedding = np.random.rand(384).astype(np.float32)
        results = await real_chromadb_tier.search(query_embedding, k=2)
        assert len(results) <= 2
        
        # Search with filters
        python_results = await real_chromadb_tier.search(
            query_embedding,
            k=5,
            filters={"lang": "python"}
        )
        assert all(r.metadata.get("lang") == "python" for r in python_results)
        
        # Get by IDs
        retrieved = await real_chromadb_tier.get_by_ids(["doc1", "doc3"])
        assert len(retrieved) == 2
        assert {r.id for r in retrieved} == {"doc1", "doc3"}

    @pytest.mark.asyncio
    async def test_real_update_and_remove(self, real_chromadb_tier):
        """Test real update and remove operations"""
        # Add a document
        doc_id = "update_test"
        embedding = np.random.rand(384).astype(np.float32)
        await real_chromadb_tier.add(
            doc_id,
            embedding,
            "Original content",
            {"version": "1.0"}
        )
        
        # Update it
        success = await real_chromadb_tier.update(
            doc_id,
            content="Updated content",
            metadata={"version": "2.0", "updated": True}
        )
        assert success is True
        
        # Retrieve and verify
        results = await real_chromadb_tier.get_by_ids([doc_id])
        assert len(results) == 1
        assert results[0].content == "Updated content"
        assert results[0].metadata["version"] == "2.0"
        
        # Remove it
        success = await real_chromadb_tier.remove(doc_id)
        assert success is True
        
        # Verify it's gone
        results = await real_chromadb_tier.get_by_ids([doc_id])
        assert len(results) == 0