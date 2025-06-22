"""
Comprehensive tests for semantic_search module
@nist-controls: SA-11, CA-7
@evidence: Semantic search engine testing
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.standards.semantic_search import (
    EmbeddingModel,
    QueryExpander,
    SearchResult,
    SemanticSearchEngine,
    VectorIndex,
    create_semantic_search_engine,
    search_standards,
)


class TestSearchResult:
    """Test SearchResult dataclass"""

    def test_search_result_creation(self):
        """Test creating a search result"""
        result = SearchResult(
            content="Test content",
            score=0.95,
            metadata={"type": "section"}
        )

        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata == {"type": "section"}
        assert result.chunk_id is None
        assert result.standard_id is None
        assert result.section_id is None

    def test_search_result_with_all_fields(self):
        """Test search result with all fields"""
        result = SearchResult(
            content="Full content",
            score=0.88,
            metadata={"type": "title", "lang": "en"},
            chunk_id="chunk_001",
            standard_id="std_001",
            section_id="sec_001"
        )

        assert result.chunk_id == "chunk_001"
        assert result.standard_id == "std_001"
        assert result.section_id == "sec_001"
        assert result.metadata["lang"] == "en"


class TestEmbeddingModel:
    """Test EmbeddingModel class"""

    @patch('src.core.standards.semantic_search.SentenceTransformer')
    def test_initialization_success(self, mock_transformer):
        """Test successful model initialization"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model

        model = EmbeddingModel("test-model")

        assert model.model_name == "test-model"
        assert model.model == mock_model
        mock_transformer.assert_called_once_with("test-model")

    @patch('src.core.standards.semantic_search.SentenceTransformer', None)
    def test_initialization_import_error(self):
        """Test initialization with missing sentence-transformers"""
        with pytest.raises(ImportError) as exc_info:
            EmbeddingModel()

        assert "sentence-transformers" in str(exc_info.value)

    @patch('src.core.standards.semantic_search.SentenceTransformer')
    def test_initialization_model_error(self, mock_transformer):
        """Test initialization with model loading error"""
        mock_transformer.side_effect = Exception("Model not found")

        with pytest.raises(Exception) as exc_info:
            EmbeddingModel("invalid-model")

        assert "Model not found" in str(exc_info.value)

    @patch('src.core.standards.semantic_search.SentenceTransformer')
    def test_encode_texts(self, mock_transformer):
        """Test encoding multiple texts"""
        mock_model = MagicMock()
        mock_embeddings = np.random.rand(3, 384)
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model

        model = EmbeddingModel()
        texts = ["text1", "text2", "text3"]
        embeddings = model.encode(texts, batch_size=16)

        assert embeddings.shape == (3, 384)
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    @patch('src.core.standards.semantic_search.SentenceTransformer')
    def test_encode_single(self, mock_transformer):
        """Test encoding single text"""
        mock_model = MagicMock()
        mock_embedding = np.random.rand(1, 384)
        mock_model.encode.return_value = mock_embedding
        mock_transformer.return_value = mock_model

        model = EmbeddingModel()
        embedding = model.encode_single("test text")

        assert embedding.shape == (384,)
        mock_model.encode.assert_called_once()

    def test_encode_without_model(self):
        """Test encoding without initialized model"""
        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model = None

        with pytest.raises(RuntimeError) as exc_info:
            model.encode(["test"])

        assert "not initialized" in str(exc_info.value)


class TestVectorIndex:
    """Test VectorIndex class"""

    def test_initialization(self):
        """Test vector index initialization"""
        index = VectorIndex(dimension=384)

        assert index.dimension == 384
        assert index.index is None
        assert index.embeddings is None
        assert index.metadata == []
        assert isinstance(index.use_faiss, bool)

    def test_build_numpy_index(self):
        """Test building index with numpy (no FAISS)"""
        index = VectorIndex()
        index.use_faiss = False

        embeddings = np.random.rand(5, 384)
        metadata = [{"id": i} for i in range(5)]

        index.build(embeddings, metadata)

        assert index.dimension == 384
        assert len(index.metadata) == 5
        assert index.embeddings is not None
        assert index.embeddings.shape == (5, 384)
        # Check normalization
        norms = np.linalg.norm(index.embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(5), rtol=1e-6)

    @patch('src.core.standards.semantic_search.faiss')
    def test_build_faiss_index(self, mock_faiss):
        """Test building index with FAISS"""
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        index = VectorIndex()
        index.use_faiss = True
        index.faiss = mock_faiss

        embeddings = np.random.rand(5, 384)
        metadata = [{"id": i} for i in range(5)]

        index.build(embeddings, metadata)

        assert index.dimension == 384
        assert len(index.metadata) == 5
        mock_faiss.IndexFlatIP.assert_called_once_with(384)
        mock_index.add.assert_called_once()

        # Check that embeddings were normalized
        call_args = mock_index.add.call_args[0][0]
        assert call_args.dtype == np.float32

    def test_build_mismatched_lengths(self):
        """Test building with mismatched embeddings and metadata"""
        index = VectorIndex()

        embeddings = np.random.rand(5, 384)
        metadata = [{"id": i} for i in range(3)]  # Wrong length

        with pytest.raises(ValueError) as exc_info:
            index.build(embeddings, metadata)

        assert "same length" in str(exc_info.value)

    def test_search_numpy(self):
        """Test searching with numpy index"""
        index = VectorIndex()
        index.use_faiss = False

        # Build index
        embeddings = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.7, 0.7, 0],
            [0, 0.7, 0.7]
        ])
        metadata = [{"id": i} for i in range(5)]
        index.build(embeddings, metadata)

        # Search for vector similar to first
        query = np.array([0.9, 0.1, 0])
        results = index.search(query, k=3)

        assert len(results) == 3
        # First result should be index 0 (most similar)
        assert results[0][0] == 0
        assert results[0][1] > 0.9  # High similarity

    @patch('src.core.standards.semantic_search.faiss')
    def test_search_faiss(self, mock_faiss):
        """Test searching with FAISS index"""
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.95, 0.85, 0.75]]),
            np.array([[0, 3, 1]])
        )

        index = VectorIndex()
        index.use_faiss = True
        index.faiss = mock_faiss
        index.index = mock_index

        query = np.array([1, 0, 0])
        results = index.search(query, k=3)

        assert len(results) == 3
        assert results[0] == (0, 0.95)
        assert results[1] == (3, 0.85)
        assert results[2] == (1, 0.75)

    def test_search_without_index(self):
        """Test searching without built index"""
        index = VectorIndex()

        with pytest.raises(RuntimeError) as exc_info:
            index.search(np.array([1, 0, 0]), k=5)

        assert "not built" in str(exc_info.value)

    def test_save_load_numpy(self, tmp_path):
        """Test saving and loading numpy index"""
        # Create and build index
        index = VectorIndex()
        index.use_faiss = False

        embeddings = np.random.rand(3, 128)
        metadata = [{"id": i, "type": "test"} for i in range(3)]
        index.build(embeddings, metadata)

        # Save
        index.save(tmp_path / "test_index")

        # Check files exist
        assert (tmp_path / "test_index" / "metadata.json").exists()
        assert (tmp_path / "test_index" / "embeddings.npy").exists()

        # Load into new index
        new_index = VectorIndex()
        new_index.use_faiss = False
        new_index.load(tmp_path / "test_index")

        assert new_index.dimension == 128
        assert len(new_index.metadata) == 3
        assert new_index.metadata[0]["id"] == 0
        assert new_index.embeddings is not None

    @patch('src.core.standards.semantic_search.faiss')
    def test_save_load_faiss(self, mock_faiss, tmp_path):
        """Test saving and loading FAISS index"""
        # Create and build index
        mock_index = MagicMock()
        mock_index.d = 128

        index = VectorIndex()
        index.use_faiss = True
        index.faiss = mock_faiss
        index.index = mock_index
        index.dimension = 128
        index.metadata = [{"id": i} for i in range(3)]

        # Save
        index.save(tmp_path / "test_index")

        # Check metadata saved
        assert (tmp_path / "test_index" / "metadata.json").exists()
        mock_faiss.write_index.assert_called_once()

        # Load into new index
        mock_faiss.read_index.return_value = mock_index
        new_index = VectorIndex()
        new_index.use_faiss = True
        new_index.faiss = mock_faiss
        new_index.load(tmp_path / "test_index")

        assert new_index.dimension == 128
        assert len(new_index.metadata) == 3


class TestSemanticSearchEngine:
    """Test SemanticSearchEngine class"""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create mock embedding model"""
        model = MagicMock(spec=EmbeddingModel)
        model.encode.return_value = np.random.rand(5, 384)
        model.encode_single.return_value = np.random.rand(384)
        return model

    @pytest.fixture
    def sample_standards(self):
        """Create sample standards for testing"""
        return [
            {
                "id": "std_001",
                "title": "Security Standard",
                "description": "Standard for application security",
                "sections": [
                    {
                        "id": "sec_001",
                        "title": "Authentication",
                        "content": "Users must be authenticated before access"
                    },
                    {
                        "id": "sec_002",
                        "title": "Authorization",
                        "content": "Access control must be implemented"
                    }
                ]
            },
            {
                "id": "std_002",
                "title": "Development Standard",
                "description": "Best practices for development",
                "sections": [
                    {
                        "id": "sec_003",
                        "title": "Testing",
                        "content": "All code must have unit tests"
                    }
                ]
            }
        ]

    def test_initialization(self, mock_embedding_model):
        """Test search engine initialization"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        assert engine.embedding_model == mock_embedding_model
        assert isinstance(engine.index, VectorIndex)
        assert engine._is_indexed is False

    def test_initialization_with_index_path(self, mock_embedding_model, tmp_path):
        """Test initialization with index path"""
        index_path = tmp_path / "test_index"
        engine = SemanticSearchEngine(
            embedding_model=mock_embedding_model,
            index_path=index_path
        )

        assert engine.index_path == index_path

    def test_index_standards(self, mock_embedding_model, sample_standards):
        """Test indexing standards"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        # Mock encode to return proper shaped array
        num_texts = 7  # 2 titles + 2 descriptions + 3 sections
        mock_embedding_model.encode.return_value = np.random.rand(num_texts, 384)

        engine.index_standards(sample_standards)

        assert engine._is_indexed is True
        mock_embedding_model.encode.assert_called_once()

        # Check that texts were extracted
        call_args = mock_embedding_model.encode.call_args[0][0]
        assert len(call_args) == num_texts
        assert "Security Standard" in call_args
        assert "Authentication" in call_args[0] or "Authentication" in ' '.join(call_args)

    def test_index_standards_with_chunking(self, mock_embedding_model):
        """Test indexing with large sections that need chunking"""
        large_content = " ".join(["word"] * 1000)  # Large content
        standards = [{
            "id": "std_large",
            "title": "Large Standard",
            "sections": [{
                "id": "sec_large",
                "title": "Large Section",
                "content": large_content
            }]
        }]

        # Mock encode to handle variable number of chunks
        mock_embedding_model.encode.return_value = np.random.rand(10, 384)

        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)
        engine.index_standards(standards, chunk_size=100)

        assert engine._is_indexed is True
        # Should have created multiple chunks
        assert len(engine.index.metadata) > 2  # More than just title and section

    def test_search(self, mock_embedding_model, sample_standards):
        """Test searching indexed content"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        # Index standards
        num_texts = 7
        mock_embedding_model.encode.return_value = np.random.rand(num_texts, 384)
        engine.index_standards(sample_standards)

        # Mock search
        engine.index = MagicMock()
        engine.index.search.return_value = [
            (0, 0.95),
            (1, 0.85),
            (2, 0.75),
            (3, 0.65)
        ]
        engine.index.metadata = [
            {"content": "Result 1", "standard_id": "std_001", "type": "title"},
            {"content": "Result 2", "standard_id": "std_001", "type": "section"},
            {"content": "Result 3", "standard_id": "std_002", "type": "title"},
            {"content": "Result 4", "standard_id": "std_002", "type": "section"}
        ]

        results = engine.search("authentication", k=3)

        assert len(results) == 3
        assert results[0].score == 0.95
        assert results[0].content == "Result 1"
        assert results[0].standard_id == "std_001"

    def test_search_with_filters(self, mock_embedding_model, sample_standards):
        """Test searching with filters"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        # Index standards
        mock_embedding_model.encode.return_value = np.random.rand(7, 384)
        engine.index_standards(sample_standards)

        # Mock search
        engine.index = MagicMock()
        engine.index.search.return_value = [
            (0, 0.95),
            (1, 0.85),
            (2, 0.75),
            (3, 0.65)
        ]
        engine.index.metadata = [
            {"content": "Title 1", "standard_id": "std_001", "type": "title"},
            {"content": "Section 1", "standard_id": "std_001", "type": "section"},
            {"content": "Title 2", "standard_id": "std_002", "type": "title"},
            {"content": "Section 2", "standard_id": "std_002", "type": "section"}
        ]

        # Filter by standard_id
        results = engine.search("test", k=10, filters={"standard_id": "std_001"})

        assert len(results) == 2
        assert all(r.standard_id == "std_001" for r in results)

    def test_search_with_min_score(self, mock_embedding_model, sample_standards):
        """Test searching with minimum score threshold"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        # Index standards
        mock_embedding_model.encode.return_value = np.random.rand(7, 384)
        engine.index_standards(sample_standards)

        # Mock search with low scores
        engine.index = MagicMock()
        engine.index.search.return_value = [
            (0, 0.85),
            (1, 0.25),  # Below threshold
            (2, 0.15),  # Below threshold
            (3, 0.65)
        ]
        engine.index.metadata = [
            {"content": f"Result {i}", "standard_id": "std_001"}
            for i in range(4)
        ]

        results = engine.search("test", k=10, min_score=0.5)

        assert len(results) == 2
        assert all(r.score >= 0.5 for r in results)

    def test_search_not_indexed(self, mock_embedding_model):
        """Test searching without indexing"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        with pytest.raises(RuntimeError) as exc_info:
            engine.search("test")

        assert "No content indexed" in str(exc_info.value)

    def test_find_similar(self, mock_embedding_model, sample_standards):
        """Test finding similar content"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        # Index standards
        mock_embedding_model.encode.return_value = np.random.rand(7, 384)
        engine.index_standards(sample_standards)

        # Mock search
        engine.index = MagicMock()
        engine.index.search.return_value = [(0, 0.98), (1, 0.92)]
        engine.index.metadata = [
            {"content": "Similar 1", "standard_id": "std_001"},
            {"content": "Similar 2", "standard_id": "std_002"}
        ]

        results = engine.find_similar("test content", k=2)

        assert len(results) == 2
        assert results[0].score == 0.98
        mock_embedding_model.encode_single.assert_called_once_with("test content")

    def test_rerank_results(self, mock_embedding_model):
        """Test reranking results with context"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        # Create initial results
        results = [
            SearchResult(
                content="Python authentication library",
                score=0.8,
                metadata={},
                standard_id="std_001"
            ),
            SearchResult(
                content="Java security framework",
                score=0.85,
                metadata={},
                standard_id="std_002"
            ),
            SearchResult(
                content="General security guidelines must be followed",
                score=0.75,
                metadata={},
                standard_id="std_003"
            )
        ]

        # Rerank with context
        context = {
            "language": "python",
            "project_type": "web",
            "compliance_level": "high"
        }

        reranked = engine.rerank_results("authentication", results, context)

        # Python result should be boosted
        assert reranked[0].content == "Python authentication library"
        assert reranked[0].score > 0.8  # Boosted

        # High compliance with "must" should be boosted
        high_compliance_result = next(r for r in reranked if "must" in r.content)
        assert high_compliance_result.score > 0.75  # Boosted

    def test_chunk_text(self, mock_embedding_model):
        """Test text chunking"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        text = "\n\n".join([
            "Paragraph 1 with some content.",
            "Paragraph 2 with more content.",
            "Paragraph 3 with even more content.",
            "Paragraph 4 with additional content."
        ])

        chunks = engine._chunk_text(text, chunk_size=10)

        assert len(chunks) > 1
        assert all("\n\n" not in chunk or chunk.count("\n\n") < 4 for chunk in chunks)

    def test_save_load_index(self, mock_embedding_model, sample_standards, tmp_path):
        """Test saving and loading index"""
        index_path = tmp_path / "test_index"
        engine = SemanticSearchEngine(
            embedding_model=mock_embedding_model,
            index_path=index_path
        )

        # Index standards
        mock_embedding_model.encode.return_value = np.random.rand(7, 384)
        engine.index_standards(sample_standards)

        # Save index
        engine.save_index()
        assert index_path.exists()

        # Create new engine and load
        new_engine = SemanticSearchEngine(
            embedding_model=mock_embedding_model,
            index_path=index_path
        )
        assert new_engine._is_indexed is True

    def test_get_index_stats(self, mock_embedding_model, sample_standards):
        """Test getting index statistics"""
        engine = SemanticSearchEngine(embedding_model=mock_embedding_model)

        # Stats before indexing
        stats = engine.get_index_stats()
        assert stats["indexed"] is False

        # Index standards
        mock_embedding_model.encode.return_value = np.random.rand(7, 384)
        engine.index_standards(sample_standards)

        # Stats after indexing
        stats = engine.get_index_stats()
        assert stats["indexed"] is True
        assert stats["total_documents"] == 7
        assert "types" in stats
        assert stats["types"]["title"] == 2  # Two titles
        assert stats["types"]["description"] == 2  # Two descriptions
        assert stats["types"]["section"] == 3  # Three sections


class TestQueryExpander:
    """Test QueryExpander class"""

    def test_initialization(self):
        """Test query expander initialization"""
        expander = QueryExpander()

        assert isinstance(expander.expansions, dict)
        assert "authentication" in expander.expansions
        assert "encryption" in expander.expansions
        assert "nist" in expander.expansions

    def test_expand_query_direct_match(self):
        """Test expanding query with direct term match"""
        expander = QueryExpander()

        expanded = expander.expand_query("authentication system")

        assert "authentication" in expanded
        assert "auth" in expanded
        assert "login" in expanded
        assert "identity" in expanded

    def test_expand_query_synonym_match(self):
        """Test expanding query with synonym match"""
        expander = QueryExpander()

        expanded = expander.expand_query("auth module")

        assert "auth" in expanded
        assert "authentication" in expanded  # Reverse expansion

    def test_expand_query_multiple_terms(self):
        """Test expanding query with multiple expandable terms"""
        expander = QueryExpander()

        expanded = expander.expand_query("api security")

        assert "api" in expanded
        assert "interface" in expanded
        assert "security" in expanded
        assert "secure" in expanded

    def test_expand_query_no_expansion(self):
        """Test query with no expandable terms"""
        expander = QueryExpander()

        expanded = expander.expand_query("unusual term xyz")

        assert expanded == "unusual term xyz"

    def test_expand_query_case_insensitive(self):
        """Test case insensitive expansion"""
        expander = QueryExpander()

        expanded = expander.expand_query("AUTHENTICATION API")

        assert "auth" in expanded
        assert "interface" in expanded

    def test_no_duplicate_expansion(self):
        """Test that existing terms aren't duplicated"""
        expander = QueryExpander()

        expanded = expander.expand_query("authentication auth login")

        # Count occurrences
        terms = expanded.split()
        assert terms.count("auth") == 1
        assert terms.count("authentication") == 1
        assert terms.count("login") == 1


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    @patch('src.core.standards.semantic_search.EmbeddingModel')
    def test_create_semantic_search_engine(self, mock_embedding_class):
        """Test creating search engine with convenience function"""
        mock_model = MagicMock()
        mock_embedding_class.return_value = mock_model

        engine = create_semantic_search_engine(
            model_name="test-model",
            index_path=Path("/tmp/index")
        )

        assert isinstance(engine, SemanticSearchEngine)
        assert engine.embedding_model == mock_model
        assert engine.index_path == Path("/tmp/index")
        mock_embedding_class.assert_called_once_with("test-model")

    @patch('src.core.standards.semantic_search.create_semantic_search_engine')
    def test_search_standards(self, mock_create_engine):
        """Test quick search function"""
        # Mock engine
        mock_engine = MagicMock()
        mock_results = [
            SearchResult(content="Result 1", score=0.9, metadata={}),
            SearchResult(content="Result 2", score=0.8, metadata={})
        ]
        mock_engine.search.return_value = mock_results
        mock_engine.rerank_results.return_value = mock_results
        mock_create_engine.return_value = mock_engine

        # Test data
        standards = [{"id": "std_001", "title": "Test Standard"}]
        context = {"language": "python"}

        # Search
        results = search_standards(
            query="security",
            standards=standards,
            k=5,
            context=context
        )

        assert len(results) == 2
        assert results[0].score == 0.9

        # Verify calls
        mock_engine.index_standards.assert_called_once_with(standards)
        mock_engine.search.assert_called_once()
        mock_engine.rerank_results.assert_called_once_with("security", mock_results, context)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_standards_indexing(self):
        """Test indexing empty standards list"""
        engine = SemanticSearchEngine()
        engine.embedding_model = MagicMock()
        engine.embedding_model.encode.return_value = np.array([]).reshape(0, 384)

        engine.index_standards([])

        assert engine._is_indexed is True
        assert len(engine.index.metadata) == 0

    def test_malformed_standards(self):
        """Test handling malformed standards"""
        engine = SemanticSearchEngine()
        engine.embedding_model = MagicMock()
        engine.embedding_model.encode.return_value = np.random.rand(1, 384)

        malformed_standards = [
            {"id": "std_001"},  # Missing title and sections
            {"title": "No ID"},  # Missing id
            {"id": "std_002", "sections": None}  # None sections
        ]

        # Should handle gracefully
        engine.index_standards(malformed_standards)
        assert engine._is_indexed is True

    def test_special_characters_in_content(self):
        """Test handling special characters"""
        engine = SemanticSearchEngine()
        engine.embedding_model = MagicMock()
        engine.embedding_model.encode.return_value = np.random.rand(2, 384)

        standards = [{
            "id": "std_001",
            "title": "Special <>&\" Characters",
            "sections": [{
                "id": "sec_001",
                "content": "Content with émojis 🔒 and símbolos ñ"
            }]
        }]

        # Should handle without errors
        engine.index_standards(standards)
        assert engine._is_indexed is True

    def test_very_long_content(self):
        """Test handling very long content"""
        engine = SemanticSearchEngine()
        engine.embedding_model = MagicMock()

        # Create very long content
        long_content = " ".join(["word"] * 10000)
        standards = [{
            "id": "std_long",
            "sections": [{
                "id": "sec_long",
                "content": long_content
            }]
        }]

        # Mock to handle multiple chunks
        engine.embedding_model.encode.return_value = np.random.rand(50, 384)

        engine.index_standards(standards, chunk_size=100)

        assert engine._is_indexed is True
        # Should have created multiple chunks
        section_chunks = [m for m in engine.index.metadata if m.get("type") == "section"]
        assert len(section_chunks) > 1

    def test_concurrent_indexing_protection(self):
        """Test protection against concurrent indexing"""
        engine = SemanticSearchEngine()
        engine.embedding_model = MagicMock()
        engine.embedding_model.encode.return_value = np.random.rand(1, 384)

        # Index once
        engine.index_standards([{"id": "std_001", "title": "Test"}])

        # Index again (should replace)
        engine.index_standards([{"id": "std_002", "title": "Test 2"}])

        assert engine._is_indexed is True
        # Should have new content
        assert any("std_002" in str(m) for m in engine.index.metadata)
