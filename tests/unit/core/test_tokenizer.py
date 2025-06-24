"""
Tests for tokenizer module
"""

from unittest.mock import Mock, patch

import pytest

from src.core.tokenizer import (
    ApproximateTokenizer,
    BaseTokenizer,
    TiktokenTokenizer,
    TokenBudget,
    TokenizerFactory,
    TokenizerType,
    TransformersTokenizer,
    count_tokens,
    get_default_tokenizer,
    split_to_chunks,
    truncate_to_tokens,
)


class TestBaseTokenizer:
    """Test the BaseTokenizer abstract class"""

    def test_abstract_methods(self):
        """Test that BaseTokenizer is abstract"""
        with pytest.raises(TypeError):
            BaseTokenizer()

    def test_concrete_implementation(self):
        """Test that concrete implementations work"""
        class ConcreteTokenizer(BaseTokenizer):
            def encode(self, text: str) -> list[int]:
                return [1, 2, 3]

            def decode(self, tokens: list[int]) -> str:
                return "test"

            def count_tokens(self, text: str) -> int:
                return 3

            def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
                return text[:max_tokens]

            def split_to_chunks(self, text: str, chunk_size: int, overlap: int = 0) -> list[str]:
                return [text]

        tokenizer = ConcreteTokenizer()
        assert tokenizer.encode("test") == [1, 2, 3]
        assert tokenizer.decode([1, 2, 3]) == "test"
        assert tokenizer.count_tokens("test") == 3


class TestTiktokenTokenizer:
    """Test the TiktokenTokenizer implementation"""

    @patch('src.core.tokenizer.tiktoken')
    def test_initialization_success(self, mock_tiktoken):
        """Test successful initialization"""
        mock_encoding = Mock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        tokenizer = TiktokenTokenizer()
        assert tokenizer.encoding == mock_encoding
        mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-3.5-turbo")

    @patch('src.core.tokenizer.tiktoken')
    def test_initialization_with_model(self, mock_tiktoken):
        """Test initialization with custom model"""
        mock_encoding = Mock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        TiktokenTokenizer(model="gpt-4")
        mock_tiktoken.encoding_for_model.assert_called_once_with("gpt-4")

    @patch('src.core.tokenizer.tiktoken')
    def test_initialization_fallback(self, mock_tiktoken):
        """Test fallback when model encoding fails"""
        mock_tiktoken.encoding_for_model.side_effect = Exception("Model not found")
        mock_encoding = Mock()
        mock_tiktoken.get_encoding.return_value = mock_encoding

        tokenizer = TiktokenTokenizer()
        assert tokenizer.encoding == mock_encoding
        mock_tiktoken.get_encoding.assert_called_once_with("cl100k_base")

    @patch('src.core.tokenizer.tiktoken')
    def test_encode(self, mock_tiktoken):
        """Test encode method"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        tokenizer = TiktokenTokenizer()
        result = tokenizer.encode("test text")
        assert result == [1, 2, 3]
        mock_encoding.encode.assert_called_once_with("test text")

    @patch('src.core.tokenizer.tiktoken')
    def test_decode(self, mock_tiktoken):
        """Test decode method"""
        mock_encoding = Mock()
        mock_encoding.decode.return_value = "decoded text"
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        tokenizer = TiktokenTokenizer()
        result = tokenizer.decode([1, 2, 3])
        assert result == "decoded text"
        mock_encoding.decode.assert_called_once_with([1, 2, 3])

    @patch('src.core.tokenizer.tiktoken')
    def test_count_tokens(self, mock_tiktoken):
        """Test count_tokens method"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        tokenizer = TiktokenTokenizer()
        result = tokenizer.count_tokens("test text")
        assert result == 5

    @patch('src.core.tokenizer.tiktoken')
    def test_truncate_to_tokens(self, mock_tiktoken):
        """Test truncate_to_tokens method"""
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoding.decode.return_value = "truncated"
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        tokenizer = TiktokenTokenizer()
        result = tokenizer.truncate_to_tokens("test text", 3)
        assert result == "truncated"
        mock_encoding.decode.assert_called_once_with([1, 2, 3])

    @patch('src.core.tokenizer.tiktoken')
    def test_split_to_chunks(self, mock_tiktoken):
        """Test split_to_chunks method"""
        mock_encoding = Mock()
        # Mock a longer text that needs chunking
        mock_encoding.encode.return_value = list(range(10))
        mock_encoding.decode.side_effect = lambda tokens: f"chunk{len(tokens)}"
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        tokenizer = TiktokenTokenizer()
        result = tokenizer.split_to_chunks("long text", chunk_size=3, overlap=1)

        # Should create chunks of size 3 with overlap 1
        assert len(result) == 4  # Chunks: [0,1,2], [2,3,4], [4,5,6], [6,7,8,9]
        assert all(chunk.startswith("chunk") for chunk in result)


class TestTransformersTokenizer:
    """Test the TransformersTokenizer implementation"""

    @patch('src.core.tokenizer.AutoTokenizer')
    def test_initialization_success(self, mock_auto_tokenizer):
        """Test successful initialization"""
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = TransformersTokenizer()
        assert tokenizer.tokenizer == mock_tokenizer
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("bert-base-uncased")

    @patch('src.core.tokenizer.AutoTokenizer')
    def test_initialization_with_model(self, mock_auto_tokenizer):
        """Test initialization with custom model"""
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        TransformersTokenizer(model_name="gpt2")
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("gpt2")

    @patch('src.core.tokenizer.AutoTokenizer')
    def test_initialization_failure(self, mock_auto_tokenizer):
        """Test initialization failure"""
        mock_auto_tokenizer.from_pretrained.side_effect = Exception("Model not found")

        with pytest.raises(Exception):
            TransformersTokenizer()

    @patch('src.core.tokenizer.AutoTokenizer')
    def test_encode(self, mock_auto_tokenizer):
        """Test encode method"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [101, 102, 103]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = TransformersTokenizer()
        result = tokenizer.encode("test text")
        assert result == [101, 102, 103]
        mock_tokenizer.encode.assert_called_once_with("test text", add_special_tokens=True)

    @patch('src.core.tokenizer.AutoTokenizer')
    def test_decode(self, mock_auto_tokenizer):
        """Test decode method"""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "decoded text"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = TransformersTokenizer()
        result = tokenizer.decode([101, 102, 103])
        assert result == "decoded text"
        mock_tokenizer.decode.assert_called_once_with([101, 102, 103], skip_special_tokens=True)

    @patch('src.core.tokenizer.AutoTokenizer')
    def test_count_tokens(self, mock_auto_tokenizer):
        """Test count_tokens method"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [101, 102, 103, 104]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = TransformersTokenizer()
        result = tokenizer.count_tokens("test text")
        assert result == 4

    @patch('src.core.tokenizer.AutoTokenizer')
    def test_truncate_to_tokens(self, mock_auto_tokenizer):
        """Test truncate_to_tokens method"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [101, 102, 103, 104, 105]
        mock_tokenizer.decode.return_value = "truncated"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = TransformersTokenizer()
        result = tokenizer.truncate_to_tokens("test text", 3)
        assert result == "truncated"
        mock_tokenizer.decode.assert_called_once_with([101, 102, 103], skip_special_tokens=True)

    @patch('src.core.tokenizer.AutoTokenizer')
    def test_split_to_chunks(self, mock_auto_tokenizer):
        """Test split_to_chunks method"""
        mock_tokenizer = Mock()
        # Mock a longer text that needs chunking
        mock_tokenizer.encode.return_value = list(range(10))
        def decode_side_effect(tokens, skip_special_tokens=True):
            _ = skip_special_tokens  # Use the parameter
            return f"chunk{len(tokens)}"
        mock_tokenizer.decode.side_effect = decode_side_effect
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = TransformersTokenizer()
        result = tokenizer.split_to_chunks("long text", chunk_size=3, overlap=1)

        # Should create chunks
        assert len(result) > 1
        assert all(chunk.startswith("chunk") for chunk in result)


class TestApproximateTokenizer:
    """Test the ApproximateTokenizer implementation"""

    def test_initialization(self):
        """Test initialization with default parameters"""
        tokenizer = ApproximateTokenizer()
        assert tokenizer.tokens_per_word == 1.3
        assert tokenizer.chars_per_token == 4.0

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters"""
        tokenizer = ApproximateTokenizer(tokens_per_word=1.5, chars_per_token=5.0)
        assert tokenizer.tokens_per_word == 1.5
        assert tokenizer.chars_per_token == 5.0

    def test_encode(self):
        """Test encode method"""
        tokenizer = ApproximateTokenizer()
        result = tokenizer.encode("test")
        # With 1.3 tokens per word, "test" (1 word) should be ~1 token
        assert len(result) == 1

        result = tokenizer.encode("test text here")
        # 3 words * 1.3 = 3.9, rounded to 3 tokens
        assert len(result) == 3

    def test_decode(self):
        """Test decode method"""
        tokenizer = ApproximateTokenizer()
        with pytest.raises(NotImplementedError):
            tokenizer.decode([0, 1, 2])

    def test_count_tokens(self):
        """Test count_tokens method"""
        tokenizer = ApproximateTokenizer()

        assert tokenizer.count_tokens("") == 0
        # "test" = 1 word * 1.3 = 1.3 word estimate, 4 chars / 4 = 1 char estimate -> avg = 1
        assert tokenizer.count_tokens("test") == 1
        # "test text" = 2 words * 1.3 = 2.6 -> 2, 9 chars / 4 = 2.25 -> 2, avg = 2
        assert tokenizer.count_tokens("test text") == 2

    def test_truncate_to_tokens(self):
        """Test truncate_to_tokens method"""
        tokenizer = ApproximateTokenizer()

        # Test truncation
        result = tokenizer.truncate_to_tokens("test text here and more words", 2)
        # Should truncate to approximately 2 tokens worth
        assert len(result) < len("test text here and more words")

        # Test no truncation needed
        result = tokenizer.truncate_to_tokens("short", 10)
        assert result == "short"

    def test_split_to_chunks(self):
        """Test split_to_chunks method"""
        tokenizer = ApproximateTokenizer(chars_per_token=4)

        # Test without overlap
        text = "a" * 20  # 20 chars = 5 tokens
        result = tokenizer.split_to_chunks(text, chunk_size=2, overlap=0)
        # 2 tokens = 8 chars per chunk
        assert result == ["a" * 8, "a" * 8, "a" * 4]

        # Test with overlap
        result = tokenizer.split_to_chunks(text, chunk_size=2, overlap=1)
        # Chunks should overlap by 1 token = 4 chars
        assert len(result) == 5  # More chunks due to overlap

        # Test empty text
        result = tokenizer.split_to_chunks("", chunk_size=10)
        assert result == []


class TestTokenizerFactory:
    """Test the TokenizerFactory class"""

    def teardown_method(self):
        """Clear the factory cache after each test"""
        TokenizerFactory._instances.clear()

    @patch('src.core.tokenizer.tiktoken')
    def test_get_tiktoken_tokenizer(self, mock_tiktoken):
        """Test getting tiktoken tokenizer"""
        mock_encoding = Mock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        tokenizer = TokenizerFactory.get_tokenizer(TokenizerType.TIKTOKEN)
        assert isinstance(tokenizer, TiktokenTokenizer)

    @patch('src.core.tokenizer.AutoTokenizer')
    def test_get_transformers_tokenizer(self, mock_auto_tokenizer):
        """Test getting transformers tokenizer"""
        mock_tokenizer = Mock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        tokenizer = TokenizerFactory.get_tokenizer(TokenizerType.TRANSFORMERS)
        assert isinstance(tokenizer, TransformersTokenizer)

    def test_get_approximate_tokenizer(self):
        """Test getting approximate tokenizer"""
        tokenizer = TokenizerFactory.get_tokenizer(TokenizerType.APPROXIMATE)
        assert isinstance(tokenizer, ApproximateTokenizer)

    def test_get_tokenizer_with_kwargs(self):
        """Test getting tokenizer with kwargs"""
        tokenizer = TokenizerFactory.get_tokenizer(TokenizerType.APPROXIMATE, chars_per_token=5)
        assert isinstance(tokenizer, ApproximateTokenizer)
        assert tokenizer.chars_per_token == 5

    def test_get_tokenizer_invalid_type(self):
        """Test getting tokenizer with invalid type"""
        with pytest.raises(ValueError):
            TokenizerFactory.get_tokenizer("invalid_type")

    @patch('src.core.tokenizer.tiktoken')
    def test_caching(self, mock_tiktoken):
        """Test that tokenizers are cached"""
        mock_encoding = Mock()
        mock_tiktoken.encoding_for_model.return_value = mock_encoding

        tokenizer1 = TokenizerFactory.get_tokenizer(TokenizerType.TIKTOKEN)
        tokenizer2 = TokenizerFactory.get_tokenizer(TokenizerType.TIKTOKEN)
        assert tokenizer1 is tokenizer2

    @patch('src.core.tokenizer.tiktoken')
    def test_fallback_on_import_error(self, mock_tiktoken):
        """Test fallback to approximate tokenizer on import error"""
        mock_tiktoken.encoding_for_model.side_effect = ImportError("tiktoken not installed")

        tokenizer = TokenizerFactory.get_tokenizer(TokenizerType.TIKTOKEN)
        assert isinstance(tokenizer, ApproximateTokenizer)

    def test_create_with_fallback(self):
        """Test create_with_fallback method"""
        with patch('src.core.tokenizer.tiktoken') as mock_tiktoken:
            mock_tiktoken.encoding_for_model.side_effect = ImportError()

            tokenizer = TokenizerFactory.create_with_fallback()
            assert isinstance(tokenizer, ApproximateTokenizer)


class TestConvenienceFunctions:
    """Test the convenience functions"""

    def test_get_default_tokenizer(self):
        """Test get_default_tokenizer function"""
        tokenizer = get_default_tokenizer()
        assert isinstance(tokenizer, BaseTokenizer)

    def test_count_tokens_with_default(self):
        """Test count_tokens function with default tokenizer"""
        count = count_tokens("test text")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_with_custom_tokenizer(self):
        """Test count_tokens function with custom tokenizer"""
        mock_tokenizer = Mock()
        mock_tokenizer.count_tokens.return_value = 42

        count = count_tokens("test text", tokenizer=mock_tokenizer)
        assert count == 42
        mock_tokenizer.count_tokens.assert_called_once_with("test text")

    def test_truncate_to_tokens_with_default(self):
        """Test truncate_to_tokens function with default tokenizer"""
        result = truncate_to_tokens("test text here", 2)
        assert isinstance(result, str)
        assert len(result) <= len("test text here")

    def test_truncate_to_tokens_with_custom_tokenizer(self):
        """Test truncate_to_tokens function with custom tokenizer"""
        mock_tokenizer = Mock()
        mock_tokenizer.truncate_to_tokens.return_value = "truncated"

        result = truncate_to_tokens("test text", 5, tokenizer=mock_tokenizer)
        assert result == "truncated"
        mock_tokenizer.truncate_to_tokens.assert_called_once_with("test text", 5)

    def test_split_to_chunks_with_default(self):
        """Test split_to_chunks function with default tokenizer"""
        chunks = split_to_chunks("test text here and more", 2)
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, str) for chunk in chunks)

    def test_split_to_chunks_with_custom_tokenizer(self):
        """Test split_to_chunks function with custom tokenizer"""
        mock_tokenizer = Mock()
        mock_tokenizer.split_to_chunks.return_value = ["chunk1", "chunk2"]

        chunks = split_to_chunks("test text", 5, overlap=1, tokenizer=mock_tokenizer)
        assert chunks == ["chunk1", "chunk2"]
        mock_tokenizer.split_to_chunks.assert_called_once_with("test text", 5, 1)


class TestTokenBudget:
    """Test the TokenBudget class"""

    def test_initialization(self):
        """Test initialization"""
        budget = TokenBudget(1000)
        assert budget.total_tokens == 1000
        assert budget.allocated_tokens == 0
        assert budget.allocations == {}

    def test_allocate_success(self):
        """Test successful allocation"""
        budget = TokenBudget(1000)
        assert budget.allocate("component1", 100)
        assert budget.allocated_tokens == 100
        assert budget.allocations["component1"] == 100

    def test_allocate_failure(self):
        """Test allocation failure when exceeding budget"""
        budget = TokenBudget(100)
        assert budget.allocate("component1", 50)
        assert not budget.allocate("component2", 60)  # Would exceed budget
        assert budget.allocated_tokens == 50  # Should not change

    def test_allocate_for_text(self):
        """Test allocation based on text"""
        mock_tokenizer = Mock()
        mock_tokenizer.count_tokens.return_value = 25

        budget = TokenBudget(100, tokenizer=mock_tokenizer)
        success, tokens = budget.allocate_for_text("test", "test text here")

        assert success
        assert tokens == 25
        assert budget.allocated_tokens == 25
        mock_tokenizer.count_tokens.assert_called_once_with("test text here")

    def test_remaining(self):
        """Test remaining tokens calculation"""
        budget = TokenBudget(100)
        budget.allocate("test", 30)
        assert budget.remaining() == 70

    def test_usage_percentage(self):
        """Test usage percentage calculation"""
        budget = TokenBudget(100)
        budget.allocate("test", 25)
        assert budget.usage_percentage() == 25.0

        # Test with zero budget
        budget = TokenBudget(0)
        assert budget.usage_percentage() == 0

    def test_get_allocation_summary(self):
        """Test allocation summary"""
        budget = TokenBudget(100)
        budget.allocate("comp1", 30)
        budget.allocate("comp2", 20)

        summary = budget.get_allocation_summary()
        assert summary["total_budget"] == 100
        assert summary["allocated"] == 50
        assert summary["remaining"] == 50
        assert summary["usage_percentage"] == 50.0
        assert summary["allocations"] == {"comp1": 30, "comp2": 20}
