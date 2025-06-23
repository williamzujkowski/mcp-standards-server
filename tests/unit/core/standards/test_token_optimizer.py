"""
Unit tests for token optimizer module
@nist-controls: SA-11, CA-7
@evidence: Token optimization testing
"""

import pytest

from src.core.standards.models import TokenOptimizationStrategy, StandardQuery
from src.core.standards.token_optimizer import (
    TokenOptimizer,
    OptimizationLevel,
    OptimizationMetrics,
    ContentSection
)


class TestTokenOptimizer:
    """Test TokenOptimizer class"""

    @pytest.fixture
    def optimizer(self):
        """Create token optimizer instance"""
        return TokenOptimizer()

    def test_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.tokenizer is not None
        assert hasattr(optimizer, 'optimize_tokens')

    def test_optimize_text_basic(self, optimizer):
        """Test basic text optimization"""
        text = "This is a test sentence with some words that could be optimized."
        
        optimized = optimizer.optimize_text(text, max_tokens=10)
        
        assert isinstance(optimized, str)
        assert len(optimized) <= len(text)
        # Should be shorter than original
        assert optimizer.tokenizer.count_tokens(optimized) <= 10

    def test_optimize_text_already_short(self, optimizer):
        """Test optimization of already short text"""
        text = "Short text"
        
        optimized = optimizer.optimize_text(text, max_tokens=100)
        
        # Should return original if already within limit
        assert optimized == text

    def test_optimize_text_empty(self, optimizer):
        """Test optimization of empty text"""
        optimized = optimizer.optimize_text("", max_tokens=10)
        assert optimized == ""

    def test_optimize_query_request(self, optimizer):
        """Test optimizing a query request"""
        request = StandardQuery(
            query="Find all security controls related to authentication and access control",
            filters={"category": ["security"]},
            max_tokens=50
        )
        
        optimized = optimizer.optimize_query(request)
        
        assert isinstance(optimized, StandardQuery)
        # Query should be optimized
        assert len(optimized.query) <= len(request.query)

    def test_optimize_with_strategy_summarize(self, optimizer):
        """Test optimization with SUMMARIZE strategy"""
        long_text = " ".join(["This is a sentence."] * 20)
        
        optimized = optimizer.optimize_with_strategy(
            long_text,
            TokenOptimizationStrategy.SUMMARIZE,
            max_tokens=20
        )
        
        assert isinstance(optimized, str)
        assert len(optimized) < len(long_text)
        assert optimizer.tokenizer.count_tokens(optimized) <= 20

    def test_optimize_with_strategy_essential_only(self, optimizer):
        """Test optimization with ESSENTIAL_ONLY strategy"""
        text = """
        The system MUST implement strong authentication.
        The system SHALL use encryption.
        Consider using additional security measures.
        The system MUST log all access attempts.
        """
        
        optimized = optimizer.optimize_with_strategy(
            text,
            TokenOptimizationStrategy.ESSENTIAL_ONLY,
            max_tokens=50
        )
        
        assert isinstance(optimized, str)
        # Should contain MUST/SHALL requirements
        assert "MUST" in optimized or "SHALL" in optimized
        # Should not contain optional text
        assert "Consider" not in optimized

    def test_optimize_with_strategy_hierarchical(self, optimizer):
        """Test optimization with HIERARCHICAL strategy"""
        text = """
        # Main Topic
        
        ## Subtopic 1
        Details about subtopic 1
        
        ## Subtopic 2
        Details about subtopic 2
        
        ### Sub-subtopic
        More detailed information
        """
        
        optimized = optimizer.optimize_with_strategy(
            text,
            TokenOptimizationStrategy.HIERARCHICAL,
            max_tokens=30
        )
        
        assert isinstance(optimized, str)
        # Should preserve hierarchy
        assert "#" in optimized
        assert len(optimized) < len(text)

    def test_optimize_with_strategy_progressive(self, optimizer):
        """Test optimization with PROGRESSIVE strategy"""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        
        optimized = optimizer.optimize_with_strategy(
            text,
            TokenOptimizationStrategy.PROGRESSIVE,
            max_tokens=10
        )
        
        assert isinstance(optimized, str)
        assert len(optimized) < len(text)
        # Should include at least first sentence
        assert "First" in optimized

    def test_optimize_standards_content(self, optimizer):
        """Test optimizing standards content"""
        content = {
            "title": "Security Standard",
            "description": "A comprehensive security standard with many requirements",
            "sections": [
                {
                    "title": "Authentication",
                    "content": "The system MUST implement multi-factor authentication"
                },
                {
                    "title": "Authorization",
                    "content": "The system SHALL implement role-based access control"
                }
            ]
        }
        
        optimized = optimizer.optimize_standards_content(content, max_tokens=50)
        
        assert isinstance(optimized, dict)
        assert "title" in optimized
        assert "sections" in optimized
        # Content should be reduced
        total_text = str(optimized)
        assert len(total_text) < len(str(content))

    def test_calculate_token_reduction(self, optimizer):
        """Test calculating token reduction"""
        original = "This is a long text with many words that will be optimized"
        optimized = "This is optimized text"
        
        reduction = optimizer.calculate_token_reduction(original, optimized)
        
        assert isinstance(reduction, dict)
        assert "original_tokens" in reduction
        assert "optimized_tokens" in reduction
        assert "reduction_percentage" in reduction
        assert reduction["reduction_percentage"] > 0

    def test_extract_key_phrases(self, optimizer):
        """Test extracting key phrases"""
        text = """
        The system MUST implement authentication.
        Users SHALL be verified.
        Access control is REQUIRED.
        """
        
        phrases = optimizer.extract_key_phrases(text)
        
        assert isinstance(phrases, list)
        assert len(phrases) > 0
        # Should extract requirement keywords
        assert any("MUST" in phrase for phrase in phrases)

    def test_compress_whitespace(self, optimizer):
        """Test whitespace compression"""
        text = "This    has     extra    whitespace\n\n\nand newlines"
        
        compressed = optimizer.compress_whitespace(text)
        
        assert "    " not in compressed
        assert "\n\n\n" not in compressed
        assert len(compressed) < len(text)

    def test_remove_redundant_words(self, optimizer):
        """Test removing redundant words"""
        text = "The the system shall shall implement the the security"
        
        cleaned = optimizer.remove_redundant_words(text)
        
        # Should remove duplicate words
        assert "the the" not in cleaned.lower()
        assert "shall shall" not in cleaned

    def test_truncate_to_sentences(self, optimizer):
        """Test truncating to complete sentences"""
        text = "First sentence. Second sentence. Third sentence that is incomplete"
        
        truncated = optimizer.truncate_to_sentences(text, max_tokens=5)
        
        assert isinstance(truncated, str)
        assert truncated.endswith(".")
        assert "incomplete" not in truncated

    def test_optimize_json_content(self, optimizer):
        """Test optimizing JSON content"""
        json_content = {
            "key1": "value1",
            "key2": "A very long value that contains lots of unnecessary information",
            "nested": {
                "subkey": "Another long value with redundant content"
            }
        }
        
        optimized = optimizer.optimize_json_content(json_content, max_tokens=20)
        
        assert isinstance(optimized, dict)
        # Structure should be preserved
        assert "key1" in optimized
        # Long values should be shortened
        if "key2" in optimized:
            assert len(str(optimized["key2"])) < len(json_content["key2"])

    def test_optimize_list_content(self, optimizer):
        """Test optimizing list content"""
        items = [
            "First item with lots of detail",
            "Second item with even more unnecessary detail",
            "Third item",
            "Fourth item with redundant information",
            "Fifth item"
        ]
        
        optimized = optimizer.optimize_list_content(items, max_tokens=20)
        
        assert isinstance(optimized, list)
        assert len(optimized) <= len(items)
        # Should preserve most important items
        assert len(optimized) > 0

    def test_create_summary(self, optimizer):
        """Test creating a summary"""
        text = """
        This is a comprehensive document about security standards.
        It contains multiple sections covering various aspects.
        The main topics include authentication, authorization, and auditing.
        Each section provides detailed requirements and guidelines.
        Implementation examples are provided throughout.
        """
        
        summary = optimizer.create_summary(text, max_tokens=20)
        
        assert isinstance(summary, str)
        assert len(summary) < len(text)
        # Should mention key topics
        assert any(word in summary.lower() for word in ["security", "authentication", "authorization"])

    def test_prioritize_content(self, optimizer):
        """Test content prioritization"""
        sections = [
            {"priority": "high", "content": "Critical security requirement"},
            {"priority": "low", "content": "Optional recommendation"},
            {"priority": "medium", "content": "Standard requirement"},
            {"priority": "high", "content": "Another critical requirement"}
        ]
        
        prioritized = optimizer.prioritize_content(sections, max_tokens=30)
        
        assert isinstance(prioritized, list)
        # High priority items should be included
        high_priority_found = any(
            "Critical" in item.get("content", "") 
            for item in prioritized
        )
        assert high_priority_found
        # Low priority might be excluded
        assert len(prioritized) <= len(sections)

    def test_optimize_with_context_preservation(self, optimizer):
        """Test optimization while preserving context"""
        text = """
        @context: security-standard-v2.0
        @nist-controls: AC-3, AU-2
        
        The system MUST implement access control.
        Additional details about implementation...
        """
        
        optimized = optimizer.optimize_with_context(text, max_tokens=30)
        
        assert isinstance(optimized, str)
        # Should preserve metadata
        assert "@context" in optimized
        assert "@nist-controls" in optimized
        # Should include key requirement
        assert "MUST" in optimized

    def test_batch_optimization(self, optimizer):
        """Test batch optimization of multiple texts"""
        texts = [
            "First document with lots of content",
            "Second document with even more content",
            "Third short doc"
        ]
        
        optimized_batch = optimizer.batch_optimize(texts, max_tokens=10)
        
        assert isinstance(optimized_batch, list)
        assert len(optimized_batch) == len(texts)
        # Each should be optimized
        for original, optimized in zip(texts, optimized_batch):
            assert len(optimized) <= len(original)

    def test_adaptive_optimization(self, optimizer):
        """Test adaptive optimization based on content type"""
        # Code content
        code_text = """
        def authenticate(user, password):
            # Check user credentials
            if user and password:
                return True
            return False
        """
        
        optimized_code = optimizer.adaptive_optimize(code_text, "code", max_tokens=20)
        assert isinstance(optimized_code, str)
        
        # Prose content
        prose_text = "This is a long narrative description of security requirements."
        optimized_prose = optimizer.adaptive_optimize(prose_text, "prose", max_tokens=10)
        assert isinstance(optimized_prose, str)
        assert len(optimized_prose) < len(prose_text)

    def test_optimize_with_importance_scores(self, optimizer):
        """Test optimization using importance scores"""
        sentences = [
            ("The system MUST authenticate users.", 1.0),
            ("Consider using biometric authentication.", 0.3),
            ("The system SHALL log access attempts.", 0.9),
            ("Additional options are available.", 0.2)
        ]
        
        optimized = optimizer.optimize_with_scores(sentences, max_tokens=20)
        
        assert isinstance(optimized, str)
        # High importance sentences should be included
        assert "MUST" in optimized
        # Low importance might be excluded
        assert "Additional options" not in optimized

    def test_progressive_detail_levels(self, optimizer):
        """Test progressive detail level optimization"""
        content = {
            "overview": "Security standard overview",
            "details": {
                "level1": "Basic requirements",
                "level2": "Detailed requirements with examples",
                "level3": "Complete implementation guide with code samples"
            }
        }
        
        # Minimal detail
        minimal = optimizer.get_progressive_detail(content, "minimal", max_tokens=10)
        assert isinstance(minimal, str)
        assert len(minimal) < 50
        
        # Medium detail
        medium = optimizer.get_progressive_detail(content, "medium", max_tokens=30)
        assert isinstance(medium, str)
        assert len(medium) > len(minimal)
        
        # Full detail
        full = optimizer.get_progressive_detail(content, "full", max_tokens=100)
        assert isinstance(full, str)
        assert len(full) > len(medium)

    def test_optimize_preserving_structure(self, optimizer):
        """Test optimization while preserving document structure"""
        structured_content = """
        # Title
        
        ## Section 1
        Content for section 1
        
        ## Section 2
        Content for section 2
        
        ### Subsection 2.1
        Detailed content
        """
        
        optimized = optimizer.optimize_preserving_structure(structured_content, max_tokens=30)
        
        assert isinstance(optimized, str)
        # Should preserve headers
        assert "# Title" in optimized
        assert "##" in optimized
        # But content should be reduced
        assert len(optimized) < len(structured_content)

    def test_intelligent_truncation(self, optimizer):
        """Test intelligent truncation"""
        text = """
        The system MUST implement authentication. [CRITICAL]
        Users should have unique identifiers. [RECOMMENDED]
        The system MUST use encryption. [CRITICAL]
        Consider implementing SSO. [OPTIONAL]
        """
        
        truncated = optimizer.intelligent_truncate(text, max_tokens=20)
        
        assert isinstance(truncated, str)
        # Should keep critical items
        assert "[CRITICAL]" in truncated
        # Might drop optional items
        token_count = optimizer.tokenizer.count_tokens(truncated)
        assert token_count <= 20

    def test_optimize_with_fallback(self, optimizer):
        """Test optimization with fallback strategies"""
        text = "A" * 1000  # Very long repetitive text
        
        # Should try multiple strategies and not fail
        optimized = optimizer.optimize_with_fallback(text, max_tokens=50)
        
        assert isinstance(optimized, str)
        assert len(optimized) < len(text)
        assert optimizer.tokenizer.count_tokens(optimized) <= 50

    def test_error_handling(self, optimizer):
        """Test error handling in optimization"""
        # Invalid inputs
        assert optimizer.optimize_text(None, 10) == ""
        assert optimizer.optimize_text(123, 10) == "123"
        
        # Negative token limit
        result = optimizer.optimize_text("test", -1)
        assert result == "test"  # Should return original
        
        # Empty strategies
        result = optimizer.optimize_with_strategy("test", None, 10)
        assert result == "test"  # Should return original