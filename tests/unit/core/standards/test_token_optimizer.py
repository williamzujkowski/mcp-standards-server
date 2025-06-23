"""
Unit tests for token optimizer module
@nist-controls: SA-11, CA-7
@evidence: Token optimization testing
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.core.standards.models import TokenOptimizationStrategy
from src.core.standards.token_optimizer import (
    ContentSection,
    EssentialOnlyStrategy,
    HierarchicalStrategy,
    OptimizationLevel,
    OptimizationMetrics,
    SummarizationStrategy,
    TokenOptimizationEngine,
)


class TestOptimizationMetrics:
    """Test OptimizationMetrics dataclass"""

    def test_metrics_creation(self):
        """Test creating optimization metrics"""
        metrics = OptimizationMetrics(
            original_tokens=1000,
            optimized_tokens=100,
            reduction_percentage=90.0,
            information_retained=0.85,
            processing_time=0.5
        )

        assert metrics.original_tokens == 1000
        assert metrics.optimized_tokens == 100
        assert metrics.reduction_percentage == 90.0
        assert metrics.information_retained == 0.85
        assert metrics.processing_time == 0.5


class TestContentSection:
    """Test ContentSection model"""

    def test_content_section_creation(self):
        """Test creating content section"""
        section = ContentSection(
            title="Test Section",
            content="Test content",
            importance=0.9,
            keywords=["test", "section"],
            concepts=["testing"],
            requirements=["Must test"],
            examples=["Example 1"],
            token_count=10
        )

        assert section.title == "Test Section"
        assert section.content == "Test content"
        assert section.importance == 0.9
        assert len(section.keywords) == 2
        assert len(section.concepts) == 1
        assert len(section.requirements) == 1
        assert len(section.examples) == 1
        assert section.token_count == 10


class TestTokenOptimizationEngine:
    """Test TokenOptimizationEngine class"""

    @pytest.fixture
    def engine(self):
        """Create token optimization engine instance"""
        return TokenOptimizationEngine()

    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.tokenizer is not None
        assert "summarize" in engine.strategies
        assert "essential" in engine.strategies
        assert "hierarchical" in engine.strategies
        assert isinstance(engine._metrics_history, list)

    @pytest.mark.asyncio
    async def test_optimize_with_summarize_strategy(self, engine):
        """Test optimization with summarize strategy"""
        content = " ".join(["This is a test sentence."] * 50)
        
        optimized, metrics = await engine.optimize(
            content,
            strategy="summarize",
            max_tokens=50,
            level=OptimizationLevel.MODERATE
        )
        
        assert isinstance(optimized, str)
        assert isinstance(metrics, OptimizationMetrics)
        assert metrics.optimized_tokens <= 50
        assert metrics.reduction_percentage > 0
        assert len(optimized) < len(content)

    @pytest.mark.asyncio
    async def test_optimize_with_essential_strategy(self, engine):
        """Test optimization with essential strategy"""
        content = """
        The system MUST implement authentication.
        The system SHALL use encryption.
        Consider using additional security measures.
        The system REQUIRED to log all access.
        @nist-controls: AC-3, AU-2
        """
        
        optimized, metrics = await engine.optimize(
            content,
            strategy="essential",
            max_tokens=100,
            level=OptimizationLevel.AGGRESSIVE
        )
        
        assert isinstance(optimized, str)
        assert isinstance(metrics, OptimizationMetrics)
        # Should contain essential requirements
        assert "MUST" in optimized or "SHALL" in optimized or "Requirements" in optimized
        assert metrics.reduction_percentage > 0

    @pytest.mark.asyncio
    async def test_optimize_with_hierarchical_strategy(self, engine):
        """Test optimization with hierarchical strategy"""
        content = """
        # Main Topic
        ## Subtopic 1
        Details about subtopic 1
        ## Subtopic 2
        Details about subtopic 2
        ### Sub-subtopic
        More detailed information
        """
        
        optimized, metrics = await engine.optimize(
            content,
            strategy="hierarchical",
            max_tokens=50,
            level=OptimizationLevel.MINIMAL
        )
        
        assert isinstance(optimized, str)
        assert isinstance(metrics, OptimizationMetrics)
        # Should preserve hierarchy
        assert "#" in optimized
        assert len(optimized) < len(content)

    @pytest.mark.asyncio
    async def test_optimize_with_invalid_strategy(self, engine):
        """Test optimization with invalid strategy"""
        with pytest.raises(ValueError) as exc_info:
            await engine.optimize(
                "test content",
                strategy="invalid_strategy",
                max_tokens=50
            )
        
        assert "Unknown strategy" in str(exc_info.value)

    def test_estimate_tokens(self, engine):
        """Test token estimation"""
        content = "This is a test sentence with several words."
        
        token_count = engine.estimate_tokens(content)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        # Rough estimate: should be less than word count * 2
        assert token_count < len(content.split()) * 2

    def test_get_metrics_summary_empty(self, engine):
        """Test getting metrics summary with no history"""
        summary = engine.get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert len(summary) == 0

    @pytest.mark.asyncio
    async def test_get_metrics_summary_with_history(self, engine):
        """Test getting metrics summary with history"""
        # Run some optimizations to build history
        await engine.optimize("Test content 1", strategy="summarize", max_tokens=10)
        await engine.optimize("Test content 2", strategy="essential", max_tokens=10)
        
        summary = engine.get_metrics_summary()
        
        assert isinstance(summary, dict)
        assert "average_reduction" in summary
        assert "average_retention" in summary
        assert "average_processing_time" in summary
        assert summary["average_reduction"] > 0

    def test_information_retention_estimation(self, engine):
        """Test information retention estimation"""
        original = "The system MUST implement authentication and authorization."
        optimized = "System MUST implement authentication."
        
        # Access private method for testing
        retention = engine._estimate_information_retention(original, optimized)
        
        assert isinstance(retention, float)
        assert 0 <= retention <= 1
        # Should retain important keywords
        assert retention > 0.5

    @pytest.mark.asyncio
    async def test_optimization_levels(self, engine):
        """Test different optimization levels"""
        content = " ".join(["This is a test sentence."] * 20)
        
        # Minimal optimization
        minimal_opt, minimal_metrics = await engine.optimize(
            content,
            strategy="summarize",
            max_tokens=100,
            level=OptimizationLevel.MINIMAL
        )
        
        # Aggressive optimization
        aggressive_opt, aggressive_metrics = await engine.optimize(
            content,
            strategy="summarize",
            max_tokens=100,
            level=OptimizationLevel.AGGRESSIVE
        )
        
        # Aggressive should reduce more
        assert aggressive_metrics.reduction_percentage >= minimal_metrics.reduction_percentage
        assert len(aggressive_opt) <= len(minimal_opt)

    @pytest.mark.asyncio
    async def test_context_passing(self, engine):
        """Test passing context to optimization"""
        content = "Test content for context passing"
        context = {"preserve_keywords": ["test", "context"]}
        
        optimized, metrics = await engine.optimize(
            content,
            strategy="summarize",
            max_tokens=10,
            context=context
        )
        
        assert isinstance(optimized, str)
        assert isinstance(metrics, OptimizationMetrics)


class TestSummarizationStrategy:
    """Test SummarizationStrategy class"""

    @pytest.fixture
    def strategy(self):
        """Create summarization strategy instance"""
        return SummarizationStrategy()

    @pytest.mark.asyncio
    async def test_optimize_short_content(self, strategy):
        """Test optimizing short content"""
        content = "Short content."
        
        result = await strategy.optimize(content, max_tokens=100, context={})
        
        # Should return original if already short
        assert result == content

    @pytest.mark.asyncio
    async def test_optimize_long_content(self, strategy):
        """Test optimizing long content"""
        content = " ".join([f"Sentence {i}." for i in range(50)])
        
        result = await strategy.optimize(content, max_tokens=50, context={})
        
        assert isinstance(result, str)
        assert len(result) < len(content)
        # Should create a summary
        assert "Summary" in result or len(result.split('.')) < 50

    def test_estimate_tokens(self, strategy):
        """Test token estimation in strategy"""
        content = "Test content for token estimation"
        
        tokens = strategy.estimate_tokens(content)
        
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_extract_sentences(self, strategy):
        """Test sentence extraction"""
        content = "First sentence. Second sentence? Third sentence! Fourth."
        
        sentences = strategy._extract_sentences(content)
        
        assert isinstance(sentences, list)
        assert len(sentences) == 4
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence?"


class TestEssentialOnlyStrategy:
    """Test EssentialOnlyStrategy class"""

    @pytest.fixture
    def strategy(self):
        """Create essential only strategy instance"""
        return EssentialOnlyStrategy()

    @pytest.mark.asyncio
    async def test_extract_requirements(self, strategy):
        """Test extracting requirements"""
        content = """
        The system MUST implement authentication.
        Users SHALL be verified.
        The system REQUIRED to log access.
        Consider using extra features.
        """
        
        result = await strategy.optimize(content, max_tokens=100, context={})
        
        assert isinstance(result, str)
        # Should contain requirements section
        assert "Requirements" in result or "MUST" in result or "SHALL" in result
        # Should not contain optional content
        assert "Consider" not in result

    @pytest.mark.asyncio
    async def test_extract_nist_controls(self, strategy):
        """Test extracting NIST controls"""
        content = """
        Some general text here.
        @nist-controls: AC-3, AU-2, IA-2(1)
        More general text.
        @nist-control: SC-8
        """
        
        result = await strategy.optimize(content, max_tokens=100, context={})
        
        assert isinstance(result, str)
        # Should contain NIST controls
        assert "NIST Controls" in result or "AC-3" in result

    @pytest.mark.asyncio
    async def test_extract_configurations(self, strategy):
        """Test extracting critical configurations"""
        content = """
        General setup instructions.
        Default value: 30 seconds
        Maximum setting: 100 connections
        Minimum configuration: 2 replicas
        """
        
        result = await strategy.optimize(content, max_tokens=100, context={})
        
        assert isinstance(result, str)
        # Should contain configurations
        if "Configurations" in result:
            assert "value" in result or "setting" in result

    def test_extract_pattern_matches(self, strategy):
        """Test pattern matching extraction"""
        content = "The system MUST do X. The system SHALL do Y. Consider doing Z."
        pattern = r'(?:MUST|SHALL)\s+[^.]+\.'
        
        matches = strategy._extract_pattern_matches(content, pattern)
        
        assert isinstance(matches, list)
        assert len(matches) == 2
        assert any("MUST" in match for match in matches)
        assert any("SHALL" in match for match in matches)


class TestHierarchicalStrategy:
    """Test HierarchicalStrategy class"""

    @pytest.fixture
    def strategy(self):
        """Create hierarchical strategy instance"""
        return HierarchicalStrategy()

    @pytest.mark.asyncio
    async def test_preserve_hierarchy(self, strategy):
        """Test preserving document hierarchy"""
        content = """
        # Title
        ## Section 1
        Content for section 1
        ## Section 2
        Content for section 2
        ### Subsection 2.1
        Detailed content
        """
        
        result = await strategy.optimize(content, max_tokens=50, context={})
        
        assert isinstance(result, str)
        # Should preserve headers
        assert "#" in result
        # Should have some structure
        assert result.count("#") >= 2

    @pytest.mark.asyncio
    async def test_optimize_sections(self, strategy):
        """Test optimizing individual sections"""
        content = """
        # Main Title
        
        ## Very Long Section
        """ + " ".join(["Long content."] * 50) + """
        
        ## Short Section
        Brief content.
        """
        
        result = await strategy.optimize(content, max_tokens=100, context={})
        
        assert isinstance(result, str)
        assert "Main Title" in result
        assert "Long Section" in result or "..." in result
        assert len(result) < len(content)

    def test_parse_sections(self, strategy):
        """Test parsing sections from content"""
        content = """# Title
## Section 1
Content 1
## Section 2
Content 2"""
        
        sections = strategy._parse_sections(content)
        
        assert isinstance(sections, list)
        assert len(sections) >= 2
        # Each section should have header and content
        for section in sections:
            assert "header" in section
            assert "content" in section
            assert "level" in section


class TestOptimizationIntegration:
    """Integration tests for token optimization"""

    @pytest.fixture
    def engine(self):
        """Create engine for integration tests"""
        return TokenOptimizationEngine()

    @pytest.mark.asyncio
    async def test_progressive_optimization(self, engine):
        """Test progressive optimization with increasing aggressiveness"""
        content = " ".join([f"This is sentence number {i}." for i in range(100)])
        
        results = []
        for level in [OptimizationLevel.MINIMAL, OptimizationLevel.MODERATE, OptimizationLevel.AGGRESSIVE]:
            optimized, metrics = await engine.optimize(
                content,
                strategy="summarize",
                max_tokens=200,
                level=level
            )
            results.append((optimized, metrics))
        
        # Each level should be more aggressive
        assert results[0][1].reduction_percentage <= results[1][1].reduction_percentage
        assert results[1][1].reduction_percentage <= results[2][1].reduction_percentage
        
        # Content should get progressively shorter
        assert len(results[0][0]) >= len(results[1][0])
        assert len(results[1][0]) >= len(results[2][0])

    @pytest.mark.asyncio
    async def test_strategy_comparison(self, engine):
        """Test different strategies on same content"""
        content = """
        # Security Standard
        
        The system MUST implement authentication.
        The system SHALL use encryption.
        
        ## Details
        Here are detailed implementation guidelines with examples.
        Consider using OAuth 2.0 for authentication.
        
        @nist-controls: AC-3, AU-2, IA-2
        """
        
        strategies = ["summarize", "essential", "hierarchical"]
        results = {}
        
        for strategy in strategies:
            optimized, metrics = await engine.optimize(
                content,
                strategy=strategy,
                max_tokens=100
            )
            results[strategy] = (optimized, metrics)
        
        # Each strategy should produce different results
        assert results["summarize"][0] != results["essential"][0]
        assert results["essential"][0] != results["hierarchical"][0]
        
        # Essential should focus on requirements
        assert "MUST" in results["essential"][0] or "Requirements" in results["essential"][0]
        
        # Hierarchical should preserve structure
        assert "#" in results["hierarchical"][0]

    @pytest.mark.asyncio
    async def test_token_budget_enforcement(self, engine):
        """Test that token budget is enforced"""
        content = " ".join(["Word"] * 1000)
        max_tokens = 50
        
        optimized, metrics = await engine.optimize(
            content,
            strategy="summarize",
            max_tokens=max_tokens,
            level=OptimizationLevel.AGGRESSIVE
        )
        
        # Should not exceed token budget (with some tolerance for tokenizer differences)
        assert metrics.optimized_tokens <= max_tokens * 1.1

    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling in optimization"""
        # Empty content
        optimized, metrics = await engine.optimize(
            "",
            strategy="summarize",
            max_tokens=50
        )
        assert optimized == ""
        assert metrics.reduction_percentage == 0
        
        # Very small token budget
        content = "This is a test sentence that needs optimization."
        optimized, metrics = await engine.optimize(
            content,
            strategy="summarize",
            max_tokens=1
        )
        assert len(optimized) < len(content)
        assert metrics.optimized_tokens <= 5  # Allow some flexibility

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, engine):
        """Test that metrics are properly tracked"""
        # Clear history
        engine._metrics_history.clear()
        
        # Run multiple optimizations
        for i in range(3):
            await engine.optimize(
                f"Test content {i}" * 10,
                strategy="summarize",
                max_tokens=20
            )
        
        # Check metrics history
        assert len(engine._metrics_history) == 3
        
        # Get summary
        summary = engine.get_metrics_summary()
        assert summary["average_reduction"] > 0
        assert summary["average_retention"] > 0
        assert summary["average_processing_time"] > 0
        assert summary["total_optimizations"] == 3