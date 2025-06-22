"""
Comprehensive tests for micro_standards module
@nist-controls: SA-11, CA-7
@evidence: Micro standards chunking testing
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.core.standards.micro_standards import (
    ChunkingContext,
    MicroStandard,
    MicroStandardIndex,
    MicroStandardsGenerator,
    generate_micro_standards,
)
from src.core.standards.models import Standard, StandardSection
from src.core.tokenizer import BaseTokenizer


class MockTokenizer(BaseTokenizer):
    """Mock tokenizer for testing"""
    def count_tokens(self, text: str) -> int:
        # Simple word-based estimation for testing
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        return ' '.join(words[:max_tokens])


class TestMicroStandard:
    """Test MicroStandard model"""

    def test_micro_standard_creation(self):
        """Test creating a micro standard"""
        micro = MicroStandard(
            id="test_001",
            standard_id="std_001",
            title="Test Standard",
            content="Test content",
            token_count=100,
            chunk_type="overview"
        )

        assert micro.id == "test_001"
        assert micro.standard_id == "std_001"
        assert micro.title == "Test Standard"
        assert micro.content == "Test content"
        assert micro.token_count == 100
        assert micro.chunk_type == "overview"
        assert isinstance(micro.topics, list)
        assert isinstance(micro.concepts, list)
        assert isinstance(micro.nist_controls, list)
        assert micro.parent_id is None
        assert isinstance(micro.child_ids, list)
        assert isinstance(micro.navigation, dict)
        assert isinstance(micro.metadata, dict)
        assert isinstance(micro.created_at, datetime)

    def test_micro_standard_with_all_fields(self):
        """Test micro standard with all fields populated"""
        micro = MicroStandard(
            id="test_002",
            standard_id="std_002",
            title="Complete Standard",
            content="Detailed content",
            token_count=200,
            chunk_type="requirement",
            topics=["security", "authentication"],
            concepts=["RBAC", "MFA"],
            nist_controls=["AC-3", "IA-2"],
            parent_id="test_001",
            child_ids=["test_003", "test_004"],
            navigation={"prev": "test_001", "next": "test_003"},
            metadata={"version": "1.0", "author": "test"}
        )

        assert len(micro.topics) == 2
        assert "security" in micro.topics
        assert len(micro.concepts) == 2
        assert "RBAC" in micro.concepts
        assert len(micro.nist_controls) == 2
        assert "AC-3" in micro.nist_controls
        assert micro.parent_id == "test_001"
        assert len(micro.child_ids) == 2
        assert micro.navigation["prev"] == "test_001"
        assert micro.metadata["version"] == "1.0"

    def test_micro_standard_to_dict(self):
        """Test converting micro standard to dictionary"""
        micro = MicroStandard(
            id="test_003",
            standard_id="std_003",
            title="Dict Test",
            content="Convert to dict",
            token_count=50,
            chunk_type="example",
            topics=["testing"],
            nist_controls=["SA-11"]
        )

        result = micro.to_dict()

        assert isinstance(result, dict)
        assert result["id"] == "test_003"
        assert result["standard_id"] == "std_003"
        assert result["title"] == "Dict Test"
        assert result["content"] == "Convert to dict"
        assert result["token_count"] == 50
        assert result["chunk_type"] == "example"
        assert result["topics"] == ["testing"]
        assert result["nist_controls"] == ["SA-11"]
        assert isinstance(result["created_at"], str)
        # Should be ISO format
        datetime.fromisoformat(result["created_at"])

    def test_micro_standard_validation(self):
        """Test micro standard validation"""
        # Missing required fields
        with pytest.raises(ValidationError):
            MicroStandard(
                id="test_004",
                standard_id="std_004",
                # Missing title, content, token_count, chunk_type
            )

    def test_chunk_types(self):
        """Test different chunk types"""
        chunk_types = ["overview", "requirement", "implementation", "example", "topic"]

        for chunk_type in chunk_types:
            micro = MicroStandard(
                id=f"test_{chunk_type}",
                standard_id="std_test",
                title=f"Test {chunk_type}",
                content=f"Content for {chunk_type}",
                token_count=100,
                chunk_type=chunk_type
            )
            assert micro.chunk_type == chunk_type


class TestMicroStandardIndex:
    """Test MicroStandardIndex model"""

    def test_index_creation(self):
        """Test creating an index"""
        index = MicroStandardIndex(
            standard_id="std_001",
            total_chunks=10,
            overview_chunk_id="chunk_001",
            chunk_hierarchy={"chunk_001": ["chunk_002", "chunk_003"]},
            topic_index={"security": ["chunk_002", "chunk_004"]},
            concept_index={"rbac": ["chunk_003", "chunk_005"]},
            control_index={"AC-3": ["chunk_002", "chunk_003"]},
            chunk_map={"chunk_001": "Overview", "chunk_002": "Security"}
        )

        assert index.standard_id == "std_001"
        assert index.total_chunks == 10
        assert index.overview_chunk_id == "chunk_001"
        assert len(index.chunk_hierarchy["chunk_001"]) == 2
        assert len(index.topic_index["security"]) == 2
        assert len(index.concept_index["rbac"]) == 2
        assert len(index.control_index["AC-3"]) == 2
        assert index.chunk_map["chunk_001"] == "Overview"

    def test_get_chunks_for_topic(self):
        """Test getting chunks for a topic"""
        index = MicroStandardIndex(
            standard_id="std_002",
            total_chunks=5,
            overview_chunk_id="chunk_001",
            chunk_hierarchy={},
            topic_index={
                "security": ["chunk_002", "chunk_003"],
                "authentication": ["chunk_003", "chunk_004"]
            },
            concept_index={},
            control_index={},
            chunk_map={}
        )

        # Existing topic
        chunks = index.get_chunks_for_topic("security")
        assert len(chunks) == 2
        assert "chunk_002" in chunks
        assert "chunk_003" in chunks

        # Case insensitive
        chunks = index.get_chunks_for_topic("SECURITY")
        assert len(chunks) == 2

        # Non-existent topic
        chunks = index.get_chunks_for_topic("nonexistent")
        assert chunks == []

    def test_get_chunks_for_control(self):
        """Test getting chunks for a NIST control"""
        index = MicroStandardIndex(
            standard_id="std_003",
            total_chunks=5,
            overview_chunk_id="chunk_001",
            chunk_hierarchy={},
            topic_index={},
            concept_index={},
            control_index={
                "AC-3": ["chunk_002", "chunk_003"],
                "IA-2": ["chunk_003", "chunk_004"],
                "SC-8": ["chunk_005"]
            },
            chunk_map={}
        )

        # Existing control
        chunks = index.get_chunks_for_control("AC-3")
        assert len(chunks) == 2
        assert "chunk_002" in chunks
        assert "chunk_003" in chunks

        # Case insensitive
        chunks = index.get_chunks_for_control("ac-3")
        assert len(chunks) == 2

        # Non-existent control
        chunks = index.get_chunks_for_control("XX-99")
        assert chunks == []

    def test_get_navigation_path(self):
        """Test getting navigation path"""
        index = MicroStandardIndex(
            standard_id="std_004",
            total_chunks=10,
            overview_chunk_id="chunk_001",
            chunk_hierarchy={
                "chunk_001": ["chunk_002", "chunk_003"],
                "chunk_002": ["chunk_004", "chunk_005"],
                "chunk_003": ["chunk_006"]
            },
            topic_index={},
            concept_index={},
            control_index={},
            chunk_map={}
        )

        # Path from leaf to root
        path = index.get_navigation_path("chunk_005")
        assert path == ["chunk_001", "chunk_002", "chunk_005"]

        # Path for root
        path = index.get_navigation_path("chunk_001")
        assert path == ["chunk_001"]

        # Path for mid-level
        path = index.get_navigation_path("chunk_002")
        assert path == ["chunk_001", "chunk_002"]

        # Orphan chunk
        path = index.get_navigation_path("chunk_999")
        assert path == ["chunk_999"]


class TestChunkingContext:
    """Test ChunkingContext dataclass"""

    def test_default_context(self):
        """Test default chunking context"""
        standard = Standard(
            id="std_001",
            title="Test Standard",
            category="security",
            sections=[]
        )

        context = ChunkingContext(standard=standard)

        assert context.standard == standard
        assert context.target_tokens == 500
        assert context.variance_allowed == 0.1
        assert context.min_tokens == 450
        assert context.max_tokens == 550
        assert context.preserve_sections is True
        assert context.extract_patterns is True

    def test_custom_context(self):
        """Test custom chunking context"""
        standard = Standard(
            id="std_002",
            title="Test Standard 2",
            category="development",
            sections=[]
        )

        context = ChunkingContext(
            standard=standard,
            target_tokens=300,
            variance_allowed=0.2,
            preserve_sections=False,
            extract_patterns=False
        )

        assert context.target_tokens == 300
        assert context.variance_allowed == 0.2
        assert context.min_tokens == 450  # Still default
        assert context.max_tokens == 550  # Still default
        assert context.preserve_sections is False
        assert context.extract_patterns is False


class TestMicroStandardsGenerator:
    """Test MicroStandardsGenerator class"""

    @pytest.fixture
    def generator(self):
        """Create generator with mock tokenizer"""
        tokenizer = MockTokenizer()
        return MicroStandardsGenerator(tokenizer=tokenizer)

    @pytest.fixture
    def sample_standard(self):
        """Create sample standard for testing"""
        sections = [
            StandardSection(
                id="sec_001",
                type="core",
                section="1",
                title="Introduction",
                content="This is an introduction to security standards. It covers basic concepts.",
                tokens=20
            ),
            StandardSection(
                id="sec_002",
                type="requirement",
                section="2",
                title="Requirements",
                content="The system MUST implement access control. The system SHALL provide audit logging. @nist-controls: AC-3, AU-2",
                tokens=25,
                nist_controls=["AC-3", "AU-2"]
            ),
            StandardSection(
                id="sec_003",
                type="implementation",
                section="3",
                title="Implementation Guide",
                content="To implement access control:\n\n```python\ndef check_access(user, resource):\n    return user.has_permission(resource)\n```\n\nThis ensures proper authorization.",
                tokens=30
            ),
            StandardSection(
                id="sec_004",
                type="example",
                section="4",
                title="Examples",
                content="Example: User authentication\n\n```javascript\nfunction authenticate(username, password) {\n    // Verify credentials\n    return verifyUser(username, password);\n}\n```",
                tokens=25
            )
        ]

        return Standard(
            id="std_test",
            title="Test Security Standard",
            version="1.0.0",
            category="security",
            description="A test standard for security",
            sections=sections,
            tags=["security", "authentication", "authorization"]
        )

    def test_generator_initialization(self, generator):
        """Test generator initialization"""
        assert generator.tokenizer is not None
        assert generator.token_optimizer is not None
        assert generator.chunk_counter == 0
        assert generator.requirement_pattern is not None
        assert generator.control_pattern is not None
        assert generator.concept_pattern is not None

    def test_generate_chunk_id(self, generator):
        """Test chunk ID generation"""
        id1 = generator._generate_chunk_id("std_001", "overview")
        id2 = generator._generate_chunk_id("std_001", "overview")

        # IDs should be unique
        assert id1 != id2
        assert len(id1) == 12  # MD5 hex digest truncated
        assert len(id2) == 12

        # Counter should increment
        assert generator.chunk_counter == 2

    def test_estimate_tokens(self, generator):
        """Test token estimation"""
        text = "This is a test sentence with several words"
        tokens = generator._estimate_tokens(text)
        assert tokens == 8  # Word count with mock tokenizer

    def test_extract_requirements(self, generator):
        """Test requirement extraction"""
        content = """
        The system MUST implement strong authentication.
        Users SHALL be verified before access.
        The application REQUIRED to log all access attempts.
        Optional: Additional security measures.
        """

        requirements = generator._extract_requirements(content)

        assert len(requirements) >= 3
        assert any("MUST implement" in req for req in requirements)
        assert any("SHALL be verified" in req for req in requirements)
        assert any("REQUIRED to log" in req for req in requirements)

    def test_extract_controls_from_text(self, generator):
        """Test NIST control extraction"""
        content = """
        @nist-controls: AC-3, AU-2, IA-2(1)
        This implements access control.
        @nist-control: SC-8
        Also see @nist-controls: SI-10, SI-11
        """

        controls = generator._extract_controls_from_text(content)

        assert "AC-3" in controls
        assert "AU-2" in controls
        assert "IA-2(1)" in controls
        assert "SC-8" in controls
        assert "SI-10" in controls
        assert "SI-11" in controls
        assert len(set(controls)) == 6

    def test_extract_concepts_from_text(self, generator):
        """Test concept extraction"""
        content = """
        Authentication is the process of verifying identity.
        Role-Based Access Control refers to permission management.
        The Security Token Service manages tokens.
        """

        concepts = generator._extract_concepts_from_text(content)

        # Should find defined terms and capitalized terms
        assert len(concepts) > 0
        assert any("Authentication" in c for c in concepts)
        # Capitalized terms
        assert any("Security Token Service" in c for c in concepts)

    def test_extract_code_blocks(self, generator):
        """Test code block extraction"""
        content = """
        Here's an example:

        ```python
        def authenticate(user):
            return user.is_valid()
        ```

        And another:

        ```javascript
        function validate() {
            return true;
        }
        ```

        ```
        plain text block
        ```
        """

        blocks = generator._extract_code_blocks(content)

        assert len(blocks) == 3
        assert blocks[0][0] == "python"
        assert "def authenticate" in blocks[0][1]
        assert blocks[1][0] == "javascript"
        assert "function validate" in blocks[1][1]
        assert blocks[2][0] == "plaintext"  # Default language

    def test_find_code_explanation(self, generator):
        """Test finding explanation for code blocks"""
        content = """
        This function validates user credentials:

        ```python
        def validate(user, password):
            return check_password(user, password)
        ```

        The validation ensures secure authentication.
        """

        code = "def validate(user, password):\n    return check_password(user, password)"

        explanation = generator._find_code_explanation(content, code)

        assert explanation is not None
        assert "validates user credentials" in explanation or "validation ensures" in explanation

    def test_create_overview_chunk(self, generator, sample_standard):
        """Test creating overview chunk"""
        context = ChunkingContext(standard=sample_standard)

        chunk = generator._create_overview_chunk(sample_standard, context)

        assert chunk.chunk_type == "overview"
        assert chunk.standard_id == sample_standard.id
        assert sample_standard.title in chunk.title
        assert "Overview" in chunk.title
        assert chunk.token_count > 0
        assert chunk.token_count <= context.max_tokens
        assert chunk.metadata.get("is_root") is True
        assert chunk.metadata.get("total_sections") == 4
        assert len(chunk.topics) > 0
        assert len(chunk.nist_controls) > 0

    def test_create_requirement_chunks(self, generator, sample_standard):
        """Test creating requirement chunks"""
        context = ChunkingContext(standard=sample_standard)

        chunks = generator._create_requirement_chunks(sample_standard, context)

        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.chunk_type == "requirement"
        assert "Requirements" in chunk.title
        assert chunk.token_count > 0
        assert "MUST" in chunk.content or "SHALL" in chunk.content
        assert len(chunk.nist_controls) > 0
        assert "AC-3" in chunk.nist_controls or "AU-2" in chunk.nist_controls

    def test_create_topic_chunks(self, generator, sample_standard):
        """Test creating topic chunks"""
        context = ChunkingContext(standard=sample_standard)

        chunks = generator._create_topic_chunks(sample_standard, context)

        assert len(chunks) > 0
        # Should have chunks for each section
        chunk_types = [c.metadata.get("section_id") for c in chunks]
        assert any("sec_001" in str(ct) for ct in chunk_types)

    def test_create_implementation_chunks(self, generator, sample_standard):
        """Test creating implementation chunks"""
        context = ChunkingContext(standard=sample_standard)

        chunks = generator._create_implementation_chunks(sample_standard, context)

        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.chunk_type == "implementation"
        assert "Implementation" in chunk.title
        assert "```" in chunk.content  # Has code block
        assert chunk.metadata.get("language") is not None

    def test_create_example_chunks(self, generator, sample_standard):
        """Test creating example chunks"""
        context = ChunkingContext(standard=sample_standard)

        chunks = generator._create_example_chunks(sample_standard, context)

        assert len(chunks) > 0
        chunk = chunks[0]
        assert chunk.chunk_type == "example"
        assert "Example" in chunk.title
        assert "```" in chunk.content  # Has code block

    def test_split_large_section(self, generator):
        """Test splitting large sections"""
        # Create a large section
        large_content = "\n\n".join([
            f"Paragraph {i}: " + " ".join(["word"] * 100)
            for i in range(10)
        ])

        large_section = StandardSection(
            id="large_001",
            type="core",
            section="1",
            title="Large Section",
            content=large_content,
            tokens=1000
        )

        standard = Standard(
            id="std_large",
            title="Large Standard",
            category="test",
            sections=[large_section]
        )

        context = ChunkingContext(standard=standard, max_tokens=200)

        chunks = generator._split_large_section(standard, large_section, context)

        assert len(chunks) > 1  # Should be split
        for chunk in chunks:
            assert chunk.token_count <= context.max_tokens
            assert "Part" in chunk.title or chunk.title.endswith(large_section.title)

    def test_build_navigation(self, generator):
        """Test building navigation links"""
        chunks = [
            MicroStandard(
                id=f"chunk_{i}",
                standard_id="std_001",
                title=f"Chunk {i}",
                content=f"Content {i}",
                token_count=100,
                chunk_type="topic"
            )
            for i in range(5)
        ]

        generator._build_navigation(chunks)

        # Check navigation links
        assert chunks[0].navigation.get("prev") is None
        assert chunks[0].navigation["next"] == "chunk_1"

        assert chunks[1].navigation["prev"] == "chunk_0"
        assert chunks[1].navigation["next"] == "chunk_2"
        assert chunks[1].navigation["up"] == "chunk_0"

        assert chunks[4].navigation["prev"] == "chunk_3"
        assert chunks[4].navigation.get("next") is None
        assert chunks[4].navigation["up"] == "chunk_0"

    def test_generate_chunks_full_process(self, generator, sample_standard):
        """Test full chunk generation process"""
        context = ChunkingContext(standard=sample_standard)

        chunks = generator.generate_chunks(sample_standard, context)

        assert len(chunks) > 0

        # Should have overview chunk first
        assert chunks[0].chunk_type == "overview"
        assert chunks[0].metadata.get("is_root") is True

        # Should have various chunk types
        chunk_types = {chunk.chunk_type for chunk in chunks}
        assert "overview" in chunk_types
        assert len(chunk_types) > 1  # Multiple types

        # All chunks should respect token limits
        for chunk in chunks:
            assert chunk.token_count <= context.max_tokens
            assert chunk.standard_id == sample_standard.id
            assert chunk.title is not None
            assert chunk.content is not None

        # Navigation should be set
        for i, chunk in enumerate(chunks):
            if i > 0:
                assert "prev" in chunk.navigation
            if i < len(chunks) - 1:
                assert "next" in chunk.navigation
            assert "up" in chunk.navigation

    def test_create_index(self, generator, sample_standard):
        """Test creating an index"""
        chunks = generator.generate_chunks(sample_standard)

        index = generator.create_index(chunks, sample_standard)

        assert index.standard_id == sample_standard.id
        assert index.total_chunks == len(chunks)
        assert index.overview_chunk_id == chunks[0].id

        # Check chunk map
        assert len(index.chunk_map) == len(chunks)
        for chunk in chunks:
            assert chunk.id in index.chunk_map
            assert index.chunk_map[chunk.id] == chunk.title

        # Check indexes are populated
        assert len(index.topic_index) > 0
        assert len(index.control_index) > 0

    def test_save_and_load_chunks(self, generator, sample_standard, tmp_path):
        """Test saving and loading chunks"""
        # Generate chunks
        chunks = generator.generate_chunks(sample_standard)

        # Save chunks
        generator.save_chunks(chunks, tmp_path)

        # Check files exist
        assert (tmp_path / f"{sample_standard.id}_index.json").exists()
        for chunk in chunks:
            assert (tmp_path / f"{chunk.id}.json").exists()

        # Load chunks
        loaded_chunks, loaded_index = generator.load_chunks(sample_standard.id, tmp_path)

        assert len(loaded_chunks) == len(chunks)
        assert loaded_index.standard_id == sample_standard.id
        assert loaded_index.total_chunks == len(chunks)

        # Verify loaded data
        loaded_ids = {c.id for c in loaded_chunks}
        original_ids = {c.id for c in chunks}
        assert loaded_ids == original_ids

    def test_edge_cases(self, generator):
        """Test edge cases"""
        # Empty standard
        empty_standard = Standard(
            id="empty",
            title="Empty Standard",
            category="test",
            sections=[]
        )

        chunks = generator.generate_chunks(empty_standard)
        assert len(chunks) >= 1  # At least overview
        assert chunks[0].chunk_type == "overview"

        # Standard with minimal content
        minimal_standard = Standard(
            id="minimal",
            title="Minimal",
            category="test",
            sections=[
                StandardSection(
                    id="min_001",
                    type="core",
                    section="1",
                    title="Short",
                    content="Very short content.",
                    tokens=3
                )
            ]
        )

        chunks = generator.generate_chunks(minimal_standard)
        assert len(chunks) >= 1

        # No chunks to index
        with pytest.raises(ValueError):
            generator.create_index([], empty_standard)

    def test_extract_patterns(self, generator):
        """Test pattern extraction methods"""
        # Test extracting all controls from standard
        sections = [
            StandardSection(
                id="s1",
                type="core",
                section="1",
                title="Section 1",
                content="@nist-controls: AC-3, AU-2",
                tokens=10
            ),
            StandardSection(
                id="s2",
                type="core",
                section="2",
                title="Section 2",
                content="@nist-controls: IA-2, SC-8",
                tokens=10
            )
        ]

        standard = Standard(
            id="pattern_test",
            title="Pattern Test",
            category="test",
            sections=sections
        )

        controls = generator._extract_all_controls(standard)
        assert len(controls) == 4
        assert "AC-3" in controls
        assert "AU-2" in controls
        assert "IA-2" in controls
        assert "SC-8" in controls

        # Test concept extraction
        concepts = generator._extract_concepts(standard)
        assert isinstance(concepts, list)

    def test_chunk_metadata(self, generator, sample_standard):
        """Test chunk metadata population"""
        chunks = generator.generate_chunks(sample_standard)

        for chunk in chunks:
            assert chunk.metadata is not None
            assert isinstance(chunk.metadata, dict)

            if chunk.chunk_type == "overview":
                assert "is_root" in chunk.metadata
                assert "total_sections" in chunk.metadata

            if chunk.chunk_type == "requirement":
                assert "requirement_count" in chunk.metadata or "sections_covered" in chunk.metadata

            if chunk.chunk_type == "implementation":
                assert "language" in chunk.metadata or "section_id" in chunk.metadata


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    @pytest.mark.asyncio
    async def test_generate_micro_standards(self, tmp_path):
        """Test async generate_micro_standards function"""
        standard = Standard(
            id="async_test",
            title="Async Test Standard",
            category="test",
            sections=[
                StandardSection(
                    id="async_001",
                    type="core",
                    section="1",
                    title="Test Section",
                    content="Test content for async generation",
                    tokens=10
                )
            ]
        )

        # Without output directory
        chunks, index = await generate_micro_standards(standard)

        assert len(chunks) > 0
        assert index.standard_id == standard.id
        assert index.total_chunks == len(chunks)

        # With output directory
        chunks2, index2 = await generate_micro_standards(standard, tmp_path)

        assert len(chunks2) > 0
        assert (tmp_path / f"{standard.id}_index.json").exists()


class TestIntegration:
    """Integration tests for micro standards"""

    @pytest.fixture
    def complex_standard(self):
        """Create a complex standard for integration testing"""
        sections = []

        # Add various section types
        for i in range(10):
            section_type = ["core", "requirement", "implementation", "example"][i % 4]
            content = f"Section {i} content. "

            if section_type == "requirement":
                content += "The system MUST do this. The system SHALL do that. @nist-controls: AC-3, AU-2"
            elif section_type == "implementation":
                content += "\n```python\ndef example():\n    pass\n```\nThis implements the requirement."
            elif section_type == "example":
                content += "For example, consider this scenario..."

            content += " " * 50  # Pad content

            sections.append(StandardSection(
                id=f"sec_{i:03d}",
                type=section_type,
                section=str(i),
                title=f"Section {i}",
                content=content,
                tokens=len(content.split()),
                tags=[f"tag{i}", f"tag{i+1}"],
                nist_controls=["AC-3", "AU-2"] if i % 2 == 0 else ["IA-2", "SC-8"]
            ))

        return Standard(
            id="complex_std",
            title="Complex Test Standard",
            version="2.0.0",
            category="security",
            description="A complex standard for comprehensive testing",
            sections=sections,
            tags=["security", "compliance", "testing"]
        )

    def test_complex_standard_processing(self, complex_standard):
        """Test processing a complex standard"""
        generator = MicroStandardsGenerator(tokenizer=MockTokenizer())

        chunks = generator.generate_chunks(complex_standard)
        index = generator.create_index(chunks, complex_standard)

        # Verify chunk distribution
        chunk_types = {}
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

        assert "overview" in chunk_types
        assert chunk_types["overview"] >= 1

        # Verify index completeness
        assert len(index.chunk_map) == len(chunks)
        assert len(index.topic_index) > 0
        assert len(index.control_index) > 0

        # Test navigation
        for chunk_id in index.chunk_map:
            path = index.get_navigation_path(chunk_id)
            assert len(path) > 0
            assert path[0] == chunks[0].id  # Should start with overview

        # Test search functionality
        security_chunks = index.get_chunks_for_topic("security")
        assert len(security_chunks) > 0

        ac3_chunks = index.get_chunks_for_control("AC-3")
        assert len(ac3_chunks) > 0

    def test_token_optimization_integration(self):
        """Test integration with token optimization"""
        # Create a section that needs optimization
        long_content = " ".join(["word"] * 1000)  # Very long content

        section = StandardSection(
            id="long_001",
            type="core",
            section="1",
            title="Long Section",
            content=long_content,
            tokens=1000
        )

        standard = Standard(
            id="opt_test",
            title="Optimization Test",
            category="test",
            sections=[section]
        )

        generator = MicroStandardsGenerator(tokenizer=MockTokenizer())
        context = ChunkingContext(standard=standard, max_tokens=100)

        chunks = generator.generate_chunks(standard, context)

        # All chunks should respect token limit
        for chunk in chunks:
            assert chunk.token_count <= context.max_tokens

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery"""
        generator = MicroStandardsGenerator(tokenizer=MockTokenizer())

        # Malformed standard
        malformed = Standard(
            id="malformed",
            title="Malformed Standard",
            category="test",
            sections=[
                StandardSection(
                    id="mal_001",
                    type="core",
                    section="1",
                    title=None,  # No title
                    content=None,  # No content
                    tokens=0
                )
            ]
        )

        # Should still generate chunks without crashing
        chunks = generator.generate_chunks(malformed)
        assert len(chunks) >= 1  # At least overview

        # Test with special characters
        special_standard = Standard(
            id="special_chars",
            title="Special <>&\" Characters",
            category="test & debug",
            sections=[
                StandardSection(
                    id="spec_001",
                    type="core",
                    section="1",
                    title="Section with <tags>",
                    content="Content with special chars: <>&\"'",
                    tokens=10
                )
            ],
            tags=["<tag1>", "tag&2"]
        )

        chunks = generator.generate_chunks(special_standard)
        assert len(chunks) > 0
        # Should handle special characters in JSON serialization
        for chunk in chunks:
            chunk_dict = chunk.to_dict()
            assert isinstance(chunk_dict, dict)
