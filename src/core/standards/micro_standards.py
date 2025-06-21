"""
Micro Standards Generator - Creates 500-token digestible chunks
@nist-controls: SI-12, AC-4
@evidence: Information chunking and controlled access
@oscal-component: standards-engine
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..tokenizer import BaseTokenizer, get_default_tokenizer
from .models import Standard, StandardSection
from .token_optimizer import TokenOptimizationEngine


class MicroStandard(BaseModel):
    """A micro-standard chunk of ~500 tokens"""
    id: str
    standard_id: str
    title: str
    content: str
    token_count: int
    chunk_type: str  # overview, requirement, implementation, example
    topics: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    nist_controls: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    child_ids: list[str] = Field(default_factory=list)
    navigation: dict[str, str] = Field(default_factory=dict)  # prev, next, up
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "standard_id": self.standard_id,
            "title": self.title,
            "content": self.content,
            "token_count": self.token_count,
            "chunk_type": self.chunk_type,
            "topics": self.topics,
            "concepts": self.concepts,
            "nist_controls": self.nist_controls,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "navigation": self.navigation,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class MicroStandardIndex(BaseModel):
    """Index for micro standards navigation and search"""
    standard_id: str
    total_chunks: int
    overview_chunk_id: str
    chunk_hierarchy: dict[str, list[str]]  # parent_id -> child_ids
    topic_index: dict[str, list[str]]  # topic -> chunk_ids
    concept_index: dict[str, list[str]]  # concept -> chunk_ids
    control_index: dict[str, list[str]]  # nist_control -> chunk_ids
    chunk_map: dict[str, str]  # chunk_id -> title
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def get_chunks_for_topic(self, topic: str) -> list[str]:
        """Get all chunk IDs for a topic"""
        return self.topic_index.get(topic.lower(), [])

    def get_chunks_for_control(self, control: str) -> list[str]:
        """Get all chunk IDs for a NIST control"""
        return self.control_index.get(control.upper(), [])

    def get_navigation_path(self, chunk_id: str) -> list[str]:
        """Get navigation path from root to chunk"""
        path = []
        current = chunk_id

        # Build path from chunk to root
        while current:
            path.append(current)
            # Find parent
            parent = None
            for p_id, children in self.chunk_hierarchy.items():
                if current in children:
                    parent = p_id
                    break
            current = parent  # type: ignore[assignment]

        return list(reversed(path))


@dataclass
class ChunkingContext:
    """Context for intelligent chunking decisions"""
    standard: Standard
    target_tokens: int = 500
    variance_allowed: float = 0.1  # ±10% variance
    min_tokens: int = 450
    max_tokens: int = 550
    preserve_sections: bool = True
    extract_patterns: bool = True


class MicroStandardsGenerator:
    """
    Generates micro standards chunks of ~500 tokens
    @nist-controls: SI-12, AC-4
    @evidence: Intelligent content chunking with navigation
    """

    def __init__(self, token_optimizer: TokenOptimizationEngine | None = None, tokenizer: BaseTokenizer | None = None):
        self.tokenizer = tokenizer or get_default_tokenizer()
        self.token_optimizer = token_optimizer or TokenOptimizationEngine(self.tokenizer)
        self.chunk_counter = 0

        # Patterns for extraction
        self.requirement_pattern = r'(?:MUST|SHALL|REQUIRED)\s+[^.]+\.'
        self.control_pattern = r'@nist-controls?:\s*([A-Z]{2}-\d+(?:\(\d+\))?(?:,\s*[A-Z]{2}-\d+(?:\(\d+\))?)*)'
        self.concept_pattern = r'(?:^|\n)(?:A|An|The)?\s*(\w+(?:\s+\w+)?)\s+(?:is|are|refers to|means)\s+'

    def generate_chunks(self, standard: Standard, context: ChunkingContext | None = None) -> list[MicroStandard]:
        """Generate micro standard chunks from a full standard"""
        if context is None:
            context = ChunkingContext(standard=standard)

        chunks = []
        self.chunk_counter = 0

        # 1. Create overview chunk
        overview_chunk = self._create_overview_chunk(standard, context)
        chunks.append(overview_chunk)

        # 2. Create requirement chunks
        requirement_chunks = self._create_requirement_chunks(standard, context)
        chunks.extend(requirement_chunks)

        # 3. Create topic-based chunks
        topic_chunks = self._create_topic_chunks(standard, context)
        chunks.extend(topic_chunks)

        # 4. Create implementation chunks
        impl_chunks = self._create_implementation_chunks(standard, context)
        chunks.extend(impl_chunks)

        # 5. Create example chunks (if any)
        example_chunks = self._create_example_chunks(standard, context)
        chunks.extend(example_chunks)

        # 6. Build navigation links
        self._build_navigation(chunks)

        return chunks

    def _generate_chunk_id(self, standard_id: str, chunk_type: str) -> str:
        """Generate unique chunk ID"""
        self.chunk_counter += 1
        base = f"{standard_id}_{chunk_type}_{self.chunk_counter:03d}"
        return hashlib.md5(base.encode()).hexdigest()[:12]

    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content"""
        return self.tokenizer.count_tokens(content)

    def _create_overview_chunk(self, standard: Standard, context: ChunkingContext) -> MicroStandard:
        """Create overview chunk with key information"""
        overview_parts = []

        # Title and description
        overview_parts.append(f"# {standard.title}")
        if standard.description:
            overview_parts.append(f"\n{standard.description}")

        # Key metadata
        overview_parts.append(f"\n**Version:** {standard.version}")
        overview_parts.append(f"**Category:** {standard.category}")
        if standard.tags:
            overview_parts.append(f"**Tags:** {', '.join(standard.tags)}")

        # Table of contents
        overview_parts.append("\n## Contents")
        section_list = []
        for section in standard.sections[:10]:  # Top 10 sections
            section_list.append(f"- {section.title}")
        overview_parts.append("\n".join(section_list))

        # Key concepts extracted
        concepts = self._extract_concepts(standard)
        if concepts:
            overview_parts.append("\n## Key Concepts")
            overview_parts.append(", ".join(concepts[:10]))

        # NIST controls summary
        controls = self._extract_all_controls(standard)
        if controls:
            overview_parts.append(f"\n## NIST Controls ({len(controls)} total)")
            overview_parts.append(", ".join(list(controls)[:10]))
            if len(controls) > 10:
                overview_parts.append(f"... and {len(controls) - 10} more")

        content = "\n".join(overview_parts)

        # Optimize to fit token limit
        if self._estimate_tokens(content) > context.max_tokens:
            # Simple truncation for now - async optimization not available in sync context
            content = self.tokenizer.truncate_to_tokens(content, context.target_tokens)

        return MicroStandard(
            id=self._generate_chunk_id(standard.id, "overview"),
            standard_id=standard.id,
            title=f"{standard.title} - Overview",
            content=content,
            token_count=self._estimate_tokens(content),
            chunk_type="overview",
            topics=standard.tags or [],
            concepts=concepts[:5],
            nist_controls=list(controls)[:10],
            metadata={
                "is_root": True,
                "total_sections": len(standard.sections)
            }
        )

    def _create_requirement_chunks(self, standard: Standard, context: ChunkingContext) -> list[MicroStandard]:
        """Create chunks for requirements"""
        chunks = []
        all_requirements = []

        # Extract all requirements from standard
        for section in standard.sections:
            requirements = self._extract_requirements(section.content)
            for req in requirements:
                all_requirements.append({
                    "requirement": req,
                    "section": section.title,
                    "section_id": section.id
                })

        if not all_requirements:
            return chunks

        # Group requirements into chunks
        current_chunk_reqs = []
        current_tokens = 0

        for req_info in all_requirements:
            req_text = f"**{req_info['section']}**\n- {req_info['requirement']}\n"
            req_tokens = self._estimate_tokens(req_text)

            if current_tokens + req_tokens > context.max_tokens and current_chunk_reqs:
                # Create chunk
                chunk = self._create_requirement_chunk_from_list(
                    standard, current_chunk_reqs, context
                )
                chunks.append(chunk)
                current_chunk_reqs = [req_info]
                current_tokens = req_tokens
            else:
                current_chunk_reqs.append(req_info)
                current_tokens += req_tokens

        # Create final chunk
        if current_chunk_reqs:
            chunk = self._create_requirement_chunk_from_list(
                standard, current_chunk_reqs, context
            )
            chunks.append(chunk)

        return chunks

    def _create_requirement_chunk_from_list(
        self,
        standard: Standard,
        requirements: list[dict[str, str]],
        context: ChunkingContext
    ) -> MicroStandard:
        """Create a requirement chunk from a list of requirements"""
        content_parts = [f"# {standard.title} - Requirements\n"]

        # Group by section
        by_section = {}
        for req in requirements:
            section = req["section"]
            if section not in by_section:
                by_section[section] = []
            by_section[section].append(req["requirement"])

        # Build content
        for section, reqs in by_section.items():
            content_parts.append(f"\n## {section}")
            for req in reqs:
                content_parts.append(f"- {req}")

        content = "\n".join(content_parts)

        # Extract concepts and controls
        concepts = []
        controls = set()
        for req in requirements:
            concepts.extend(self._extract_concepts_from_text(req["requirement"]))
            controls.update(self._extract_controls_from_text(req["requirement"]))

        return MicroStandard(
            id=self._generate_chunk_id(standard.id, "requirements"),
            standard_id=standard.id,
            title=f"{standard.title} - Requirements ({len(requirements)} items)",
            content=content,
            token_count=self._estimate_tokens(content),
            chunk_type="requirement",
            topics=[req["section"] for req in requirements],
            concepts=list(set(concepts))[:10],
            nist_controls=list(controls),
            metadata={
                "requirement_count": len(requirements),
                "sections_covered": list(by_section.keys())
            }
        )

    def _create_topic_chunks(self, standard: Standard, context: ChunkingContext) -> list[MicroStandard]:
        """Create topic-based chunks from sections"""
        chunks = []

        for section in standard.sections:
            # Skip if section is too small
            if self._estimate_tokens(section.content) < 100:
                continue

            # Check if section needs splitting
            section_tokens = self._estimate_tokens(section.content)

            if section_tokens <= context.max_tokens:
                # Create single chunk for section
                chunk = self._create_topic_chunk_from_section(standard, section, context)
                chunks.append(chunk)
            else:
                # Split section into multiple chunks
                sub_chunks = self._split_large_section(standard, section, context)
                chunks.extend(sub_chunks)

        return chunks

    def _create_topic_chunk_from_section(
        self,
        standard: Standard,
        section: StandardSection,
        context: ChunkingContext,
        subtitle: str | None = None
    ) -> MicroStandard:
        """Create a topic chunk from a section"""
        title = f"{standard.title} - {section.title}"
        if subtitle:
            title += f" ({subtitle})"

        # Extract metadata
        concepts = self._extract_concepts_from_text(section.content)
        controls = self._extract_controls_from_text(section.content)

        # Add section header
        content = f"# {section.title}\n\n{section.content}"

        # Optimize if needed
        if self._estimate_tokens(content) > context.target_tokens:
            # Simple truncation for now - async optimization not available in sync context
            content = self.tokenizer.truncate_to_tokens(content, context.target_tokens)

        return MicroStandard(
            id=self._generate_chunk_id(standard.id, "topic"),
            standard_id=standard.id,
            title=title,
            content=content,
            token_count=self._estimate_tokens(content),
            chunk_type="topic",
            topics=[section.title] + (section.tags or []),
            concepts=concepts[:10],
            nist_controls=list(controls),
            metadata={
                "section_id": section.id,
                "original_tokens": self._estimate_tokens(section.content)
            }
        )

    def _split_large_section(
        self,
        standard: Standard,
        section: StandardSection,
        context: ChunkingContext
    ) -> list[MicroStandard]:
        """Split a large section into multiple chunks"""
        chunks = []

        # Try to split by paragraphs
        paragraphs = section.content.split('\n\n')

        current_content = []
        current_tokens = 0
        chunk_num = 1

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)

            if current_tokens + para_tokens > context.max_tokens and current_content:
                # Create chunk
                content = "\n\n".join(current_content)
                chunk = self._create_topic_chunk_from_section(
                    standard,
                    StandardSection(
                        id=f"{section.id}_part{chunk_num}",
                        title=section.title,
                        content=content,
                        order=section.order
                    ),
                    context,
                    subtitle=f"Part {chunk_num}"
                )
                chunks.append(chunk)

                current_content = [para]
                current_tokens = para_tokens
                chunk_num += 1
            else:
                current_content.append(para)
                current_tokens += para_tokens

        # Create final chunk
        if current_content:
            content = "\n\n".join(current_content)
            chunk = self._create_topic_chunk_from_section(
                standard,
                StandardSection(
                    id=f"{section.id}_part{chunk_num}",
                    title=section.title,
                    content=content,
                    order=section.order
                ),
                context,
                subtitle=f"Part {chunk_num}" if chunk_num > 1 else None
            )
            chunks.append(chunk)

        return chunks

    def _create_implementation_chunks(self, standard: Standard, context: ChunkingContext) -> list[MicroStandard]:
        """Create implementation-focused chunks"""
        chunks = []
        implementations = []

        # Look for implementation sections
        for section in standard.sections:
            if any(keyword in section.title.lower() for keyword in ["implement", "example", "how to", "guide"]):
                implementations.append(section)

        # Also extract code blocks
        for section in standard.sections:
            code_blocks = self._extract_code_blocks(section.content)
            for i, (lang, code) in enumerate(code_blocks):
                impl_content = f"# Implementation: {section.title}\n\n"
                impl_content += f"```{lang}\n{code}\n```"

                # Add explanation if available
                explanation = self._find_code_explanation(section.content, code)
                if explanation:
                    impl_content += f"\n\n{explanation}"

                if self._estimate_tokens(impl_content) <= context.max_tokens:
                    chunk = MicroStandard(
                        id=self._generate_chunk_id(standard.id, "implementation"),
                        standard_id=standard.id,
                        title=f"{standard.title} - Implementation Example {i+1}",
                        content=impl_content,
                        token_count=self._estimate_tokens(impl_content),
                        chunk_type="implementation",
                        topics=[section.title, lang],
                        concepts=self._extract_concepts_from_text(explanation or ""),
                        nist_controls=self._extract_controls_from_text(section.content),
                        metadata={
                            "language": lang,
                            "section_id": section.id
                        }
                    )
                    chunks.append(chunk)

        return chunks

    def _create_example_chunks(self, standard: Standard, context: ChunkingContext) -> list[MicroStandard]:
        """Create example-focused chunks"""
        chunks = []

        # Look for example patterns
        for section in standard.sections:
            examples = self._extract_examples(section.content)

            for i, example in enumerate(examples):
                if self._estimate_tokens(example) > 50:  # Skip tiny examples
                    example_content = f"# Example: {section.title}\n\n{example}"

                    if self._estimate_tokens(example_content) <= context.max_tokens:
                        chunk = MicroStandard(
                            id=self._generate_chunk_id(standard.id, "example"),
                            standard_id=standard.id,
                            title=f"{standard.title} - Example {i+1}",
                            content=example_content,
                            token_count=self._estimate_tokens(example_content),
                            chunk_type="example",
                            topics=[section.title],
                            concepts=[],
                            nist_controls=self._extract_controls_from_text(example),
                            metadata={
                                "section_id": section.id,
                                "example_index": i
                            }
                        )
                        chunks.append(chunk)

        return chunks

    def _extract_requirements(self, content: str) -> list[str]:
        """Extract requirement statements"""
        import re
        requirements = re.findall(self.requirement_pattern, content, re.IGNORECASE)
        return [req.strip() for req in requirements]

    def _extract_concepts(self, standard: Standard) -> list[str]:
        """Extract key concepts from entire standard"""
        all_concepts = []

        for section in standard.sections:
            concepts = self._extract_concepts_from_text(section.content)
            all_concepts.extend(concepts)

        # Count frequency and return most common
        concept_counts = {}
        for concept in all_concepts:
            concept_lower = concept.lower()
            concept_counts[concept_lower] = concept_counts.get(concept_lower, 0) + 1

        # Sort by frequency
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in sorted_concepts[:20]]

    def _extract_concepts_from_text(self, text: str) -> list[str]:
        """Extract concepts from text"""
        import re
        concepts = []

        # Look for definitions
        matches = re.findall(self.concept_pattern, text, re.IGNORECASE)
        concepts.extend([match.strip() for match in matches])

        # Also look for capitalized terms
        cap_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        concepts.extend(cap_terms)

        return list(set(concepts))[:10]

    def _extract_all_controls(self, standard: Standard) -> set[str]:
        """Extract all NIST controls from standard"""
        controls = set()

        for section in standard.sections:
            section_controls = self._extract_controls_from_text(section.content)
            controls.update(section_controls)

        return controls

    def _extract_controls_from_text(self, text: str) -> list[str]:
        """Extract NIST controls from text"""
        import re
        controls = []

        matches = re.findall(self.control_pattern, text)
        for match in matches:
            # Split multiple controls
            control_list = [c.strip() for c in match.split(',')]
            controls.extend(control_list)

        return list(set(controls))

    def _extract_code_blocks(self, content: str) -> list[tuple[str, str]]:
        """Extract code blocks with language"""
        import re
        code_blocks = []

        pattern = r'```(\w*)\n([\s\S]*?)```'
        matches = re.findall(pattern, content)

        for lang, code in matches:
            lang = lang or "plaintext"
            code_blocks.append((lang, code.strip()))

        return code_blocks

    def _find_code_explanation(self, content: str, code: str) -> str | None:
        """Find explanation for a code block"""
        # Look for paragraph before or after code block
        code_position = content.find(code)
        if code_position == -1:
            return None

        # Get surrounding text
        before = content[:code_position].strip()
        after = content[code_position + len(code):].strip()

        # Look for explanation patterns
        before_paras = before.split('\n\n')
        if before_paras:
            last_para = before_paras[-1]
            if len(last_para) > 50:
                return last_para

        after_paras = after.split('\n\n')
        if after_paras:
            first_para = after_paras[0]
            if len(first_para) > 50 and not first_para.startswith('```'):
                return first_para

        return None

    def _extract_examples(self, content: str) -> list[str]:
        """Extract example sections"""
        import re
        examples = []

        # Look for example patterns
        example_pattern = r'(?:Example|e\.g\.|For example)[:\s]+([^.]+(?:\.[^.]+)*)'
        matches = re.findall(example_pattern, content, re.IGNORECASE)
        examples.extend(matches)

        # Also include code blocks as examples
        code_blocks = self._extract_code_blocks(content)
        for lang, code in code_blocks:
            examples.append(f"```{lang}\n{code}\n```")

        return examples

    def _build_navigation(self, chunks: list[MicroStandard]) -> None:
        """Build navigation links between chunks"""
        if not chunks:
            return

        # Set up prev/next navigation
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.navigation["prev"] = chunks[i-1].id
            if i < len(chunks) - 1:
                chunk.navigation["next"] = chunks[i+1].id

            # Link to overview
            chunk.navigation["up"] = chunks[0].id

    def create_index(self, chunks: list[MicroStandard], standard: Standard) -> MicroStandardIndex:
        """Create an index for the micro standards"""
        if not chunks:
            raise ValueError("No chunks to index")

        index = MicroStandardIndex(
            standard_id=standard.id,
            total_chunks=len(chunks),
            overview_chunk_id=chunks[0].id,
            chunk_hierarchy={},
            topic_index={},
            concept_index={},
            control_index={},
            chunk_map={}
        )

        # Build indexes
        for chunk in chunks:
            # Chunk map
            index.chunk_map[chunk.id] = chunk.title

            # Topic index
            for topic in chunk.topics:
                topic_lower = topic.lower()
                if topic_lower not in index.topic_index:
                    index.topic_index[topic_lower] = []
                index.topic_index[topic_lower].append(chunk.id)

            # Concept index
            for concept in chunk.concepts:
                concept_lower = concept.lower()
                if concept_lower not in index.concept_index:
                    index.concept_index[concept_lower] = []
                index.concept_index[concept_lower].append(chunk.id)

            # Control index
            for control in chunk.nist_controls:
                control_upper = control.upper()
                if control_upper not in index.control_index:
                    index.control_index[control_upper] = []
                index.control_index[control_upper].append(chunk.id)

            # Hierarchy (simplified - overview is parent of all)
            if chunk.parent_id:
                if chunk.parent_id not in index.chunk_hierarchy:
                    index.chunk_hierarchy[chunk.parent_id] = []
                index.chunk_hierarchy[chunk.parent_id].append(chunk.id)

        return index

    def save_chunks(self, chunks: list[MicroStandard], output_dir: Path) -> None:
        """Save chunks to disk"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save individual chunks
        for chunk in chunks:
            chunk_file = output_dir / f"{chunk.id}.json"
            with open(chunk_file, 'w') as f:
                json.dump(chunk.to_dict(), f, indent=2)

        # Save index
        if chunks and chunks[0].standard_id:
            standard = Standard(
                id=chunks[0].standard_id,
                title=chunks[0].title.split(' - ')[0],
                category="",
                sections=[]
            )
            index = self.create_index(chunks, standard)
            index_file = output_dir / f"{standard.id}_index.json"
            with open(index_file, 'w') as f:
                json.dump(index.model_dump(), f, indent=2, default=str)

    def load_chunks(self, standard_id: str, input_dir: Path) -> tuple[list[MicroStandard], MicroStandardIndex]:
        """Load chunks from disk"""
        input_dir = Path(input_dir)

        # Load index
        index_file = input_dir / f"{standard_id}_index.json"
        with open(index_file) as f:
            index_data = json.load(f)
            index = MicroStandardIndex(**index_data)

        # Load chunks
        chunks = []
        for chunk_id in index.chunk_map:
            chunk_file = input_dir / f"{chunk_id}.json"
            with open(chunk_file) as f:
                chunk_data = json.load(f)
                # Convert datetime string back to datetime
                chunk_data['created_at'] = datetime.fromisoformat(chunk_data['created_at'])
                chunk = MicroStandard(**chunk_data)
                chunks.append(chunk)

        return chunks, index


# Convenience functions
async def generate_micro_standards(
    standard: Standard,
    output_dir: Path | None = None
) -> tuple[list[MicroStandard], MicroStandardIndex]:
    """Generate micro standards from a full standard"""
    generator = MicroStandardsGenerator()
    chunks = generator.generate_chunks(standard)
    index = generator.create_index(chunks, standard)

    if output_dir:
        generator.save_chunks(chunks, output_dir)

    return chunks, index
