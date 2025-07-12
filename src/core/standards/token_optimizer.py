"""
Token Usage Optimization for MCP Standards Server.

This module provides comprehensive token optimization strategies including:
- Multiple format variants (full, condensed, reference, summary)
- Dynamic loading with progressive disclosure
- Token counting and budgeting for different models
- Advanced compression techniques
- Context-aware section selection
"""

import hashlib
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


class StandardFormat(Enum):
    """Available format variants for standards."""

    FULL = "full"  # Complete documentation with examples
    CONDENSED = "condensed"  # Key points and essential information
    REFERENCE = "reference"  # Quick lookup with minimal tokens
    SUMMARY = "summary"  # One-paragraph executive summary
    CUSTOM = "custom"  # User-defined format


class ModelType(Enum):
    """Supported model types for token counting."""

    GPT4 = "gpt-4"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE = "claude"
    CLAUDE_INSTANT = "claude-instant"
    CUSTOM = "custom"


@dataclass
class TokenBudget:
    """Token budget configuration."""

    total: int
    reserved_for_context: int = 1000
    reserved_for_response: int = 2000
    warning_threshold: float = 0.8  # Warn when 80% of budget used

    @property
    def available(self) -> int:
        """Calculate available tokens for content."""
        return self.total - self.reserved_for_context - self.reserved_for_response

    @property
    def warning_limit(self) -> int:
        """Calculate warning threshold."""
        return int(self.available * self.warning_threshold)


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    format_used: StandardFormat
    sections_included: list[str]
    sections_excluded: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class StandardSection:
    """Represents a section of a standard."""

    id: str
    title: str
    content: str
    priority: int = 5  # 1-10, higher is more important
    token_count: int = 0
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Calculate token count if not provided."""
        if self.token_count == 0:
            # Simple approximation: 1 token ≈ 4 characters
            self.token_count = len(self.content) // 4


class TokenCounter:
    """Accurate token counting for different models."""

    def __init__(self, model_type: ModelType = ModelType.GPT4) -> None:
        self.model_type = model_type
        self._encoders: dict[str, Any] = {}
        self._init_encoders()

    def _init_encoders(self) -> None:
        """Initialize token encoders for different models."""
        try:
            # GPT models use tiktoken
            if self.model_type in [ModelType.GPT4, ModelType.GPT35_TURBO]:
                encoding_name = (
                    "cl100k_base"
                    if self.model_type == ModelType.GPT4
                    else "cl100k_base"
                )
                self._encoders["default"] = tiktoken.get_encoding(encoding_name)
            else:
                # For Claude and others, use approximation
                self._encoders["default"] = None
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken encoder: {e}")
            self._encoders["default"] = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text for the configured model."""
        if self._encoders.get("default"):
            try:
                return len(self._encoders["default"].encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}")

        # Fallback approximation
        if self.model_type in [ModelType.CLAUDE, ModelType.CLAUDE_INSTANT]:
            # Claude typically uses ~1 token per 3.5 characters
            return len(text) // 3
        else:
            # Default approximation: 1 token ≈ 4 characters
            return len(text) // 4

    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """Count tokens for multiple texts efficiently."""
        return [self.count_tokens(text) for text in texts]


class CompressionTechniques:
    """Collection of compression techniques."""

    @staticmethod
    def remove_redundancy(text: str) -> str:
        """Remove redundant information from text."""
        # Remove multiple spaces within lines (but not newlines)
        text = re.sub(r"[ \t]+", " ", text)

        # Remove redundant newlines (3+ becomes 2)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove trailing spaces from each line while preserving newline structure
        # Split on \n\n first to preserve paragraph breaks
        paragraphs = text.split("\n\n")
        cleaned_paragraphs = []

        for paragraph in paragraphs:
            # Clean each paragraph
            lines = [line.rstrip() for line in paragraph.split("\n")]
            cleaned_paragraph = "\n".join(line for line in lines if line.strip())
            if cleaned_paragraph.strip():
                cleaned_paragraphs.append(cleaned_paragraph)

        return "\n\n".join(cleaned_paragraphs).strip()

    @staticmethod
    def use_abbreviations(text: str) -> str:
        """Replace common terms with abbreviations."""
        abbreviations = {
            "implementation": "impl",
            "configuration": "config",
            "documentation": "docs",
            "application": "app",
            "development": "dev",
            "production": "prod",
            "environment": "env",
            "repository": "repo",
            "authentication": "auth",
            "authorization": "authz",
            "administrator": "admin",
            "database": "db",
            "infrastructure": "infra",
            "kubernetes": "k8s",
            "javascript": "js",
            "typescript": "ts",
            "python": "py",
            "requirements": "reqs",
            "dependencies": "deps",
            "performance": "perf",
            "optimization": "opt",
            "security": "sec",
            "vulnerability": "vuln",
            "compliance": "compl",
            "standard": "std",
            "reference": "ref",
            "example": "ex",
            "information": "info",
            "management": "mgmt",
            "monitoring": "mon",
            "observability": "o11y",
            "continuous integration": "CI",
            "continuous deployment": "CD",
            "pull request": "PR",
            "merge request": "MR",
            "quality assurance": "QA",
            "user interface": "UI",
            "user experience": "UX",
            "application programming interface": "API",
            "software development kit": "SDK",
            "command line interface": "CLI",
            "graphical user interface": "GUI",
        }

        # Case-insensitive replacement
        for full, abbr in abbreviations.items():
            text = re.sub(rf"\b{full}\b", abbr, text, flags=re.IGNORECASE)

        return text

    @staticmethod
    def compress_code_examples(text: str) -> str:
        """Compress code examples while maintaining readability."""
        # Find code blocks
        code_pattern = r"```[\s\S]*?```"

        def compress_code(match: re.Match[str]) -> str:
            code = match.group(0)
            lines = code.split("\n")

            # Keep language identifier
            if len(lines) > 2:
                compressed = [lines[0]]  # ```language

                # Remove empty lines and excessive whitespace
                for line in lines[1:-1]:
                    stripped = line.strip()
                    if (
                        stripped
                        and not stripped.startswith("//")
                        and not stripped.startswith("#")
                    ):
                        # Reduce indentation to 2 spaces
                        indent_level = (len(line) - len(line.lstrip())) // 4
                        compressed.append("  " * indent_level + stripped)

                compressed.append(lines[-1])  # ```
                return "\n".join(compressed)
            return code

        return re.sub(code_pattern, compress_code, text)

    @staticmethod
    def create_lookup_table(
        text: str, patterns: list[str]
    ) -> tuple[str, dict[str, str]]:
        """Create lookup tables for repeated patterns."""
        lookup_table = {}
        compressed_text = text

        # Find repeated phrases (>10 chars appearing 2+ times)
        words = text.split()
        phrase_counts: defaultdict[str, int] = defaultdict(int)

        # Count 3-word phrases
        for i in range(len(words) - 2):
            phrase = " ".join(words[i : i + 3])
            if len(phrase) > 10:  # Lowered from 20 to 10
                phrase_counts[phrase] += 1

        # Create lookup entries for frequent phrases
        lookup_id = 1
        for phrase, count in phrase_counts.items():
            if count >= 2:  # Lowered from 3 to 2
                key = f"[L{lookup_id}]"
                lookup_table[key] = phrase
                compressed_text = compressed_text.replace(phrase, key)
                lookup_id += 1

        return compressed_text, lookup_table

    @staticmethod
    def extract_essential_only(text: str) -> str:
        """Extract only essential information."""
        lines = text.split("\n")
        essential_lines = []

        # Keywords that indicate essential information
        essential_keywords = [
            "must",
            "should",
            "required",
            "critical",
            "important",
            "warning",
            "error",
            "security",
            "performance",
            "best practice",
            "do not",
            "don't",
            "avoid",
            "never",
            "always",
        ]

        for line in lines:
            lower_line = line.lower()
            # Keep headers
            if line.strip().startswith("#"):
                essential_lines.append(line)
            # Keep lines with essential keywords
            elif any(keyword in lower_line for keyword in essential_keywords):
                essential_lines.append(line)
            # Keep bullet points with key information
            elif line.strip().startswith(("- ", "* ", "1.", "2.", "3.")):
                if len(line.strip()) > 10:  # Avoid trivial bullets
                    essential_lines.append(line)

        return "\n".join(essential_lines)


class TokenOptimizer:
    """Main token optimization engine."""

    def __init__(
        self,
        model_type: ModelType = ModelType.GPT4,
        default_budget: TokenBudget | None = None,
    ):
        self.model_type = model_type
        self.token_counter = TokenCounter(model_type)
        self.compression = CompressionTechniques()
        self.default_budget = default_budget or TokenBudget(total=8000)

        # Cache for formatted content
        self._format_cache: dict[str, tuple[float, str, CompressionResult]] = {}
        self._cache_ttl = 3600  # 1 hour

        # Lookup tables for compression
        self._global_lookup_table: dict[str, str] = {}

    def optimize_standard(
        self,
        standard: dict[str, Any],
        format_type: StandardFormat = StandardFormat.CONDENSED,
        budget: TokenBudget | None = None,
        required_sections: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, CompressionResult]:
        """Optimize a standard for token usage."""
        budget = budget or self.default_budget

        # Check cache
        cache_key = self._get_cache_key(
            standard, format_type, required_sections, context
        )
        if cache_key in self._format_cache:
            cached_time, cached_content, cached_result = self._format_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_content, cached_result

        # Parse standard into sections
        sections = self._parse_standard_sections(standard)

        # Select format strategy
        if format_type == StandardFormat.FULL:
            content, result = self._format_full(sections, budget, required_sections)
        elif format_type == StandardFormat.CONDENSED:
            content, result = self._format_condensed(
                sections, budget, required_sections
            )
        elif format_type == StandardFormat.REFERENCE:
            content, result = self._format_reference(
                sections, budget, required_sections
            )
        elif format_type == StandardFormat.SUMMARY:
            content, result = self._format_summary(sections, budget)
        else:
            content, result = self._format_custom(
                sections, budget, required_sections, context
            )

        # Cache result
        self._format_cache[cache_key] = (time.time(), content, result)

        return content, result

    def _parse_standard_sections(
        self, standard: dict[str, Any]
    ) -> list[StandardSection]:
        """Parse standard into sections."""
        sections: list[StandardSection] = []
        content = standard.get("content", "")

        # Handle different content formats
        if isinstance(content, dict):
            # If content is a dictionary, convert it to a string representation
            content_parts = []
            for key, value in content.items():
                if isinstance(value, list):
                    content_parts.append(
                        f"## {key.title()}\n" + "\n".join(f"- {item}" for item in value)
                    )
                else:
                    content_parts.append(f"## {key.title()}\n{value}")
            content = "\n\n".join(content_parts)
        elif not isinstance(content, str):
            # Convert non-string, non-dict content to string
            content = str(content)

        if not content.strip():
            return sections

        # Common section patterns
        section_patterns = [
            ("overview", r"#{1,3}\s*Overview.*?(?=\n#{1,3}|\Z)", 8),
            ("requirements", r"#{1,3}\s*Requirements.*?(?=\n#{1,3}|\Z)", 9),
            ("implementation", r"#{1,3}\s*Implementation.*?(?=\n#{1,3}|\Z)", 7),
            ("examples", r"#{1,3}\s*Examples?.*?(?=\n#{1,3}|\Z)", 5),
            ("best_practices", r"#{1,3}\s*Best Practices.*?(?=\n#{1,3}|\Z)", 8),
            ("security", r"#{1,3}\s*Security.*?(?=\n#{1,3}|\Z)", 9),
            ("performance", r"#{1,3}\s*Performance.*?(?=\n#{1,3}|\Z)", 7),
            ("testing", r"#{1,3}\s*Testing.*?(?=\n#{1,3}|\Z)", 6),
            ("references", r"#{1,3}\s*References.*?(?=\n#{1,3}|\Z)", 3),
        ]

        # Track covered character ranges to avoid duplicates
        covered_ranges: list[tuple[int, int]] = []

        for section_id, pattern, priority in section_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                start, end = match.span()
                section_content = match.group(0)

                # Check if this range overlaps with already covered content
                overlaps = any(
                    not (end <= r_start or start >= r_end)
                    for r_start, r_end in covered_ranges
                )

                if not overlaps:
                    sections.append(
                        StandardSection(
                            id=section_id,
                            title=section_id.replace("_", " ").title(),
                            content=section_content,
                            priority=priority,
                            token_count=self.token_counter.count_tokens(
                                section_content
                            ),
                        )
                    )
                    covered_ranges.append((start, end))

        # If no sections were found, treat entire content as one section
        if not sections:
            sections.append(
                StandardSection(
                    id="content",
                    title="Standard Content",
                    content=content,
                    priority=5,
                    token_count=self.token_counter.count_tokens(content),
                )
            )
        else:
            # Add any significant uncovered content as "other"
            # Sort covered ranges and find gaps
            covered_ranges.sort()
            uncovered_parts = []

            # Check content before first range
            if covered_ranges and covered_ranges[0][0] > 0:
                uncovered_content = content[: covered_ranges[0][0]].strip()
                if len(uncovered_content) > 50:  # Only include substantial content
                    uncovered_parts.append(uncovered_content)

            # Check gaps between ranges
            for i in range(len(covered_ranges) - 1):
                gap_start = covered_ranges[i][1]
                gap_end = covered_ranges[i + 1][0]
                gap_content = content[gap_start:gap_end].strip()
                if len(gap_content) > 50:  # Only include substantial content
                    uncovered_parts.append(gap_content)

            # Check content after last range
            if covered_ranges and covered_ranges[-1][1] < len(content):
                uncovered_content = content[covered_ranges[-1][1] :].strip()
                if len(uncovered_content) > 50:  # Only include substantial content
                    uncovered_parts.append(uncovered_content)

            # Add uncovered content as "other" section if any
            if uncovered_parts:
                remaining_content = "\n\n".join(uncovered_parts)
                sections.append(
                    StandardSection(
                        id="other",
                        title="Additional Information",
                        content=remaining_content,
                        priority=4,
                        token_count=self.token_counter.count_tokens(remaining_content),
                    )
                )

        return sections

    def _format_full(
        self,
        sections: list[StandardSection],
        budget: TokenBudget,
        required_sections: list[str] | None,
    ) -> tuple[str, CompressionResult]:
        """Format standard in full format."""
        original_tokens = sum(s.token_count for s in sections)

        # If under budget, return everything
        if original_tokens <= budget.available:
            content = "\n\n".join(s.content for s in sections)
            return content, CompressionResult(
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=1.0,
                format_used=StandardFormat.FULL,
                sections_included=[s.id for s in sections],
                sections_excluded=[],
            )

        # Otherwise, apply light compression
        compressed_sections = []
        total_tokens = 0
        included = []
        excluded = []

        # Sort by priority
        sorted_sections = sorted(sections, key=lambda s: s.priority, reverse=True)

        for section in sorted_sections:
            # Apply light compression
            compressed_content = self.compression.remove_redundancy(section.content)
            compressed_tokens = self.token_counter.count_tokens(compressed_content)

            if total_tokens + compressed_tokens <= budget.available:
                compressed_sections.append(compressed_content)
                total_tokens += compressed_tokens
                included.append(section.id)
            else:
                excluded.append(section.id)

        content = "\n\n".join(compressed_sections)

        return content, CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=total_tokens,
            compression_ratio=(
                total_tokens / original_tokens if original_tokens > 0 else 0
            ),
            format_used=StandardFormat.FULL,
            sections_included=included,
            sections_excluded=excluded,
            warnings=["Some sections excluded due to token budget"] if excluded else [],
        )

    def _format_condensed(
        self,
        sections: list[StandardSection],
        budget: TokenBudget,
        required_sections: list[str] | None,
    ) -> tuple[str, CompressionResult]:
        """Format standard in condensed format."""
        original_tokens = sum(s.token_count for s in sections)

        compressed_sections = []
        total_tokens = 0
        included = []
        excluded = []

        # Prioritize required sections
        if required_sections:
            priority_sections = [s for s in sections if s.id in required_sections]
            other_sections = [s for s in sections if s.id not in required_sections]
            sorted_sections = priority_sections + sorted(
                other_sections, key=lambda s: s.priority, reverse=True
            )
        else:
            sorted_sections = sorted(sections, key=lambda s: s.priority, reverse=True)

        for section in sorted_sections:
            # Apply multiple compression techniques
            compressed_content = section.content
            compressed_content = self.compression.remove_redundancy(compressed_content)
            compressed_content = self.compression.use_abbreviations(compressed_content)
            compressed_content = self.compression.compress_code_examples(
                compressed_content
            )
            compressed_content = self.compression.extract_essential_only(
                compressed_content
            )

            compressed_tokens = self.token_counter.count_tokens(compressed_content)

            if total_tokens + compressed_tokens <= budget.available:
                compressed_sections.append(f"## {section.title}\n{compressed_content}")
                total_tokens += compressed_tokens
                included.append(section.id)
            else:
                excluded.append(section.id)

        content = "\n\n".join(compressed_sections)

        return content, CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=total_tokens,
            compression_ratio=(
                total_tokens / original_tokens if original_tokens > 0 else 0
            ),
            format_used=StandardFormat.CONDENSED,
            sections_included=included,
            sections_excluded=excluded,
            warnings=(
                ["Required sections may be compressed"] if required_sections else []
            ),
        )

    def _format_reference(
        self,
        sections: list[StandardSection],
        budget: TokenBudget,
        required_sections: list[str] | None,
    ) -> tuple[str, CompressionResult]:
        """Format standard as quick reference."""
        original_tokens = sum(s.token_count for s in sections)

        reference_parts = []

        # Extract key points from each section
        for section in sections:
            if required_sections and section.id not in required_sections:
                continue

            # Extract bullet points and key information
            lines = section.content.split("\n")
            key_points = []

            for line in lines:
                stripped = line.strip()
                # Skip headers - we'll use section.title instead
                if stripped.startswith("#"):
                    continue
                # Bullet points
                elif stripped.startswith(("- ", "* ", "• ")):
                    # Compress the bullet point
                    compressed = self.compression.use_abbreviations(stripped)
                    if len(compressed) > 80:
                        compressed = compressed[:77] + "..."
                    key_points.append(compressed)
                # Numbered items
                elif re.match(r"^\d+\.", stripped):
                    compressed = self.compression.use_abbreviations(stripped)
                    if len(compressed) > 80:
                        compressed = compressed[:77] + "..."
                    key_points.append(compressed)

            if key_points:
                reference_parts.append(
                    f"## {section.title}\n" + "\n".join(key_points[:5])
                )

        content = "\n\n".join(reference_parts)
        compressed_tokens = self.token_counter.count_tokens(content)

        # If still over budget, truncate
        if compressed_tokens > budget.available:
            # Calculate how much to keep
            keep_ratio = budget.available / compressed_tokens
            keep_chars = int(len(content) * keep_ratio * 0.9)  # 90% to be safe
            content = (
                content[:keep_chars] + "\n\n[Content truncated due to token limit]"
            )
            compressed_tokens = self.token_counter.count_tokens(content)

        return content, CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=(
                compressed_tokens / original_tokens if original_tokens > 0 else 0
            ),
            format_used=StandardFormat.REFERENCE,
            sections_included=[
                s.id
                for s in sections
                if not required_sections or s.id in required_sections
            ],
            sections_excluded=[
                s.id
                for s in sections
                if required_sections and s.id not in required_sections
            ],
            warnings=["Reference format shows key points only"],
        )

    def _format_summary(
        self, sections: list[StandardSection], budget: TokenBudget
    ) -> tuple[str, CompressionResult]:
        """Format standard as executive summary."""
        original_tokens = sum(s.token_count for s in sections)

        # Create a one-paragraph summary
        summary_parts = []

        # Find the most important content
        priority_sections = sorted(sections, key=lambda s: s.priority, reverse=True)[:3]

        for section in priority_sections:
            # Extract first substantial paragraph or key points
            lines = section.content.split("\n")
            for line in lines:
                stripped = line.strip()
                if len(stripped) > 50 and not stripped.startswith("#"):
                    # Compress and add
                    compressed = self.compression.use_abbreviations(stripped)
                    compressed = self.compression.remove_redundancy(compressed)
                    summary_parts.append(compressed)
                    break

        # Combine into a single paragraph
        summary = " ".join(summary_parts)

        # Ensure it fits in budget
        if self.token_counter.count_tokens(summary) > budget.available:
            # Truncate to fit
            keep_chars = int(budget.available * 3.5)  # Approximate chars from tokens
            summary = summary[: keep_chars - 3] + "..."

        # Add metadata
        content = f"**Summary**: {summary}\n\n**Sections covered**: {', '.join(s.title for s in priority_sections)}"

        compressed_tokens = self.token_counter.count_tokens(content)

        return content, CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=(
                compressed_tokens / original_tokens if original_tokens > 0 else 0
            ),
            format_used=StandardFormat.SUMMARY,
            sections_included=[s.id for s in priority_sections],
            sections_excluded=[s.id for s in sections if s not in priority_sections],
            warnings=["Summary format provides high-level overview only"],
        )

    def _format_custom(
        self,
        sections: list[StandardSection],
        budget: TokenBudget,
        required_sections: list[str] | None,
        context: dict[str, Any] | None,
    ) -> tuple[str, CompressionResult]:
        """Format standard with custom rules based on context."""
        # Default to condensed if no context
        if not context:
            return self._format_condensed(sections, budget, required_sections)

        # Analyze context to determine best approach
        query_type = context.get("query_type", "general")
        user_expertise = context.get("user_expertise", "intermediate")
        focus_areas = context.get("focus_areas", [])

        # Adjust section priorities based on context
        for section in sections:
            # Boost sections related to focus areas
            if any(area.lower() in section.content.lower() for area in focus_areas):
                section.priority += 2

            # Adjust based on expertise
            if user_expertise == "beginner" and section.id == "examples":
                section.priority += 3
            elif user_expertise == "expert" and section.id in ["overview", "examples"]:
                section.priority -= 2

        # Use appropriate format based on query type
        if query_type == "quick_lookup":
            return self._format_reference(sections, budget, required_sections)
        elif query_type == "detailed_explanation":
            return self._format_full(sections, budget, required_sections)
        else:
            return self._format_condensed(sections, budget, required_sections)

    def progressive_load(
        self, standard: dict[str, Any], initial_sections: list[str], max_depth: int = 3
    ) -> list[list[tuple[str, int]]]:
        """Generate progressive loading plan for standard."""
        sections = self._parse_standard_sections(standard)

        # Build dependency graph
        dependency_graph: dict[str, dict[str, Any]] = {}
        for section in sections:
            dependency_graph[section.id] = {
                "section": section,
                "dependencies": section.dependencies,
                "dependents": [],
            }

        # Find dependents
        for section_id, data in dependency_graph.items():
            for dep in data["dependencies"]:
                if dep in dependency_graph:
                    dependency_graph[dep]["dependents"].append(section_id)

        # Generate loading order
        loading_plan = []
        loaded = set()
        to_load = set(initial_sections)
        depth = 0

        while to_load and depth < max_depth:
            current_batch = []

            for section_id in to_load:
                if section_id in dependency_graph and section_id not in loaded:
                    section = dependency_graph[section_id]["section"]
                    current_batch.append((section_id, section.token_count))
                    loaded.add(section_id)

            if current_batch:
                loading_plan.append(current_batch)

            # Find next batch (dependencies and high-priority dependents)
            next_batch: set[str] = set()
            for section_id in to_load:
                if section_id in dependency_graph:
                    # Add dependencies
                    next_batch.update(dependency_graph[section_id]["dependencies"])
                    # Add high-priority dependents
                    for dep in dependency_graph[section_id]["dependents"]:
                        if dependency_graph[dep]["section"].priority >= 7:
                            next_batch.add(dep)

            to_load = next_batch - loaded
            depth += 1

        # If we only have the initial sections and no expansion occurred,
        # add a fallback batch with high-priority sections
        if len(loading_plan) == 1 and len(loaded) == len(initial_sections):
            high_priority_sections = [
                s for s in sections if s.id not in loaded and s.priority >= 7
            ]
            if high_priority_sections:
                # Sort by priority and take up to 3 additional sections
                high_priority_sections.sort(key=lambda s: s.priority, reverse=True)
                fallback_batch = [
                    (section.id, section.token_count)
                    for section in high_priority_sections[:3]
                ]
                if fallback_batch:
                    loading_plan.append(fallback_batch)

        # Return the loading plan with batches preserved
        return loading_plan

    def estimate_tokens(
        self,
        standards: list[dict[str, Any]],
        format_type: StandardFormat = StandardFormat.CONDENSED,
    ) -> dict[str, Any]:
        """Estimate token usage for multiple standards."""
        estimates = []
        total_original = 0
        total_compressed = 0

        for standard in standards:
            sections = self._parse_standard_sections(standard)
            original = sum(s.token_count for s in sections)

            # Estimate compressed size based on format
            compression_factors = {
                StandardFormat.FULL: 0.9,
                StandardFormat.CONDENSED: 0.5,
                StandardFormat.REFERENCE: 0.2,
                StandardFormat.SUMMARY: 0.05,
            }

            factor = compression_factors.get(format_type, 0.5)
            compressed = int(original * factor)

            estimates.append(
                {
                    "standard_id": standard.get("id", "unknown"),
                    "original_tokens": original,
                    "estimated_compressed": compressed,
                    "compression_ratio": factor,
                }
            )

            total_original += original
            total_compressed += compressed

        return {
            "standards": estimates,
            "total_original": total_original,
            "total_compressed": total_compressed,
            "overall_compression": (
                total_compressed / total_original if total_original > 0 else 0
            ),
            "format_used": format_type.value,
        }

    def auto_select_format(
        self,
        standard: dict[str, Any],
        budget: TokenBudget,
        context: dict[str, Any] | None = None,
    ) -> StandardFormat:
        """Automatically select best format based on budget and context."""
        sections = self._parse_standard_sections(standard)
        total_tokens = sum(s.token_count for s in sections)

        # Calculate available token ratio
        token_ratio = budget.available / total_tokens if total_tokens > 0 else 1.0

        # Context hints
        if context:
            if context.get("query_type") == "quick_lookup":
                return StandardFormat.REFERENCE
            elif context.get("need_examples") and token_ratio > 0.7:
                return StandardFormat.FULL

        # Select based on token ratio
        if token_ratio >= 0.8:
            return StandardFormat.FULL
        elif token_ratio >= 0.4:
            return StandardFormat.CONDENSED
        elif token_ratio >= 0.15:
            return StandardFormat.REFERENCE
        else:
            return StandardFormat.SUMMARY

    def get_compression_stats(self) -> dict[str, Any]:
        """Get statistics about compression performance."""
        cache_stats = {
            "cache_size": len(self._format_cache),
            "cache_hits": getattr(self, "_cache_hits", 0),
            "cache_misses": getattr(self, "_cache_misses", 0),
        }

        if self._format_cache:
            # Analyze cached results
            compression_ratios = []
            format_usage: defaultdict[str, int] = defaultdict(int)

            for _, (_, _, result) in self._format_cache.items():
                compression_ratios.append(result.compression_ratio)
                format_usage[result.format_used.value] += 1

            cache_stats.update(
                {
                    "average_compression_ratio": sum(compression_ratios)
                    / len(compression_ratios),
                    "format_usage": dict(format_usage),
                }
            )

        return cache_stats

    def _get_cache_key(
        self,
        standard: dict[str, Any],
        format_type: StandardFormat,
        required_sections: list[str] | None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Generate cache key for formatted content."""
        key_parts = [
            standard.get("id", "unknown"),
            format_type.value,
            ",".join(sorted(required_sections)) if required_sections else "all",
        ]

        # Include context for CUSTOM format to avoid cache collisions
        if format_type == StandardFormat.CUSTOM and context:
            context_key = []
            for key in sorted(context.keys()):
                context_key.append(f"{key}:{context[key]}")
            key_parts.append("ctx:" + "|".join(context_key))

        key_string = "|".join(key_parts)
        # Use SHA-256 with application-specific salt
        salted_key = f"mcp_token_opt_v1:{key_string}"
        return hashlib.sha256(salted_key.encode()).hexdigest()


class DynamicLoader:
    """Handles dynamic loading of standard sections."""

    def __init__(self, optimizer: TokenOptimizer) -> None:
        self.optimizer = optimizer
        self._loaded_sections: defaultdict[str, set[str]] = defaultdict(set)
        self._loading_history: defaultdict[str, list[dict[str, Any]]] = defaultdict(
            list
        )

    def load_section(
        self, standard_id: str, section_id: str, budget: TokenBudget
    ) -> tuple[str, int]:
        """Load a specific section dynamically."""
        # Track loading
        self._loaded_sections[standard_id].add(section_id)
        self._loading_history[standard_id].append(
            {
                "section_id": section_id,
                "timestamp": time.time(),
                "budget_used": 0,  # Will be updated
            }
        )

        # In practice, this would fetch from storage
        # For now, return placeholder
        content = f"[Section {section_id} content for {standard_id}]"
        tokens = self.optimizer.token_counter.count_tokens(content)

        # Update history
        self._loading_history[standard_id][-1]["budget_used"] = tokens

        return content, tokens

    def get_loading_suggestions(
        self, standard_id: str, context: dict[str, Any]
    ) -> list[str]:
        """Suggest next sections to load based on context."""
        loaded = self._loaded_sections.get(standard_id, set())

        # Analyze context to suggest sections
        suggestions = []

        # Based on recent queries
        if "recent_queries" in context:
            for query in context["recent_queries"]:
                if "security" in query.lower() and "security" not in loaded:
                    suggestions.append("security")
                elif "example" in query.lower() and "examples" not in loaded:
                    suggestions.append("examples")
                elif "performance" in query.lower() and "performance" not in loaded:
                    suggestions.append("performance")

        # Based on user expertise
        expertise = context.get("user_expertise", "intermediate")
        if expertise == "beginner" and "examples" not in loaded:
            suggestions.append("examples")
        elif expertise == "expert" and "implementation" not in loaded:
            suggestions.append("implementation")

        return list(set(suggestions))  # Remove duplicates

    def get_loading_stats(self, standard_id: str) -> dict[str, Any]:
        """Get statistics about dynamic loading for a standard."""
        loaded = self._loaded_sections.get(standard_id, set())
        history = self._loading_history.get(standard_id, [])

        total_tokens = sum(h["budget_used"] for h in history)

        return {
            "sections_loaded": list(loaded),
            "total_sections": len(loaded),
            "total_tokens_used": total_tokens,
            "loading_events": len(history),
            "average_tokens_per_section": total_tokens / len(loaded) if loaded else 0,
        }


# Utility functions
def create_token_optimizer(
    model_type: str | ModelType = ModelType.GPT4, default_budget: int | None = None
) -> TokenOptimizer:
    """Create a token optimizer instance."""
    if isinstance(model_type, str):
        model_type = ModelType(model_type)

    budget = None
    if default_budget:
        budget = TokenBudget(total=default_budget)

    return TokenOptimizer(model_type=model_type, default_budget=budget)


def estimate_token_savings(
    original_text: str, optimizer: TokenOptimizer, formats: list[StandardFormat]
) -> dict[str, Any]:
    """Estimate token savings across different formats."""
    original_tokens = optimizer.token_counter.count_tokens(original_text)

    savings = {}
    for format_type in formats:
        # Create mock standard
        mock_standard = {"content": original_text}

        # Optimize
        _, result = optimizer.optimize_standard(mock_standard, format_type=format_type)

        savings[format_type.value] = {
            "tokens": result.compressed_tokens,
            "reduction": original_tokens - result.compressed_tokens,
            "percentage": (
                ((original_tokens - result.compressed_tokens) / original_tokens * 100)
                if original_tokens > 0
                else 0
            ),
        }

    return {"original_tokens": original_tokens, "format_savings": savings}
