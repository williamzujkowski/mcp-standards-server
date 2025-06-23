"""
Token optimization engine for achieving 90% token reduction
@nist-controls: SI-10, SI-12, AC-4
@evidence: Intelligent content optimization and filtering
@oscal-component: standards-engine
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

from ..tokenizer import BaseTokenizer, get_default_tokenizer


class OptimizationLevel(str, Enum):
    """Optimization levels for content reduction"""
    MINIMAL = "minimal"      # ~30% reduction
    MODERATE = "moderate"    # ~60% reduction
    AGGRESSIVE = "aggressive"  # ~90% reduction


@dataclass
class OptimizationMetrics:
    """Metrics for optimization effectiveness"""
    original_tokens: int
    optimized_tokens: int
    reduction_percentage: float
    information_retained: float
    processing_time: float


class ContentSection(BaseModel):
    """Represents a section of content with metadata"""
    title: str
    content: str
    importance: float  # 0.0 to 1.0
    keywords: list[str]
    concepts: list[str]
    requirements: list[str]
    examples: list[str]
    token_count: int


class OptimizationStrategy(ABC):
    """Base class for optimization strategies"""

    def __init__(self, tokenizer: BaseTokenizer | None = None):
        self.tokenizer = tokenizer or get_default_tokenizer()

    @abstractmethod
    async def optimize(self, content: str, max_tokens: int, context: dict[str, Any]) -> str:
        """Optimize content to fit within token limit"""
        pass

    def estimate_tokens(self, content: str) -> int:
        """Estimate token count for content"""
        return self.tokenizer.count_tokens(content)


class SummarizationStrategy(OptimizationStrategy):
    """
    Intelligent summarization using extraction and abstraction
    @nist-controls: SI-10
    @evidence: Content summarization with information preservation
    """

    def __init__(self, tokenizer: BaseTokenizer | None = None):
        super().__init__(tokenizer)
        self.requirement_keywords = {
            "MUST", "SHALL", "REQUIRED", "MUST NOT", "SHALL NOT",
            "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", "OPTIONAL"
        }
        self.security_keywords = {
            "authentication", "authorization", "encryption", "security",
            "compliance", "audit", "logging", "access control", "validation"
        }

    async def optimize(self, content: str, max_tokens: int, context: dict[str, Any]) -> str:
        """Create intelligent summary preserving critical information"""
        sections = self._parse_content(content)
        prioritized = self._prioritize_sections(sections, context)

        result = []
        token_budget = max_tokens

        # Always include requirements
        requirements = self._extract_requirements(content)
        if requirements:
            req_summary = self._summarize_requirements(requirements)
            result.append("## Key Requirements\n" + req_summary)
            token_budget -= self.estimate_tokens(req_summary)

        # Add sections by priority
        for section in prioritized:
            if token_budget <= 0:
                break

            if section.importance >= 0.8:
                # High importance - include more detail
                summarized = self._detailed_summary(section)
            elif section.importance >= 0.5:
                # Medium importance - moderate summary
                summarized = self._moderate_summary(section)
            else:
                # Low importance - brief mention
                summarized = self._brief_summary(section)

            section_tokens = self.estimate_tokens(summarized)
            if section_tokens <= token_budget:
                result.append(summarized)
                token_budget -= section_tokens

        return "\n\n".join(result)

    def _parse_content(self, content: str) -> list[ContentSection]:
        """Parse content into structured sections"""
        sections = []

        # Split by headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        parts = re.split(header_pattern, content, flags=re.MULTILINE)

        for i in range(1, len(parts), 3):
            if i + 2 < len(parts):
                len(parts[i])
                title = parts[i + 1].strip()
                section_content = parts[i + 2].strip()

                section = ContentSection(
                    title=title,
                    content=section_content,
                    importance=self._calculate_importance(title, section_content),
                    keywords=self._extract_keywords(section_content),
                    concepts=self._extract_concepts(section_content),
                    requirements=self._extract_requirements(section_content),
                    examples=self._extract_examples(section_content),
                    token_count=self.estimate_tokens(section_content)
                )
                sections.append(section)

        return sections

    def _calculate_importance(self, title: str, content: str) -> float:
        """Calculate section importance based on content analysis"""
        score = 0.5  # Base score

        # Title analysis
        title_lower = title.lower()
        if any(keyword in title_lower for keyword in ["requirement", "security", "compliance"]):
            score += 0.2
        if any(keyword in title_lower for keyword in ["example", "optional", "additional"]):
            score -= 0.2

        # Content analysis
        content_lower = content.lower()
        requirement_count = sum(1 for kw in self.requirement_keywords if kw in content)
        security_count = sum(1 for kw in self.security_keywords if kw in content_lower)

        score += min(0.3, requirement_count * 0.05)
        score += min(0.2, security_count * 0.03)

        return max(0.0, min(1.0, score))

    def _extract_requirements(self, content: str) -> list[str]:
        """Extract requirement statements"""
        requirements = []
        sentences = content.split('.')

        for sentence in sentences:
            if any(keyword in sentence.upper() for keyword in self.requirement_keywords):
                requirements.append(sentence.strip() + '.')

        return requirements

    def _extract_keywords(self, content: str) -> list[str]:
        """Extract important keywords from content"""
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', content)
        return list(set(words))[:10]

    def _extract_concepts(self, content: str) -> list[str]:
        """Extract high-level concepts"""
        concepts = []

        # Look for definition patterns
        definition_patterns = [
            r'(\w+)\s+is\s+(?:a|an|the)\s+(\w+)',
            r'(\w+)\s+refers\s+to\s+(\w+)',
            r'(\w+)\s+means\s+(\w+)'
        ]

        for pattern in definition_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            concepts.extend([match[0] for match in matches])

        return list(set(concepts))[:5]

    def _extract_examples(self, content: str) -> list[str]:
        """Extract example code or configurations"""
        examples = []

        # Code block pattern
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        examples.extend(code_blocks)

        # Example patterns
        example_pattern = r'(?:Example|e\.g\.|For example)[:\s]+([^.]+\.)'
        example_matches = re.findall(example_pattern, content, re.IGNORECASE)
        examples.extend(example_matches)

        return examples[:3]

    def _prioritize_sections(self, sections: list[ContentSection], context: dict[str, Any]) -> list[ContentSection]:
        """Prioritize sections based on context and importance"""
        # Adjust importance based on context
        if context.get("focus_area"):
            focus = context["focus_area"].lower()
            for section in sections:
                if focus in section.title.lower() or focus in section.content.lower():
                    section.importance = min(1.0, section.importance + 0.2)

        # Sort by importance
        return sorted(sections, key=lambda s: s.importance, reverse=True)

    def _summarize_requirements(self, requirements: list[str]) -> str:
        """Create concise summary of requirements"""
        if not requirements:
            return ""

        # Group similar requirements
        grouped: dict[str, list[str]] = {}
        for req in requirements:
            key = self._get_requirement_key(req)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(req)

        summary_parts = []
        for key, reqs in grouped.items():
            if len(reqs) == 1:
                summary_parts.append(f"- {reqs[0]}")
            else:
                summary_parts.append(f"- {key}: {len(reqs)} requirements")

        return "\n".join(summary_parts)

    def _get_requirement_key(self, requirement: str) -> str:
        """Extract key concept from requirement"""
        for keyword in self.security_keywords:
            if keyword in requirement.lower():
                return keyword.capitalize()
        return "General"

    def _detailed_summary(self, section: ContentSection) -> str:
        """Create detailed summary for high-importance sections"""
        summary_parts = [f"### {section.title}"]

        if section.requirements:
            summary_parts.append("\n**Requirements:**")
            for req in section.requirements[:5]:
                summary_parts.append(f"- {req}")

        if section.concepts:
            summary_parts.append("\n**Key Concepts:** " + ", ".join(section.concepts))

        # Include first paragraph
        first_para = section.content.split('\n\n')[0]
        if len(first_para) > 200:
            first_para = first_para[:200] + "..."
        summary_parts.append(f"\n{first_para}")

        return "\n".join(summary_parts)

    def _moderate_summary(self, section: ContentSection) -> str:
        """Create moderate summary for medium-importance sections"""
        summary_parts = [f"### {section.title}"]

        if section.requirements:
            summary_parts.append(f"\n{len(section.requirements)} requirements defined.")

        if section.keywords:
            summary_parts.append("**Topics:** " + ", ".join(section.keywords[:5]))

        return "\n".join(summary_parts)

    def _brief_summary(self, section: ContentSection) -> str:
        """Create brief summary for low-importance sections"""
        return f"- **{section.title}**: {section.token_count} tokens, {len(section.requirements)} requirements"

    def _truncate_to_fit(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token limit"""
        return self.tokenizer.truncate_to_tokens(content, max_tokens)

    def _extract_sentences(self, content: str) -> list[str]:
        """Extract sentences from content"""
        # Simple sentence extraction based on punctuation
        import re
        # Split on sentence-ending punctuation while keeping the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', content)
        return [s.strip() for s in sentences if s.strip()]


class EssentialOnlyStrategy(OptimizationStrategy):
    """
    Extract only essential requirements and compliance information
    @nist-controls: CM-2, CM-6
    @evidence: Configuration baseline extraction
    """

    def __init__(self, tokenizer: BaseTokenizer | None = None):
        super().__init__(tokenizer)
        self.essential_patterns = [
            # Requirements
            r'(?:MUST|SHALL|REQUIRED)\s+[^.]+\.',
            # Security controls
            r'@nist-controls?:\s*[A-Z]{2}-\d+(?:\(\d+\))?(?:,\s*[A-Z]{2}-\d+(?:\(\d+\))?)*',
            # Compliance statements
            r'(?:compliant?|compliance)\s+with\s+[^.]+\.',
            # Critical configurations
            r'(?:default|minimum|maximum)\s+(?:value|setting|configuration):\s*[^.]+\.',
        ]

    async def optimize(self, content: str, max_tokens: int, context: dict[str, Any]) -> str:
        """Extract only essential information"""
        essentials = []

        # Extract requirements
        requirements = self._extract_pattern_matches(content, self.essential_patterns[0])
        if requirements:
            essentials.append("## Requirements\n" + "\n".join(f"- {req}" for req in requirements))

        # Extract NIST controls
        controls = self._extract_pattern_matches(content, self.essential_patterns[1])
        if controls:
            essentials.append("## NIST Controls\n" + "\n".join(f"- {control}" for control in controls))

        # Extract compliance statements
        compliance = self._extract_pattern_matches(content, self.essential_patterns[2])
        if compliance:
            essentials.append("## Compliance\n" + "\n".join(f"- {comp}" for comp in compliance))

        # Extract configurations
        configs = self._extract_pattern_matches(content, self.essential_patterns[3])
        if configs:
            essentials.append("## Critical Configurations\n" + "\n".join(f"- {config}" for config in configs))

        result = "\n\n".join(essentials)

        # Ensure we fit within token budget
        if self.estimate_tokens(result) > max_tokens:
            result = self._truncate_to_fit(result, max_tokens)

        return result

    def _extract_pattern_matches(self, content: str, pattern: str) -> list[str]:
        """Extract all matches for a pattern"""
        matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
        return [match.strip() for match in matches]

    def _truncate_to_fit(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit token budget while preserving structure"""
        lines = content.split('\n')
        result: list[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = self.estimate_tokens(line)
            if current_tokens + line_tokens > max_tokens:
                if result:
                    result.append("... (truncated for token limit)")
                break
            result.append(line)
            current_tokens += line_tokens

        return '\n'.join(result)


class HierarchicalStrategy(OptimizationStrategy):
    """
    Create hierarchical content structure for progressive disclosure
    @nist-controls: SI-12
    @evidence: Information organization and presentation
    """

    def __init__(self, tokenizer: BaseTokenizer | None = None):
        super().__init__(tokenizer)
        self.max_depth = 3
        self.tokens_per_level = {
            1: 0.4,  # 40% for top level
            2: 0.4,  # 40% for second level
            3: 0.2   # 20% for details
        }

    async def optimize(self, content: str, max_tokens: int, context: dict[str, Any]) -> str:
        """Create hierarchical summary with expandable sections"""
        hierarchy = self._build_hierarchy(content)

        # Allocate tokens per level
        level_budgets = {
            level: int(max_tokens * ratio)
            for level, ratio in self.tokens_per_level.items()
        }

        result = []

        # Level 1: Overview
        overview = self._create_overview(hierarchy)
        result.append(overview)

        # Level 2: Main sections
        for section in hierarchy.get("sections", [])[:5]:  # Top 5 sections
            section_summary = self._create_section_summary(section, level_budgets[2] // 5)
            result.append(section_summary)

        # Level 3: Key details (if budget allows)
        if level_budgets[3] > 100:
            details = self._extract_key_details(hierarchy)
            result.append("\n## Key Details\n" + details)

        return "\n\n".join(result)

    def _build_hierarchy(self, content: str) -> dict[str, Any]:
        """Build hierarchical structure from content"""
        hierarchy: dict[str, Any] = {
            "title": "Document Summary",
            "overview": "",
            "sections": [],
            "details": []
        }

        # Parse sections
        sections = re.split(r'^#{1,3}\s+', content, flags=re.MULTILINE)

        for section in sections[1:]:  # Skip first empty split
            lines = section.strip().split('\n')
            if lines:
                title = lines[0]
                content = '\n'.join(lines[1:])

                hierarchy["sections"].append({
                    "title": title,
                    "content": content,
                    "subsections": self._parse_subsections(content)
                })

        # Create overview from first paragraph or section
        hierarchy["overview"] = self._extract_overview(content)

        return hierarchy

    def _parse_subsections(self, content: str) -> list[dict[str, str]]:
        """Parse subsections from content"""
        subsections = []

        # Look for bullet points or numbered lists
        list_items = re.findall(r'^\s*[-*\d]+\.\s+(.+)$', content, re.MULTILINE)
        for item in list_items[:5]:  # Top 5 items
            subsections.append({
                "title": item[:50] + "..." if len(item) > 50 else item,
                "content": item
            })

        return subsections

    def _create_overview(self, hierarchy: dict[str, Any]) -> str:
        """Create top-level overview"""
        overview_parts = ["# Document Overview\n"]

        if hierarchy["overview"]:
            overview_parts.append(hierarchy["overview"])

        overview_parts.append("\n## Table of Contents")
        for i, section in enumerate(hierarchy["sections"][:10], 1):
            overview_parts.append(f"{i}. {section['title']}")

        return "\n".join(overview_parts)

    def _create_section_summary(self, section: dict[str, Any], token_budget: int) -> str:
        """Create summary for a section"""
        summary_parts = [f"## {section['title']}"]

        # First paragraph
        paragraphs = section["content"].split('\n\n')
        if paragraphs:
            first_para = paragraphs[0][:200]
            if len(paragraphs[0]) > 200:
                first_para += "..."
            summary_parts.append(first_para)

        # Key points from subsections
        if section["subsections"]:
            summary_parts.append("\n**Key Points:**")
            for sub in section["subsections"][:3]:
                summary_parts.append(f"- {sub['title']}")

        return "\n".join(summary_parts)

    def _extract_overview(self, content: str) -> str:
        """Extract overview from content"""
        # Try to find an explicit overview or introduction
        overview_match = re.search(
            r'(?:overview|introduction|abstract|summary)[:\s]*([^#]+)',
            content,
            re.IGNORECASE
        )

        if overview_match:
            return overview_match.group(1).strip()[:300]

        # Fall back to first paragraph
        paragraphs = content.split('\n\n')
        if paragraphs:
            return paragraphs[0][:300]

        return "No overview available."

    def _extract_key_details(self, hierarchy: dict[str, Any]) -> str:
        """Extract important details across all sections"""
        details = []

        for section in hierarchy["sections"]:
            # Look for important patterns
            important_patterns = [
                r'(?:important|critical|essential|key):\s*([^.]+\.)',
                r'(?:note|warning|caution):\s*([^.]+\.)'
            ]

            for pattern in important_patterns:
                matches = re.findall(pattern, section["content"], re.IGNORECASE)
                details.extend(matches)

        if details:
            return "\n".join(f"- {detail}" for detail in details[:5])

        return "No critical details identified."

    def _parse_sections(self, content: str) -> list[dict[str, Any]]:
        """Parse content into sections"""
        sections = []

        # Split by headers
        import re
        lines = content.split('\n')
        current_section = None

        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                if current_section:
                    sections.append(current_section)

                level = len(header_match.group(1))
                header = header_match.group(2)
                current_section = {
                    "header": header,
                    "content": "",
                    "level": level
                }
            elif current_section:
                current_section["content"] = str(current_section["content"]) + line + '\n'

        if current_section:
            sections.append(current_section)

        return sections



class TokenOptimizationEngine:
    """
    Main engine for token optimization
    @nist-controls: SI-10, SI-12, AC-4
    @evidence: Comprehensive content optimization system
    """

    def __init__(self, tokenizer: BaseTokenizer | None = None):
        self.tokenizer = tokenizer or get_default_tokenizer()
        self.strategies = {
            "summarize": SummarizationStrategy(self.tokenizer),
            "essential": EssentialOnlyStrategy(self.tokenizer),
            "hierarchical": HierarchicalStrategy(self.tokenizer)
        }
        self._metrics_history: list[OptimizationMetrics] = []

    async def optimize(
        self,
        content: str,
        strategy: str = "summarize",
        max_tokens: int = 1000,
        context: dict[str, Any] | None = None,
        level: OptimizationLevel = OptimizationLevel.MODERATE
    ) -> tuple[str, OptimizationMetrics]:
        """
        Optimize content using specified strategy

        Returns:
            Tuple of (optimized_content, metrics)
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        if context is None:
            context = {}

        # Add optimization level to context
        context["optimization_level"] = level

        # Measure performance
        import time
        start_time = time.time()

        # Get original metrics
        original_tokens = self.estimate_tokens(content)

        # Apply strategy
        optimizer = self.strategies[strategy]
        optimized = await optimizer.optimize(content, max_tokens, context)

        # Calculate metrics
        optimized_tokens = self.estimate_tokens(optimized)
        reduction = 1 - (optimized_tokens / original_tokens) if original_tokens > 0 else 0

        metrics = OptimizationMetrics(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            reduction_percentage=reduction * 100,
            information_retained=self._estimate_information_retention(content, optimized),
            processing_time=time.time() - start_time
        )

        self._metrics_history.append(metrics)

        return optimized, metrics

    def estimate_tokens(self, content: str) -> int:
        """Estimate token count"""
        return self.tokenizer.count_tokens(content)

    def _estimate_information_retention(self, original: str, optimized: str) -> float:
        """Estimate how much information was retained"""
        # Simple estimation based on keyword preservation
        original_words = set(original.lower().split())
        optimized_words = set(optimized.lower().split())

        # Important keywords to check
        important_keywords = {
            "must", "shall", "required", "security", "compliance",
            "authentication", "authorization", "encryption", "audit"
        }

        original_important = original_words & important_keywords
        optimized_important = optimized_words & important_keywords

        if not original_important:
            return 1.0

        return len(optimized_important) / len(original_important)

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of optimization metrics"""
        if not self._metrics_history:
            return {}

        avg_reduction = sum(m.reduction_percentage for m in self._metrics_history) / len(self._metrics_history)
        avg_retention = sum(m.information_retained for m in self._metrics_history) / len(self._metrics_history)
        avg_time = sum(m.processing_time for m in self._metrics_history) / len(self._metrics_history)

        return {
            "average_reduction": f"{avg_reduction:.1f}%",
            "average_retention": f"{avg_retention:.1%}",
            "average_processing_time": f"{avg_time:.3f}s",
            "total_optimizations": len(self._metrics_history)
        }


# Convenience function for standalone use
async def optimize_content(
    content: str,
    strategy: str = "summarize",
    max_tokens: int = 1000,
    level: OptimizationLevel = OptimizationLevel.MODERATE
) -> tuple[str, OptimizationMetrics]:
    """Convenience function to optimize content"""
    engine = TokenOptimizationEngine()
    return await engine.optimize(content, strategy, max_tokens, level=level)
