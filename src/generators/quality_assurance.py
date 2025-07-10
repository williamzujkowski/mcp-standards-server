"""
Quality Assurance System

Comprehensive quality assessment for generated standards.
"""

import re
from collections import Counter
from datetime import datetime
from typing import Any

from .metadata import StandardMetadata


class QualityAssuranceSystem:
    """Quality assurance system for standards documents."""

    def __init__(self) -> None:
        """Initialize the QA system."""
        self.quality_metrics = [
            "completeness",
            "consistency",
            "clarity",
            "compliance_coverage",
            "implementation_guidance",
            "maintainability",
        ]

    def assess_standard(
        self, content: str, metadata: StandardMetadata
    ) -> dict[str, Any]:
        """
        Assess the quality of a standard document.

        Args:
            content: The standard document content
            metadata: Standard metadata

        Returns:
            Quality assessment results
        """
        results = {
            "overall_score": 0,
            "scores": {},
            "recommendations": [],
            "assessment_date": datetime.now().isoformat(),
        }

        # Run all quality checks
        completeness_score = self._assess_completeness(content, metadata)
        consistency_score = self._assess_consistency(content, metadata)
        clarity_score = self._assess_clarity(content)
        compliance_score = self._assess_compliance_coverage(content, metadata)
        implementation_score = self._assess_implementation_guidance(content, metadata)
        maintainability_score = self._assess_maintainability(content, metadata)

        # Store individual scores
        results["scores"] = {
            "completeness": completeness_score,
            "consistency": consistency_score,
            "clarity": clarity_score,
            "compliance_coverage": compliance_score,
            "implementation_guidance": implementation_score,
            "maintainability": maintainability_score,
        }

        # Calculate overall score (weighted average)
        weights = {
            "completeness": 0.25,
            "consistency": 0.15,
            "clarity": 0.20,
            "compliance_coverage": 0.15,
            "implementation_guidance": 0.15,
            "maintainability": 0.10,
        }

        scores_dict = results["scores"]
        if isinstance(scores_dict, dict):
            overall_score = sum(
                scores_dict.get(metric, 0.0) * weights[metric]
                for metric in self.quality_metrics
            )
        else:
            overall_score = 0.0
        results["overall_score"] = round(overall_score, 2)

        # Generate recommendations
        scores_dict = results["scores"]
        if isinstance(scores_dict, dict):
            results["recommendations"] = self._generate_recommendations(scores_dict)
        else:
            results["recommendations"] = []

        return results

    def _assess_completeness(self, content: str, metadata: StandardMetadata) -> float:
        """Assess document completeness."""
        score: float = 0.0
        max_score = 100

        # Required sections check
        required_sections = [
            "Purpose",
            "Scope",
            "Implementation",
            "Compliance",
            "References",
            "Appendix",
        ]

        sections_present = 0
        for section in required_sections:
            if section.lower() in content.lower():
                sections_present += 1

        score += (sections_present / len(required_sections)) * 30

        # Content length check
        if len(content) > 2000:
            score += 20
        elif len(content) > 1000:
            score += 10

        # Metadata completeness
        metadata_fields = [
            metadata.description,
            metadata.author,
            metadata.tags,
            metadata.nist_controls,
            metadata.compliance_frameworks,
        ]

        filled_fields = sum(1 for field in metadata_fields if field)
        score += (filled_fields / len(metadata_fields)) * 20

        # Examples and code blocks
        if "```" in content:
            score += 15

        # Images and diagrams
        if "![" in content or "```mermaid" in content:
            score += 15

        return min(score, max_score)

    def _assess_consistency(self, content: str, metadata: StandardMetadata) -> float:
        """Assess document consistency."""
        score: float = 0.0
        max_score = 100

        # Heading consistency
        headings = re.findall(r"^(#+)\s+(.+)$", content, re.MULTILINE)
        if headings:
            # Check heading level consistency
            level_changes = []
            for i in range(1, len(headings)):
                prev_level = len(headings[i - 1][0])
                curr_level = len(headings[i][0])
                level_changes.append(abs(curr_level - prev_level))

            if level_changes:
                consistency_ratio = sum(
                    1 for change in level_changes if change <= 1
                ) / len(level_changes)
                score += consistency_ratio * 25

        # Terminology consistency
        terms = self._extract_technical_terms(content)
        term_counts = Counter(terms)

        # Check for consistent term usage
        if term_counts:
            most_common = term_counts.most_common(10)
            consistent_terms = sum(1 for term, count in most_common if count > 1)
            score += (consistent_terms / len(most_common)) * 25

        # Formatting consistency
        code_blocks = re.findall(r"```(\w+)?", content)
        if code_blocks:
            language_consistency = len(set(code_blocks)) / len(code_blocks)
            score += (1 - language_consistency) * 25

        # Reference consistency
        references = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
        if references:
            # Check for consistent reference formatting
            score += 25

        return min(score, max_score)

    def _assess_clarity(self, content: str) -> float:
        """Assess document clarity."""
        score: float = 0.0
        max_score = 100

        # Sentence length analysis
        sentences = re.split(r"[.!?]+", content)
        sentence_lengths = [
            len(sentence.split()) for sentence in sentences if sentence.strip()
        ]

        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            # Optimal sentence length is 15-20 words
            if 15 <= avg_length <= 20:
                score += 25
            elif 10 <= avg_length <= 25:
                score += 20
            else:
                score += 10

        # Paragraph structure
        paragraphs = content.split("\n\n")
        paragraph_lengths = [len(p.split()) for p in paragraphs if p.strip()]

        if paragraph_lengths:
            avg_para_length = sum(paragraph_lengths) / len(paragraph_lengths)
            # Optimal paragraph length is 50-150 words
            if 50 <= avg_para_length <= 150:
                score += 25
            elif 30 <= avg_para_length <= 200:
                score += 20
            else:
                score += 10

        # Use of active voice (simple heuristic)
        passive_indicators = ["is", "are", "was", "were", "been", "being"]
        active_indicators = ["implement", "configure", "deploy", "execute"]

        passive_count = sum(content.lower().count(word) for word in passive_indicators)
        active_count = sum(content.lower().count(word) for word in active_indicators)

        if active_count > passive_count:
            score += 25
        elif active_count > 0:
            score += 15

        # Clarity keywords
        clarity_keywords = [
            "however",
            "therefore",
            "furthermore",
            "specifically",
            "for example",
        ]
        clarity_count = sum(
            content.lower().count(keyword) for keyword in clarity_keywords
        )

        if clarity_count > 0:
            score += 25

        return min(score, max_score)

    def _assess_compliance_coverage(
        self, content: str, metadata: StandardMetadata
    ) -> float:
        """Assess compliance framework coverage."""
        score: float = 0.0
        max_score = 100

        # NIST controls coverage
        if metadata.nist_controls:
            controls_mentioned = sum(
                1 for control in metadata.nist_controls if control in content
            )
            coverage_ratio = controls_mentioned / len(metadata.nist_controls)
            score += coverage_ratio * 40

        # Compliance frameworks coverage
        if metadata.compliance_frameworks:
            frameworks_mentioned = sum(
                1
                for framework in metadata.compliance_frameworks
                if framework in content
            )
            coverage_ratio = frameworks_mentioned / len(metadata.compliance_frameworks)
            score += coverage_ratio * 30

        # Risk assessment
        risk_keywords = ["risk", "threat", "vulnerability", "mitigation", "assessment"]
        risk_mentions = sum(content.lower().count(keyword) for keyword in risk_keywords)

        if risk_mentions > 5:
            score += 30
        elif risk_mentions > 0:
            score += 20

        return min(score, max_score)

    def _assess_implementation_guidance(
        self, content: str, metadata: StandardMetadata
    ) -> float:
        """Assess implementation guidance quality."""
        score: float = 0.0
        max_score = 100

        # Code examples
        code_blocks = re.findall(r"```[^`]+```", content, re.DOTALL)
        if code_blocks:
            score += min(len(code_blocks) * 10, 30)

        # Step-by-step instructions
        numbered_lists = re.findall(r"^\d+\.", content, re.MULTILINE)
        if numbered_lists:
            score += min(len(numbered_lists) * 2, 20)

        # Configuration examples
        config_keywords = ["configuration", "config", "setup", "install", "deploy"]
        config_mentions = sum(
            content.lower().count(keyword) for keyword in config_keywords
        )

        if config_mentions > 3:
            score += 20
        elif config_mentions > 0:
            score += 10

        # Tools and technologies
        if metadata.type == "technical":
            tech_keywords = ["tool", "framework", "library", "platform", "technology"]
            tech_mentions = sum(
                content.lower().count(keyword) for keyword in tech_keywords
            )

            if tech_mentions > 5:
                score += 30
            elif tech_mentions > 0:
                score += 20

        return min(score, max_score)

    def _assess_maintainability(
        self, content: str, metadata: StandardMetadata
    ) -> float:
        """Assess document maintainability."""
        score: float = 0.0
        max_score = 100

        # Version information
        if metadata.version:
            score += 25

        # Author information
        if metadata.author:
            score += 20

        # Update history
        if "changelog" in content.lower() or "history" in content.lower():
            score += 20

        # Dependencies documentation
        if metadata.dependencies:
            score += 15

        # Review process
        if metadata.review_status != "draft":
            score += 20

        return min(score, max_score)

    def _extract_technical_terms(self, content: str) -> list[str]:
        """Extract technical terms from content."""
        # Simple extraction based on capitalized words and technical patterns
        terms = []

        # Find capitalized words (likely technical terms)
        capitalized = re.findall(r"\b[A-Z][a-z]+\b", content)
        terms.extend(capitalized)

        # Find acronyms
        acronyms = re.findall(r"\b[A-Z]{2,}\b", content)
        terms.extend(acronyms)

        # Find code-like terms
        code_terms = re.findall(r"`([^`]+)`", content)
        terms.extend(code_terms)

        return terms

    def _generate_recommendations(self, scores: dict[str, float]) -> list[str]:
        """Generate improvement recommendations based on scores."""
        recommendations = []

        for metric, score in scores.items():
            if score < 60:
                recommendations.extend(self._get_metric_recommendations(metric, score))

        return recommendations

    def _get_metric_recommendations(self, metric: str, score: float) -> list[str]:
        """Get specific recommendations for a metric."""
        recommendations = []

        if metric == "completeness":
            recommendations.extend(
                [
                    "Add missing required sections (Purpose, Scope, Implementation, Compliance)",
                    "Include more detailed content and examples",
                    "Add code examples and configuration samples",
                    "Include diagrams or visual aids",
                ]
            )

        elif metric == "consistency":
            recommendations.extend(
                [
                    "Ensure consistent heading hierarchy",
                    "Use consistent terminology throughout",
                    "Standardize code block formatting",
                    "Maintain consistent reference format",
                ]
            )

        elif metric == "clarity":
            recommendations.extend(
                [
                    "Simplify complex sentences",
                    "Use active voice where possible",
                    "Add transition words and phrases",
                    "Break up long paragraphs",
                ]
            )

        elif metric == "compliance_coverage":
            recommendations.extend(
                [
                    "Reference all NIST controls mentioned in metadata",
                    "Include compliance framework mappings",
                    "Add risk assessment content",
                    "Include threat modeling information",
                ]
            )

        elif metric == "implementation_guidance":
            recommendations.extend(
                [
                    "Add step-by-step implementation instructions",
                    "Include more code examples",
                    "Add configuration templates",
                    "Reference specific tools and technologies",
                ]
            )

        elif metric == "maintainability":
            recommendations.extend(
                [
                    "Add version history and changelog",
                    "Include author and reviewer information",
                    "Document dependencies and prerequisites",
                    "Establish review and update processes",
                ]
            )

        return recommendations
