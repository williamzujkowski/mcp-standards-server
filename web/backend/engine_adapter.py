"""
Adapter for the StandardsEngine to work with the web UI
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SearchResult:
    """Search result with standard and metadata"""

    standard: Any
    score: float
    highlights: dict[str, list[str]]


@dataclass
class AnalysisResult:
    """Analysis result with recommendations"""

    standard: Any
    relevance_score: float
    confidence: float
    reasoning: str
    implementation_notes: str


class StandardsEngine:
    """Adapter for the standards engine with async support"""

    def __init__(self):
        self.standards_path = Path(__file__).parent.parent.parent / "data" / "standards"
        self._standards_cache = {}
        self._categories_cache = []
        self._tags_cache = set()

    async def initialize(self):
        """Initialize the engine and load standards"""
        # Load standards from YAML files
        await self._load_standards()

    async def _load_standards(self):
        """Load all standards from YAML files"""
        import yaml

        for yaml_file in self.standards_path.glob("*.yaml"):
            if yaml_file.name == "import_metadata.json":
                continue

            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)

                category = (
                    yaml_file.stem.replace("_STANDARDS", "")
                    .replace("_", " ")
                    .title()
                )
                
                standards_list = []
                
                if isinstance(data, dict):
                    if "standards" in data and isinstance(data["standards"], dict):
                        # Format: standards -> category_name -> sections -> section -> standards -> [list]
                        for std_key, std_category in data["standards"].items():
                            if isinstance(std_category, dict) and "sections" in std_category:
                                for section_key, section in std_category["sections"].items():
                                    if isinstance(section, dict) and "standards" in section:
                                        for std in section["standards"]:
                                            if isinstance(std, dict):
                                                standards_list.append(std)
                    elif "standards" in data and isinstance(data["standards"], list):
                        # Format: standards -> [list]
                        for std in data["standards"]:
                            if isinstance(std, dict):
                                standards_list.append(std)
                    else:
                        # Flat YAML file - treat the whole data as a single standard
                        if any(key in data for key in ["id", "title", "description", "author"]):
                            # Create a standard from the flat structure
                            standard_data = {
                                "id": data.get("id", yaml_file.stem.lower()),
                                "title": data.get("title", category),
                                "description": data.get("description", f"Standards for {category}"),
                                "tags": data.get("tags", []),
                                "version": data.get("version", "1.0.0"),
                                "priority": data.get("priority", "medium"),
                                "examples": data.get("code_examples", []),
                                "rules": {k: v for k, v in data.items() if k not in ["id", "title", "description", "tags", "version", "priority", "code_examples"]},
                                "metadata": {"author": data.get("author"), "created_date": data.get("created_date")}
                            }
                            standards_list.append(standard_data)
                
                if standards_list:
                    self._standards_cache[category] = []
                    
                    for std in standards_list:
                        if isinstance(std, dict):
                            # Create standard object
                            standard = type(
                                "Standard",
                                (),
                                {
                                    "id": std.get("id", f"{category.lower().replace(' ', '_')}_{len(self._standards_cache[category])}"),
                                    "title": std.get("title", ""),
                                    "description": std.get("description", ""),
                                    "category": category,
                                    "subcategory": std.get("subcategory", ""),
                                    "tags": std.get("tags", []),
                                    "priority": std.get("priority", "medium"),
                                    "version": std.get("version", "1.0.0"),
                                    "examples": std.get("examples", std.get("implementation_examples", [])),
                                    "rules": std.get("rules", {}),
                                    "created_at": None,
                                    "updated_at": None,
                                    "metadata": std.get("metadata", {}),
                                },
                            )()

                            self._standards_cache[category].append(standard)

                            # Collect tags
                            for tag in standard.tags:
                                self._tags_cache.add(tag)
                    
                    # Add category if it has standards
                    if self._standards_cache[category]:
                        self._categories_cache.append(
                            type(
                                "Category",
                                (),
                                {
                                    "name": category,
                                    "description": data.get("description", ""),
                                    "count": len(self._standards_cache[category]),
                                },
                            )()
                        )

            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")

    async def get_all_standards(self) -> list[Any]:
        """Get all standards as a flat list"""
        all_standards = []
        for standards_list in self._standards_cache.values():
            all_standards.extend(standards_list)
        return all_standards

    async def get_standard_by_id(self, standard_id: str) -> Any | None:
        """Get a specific standard by ID"""
        for standards_list in self._standards_cache.values():
            for standard in standards_list:
                if standard.id == standard_id:
                    return standard
        return None

    async def search_standards(
        self,
        query: str,
        category: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """Search standards with optional filters"""
        results = []
        query_lower = query.lower()

        for cat, standards_list in self._standards_cache.items():
            if category and cat != category:
                continue

            for standard in standards_list:
                # Simple text matching
                score = 0.0
                highlights = {"title": [], "description": []}

                if query_lower in standard.title.lower():
                    score += 0.5
                    highlights["title"].append(query)

                if query_lower in standard.description.lower():
                    score += 0.3
                    highlights["description"].append(query)

                if tags:
                    matching_tags = set(tags) & set(standard.tags)
                    if matching_tags:
                        score += 0.2 * len(matching_tags)

                if score > 0:
                    results.append(
                        SearchResult(
                            standard=standard,
                            score=min(score, 1.0),
                            highlights=highlights,
                        )
                    )

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]

    async def analyze_project(self, context: Any) -> list[AnalysisResult]:
        """Analyze project context and return recommendations"""
        recommendations = []

        # Simple rule-based matching
        for standards_list in self._standards_cache.values():
            for standard in standards_list:
                score = 0.0
                reasoning = []

                # Match languages
                if context.languages:
                    for lang in context.languages:
                        if lang.lower() in " ".join(standard.tags).lower():
                            score += 0.3
                            reasoning.append(f"Matches language: {lang}")

                # Match frameworks
                if context.frameworks:
                    for fw in context.frameworks:
                        if fw.lower() in " ".join(standard.tags).lower():
                            score += 0.3
                            reasoning.append(f"Matches framework: {fw}")

                # Match project type
                if context.project_type:
                    if context.project_type.lower() in standard.description.lower():
                        score += 0.2
                        reasoning.append(
                            f"Relevant for {context.project_type} projects"
                        )

                # Match compliance requirements
                if context.compliance_requirements:
                    for req in context.compliance_requirements:
                        if req.lower() in " ".join(standard.tags).lower():
                            score += 0.4
                            reasoning.append(f"Addresses compliance: {req}")

                if score > 0.2:  # Threshold for relevance
                    recommendations.append(
                        AnalysisResult(
                            standard=standard,
                            relevance_score=min(score, 1.0),
                            confidence=0.8 if score > 0.5 else 0.6,
                            reasoning=" ".join(reasoning),
                            implementation_notes=f"Consider implementing {standard.title} to improve your {context.project_type} project.",
                        )
                    )

        # Sort by relevance
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        return recommendations[:20]  # Top 20 recommendations

    async def get_categories(self) -> list[Any]:
        """Get all categories"""
        return self._categories_cache

    async def get_tags(self) -> list[str]:
        """Get all unique tags"""
        return sorted(self._tags_cache)
