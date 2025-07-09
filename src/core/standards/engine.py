"""Standards engine for managing and accessing standards."""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from .models import Standard, StandardMetadata, Requirement, Evidence
from .semantic_search import SemanticSearch, create_search_engine
from .rule_engine import RuleEngine
from .token_optimizer import TokenOptimizer
from .sync import StandardsSynchronizer

logger = logging.getLogger(__name__)


@dataclass
class StandardsEngineConfig:
    """Configuration for StandardsEngine."""
    data_dir: Path
    enable_semantic_search: bool = True
    enable_rule_engine: bool = True
    enable_token_optimization: bool = True
    enable_caching: bool = True
    github_repo: str = "williamzujkowski/standards"
    github_branch: str = "main"


class StandardsEngine:
    """Main engine for managing standards and their operations."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        enable_semantic_search: bool = True,
        enable_rule_engine: bool = True,
        enable_token_optimization: bool = True,
        enable_caching: bool = True,
        config: Optional[StandardsEngineConfig] = None
    ):
        self.data_dir = Path(data_dir)
        self.config = config or StandardsEngineConfig(
            data_dir=self.data_dir,
            enable_semantic_search=enable_semantic_search,
            enable_rule_engine=enable_rule_engine,
            enable_token_optimization=enable_token_optimization,
            enable_caching=enable_caching
        )
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.semantic_search: Optional[SemanticSearch] = None
        self.rule_engine: Optional[RuleEngine] = None
        self.token_optimizer: Optional[TokenOptimizer] = None
        self.sync: Optional[StandardsSynchronizer] = None
        
        # Local storage
        self._standards_cache: Dict[str, Standard] = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the engine and all components."""
        if self._initialized:
            return
            
        logger.info("Initializing StandardsEngine...")
        
        # Initialize semantic search
        if self.config.enable_semantic_search:
            self.semantic_search = await create_search_engine(
                data_dir=self.data_dir / "search",
                enable_cache=self.config.enable_caching
            )
        
        # Initialize rule engine
        if self.config.enable_rule_engine:
            self.rule_engine = RuleEngine()
        
        # Initialize token optimizer
        if self.config.enable_token_optimization:
            from .token_optimizer import create_token_optimizer
            self.token_optimizer = create_token_optimizer()
        
        # Initialize sync
        self.sync = StandardsSynchronizer(
            cache_dir=self.data_dir / "cache"
        )
        
        # Load standards
        await self._load_standards()
        
        self._initialized = True
        logger.info("StandardsEngine initialized successfully")
    
    async def _load_standards(self):
        """Load standards from local storage."""
        try:
            if self.sync:
                await self.sync.sync()
            
            # Load from YAML files
            import yaml
            
            for yaml_file in self.data_dir.glob("*.yaml"):
                if yaml_file.name == "import_metadata.json":
                    continue
                    
                try:
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                        
                    if isinstance(data, dict) and 'standards' in data:
                        category = yaml_file.stem.replace('_STANDARDS', '').replace('_', ' ').title()
                        
                        for std_data in data['standards']:
                            standard = Standard(
                                id=std_data.get('id', ''),
                                title=std_data.get('title', ''),
                                description=std_data.get('description', ''),
                                category=category,
                                subcategory=std_data.get('subcategory', ''),
                                tags=std_data.get('tags', []),
                                priority=std_data.get('priority', 'medium'),
                                version=std_data.get('version', '1.0.0'),
                                examples=std_data.get('examples', []),
                                rules=std_data.get('rules', {}),
                                metadata=StandardMetadata(
                                    **std_data.get('metadata', {})
                                )
                            )
                            
                            self._standards_cache[standard.id] = standard
                            
                except Exception as e:
                    logger.error(f"Error loading {yaml_file}: {e}")
            
            # Index in semantic search if available
            if self.semantic_search:
                documents = []
                for standard in self._standards_cache.values():
                    documents.append({
                        'id': standard.id,
                        'title': standard.title,
                        'description': standard.description,
                        'category': standard.category,
                        'tags': standard.tags,
                        'content': f"{standard.title}\n{standard.description}",
                        'metadata': {
                            'category': standard.category,
                            'subcategory': standard.subcategory,
                            'tags': standard.tags,
                            'priority': standard.priority,
                            'version': standard.version
                        }
                    })
                
                await self.semantic_search.index_documents_batch(documents)
            
        except Exception as e:
            logger.error(f"Error loading standards: {e}")
    
    async def get_standard(self, standard_id: str, version: Optional[str] = None) -> Optional[Standard]:
        """Get a specific standard by ID."""
        if not self._initialized:
            await self.initialize()
            
        return self._standards_cache.get(standard_id)
    
    async def list_standards(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        offset: int = 0,
        limit: int = 100
    ) -> List[Standard]:
        """List standards with optional filtering."""
        if not self._initialized:
            await self.initialize()
            
        standards = list(self._standards_cache.values())
        
        # Apply filters
        if category:
            standards = [s for s in standards if s.category == category]
            
        if tags:
            standards = [s for s in standards if any(tag in s.tags for tag in tags)]
        
        # Apply pagination
        return standards[offset:offset + limit]
    
    async def get_requirements(
        self,
        standard_id: str,
        requirement_ids: Optional[List[str]] = None
    ) -> List[Requirement]:
        """Get requirements for a standard."""
        if not self._initialized:
            await self.initialize()
            
        standard = self._standards_cache.get(standard_id)
        if not standard:
            return []
        
        # Extract requirements from standard rules
        requirements = []
        for rule_id, rule_data in standard.rules.items():
            if requirement_ids and rule_id not in requirement_ids:
                continue
                
            req = Requirement(
                id=rule_id,
                standard_id=standard_id,
                title=rule_data.get('title', rule_id),
                description=rule_data.get('description', ''),
                priority=rule_data.get('priority', 'medium'),
                category=rule_data.get('category', 'general'),
                tags=rule_data.get('tags', [])
            )
            requirements.append(req)
        
        return requirements
    
    async def update_standard(
        self,
        standard_id: str,
        updates: Dict[str, Any]
    ) -> Standard:
        """Update a standard."""
        if not self._initialized:
            await self.initialize()
            
        standard = self._standards_cache.get(standard_id)
        if not standard:
            raise ValueError(f"Standard not found: {standard_id}")
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(standard, key):
                setattr(standard, key, value)
        
        # Update cache
        self._standards_cache[standard_id] = standard
        
        # Re-index in semantic search
        if self.semantic_search:
            await self.semantic_search.index_documents([{
                'id': standard.id,
                'title': standard.title,
                'description': standard.description,
                'category': standard.category,
                'tags': standard.tags,
                'content': f"{standard.title}\n{standard.description}",
                'metadata': {
                    'category': standard.category,
                    'subcategory': standard.subcategory,
                    'tags': standard.tags,
                    'priority': standard.priority,
                    'version': standard.version
                }
            }])
        
        return standard
    
    async def search_standards(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search standards using semantic search."""
        if not self._initialized:
            await self.initialize()
            
        if not self.semantic_search:
            # Fallback to simple text search
            return await self._simple_search(query, category, tags, limit)
        
        # Prepare filters
        filters = {}
        if category:
            filters['category'] = category
        if tags:
            filters['tags'] = tags
        
        # Perform semantic search
        results = await self.semantic_search.search(
            query=query,
            k=limit,
            threshold=threshold,
            filters=filters
        )
        
        # Enrich results with full standard objects
        enriched_results = []
        for result in results:
            standard = self._standards_cache.get(result.get('id'))
            if standard:
                enriched_results.append({
                    'standard': standard,
                    'score': result.get('score', 0.0),
                    'highlights': result.get('highlights', {}),
                    'metadata': result.get('metadata', {})
                })
        
        return enriched_results
    
    async def _simple_search(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Simple text-based search fallback."""
        results = []
        query_lower = query.lower()
        
        for standard in self._standards_cache.values():
            if category and standard.category != category:
                continue
                
            if tags and not any(tag in standard.tags for tag in tags):
                continue
            
            score = 0.0
            
            if query_lower in standard.title.lower():
                score += 0.5
            if query_lower in standard.description.lower():
                score += 0.3
            
            if score > 0:
                results.append({
                    'standard': standard,
                    'score': score,
                    'highlights': {},
                    'metadata': {}
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    async def get_applicable_standards(
        self,
        project_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get standards applicable to a project context."""
        if not self._initialized:
            await self.initialize()
            
        if not self.rule_engine:
            return []
        
        # Use rule engine to determine applicable standards
        applicable = []
        for standard in self._standards_cache.values():
            if await self.rule_engine.is_applicable(standard, project_context):
                applicable.append({
                    'standard': standard,
                    'confidence': 0.8,  # Default confidence
                    'reasoning': 'Rule-based match'
                })
        
        return applicable
    
    async def close(self):
        """Close the engine and clean up resources."""
        if self.semantic_search:
            await self.semantic_search.close()
        
        self._initialized = False
        self._standards_cache.clear()
        logger.info("StandardsEngine closed")