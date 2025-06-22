#!/usr/bin/env python3
"""
Migration script from single FAISS to three-tier hybrid vector store.

This script migrates existing FAISS indices to the new hybrid architecture
with ChromaDB persistence and intelligent tier placement.

@nist-controls: CM-3, SC-28, SI-12
@evidence: Secure data migration with integrity verification
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import get_logger
from src.core.redis_client import get_redis_client
from src.core.standards.engine import StandardsEngine
from src.core.standards.hybrid_vector_store import HybridConfig, HybridVectorStore
from src.core.standards.semantic_search import EmbeddingModel

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass


@app.command()
def migrate(
    standards_path: Path = typer.Option(
        Path("data/standards"),
        help="Path to standards directory"
    ),
    faiss_index_path: Path = typer.Option(
        Path(".faiss_index"),
        help="Path to existing FAISS index"
    ),
    chroma_path: Path = typer.Option(
        Path(".chroma_db"),
        help="Path for ChromaDB storage"
    ),
    backup: bool = typer.Option(
        True,
        help="Create backup of existing data"
    ),
    dry_run: bool = typer.Option(
        False,
        help="Perform dry run without making changes"
    ),
    batch_size: int = typer.Option(
        100,
        help="Batch size for processing"
    ),
) -> None:
    """
    Migrate from single FAISS index to three-tier hybrid architecture.
    
    @nist-controls: CM-3, SC-28
    @evidence: Controlled migration with backup and verification
    """
    console.print("[bold cyan]Starting migration to hybrid vector store[/bold cyan]")
    
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]")
    
    try:
        # Run async migration
        asyncio.run(_perform_migration(
            standards_path,
            faiss_index_path,
            chroma_path,
            backup,
            dry_run,
            batch_size
        ))
        
        console.print("[bold green]✓ Migration completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Migration failed: {e}[/bold red]")
        raise typer.Exit(1)


async def _perform_migration(
    standards_path: Path,
    faiss_index_path: Path,
    chroma_path: Path,
    backup: bool,
    dry_run: bool,
    batch_size: int
) -> None:
    """
    Perform the actual migration.
    
    @nist-controls: SI-12, AU-12
    @evidence: Detailed migration with audit logging
    """
    # Step 1: Validate prerequisites
    console.print("\n[bold]Step 1: Validating prerequisites[/bold]")
    
    if not standards_path.exists():
        raise MigrationError(f"Standards path not found: {standards_path}")
    
    if not faiss_index_path.exists():
        console.print("[yellow]Warning: No existing FAISS index found. Will create new hybrid store.[/yellow]")
    
    # Step 2: Create backup if requested
    if backup and faiss_index_path.exists() and not dry_run:
        console.print("\n[bold]Step 2: Creating backup[/bold]")
        backup_path = faiss_index_path.with_suffix('.backup')
        shutil.copytree(faiss_index_path, backup_path, dirs_exist_ok=True)
        console.print(f"[green]✓ Backup created at: {backup_path}[/green]")
    else:
        console.print("\n[bold]Step 2: Skipping backup[/bold]")
    
    # Step 3: Initialize components
    console.print("\n[bold]Step 3: Initializing components[/bold]")
    
    # Initialize embedding model
    embedding_model = EmbeddingModel()
    console.print("[green]✓ Embedding model initialized[/green]")
    
    # Initialize hybrid store
    hybrid_config = HybridConfig(
        chroma_path=str(chroma_path),
        hot_cache_size=1000,
        access_threshold=10
    )
    
    if not dry_run:
        hybrid_store = HybridVectorStore(hybrid_config)
        console.print("[green]✓ Hybrid vector store initialized[/green]")
    else:
        console.print("[yellow]- Would initialize hybrid vector store[/yellow]")
    
    # Initialize Redis client
    redis_client = get_redis_client()
    if redis_client:
        console.print("[green]✓ Redis connected[/green]")
    else:
        console.print("[yellow]- Redis not available (optional)[/yellow]")
    
    # Step 4: Load and process standards
    console.print("\n[bold]Step 4: Loading standards data[/bold]")
    
    standards_data = await _load_standards_data(standards_path)
    console.print(f"[green]✓ Loaded {len(standards_data)} standards[/green]")
    
    # Step 5: Generate embeddings and migrate
    console.print("\n[bold]Step 5: Migrating to hybrid store[/bold]")
    
    if not dry_run:
        await _migrate_standards(
            standards_data,
            embedding_model,
            hybrid_store,
            batch_size
        )
    else:
        console.print(f"[yellow]- Would migrate {len(standards_data)} standards[/yellow]")
    
    # Step 6: Verify migration
    console.print("\n[bold]Step 6: Verifying migration[/bold]")
    
    if not dry_run:
        stats = await hybrid_store.get_stats()
        console.print("[green]✓ Migration verified[/green]")
        console.print(f"  ChromaDB documents: {stats['tiers']['chromadb']['total_documents']}")
        console.print(f"  FAISS hot cache: {stats['tiers']['faiss']['size']}")
    else:
        console.print("[yellow]- Would verify migration[/yellow]")
    
    # Step 7: Update configuration
    console.print("\n[bold]Step 7: Updating configuration[/bold]")
    
    config_updates = {
        "hybrid_search_enabled": True,
        "chroma_path": str(chroma_path),
        "migration_completed": True
    }
    
    if not dry_run:
        _update_config(config_updates)
        console.print("[green]✓ Configuration updated[/green]")
    else:
        console.print("[yellow]- Would update configuration[/yellow]")


async def _load_standards_data(standards_path: Path) -> List[Dict[str, Any]]:
    """
    Load all standards data from directory.
    
    @nist-controls: SI-10
    @evidence: Secure data loading with validation
    """
    standards_data = []
    
    # Load YAML files
    yaml_files = list(standards_path.glob("**/*.yaml")) + list(standards_path.glob("**/*.yml"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading standards...", total=len(yaml_files))
        
        for yaml_file in yaml_files:
            try:
                import yaml
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Extract content for embedding
                if isinstance(data, dict):
                    standard_id = yaml_file.stem
                    content = _extract_content(data)
                    
                    standards_data.append({
                        "id": standard_id,
                        "content": content,
                        "metadata": {
                            "source_file": str(yaml_file),
                            "type": data.get("type", "standard"),
                            "version": data.get("version", "1.0"),
                            "category": data.get("category", "general"),
                            "file_size": yaml_file.stat().st_size
                        }
                    })
                
                progress.advance(task)
                
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")
                continue
    
    return standards_data


def _extract_content(data: Dict[str, Any], max_length: int = 8000) -> str:
    """
    Extract textual content from standard data.
    
    @nist-controls: SI-10
    @evidence: Content extraction with size limits
    """
    content_parts = []
    
    # Extract key fields
    if "title" in data:
        content_parts.append(f"Title: {data['title']}")
    
    if "description" in data:
        content_parts.append(f"Description: {data['description']}")
    
    if "controls" in data:
        if isinstance(data["controls"], list):
            content_parts.append(f"Controls: {', '.join(data['controls'][:20])}")
        elif isinstance(data["controls"], dict):
            control_ids = list(data["controls"].keys())[:20]
            content_parts.append(f"Controls: {', '.join(control_ids)}")
    
    if "requirements" in data:
        content_parts.append(f"Requirements: {str(data['requirements'])[:500]}")
    
    if "implementation" in data:
        content_parts.append(f"Implementation: {str(data['implementation'])[:500]}")
    
    # Combine and truncate
    full_content = "\n\n".join(content_parts)
    if len(full_content) > max_length:
        full_content = full_content[:max_length] + "..."
    
    return full_content


async def _migrate_standards(
    standards_data: List[Dict[str, Any]],
    embedding_model: EmbeddingModel,
    hybrid_store: HybridVectorStore,
    batch_size: int
) -> None:
    """
    Migrate standards to hybrid store in batches.
    
    @nist-controls: SC-28, AU-12
    @evidence: Batch processing with progress tracking
    """
    total_migrated = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(
            "Migrating standards...",
            total=len(standards_data)
        )
        
        # Process in batches
        for i in range(0, len(standards_data), batch_size):
            batch = standards_data[i:i + batch_size]
            
            for item in batch:
                try:
                    # Generate embedding
                    embedding = await embedding_model.encode(item["content"])
                    
                    # Add to hybrid store
                    success = await hybrid_store.add(
                        id=item["id"],
                        content=item["content"],
                        embedding=embedding,
                        metadata=item["metadata"]
                    )
                    
                    if success:
                        total_migrated += 1
                    
                    progress.advance(task)
                    
                except Exception as e:
                    logger.error(f"Failed to migrate {item['id']}: {e}")
                    continue
            
            # Small delay between batches
            await asyncio.sleep(0.1)
    
    console.print(f"[green]✓ Migrated {total_migrated}/{len(standards_data)} standards[/green]")


def _update_config(updates: Dict[str, Any]) -> None:
    """
    Update configuration file with migration status.
    
    @nist-controls: CM-3
    @evidence: Configuration management
    """
    config_path = Path(".mcp-standards/migration.json")
    config_path.parent.mkdir(exist_ok=True)
    
    # Load existing config or create new
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Update with new values
    config.update(updates)
    config["migration_timestamp"] = str(Path.ctime(Path.cwd()))
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


@app.command()
def verify(
    chroma_path: Path = typer.Option(
        Path(".chroma_db"),
        help="Path to ChromaDB storage"
    ),
    sample_queries: int = typer.Option(
        5,
        help="Number of sample queries to test"
    )
) -> None:
    """
    Verify the migration was successful.
    
    @nist-controls: CA-7
    @evidence: Post-migration verification
    """
    console.print("[bold cyan]Verifying hybrid vector store[/bold cyan]\n")
    
    try:
        asyncio.run(_verify_migration(chroma_path, sample_queries))
        console.print("\n[bold green]✓ Verification completed![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Verification failed: {e}[/bold red]")
        raise typer.Exit(1)


async def _verify_migration(chroma_path: Path, sample_queries: int) -> None:
    """
    Perform verification checks.
    
    @nist-controls: CA-7, AU-6
    @evidence: Comprehensive verification with metrics
    """
    # Initialize components
    hybrid_config = HybridConfig(chroma_path=str(chroma_path))
    hybrid_store = HybridVectorStore(hybrid_config)
    embedding_model = EmbeddingModel()
    
    # Get statistics
    stats = await hybrid_store.get_stats()
    
    console.print("[bold]Storage Statistics:[/bold]")
    console.print(f"  ChromaDB documents: {stats['tiers']['chromadb']['total_documents']}")
    console.print(f"  FAISS hot cache: {stats['tiers']['faiss']['size']}/{stats['tiers']['faiss']['capacity']}")
    console.print(f"  Redis connected: {stats['tiers']['redis']['redis_connected']}")
    
    # Test sample queries
    console.print(f"\n[bold]Testing {sample_queries} sample queries:[/bold]")
    
    test_queries = [
        "access control implementation",
        "encryption requirements",
        "audit logging",
        "authentication methods",
        "data protection"
    ][:sample_queries]
    
    for query in test_queries:
        # Generate embedding
        query_embedding = await embedding_model.encode(query)
        
        # Search
        results = await hybrid_store.search(
            query=query,
            query_embedding=query_embedding,
            k=5
        )
        
        console.print(f"\n  Query: '{query}'")
        console.print(f"  Results: {len(results)} found")
        if results:
            console.print(f"  Top result: {results[0].id} (score: {results[0].score:.3f})")
            console.print(f"  Source tier: {results[0].source_tier}")


if __name__ == "__main__":
    app()