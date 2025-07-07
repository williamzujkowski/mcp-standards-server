#!/usr/bin/env python3
"""
MCP Standards Server CLI

Main command-line interface for managing and syncing standards.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.standards.sync import StandardsSynchronizer, sync_standards, check_for_updates


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog='mcp-standards',
        description='MCP Standards Server - Manage and sync development standards',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '-c', '--config',
        type=Path,
        help='Path to sync configuration file'
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )
    
    # Sync command
    sync_parser = subparsers.add_parser(
        'sync',
        help='Synchronize standards from GitHub repository'
    )
    sync_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force sync even if files are up to date'
    )
    sync_parser.add_argument(
        '--check',
        action='store_true',
        help='Check for updates without downloading'
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show sync status and statistics'
    )
    status_parser.add_argument(
        '--json',
        action='store_true',
        help='Output status in JSON format'
    )
    
    # Cache command
    cache_parser = subparsers.add_parser(
        'cache',
        help='Manage local cache'
    )
    cache_parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear all cached files'
    )
    cache_parser.add_argument(
        '--list',
        action='store_true',
        help='List cached files'
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Show or validate configuration'
    )
    config_parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate configuration file'
    )
    config_parser.add_argument(
        '--show',
        action='store_true',
        help='Show current configuration'
    )
    
    return parser


def cmd_sync(args: argparse.Namespace) -> int:
    """Handle sync command."""
    synchronizer = StandardsSynchronizer(config_path=args.config)
    
    if args.check:
        # Check for updates
        print("Checking for updates...")
        updates = synchronizer.check_updates()
        
        if updates['outdated_files']:
            print(f"\nOutdated files ({len(updates['outdated_files'])}):")
            for file in updates['outdated_files']:
                age_hours = file['age_hours']
                print(f"  - {file['path']} (last synced {age_hours:.1f} hours ago)")
        else:
            print("\nAll files are up to date!")
        
        print(f"\nTotal cached files: {updates['total_cached']}")
        print(f"Cache TTL: {updates['cache_ttl_hours']} hours")
        
        return 0
    
    # Perform sync
    print("Starting standards synchronization...")
    if args.force:
        print("Force sync enabled - all files will be re-downloaded")
    
    try:
        result = sync_standards(force=args.force, config_path=args.config)
        
        # Display results
        print(f"\nSync completed with status: {result.status.value}")
        print(f"Duration: {result.duration.total_seconds():.2f} seconds")
        print(f"Files synced: {len(result.synced_files)}/{result.total_files}")
        
        if result.synced_files:
            print("\nSynced files:")
            for file in result.synced_files:
                print(f"  - {file.path} ({file.size} bytes)")
        
        if result.failed_files:
            print(f"\nFailed files ({len(result.failed_files)}):")
            for path, error in result.failed_files:
                print(f"  - {path}: {error}")
        
        if result.message:
            print(f"\n{result.message}")
        
        return 0 if result.status.value in ['success', 'partial'] else 1
        
    except Exception as e:
        print(f"Error during sync: {e}", file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Handle status command."""
    synchronizer = StandardsSynchronizer(config_path=args.config)
    status = synchronizer.get_sync_status()
    
    if args.json:
        # JSON output
        print(json.dumps(status, indent=2, default=str))
    else:
        # Human-readable output
        print("MCP Standards Server - Sync Status\n")
        print(f"Total files cached: {status['total_files']}")
        print(f"Total cache size: {status['total_size_mb']:.2f} MB")
        
        # Rate limit info
        rate_limit = status['rate_limit']
        print(f"\nGitHub API Rate Limit:")
        print(f"  Remaining: {rate_limit['remaining']}/{rate_limit['limit']}")
        if rate_limit['reset_time']:
            print(f"  Resets at: {rate_limit['reset_time']}")
        
        # Recent syncs
        if status['last_sync_times']:
            print("\nRecent syncs:")
            recent = sorted(
                [(p, t) for p, t in status['last_sync_times'].items() if t],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for path, sync_time in recent:
                print(f"  - {path}: {sync_time}")
        
        # Configuration summary
        config = status['config']
        repo = config.get('repository', {})
        print(f"\nRepository: {repo.get('owner')}/{repo.get('repo')}")
        print(f"Branch: {repo.get('branch')}")
        print(f"Path: {repo.get('path')}")
    
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    """Handle cache command."""
    synchronizer = StandardsSynchronizer(config_path=args.config)
    
    if args.clear:
        print("Clearing cache...")
        synchronizer.clear_cache()
        print("Cache cleared successfully!")
        return 0
    
    if args.list:
        cached_files = synchronizer.get_cached_standards()
        
        if cached_files:
            print(f"Cached files ({len(cached_files)}):\n")
            for file_path in sorted(cached_files):
                size = file_path.stat().st_size / 1024  # KB
                print(f"  - {file_path.name} ({size:.1f} KB)")
        else:
            print("No cached files found.")
        
        return 0
    
    # Default: show cache info
    status = synchronizer.get_sync_status()
    print(f"Cache location: {synchronizer.cache_dir}")
    print(f"Total files: {status['total_files']}")
    print(f"Total size: {status['total_size_mb']:.2f} MB")
    
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Handle config command."""
    synchronizer = StandardsSynchronizer(config_path=args.config)
    
    if args.validate:
        # Validate configuration
        try:
            # Check required fields
            required = ['repository', 'sync', 'cache']
            missing = [field for field in required if field not in synchronizer.config]
            
            if missing:
                print(f"Invalid configuration: missing fields {missing}")
                return 1
            
            # Validate repository config
            repo = synchronizer.config['repository']
            repo_required = ['owner', 'repo', 'branch', 'path']
            repo_missing = [field for field in repo_required if field not in repo]
            
            if repo_missing:
                print(f"Invalid repository configuration: missing {repo_missing}")
                return 1
            
            print("Configuration is valid!")
            return 0
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return 1
    
    if args.show:
        # Show configuration
        import yaml
        print("Current configuration:\n")
        print(yaml.dump(synchronizer.config, default_flow_style=False))
        return 0
    
    # Default: show config file path
    print(f"Configuration file: {synchronizer.config_path}")
    if synchronizer.config_path.exists():
        print("Status: Found")
    else:
        print("Status: Using default configuration")
    
    return 0


def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle commands
    if args.command == 'sync':
        return cmd_sync(args)
    elif args.command == 'status':
        return cmd_status(args)
    elif args.command == 'cache':
        return cmd_cache(args)
    elif args.command == 'config':
        return cmd_config(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())