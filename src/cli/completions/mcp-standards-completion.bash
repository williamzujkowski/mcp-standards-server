#!/bin/bash
# Bash completion script for mcp-standards
# 
# Installation:
#   - Copy to /etc/bash_completion.d/mcp-standards
#   OR
#   - Source in your .bashrc: source /path/to/mcp-standards-completion.bash

_mcp_standards_completions() {
    local cur prev words cword
    _init_completion || return

    local commands="sync status cache config query validate serve"
    local global_opts="-h --help -v --verbose -c --config --no-color --json --version"

    # Complete command if we're at position 1
    if [[ $cword -eq 1 ]]; then
        COMPREPLY=($(compgen -W "$commands $global_opts" -- "$cur"))
        return
    fi

    # Get the command (skip global options)
    local cmd=""
    local i
    for ((i=1; i < cword; i++)); do
        if [[ " $commands " =~ " ${words[i]} " ]]; then
            cmd="${words[i]}"
            break
        fi
    done

    # If no command found, continue with global options
    if [[ -z "$cmd" ]]; then
        COMPREPLY=($(compgen -W "$commands $global_opts" -- "$cur"))
        return
    fi

    # Command-specific completions
    case "$cmd" in
        sync)
            case "$prev" in
                --include|--exclude)
                    # Suggest common patterns
                    COMPREPLY=($(compgen -W "*.yaml *.yml *.json *.md" -- "$cur"))
                    ;;
                --parallel|--retry|--timeout)
                    # Numeric values
                    COMPREPLY=($(compgen -W "1 2 3 4 5 10 15 20 30 60" -- "$cur"))
                    ;;
                *)
                    local sync_opts="-f --force --check --include --exclude --parallel --retry --timeout"
                    COMPREPLY=($(compgen -W "$sync_opts" -- "$cur"))
                    ;;
            esac
            ;;

        status)
            local status_opts="--json --detailed --check-health --summary"
            COMPREPLY=($(compgen -W "$status_opts" -- "$cur"))
            ;;

        cache)
            case "$prev" in
                --export|--import)
                    # File completion
                    _filedir
                    ;;
                *)
                    local cache_opts="--list --clear --clear-outdated --analyze --verify --export --import --size-limit"
                    COMPREPLY=($(compgen -W "$cache_opts" -- "$cur"))
                    ;;
            esac
            ;;

        config)
            case "$prev" in
                --get)
                    # Common config keys
                    local keys="repository.owner repository.repo repository.branch sync.cache_ttl_hours cache.directory"
                    COMPREPLY=($(compgen -W "$keys" -- "$cur"))
                    ;;
                --set)
                    # Config keys for --set
                    local keys="repository.owner repository.repo sync.cache_ttl_hours cache.max_size_mb"
                    COMPREPLY=($(compgen -W "$keys" -- "$cur"))
                    ;;
                *)
                    local config_opts="--init --show --validate --edit --get --set --schema --migrate"
                    COMPREPLY=($(compgen -W "$config_opts" -- "$cur"))
                    ;;
            esac
            ;;

        query)
            case "$prev" in
                --project-type)
                    COMPREPLY=($(compgen -W "web-application api cli library mobile desktop microservice" -- "$cur"))
                    ;;
                --framework)
                    # Common frameworks
                    local frameworks="react vue angular express fastapi django flask spring rails"
                    COMPREPLY=($(compgen -W "$frameworks" -- "$cur"))
                    ;;
                --language)
                    # Common languages
                    local languages="javascript typescript python java go rust ruby php c++ swift kotlin"
                    COMPREPLY=($(compgen -W "$languages" -- "$cur"))
                    ;;
                --format)
                    COMPREPLY=($(compgen -W "text json yaml markdown" -- "$cur"))
                    ;;
                --requirements)
                    COMPREPLY=($(compgen -W "accessibility security performance compliance" -- "$cur"))
                    ;;
                --context)
                    # JSON files
                    _filedir json
                    ;;
                *)
                    local query_opts="--project-type --framework --language --requirements --tags --format --detailed --token-budget --semantic --context"
                    COMPREPLY=($(compgen -W "$query_opts" -- "$cur"))
                    ;;
            esac
            ;;

        validate)
            case "$prev" in
                --format)
                    COMPREPLY=($(compgen -W "text json junit sarif" -- "$cur"))
                    ;;
                --severity|--fail-on)
                    COMPREPLY=($(compgen -W "error warning info" -- "$cur"))
                    ;;
                --output)
                    _filedir
                    ;;
                --standards)
                    _filedir 'yaml|yml|json'
                    ;;
                --config)
                    _filedir 'yaml|yml'
                    ;;
                --parallel)
                    COMPREPLY=($(compgen -W "1 2 4 8 16" -- "$cur"))
                    ;;
                --ignore)
                    # Common ignore patterns
                    COMPREPLY=($(compgen -W "*.test.js *.spec.js node_modules/ dist/ build/" -- "$cur"))
                    ;;
                *)
                    # If no option, complete with files/directories
                    if [[ "$cur" == -* ]]; then
                        local validate_opts="--fix --dry-run --format --severity --fail-on --output --standards --auto-detect --ignore --config --parallel"
                        COMPREPLY=($(compgen -W "$validate_opts" -- "$cur"))
                    else
                        _filedir
                    fi
                    ;;
            esac
            ;;

        serve)
            case "$prev" in
                --port)
                    COMPREPLY=($(compgen -W "3000 3001 8080 8081 8082" -- "$cur"))
                    ;;
                --host)
                    COMPREPLY=($(compgen -W "localhost 0.0.0.0 127.0.0.1" -- "$cur"))
                    ;;
                --log-level)
                    COMPREPLY=($(compgen -W "debug info warning error" -- "$cur"))
                    ;;
                --workers)
                    COMPREPLY=($(compgen -W "auto 1 2 4 8 16" -- "$cur"))
                    ;;
                --auth)
                    COMPREPLY=($(compgen -W "none token oauth" -- "$cur"))
                    ;;
                *)
                    local serve_opts="--host --port --stdio --socket --daemon --log-level --workers --auth --tls-cert --tls-key"
                    COMPREPLY=($(compgen -W "$serve_opts" -- "$cur"))
                    ;;
            esac
            ;;
    esac
}

# Helper function for project types
_mcp_project_types() {
    echo "web-application api cli library mobile desktop microservice monolith data-pipeline ml-model"
}

# Helper function for frameworks
_mcp_frameworks() {
    echo "react vue angular svelte nextjs nuxt gatsby express fastapi django flask spring rails laravel symfony"
}

# Helper function for languages  
_mcp_languages() {
    echo "javascript typescript python java go rust ruby php c cpp csharp swift kotlin scala elixir"
}

# Register the completion function
complete -F _mcp_standards_completions mcp-standards