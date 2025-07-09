# Installation Guide

This guide covers all the ways to install and set up the MCP Standards Server.

## Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 512MB minimum, 2GB recommended
- **Disk Space**: 100MB for installation, 500MB+ for standards cache
- **Network**: Internet connection for syncing standards
- **Redis** (optional): For enhanced caching performance
- **Node.js 16+** (optional): For web UI development

## Installation Methods

### Method 1: From Source (Currently Available)

Clone and install from the GitHub repository:

```bash
# Clone the repository
git clone https://github.com/williamzujkowski/mcp-standards-server.git
cd mcp-standards-server

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[full]"

# For development with testing tools
pip install -e ".[test]"
```

### Method 2: Optional Dependencies

#### Redis Installation

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis

# Windows (using WSL or Docker)
docker run -d -p 6379:6379 redis:latest
```

#### Web UI Dependencies

```bash
# Install Node.js dependencies for web UI
cd src/web
npm install
npm run build
```

### Method 3: Running the MCP Server

```bash
# Start the MCP server
python -m src

# Or use the CLI
mcp-standards --help

# Start with web UI
mcp-standards web --port 8080
```

### Method 4: Using Docker

Run MCP Standards Server in a container:

```bash
# Pull the image
docker pull mcp-standards/server:latest

# Run the container
docker run -it --rm \
  -v ~/.cache/mcp-standards:/root/.cache/mcp-standards \
  -v $(pwd):/workspace \
  mcp-standards/server:latest \
  validate /workspace
```

Or build your own image:

```dockerfile
FROM python:3.11-slim

RUN pip install mcp-standards-server

WORKDIR /workspace

ENTRYPOINT ["mcp-standards"]
```

### Method 5: Package Managers

#### Future Installation Methods

The following installation methods are planned for future releases:

- **pip**: `pip install mcp-standards-server`
- **pipx**: `pipx install mcp-standards-server`
- **Homebrew**: `brew install mcp-standards-server`
- **Docker Hub**: `docker pull mcp-standards/server`

## Verify Installation

After installation, verify everything is working:

```bash
# Show help
python -m src --help

# Or if installed in development mode
mcp-standards --help

# Test the server
python -m src
```

Expected output:
```
MCP Standards Server

Usage: mcp-standards [OPTIONS] COMMAND [ARGS]...

Commands:
  query     Query standards based on context
  validate  Validate code against standards
  sync      Synchronize standards from repository
  serve     Start the MCP server
  web       Start the web UI
  cache     Manage the standards cache
```

## Post-Installation Setup

### 1. Initialize Configuration

Run the interactive setup:

```bash
mcp-standards config --init
```

This will create a configuration file at `~/.config/mcp-standards/config.yaml`.

### 2. Configure GitHub Authentication (Optional)

For higher API rate limits, configure authentication:

```bash
# Set via environment variable
export MCP_STANDARDS_REPOSITORY_AUTH_TOKEN=ghp_your_token_here

# Or add to configuration
mcp-standards config --set repository.auth.type token
mcp-standards config --set repository.auth.token ghp_your_token_here
```

### 3. Sync Standards

Download the standards to your local cache:

```bash
mcp-standards sync
```

### 4. Install Shell Completions (Optional)

#### Bash

```bash
# Download completion script
curl -o ~/.local/share/bash-completion/completions/mcp-standards \
  https://raw.githubusercontent.com/williamzujkowski/mcp-standards-server/main/completions/bash/mcp-standards

# Add to .bashrc if not auto-loaded
echo 'source ~/.local/share/bash-completion/completions/mcp-standards' >> ~/.bashrc
```

#### Zsh

```bash
# Download completion script
curl -o ~/.zsh/completions/_mcp-standards \
  https://raw.githubusercontent.com/williamzujkowski/mcp-standards-server/main/completions/zsh/_mcp-standards

# Add to .zshrc
echo 'fpath=(~/.zsh/completions $fpath)' >> ~/.zshrc
echo 'autoload -U compinit && compinit' >> ~/.zshrc
```

#### Fish

```bash
# Download completion script
curl -o ~/.config/fish/completions/mcp-standards.fish \
  https://raw.githubusercontent.com/williamzujkowski/mcp-standards-server/main/completions/fish/mcp-standards.fish
```

### 5. Install Man Pages (Optional)

```bash
# System-wide (requires sudo)
sudo cp docs/man/*.1 /usr/local/share/man/man1/
sudo mandb

# User-only
mkdir -p ~/.local/share/man/man1
cp docs/man/*.1 ~/.local/share/man/man1/
mandb ~/.local/share/man
```

## Troubleshooting Installation

### Command Not Found

If `mcp-standards` is not found after installation:

1. **Check installation location**:
   ```bash
   pip show mcp-standards-server | grep Location
   ```

2. **Add to PATH**:
   ```bash
   # For user installation
   export PATH="$HOME/.local/bin:$PATH"
   
   # Add to .bashrc/.zshrc to make permanent
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
   ```

3. **Use python -m**:
   ```bash
   python -m src --help
   ```

### Permission Denied

If you get permission errors:

1. **Use --user flag**:
   ```bash
   pip install --user mcp-standards-server
   ```

2. **Use virtual environment**:
   ```bash
   python -m venv mcp-env
   source mcp-env/bin/activate  # On Windows: mcp-env\Scripts\activate
   pip install mcp-standards-server
   ```

3. **Use pipx** (recommended):
   ```bash
   pipx install mcp-standards-server
   ```

### Dependency Conflicts

If you encounter dependency conflicts:

1. **Use isolated environment**:
   ```bash
   pipx install mcp-standards-server
   ```

2. **Create fresh virtual environment**:
   ```bash
   python -m venv fresh-env
   source fresh-env/bin/activate
   pip install mcp-standards-server
   ```

3. **Upgrade pip and setuptools**:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install mcp-standards-server
   ```

## Platform-Specific Notes

### Windows

- Use `py` instead of `python` if needed
- Paths use backslashes: `C:\Users\Name\.config\mcp-standards`
- Consider using Windows Terminal for better color support
- Git Bash or WSL recommended for Unix-like experience

### macOS

- May need to install Xcode Command Line Tools
- Use Homebrew for easier management
- Check for M1/M2 compatibility if using Apple Silicon

### Linux

- May need to install python3-pip package
- Some distributions require python3-venv
- Consider using system package manager if available

## Upgrading

### Using pip

```bash
pip install --upgrade mcp-standards-server
```

### Using pipx

```bash
pipx upgrade mcp-standards-server
```

### From source

```bash
cd mcp-standards-server
git pull
pip install --upgrade .
```

## Uninstalling

### Using pip

```bash
pip uninstall mcp-standards-server
```

### Using pipx

```bash
pipx uninstall mcp-standards-server
```

### Clean up data

```bash
# Remove configuration
rm -rf ~/.config/mcp-standards

# Remove cache
rm -rf ~/.cache/mcp-standards

# Remove logs
rm -rf ~/.local/share/mcp-standards
```

## Next Steps

Now that you have MCP Standards Server installed:

1. Follow the [Quick Start Guide](./quickstart.md)
2. Learn about [Configuration Options](./configuration.md)
3. Explore [CLI Commands](../reference/cli-commands.md)
4. Set up [IDE Integration](./ide-integration.md)

Need help? Check the [Troubleshooting Guide](../reference/troubleshooting.md) or [open an issue](https://github.com/williamzujkowski/mcp-standards-server/issues).