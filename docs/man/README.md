# Man Pages Installation

This directory contains manual pages for the MCP Standards Server CLI commands.

## Installation

### System-wide Installation (requires root/sudo)

```bash
# Linux/macOS
sudo cp mcp-standards*.1 /usr/local/share/man/man1/
sudo mandb  # or makewhatis on some systems

# Alternative location
sudo cp mcp-standards*.1 /usr/share/man/man1/
```

### User Installation (no root required)

```bash
# Create user man directory
mkdir -p ~/.local/share/man/man1

# Copy man pages
cp mcp-standards*.1 ~/.local/share/man/man1/

# Add to MANPATH (add to ~/.bashrc or ~/.zshrc)
export MANPATH="$HOME/.local/share/man:$MANPATH"

# Update man database
mandb ~/.local/share/man
```

### Package Manager Installation

When installed via package managers, man pages are typically installed automatically:

```bash
# Debian/Ubuntu package
sudo dpkg -i mcp-standards-server.deb

# RPM package
sudo rpm -i mcp-standards-server.rpm

# Homebrew (macOS)
brew install mcp-standards-server
```

## Available Man Pages

- `mcp-standards(1)` - Main command overview
- `mcp-standards-sync(1)` - Sync command details
- `mcp-standards-validate(1)` - Validate command details
- `mcp-standards-serve(1)` - Server command details
- `mcp-standards-query(1)` - Query command details
- `mcp-standards-cache(1)` - Cache management
- `mcp-standards-config(1)` - Configuration management

## Viewing Man Pages

```bash
# View main page
man mcp-standards

# View specific command
man mcp-standards-validate

# Search for keyword
man -k mcp-standards

# View without installing
man ./mcp-standards.1
```

## Building Man Pages

### From Markdown

```bash
# Install pandoc
sudo apt install pandoc  # or brew install pandoc

# Convert markdown to man page
pandoc -s -t man docs/cli/README.md -o mcp-standards.1
```

### From AsciiDoc

```bash
# Install asciidoctor
gem install asciidoctor

# Convert asciidoc to man page
asciidoctor -b manpage -o mcp-standards.1 mcp-standards.adoc
```

### Formatting Guidelines

Man pages follow specific formatting conventions:

- `.TH` - Title header
- `.SH` - Section header
- `.SS` - Subsection header
- `.TP` - Tagged paragraph
- `.BR` - Bold/Roman alternating
- `.B` - Bold text
- `.I` - Italic text
- `.nf`/`.fi` - No fill (preformatted text)

## Testing Man Pages

```bash
# Check formatting
man -l mcp-standards.1

# Check for errors
groff -man -T ascii mcp-standards.1 > /dev/null

# Preview as PDF
groff -man -T pdf mcp-standards.1 > mcp-standards.pdf
```

## Distribution

### In Python Package

```python
# setup.py
setup(
    name='mcp-standards-server',
    # ... other config ...
    data_files=[
        ('share/man/man1', [
            'docs/man/mcp-standards.1',
            'docs/man/mcp-standards-sync.1',
            'docs/man/mcp-standards-validate.1',
            # ... other man pages
        ])
    ]
)
```

### In Makefile

```makefile
MANDIR = /usr/local/share/man/man1

install-man:
	install -d $(DESTDIR)$(MANDIR)
	install -m 644 docs/man/*.1 $(DESTDIR)$(MANDIR)
	mandb

uninstall-man:
	rm -f $(DESTDIR)$(MANDIR)/mcp-standards*.1
	mandb
```

## Contributing

When adding new commands or options:

1. Update the relevant man page
2. Follow existing formatting conventions
3. Test the man page renders correctly
4. Update this README if adding new pages

## Quick Reference

Common man page sections:

1. User commands
2. System calls
3. Library functions
4. Special files
5. File formats
6. Games
7. Miscellaneous
8. System administration

MCP Standards uses section 1 (user commands) for all CLI documentation.