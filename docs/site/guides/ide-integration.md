# IDE Integration Guide

Integrate MCP Standards Server with your favorite IDE for real-time validation and suggestions.

## Supported IDEs

- **Visual Studio Code** - Official extension available
- **JetBrains IDEs** - Plugin for IntelliJ, PyCharm, WebStorm, etc.
- **Vim/Neovim** - Language Server Protocol support
- **Emacs** - LSP integration
- **Sublime Text** - Package available

## Visual Studio Code

### Installation

1. Install the MCP Standards extension from VS Code Marketplace
2. Ensure MCP Standards Server is installed: `pip install mcp-standards-server`
3. Start the MCP server: `mcp-standards serve`

### Configuration

Add to your VS Code settings (`settings.json`):

```json
{
  "mcpStandards.serverUrl": "http://localhost:8080",
  "mcpStandards.enableRealTimeValidation": true,
  "mcpStandards.autoFixOnSave": true,
  "mcpStandards.showInlineHints": true,
  "mcpStandards.validationSeverity": "warning"
}
```

### Features

- **Real-time validation** as you type
- **Auto-fix on save** for common issues
- **Inline hints** and suggestions
- **Quick fixes** via code actions
- **Standards explorer** in sidebar
- **Project analysis** with applicable standards

### Commands

Access via Command Palette (`Ctrl+Shift+P`):

- `MCP Standards: Validate Current File`
- `MCP Standards: Validate Workspace`
- `MCP Standards: Fix All Issues`
- `MCP Standards: Show Applicable Standards`
- `MCP Standards: Refresh Standards`

### Workspace Configuration

Create `.vscode/settings.json` in your project:

```json
{
  "mcpStandards.projectType": "web_application",
  "mcpStandards.framework": "react",
  "mcpStandards.enabledStandards": [
    "react-patterns",
    "typescript-strict",
    "accessibility-wcag"
  ],
  "mcpStandards.excludePatterns": [
    "**/node_modules/**",
    "**/*.min.js",
    "**/dist/**"
  ]
}
```

## JetBrains IDEs

### Installation

1. Go to `File > Settings > Plugins`
2. Search for "MCP Standards"
3. Install and restart IDE
4. Configure server connection in `Settings > Tools > MCP Standards`

### Configuration

**Settings > Tools > MCP Standards:**

- **Server URL:** `http://localhost:8080`
- **Enable real-time validation:** ✓
- **Auto-fix on save:** ✓
- **Validation level:** Warning
- **Show notifications:** ✓

### Features

- **Code inspections** with standards violations
- **Quick fixes** and suggestions
- **Project analysis** tool window
- **Standards documentation** in tooltips
- **Integration with Code Cleanup**

### Custom Inspection Profiles

Create project-specific inspection profiles:

1. `File > Settings > Editor > Inspections`
2. Create new profile: "MCP Standards"
3. Enable MCP Standards inspections
4. Configure severity levels
5. Apply to project

## Vim/Neovim

### Setup with CoC (Conquer of Completion)

1. Install CoC: [coc.nvim](https://github.com/neoclide/coc.nvim)
2. Install MCP Standards language server:
   ```bash
   pip install mcp-standards-lsp
   ```
3. Configure CoC (`~/.vim/coc-settings.json`):

```json
{
  "languageserver": {
    "mcp-standards": {
      "command": "mcp-standards-lsp",
      "args": ["--stdio"],
      "filetypes": ["python", "javascript", "typescript", "go", "rust"],
      "settings": {
        "mcpStandards": {
          "serverUrl": "http://localhost:8080",
          "enableRealTimeValidation": true
        }
      }
    }
  }
}
```

### Setup with Native LSP (Neovim 0.5+)

```lua
-- ~/.config/nvim/lua/lsp-config.lua
local lspconfig = require('lspconfig')

lspconfig.mcp_standards.setup{
  cmd = {'mcp-standards-lsp', '--stdio'},
  filetypes = {'python', 'javascript', 'typescript', 'go', 'rust'},
  settings = {
    mcpStandards = {
      serverUrl = 'http://localhost:8080',
      enableRealTimeValidation = true
    }
  }
}
```

### Key Mappings

Add to your Vim configuration:

```vim
" Validate current file
nnoremap <leader>mv :call CocAction('runCommand', 'mcp-standards.validate')<CR>

" Show applicable standards
nnoremap <leader>ms :call CocAction('runCommand', 'mcp-standards.showStandards')<CR>

" Fix current line
nnoremap <leader>mf :call CocAction('codeAction', 'line')<CR>
```

## Emacs

### Setup with LSP Mode

1. Install `lsp-mode` and `mcp-standards-lsp`
2. Add to your Emacs configuration:

```elisp
;; ~/.emacs.d/init.el
(use-package lsp-mode
  :hook ((python-mode js-mode typescript-mode go-mode rust-mode) . lsp)
  :config
  (lsp-register-client
   (make-lsp-client
    :new-connection (lsp-stdio-connection "mcp-standards-lsp")
    :major-modes '(python-mode js-mode typescript-mode go-mode rust-mode)
    :server-id 'mcp-standards)))

;; Optional: UI improvements
(use-package lsp-ui :commands lsp-ui-mode)
(use-package company-lsp :commands company-lsp)
```

### Key Bindings

```elisp
;; Add to your configuration
(define-key lsp-mode-map (kbd "C-c m v") #'lsp-mcp-standards-validate)
(define-key lsp-mode-map (kbd "C-c m s") #'lsp-mcp-standards-show-applicable)
(define-key lsp-mode-map (kbd "C-c m f") #'lsp-execute-code-action)
```

## Sublime Text

### Installation

1. Install Package Control if not already installed
2. Install "MCP Standards" package
3. Configure in `Preferences > Package Settings > MCP Standards`

### Configuration

```json
{
  "server_url": "http://localhost:8080",
  "enable_real_time_validation": true,
  "auto_fix_on_save": true,
  "validation_severity": "warning",
  "show_in_status_bar": true
}
```

## Generic Language Server Integration

For editors supporting Language Server Protocol:

### Installation

```bash
pip install mcp-standards-lsp
```

### Configuration Template

```json
{
  "command": "mcp-standards-lsp",
  "args": ["--stdio"],
  "filetypes": ["python", "javascript", "typescript", "go", "rust"],
  "initializationOptions": {
    "serverUrl": "http://localhost:8080",
    "enableRealTimeValidation": true,
    "autoFixOnSave": true
  }
}
```

## Advanced Configuration

### Project-Specific Standards

Create `.mcp-standards.json` in your project root:

```json
{
  "projectType": "web_application",
  "language": "typescript",
  "framework": "react",
  "standards": {
    "required": [
      "react-patterns",
      "typescript-strict"
    ],
    "optional": [
      "accessibility-wcag",
      "performance-optimization"
    ]
  },
  "validation": {
    "severity": "warning",
    "autoFix": true,
    "excludePatterns": [
      "**/node_modules/**",
      "**/*.test.ts"
    ]
  }
}
```

### Custom Standards Development

For developing custom standards with IDE support:

```json
{
  "development": {
    "customStandardsPath": "./standards",
    "watchForChanges": true,
    "validateCustomStandards": true
  }
}
```

## Troubleshooting IDE Integration

### Common Issues

**Extension not working:**
1. Check MCP server is running: `curl http://localhost:8080/health`
2. Verify extension is enabled in IDE
3. Check extension logs for errors
4. Restart IDE and server

**Slow performance:**
1. Reduce validation frequency
2. Exclude large directories
3. Increase server workers
4. Enable incremental validation

**Network issues:**
1. Check firewall settings
2. Verify server URL in configuration
3. Test connection manually

### Debugging

Enable debug mode in IDE extension:

```json
{
  "mcpStandards.debug": true,
  "mcpStandards.logLevel": "debug"
}
```

Check logs:
- **VS Code:** Output panel > MCP Standards
- **JetBrains:** Help > Show Log in Explorer
- **Vim/Neovim:** `:CocInfo` or LSP logs

## Best Practices

### Team Configuration

1. **Shared settings:** Commit IDE configuration to repository
2. **Consistent standards:** Use same standards across team
3. **Pre-commit hooks:** Validate before commits
4. **Documentation:** Document team-specific setup

### Performance Tips

1. **Selective validation:** Only validate changed files
2. **Background processing:** Use async validation
3. **Cache optimization:** Warm cache for common standards
4. **Resource limits:** Configure appropriate limits

### Security Considerations

1. **Server access:** Restrict server to localhost in development
2. **Authentication:** Use API keys in shared environments
3. **Network security:** Use HTTPS in production
4. **Code privacy:** Be aware of what code is analyzed

---

For more help with IDE integration, visit our [troubleshooting guide](../reference/troubleshooting.md) or join the [community discussion](https://github.com/williamzujkowski/mcp-standards-server/discussions).
