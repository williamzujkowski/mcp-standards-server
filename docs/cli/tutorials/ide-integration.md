# IDE Integration Guide

This guide covers how to integrate MCP Standards Server with popular IDEs and editors for real-time standards validation and assistance.

## Table of Contents

1. [VS Code](#vs-code)
2. [JetBrains IDEs](#jetbrains-ides)
3. [Neovim](#neovim)
4. [Sublime Text](#sublime-text)
5. [Emacs](#emacs)
6. [Generic LSP Integration](#generic-lsp-integration)

## VS Code

### Method 1: Official Extension (Recommended)

1. **Install Extension**:
   ```bash
   code --install-extension mcp-standards.vscode-mcp-standards
   ```

2. **Configure Extension**:
   ```json
   // .vscode/settings.json
   {
     "mcp-standards.server.mode": "local",
     "mcp-standards.server.port": 3000,
     "mcp-standards.validation.onSave": true,
     "mcp-standards.validation.onType": false,
     "mcp-standards.validation.delay": 500,
     "mcp-standards.suggestions.enabled": true,
     "mcp-standards.token.budget": 4000,
     "mcp-standards.config.path": ".mcp-standards.yaml"
   }
   ```

3. **Start Server**:
   The extension automatically starts the server, or manually:
   ```bash
   mcp-standards serve --stdio
   ```

### Method 2: Custom Tasks

Create custom tasks for standards operations:

```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "MCP: Validate Current File",
      "type": "shell",
      "command": "mcp-standards",
      "args": [
        "validate",
        "${file}",
        "--format", "json"
      ],
      "problemMatcher": {
        "owner": "mcp-standards",
        "fileLocation": ["relative", "${workspaceFolder}"],
        "pattern": {
          "regexp": "^(.+):(\\d+):(\\d+):\\s+(error|warning|info)\\s+(.+)\\s+\\((.+)\\)$",
          "file": 1,
          "line": 2,
          "column": 3,
          "severity": 4,
          "message": 5,
          "code": 6
        }
      },
      "presentation": {
        "reveal": "never",
        "panel": "dedicated"
      }
    },
    {
      "label": "MCP: Fix Current File",
      "type": "shell",
      "command": "mcp-standards",
      "args": [
        "validate",
        "${file}",
        "--fix"
      ]
    },
    {
      "label": "MCP: Query Standards",
      "type": "shell",
      "command": "mcp-standards",
      "args": [
        "query",
        "--semantic",
        "${input:query}"
      ],
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    }
  ],
  "inputs": [
    {
      "id": "query",
      "type": "promptString",
      "description": "Enter your standards query"
    }
  ]
}
```

### Method 3: Code Actions Provider

Create a custom code actions provider:

```typescript
// extensions/mcp-standards/src/codeActions.ts
import * as vscode from 'vscode';
import { exec } from 'child_process';

export class MCPCodeActionProvider implements vscode.CodeActionProvider {
  provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range
  ): vscode.CodeAction[] {
    const actions: vscode.CodeAction[] = [];
    
    // Quick fix action
    const fixAction = new vscode.CodeAction(
      'Fix with MCP Standards',
      vscode.CodeActionKind.QuickFix
    );
    fixAction.command = {
      command: 'mcp-standards.fix',
      title: 'Fix with MCP Standards',
      arguments: [document.uri, range]
    };
    actions.push(fixAction);
    
    // Query standards action
    const queryAction = new vscode.CodeAction(
      'Query Applicable Standards',
      vscode.CodeActionKind.Source
    );
    queryAction.command = {
      command: 'mcp-standards.query',
      title: 'Query Standards'
    };
    actions.push(queryAction);
    
    return actions;
  }
}
```

### VS Code Snippets

Create standards-compliant snippets:

```json
// .vscode/mcp-snippets.code-snippets
{
  "React Component (MCP Standards)": {
    "prefix": "rcmcp",
    "body": [
      "import React from 'react';",
      "import PropTypes from 'prop-types';",
      "",
      "/**",
      " * ${1:ComponentName} - ${2:Brief description}",
      " * @component",
      " */",
      "export const ${1:ComponentName} = ({ ${3:props} }) => {",
      "  return (",
      "    <div className=\"${4:className}\" role=\"${5:region}\" aria-label=\"${6:label}\">",
      "      ${7:content}",
      "    </div>",
      "  );",
      "};",
      "",
      "${1:ComponentName}.propTypes = {",
      "  ${8:propTypes}",
      "};",
      "",
      "${1:ComponentName}.defaultProps = {",
      "  ${9:defaultProps}",
      "};",
      "",
      "export default ${1:ComponentName};"
    ],
    "description": "React component following MCP standards"
  }
}
```

## JetBrains IDEs

### IntelliJ IDEA / WebStorm / PyCharm

1. **Install Plugin**:
   - Open Settings → Plugins
   - Search for "MCP Standards"
   - Install and restart

2. **Configure External Tool**:
   ```xml
   <!-- .idea/externalTools.xml -->
   <toolSet name="MCP Standards">
     <tool name="Validate File" showInMainMenu="true" showInEditor="true">
       <exec>
         <option name="COMMAND" value="mcp-standards" />
         <option name="PARAMETERS" value="validate $FilePath$ --format json" />
         <option name="WORKING_DIRECTORY" value="$ProjectFileDir$" />
       </exec>
       <filter>
         <option name="NAME" value="MCP Output" />
         <option name="DESCRIPTION" value="Parse MCP validation output" />
         <option name="REGEXP" value="$FILE_PATH$:$LINE$:$COLUMN$: $MESSAGE$" />
       </filter>
     </tool>
   </toolSet>
   ```

3. **File Watcher**:
   ```xml
   <!-- .idea/watcherTasks.xml -->
   <TaskOptions>
     <option name="arguments" value="validate $FilePath$ --fix" />
     <option name="checkSyntaxErrors" value="true" />
     <option name="description" value="MCP Standards Auto-fix" />
     <option name="exitCodeBehavior" value="ERROR" />
     <option name="program" value="mcp-standards" />
     <option name="runOnExternalChanges" value="false" />
     <option name="scopeName" value="Project Files" />
     <option name="trackOnlyRoot" value="false" />
     <option name="workingDir" value="$ProjectFileDir$" />
   </TaskOptions>
   ```

4. **Live Templates**:
   ```xml
   <!-- .idea/templates/MCP_Standards.xml -->
   <templateSet group="MCP Standards">
     <template name="mcp-func" value="/**&#10; * $DESC$&#10; * @param {$TYPE$} $PARAM$ - $PARAM_DESC$&#10; * @returns {$RETURN_TYPE$} $RETURN_DESC$&#10; * @throws {$ERROR_TYPE$} $ERROR_DESC$&#10; */&#10;export function $NAME$($PARAMS$) {&#10;  $END$&#10;}" description="MCP Standards compliant function" toReformat="true" toShortenFQNames="true">
       <variable name="DESC" expression="" defaultValue="&quot;Function description&quot;" alwaysStopAt="true" />
       <variable name="NAME" expression="" defaultValue="&quot;functionName&quot;" alwaysStopAt="true" />
       <variable name="PARAMS" expression="" defaultValue="&quot;&quot;" alwaysStopAt="true" />
       <context>
         <option name="JAVASCRIPT" value="true" />
         <option name="TYPESCRIPT" value="true" />
       </context>
     </template>
   </templateSet>
   ```

### Custom Inspection

Create custom inspections:

```java
// MCPStandardsInspection.java
public class MCPStandardsInspection extends LocalInspectionTool {
    @Override
    public ProblemDescriptor[] checkFile(@NotNull PsiFile file, 
                                        @NotNull InspectionManager manager, 
                                        boolean isOnTheFly) {
        List<ProblemDescriptor> problems = new ArrayList<>();
        
        // Run MCP validation
        String result = runMCPValidation(file.getVirtualFile().getPath());
        List<ValidationIssue> issues = parseValidationResult(result);
        
        for (ValidationIssue issue : issues) {
            PsiElement element = file.findElementAt(issue.getOffset());
            if (element != null) {
                problems.add(manager.createProblemDescriptor(
                    element,
                    issue.getMessage(),
                    new MCPQuickFix(issue),
                    ProblemHighlightType.GENERIC_ERROR_OR_WARNING,
                    isOnTheFly
                ));
            }
        }
        
        return problems.toArray(new ProblemDescriptor[0]);
    }
}
```

## Neovim

### LSP Configuration

```lua
-- ~/.config/nvim/lua/mcp-standards.lua
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Define MCP Standards LSP
if not configs.mcp_standards then
  configs.mcp_standards = {
    default_config = {
      cmd = {'mcp-standards', 'serve', '--stdio'},
      filetypes = {'javascript', 'typescript', 'python', 'yaml'},
      root_dir = function(fname)
        return lspconfig.util.find_git_ancestor(fname) or 
               lspconfig.util.path.dirname(fname)
      end,
      settings = {
        mcp = {
          validation = {
            onSave = true,
            onType = false
          },
          suggestions = {
            enabled = true
          }
        }
      }
    }
  }
end

-- Setup LSP
lspconfig.mcp_standards.setup({
  on_attach = function(client, bufnr)
    -- Enable completion
    require('completion').on_attach(client, bufnr)
    
    -- Keybindings
    local opts = { noremap=true, silent=true, buffer=bufnr }
    vim.keymap.set('n', '<leader>sf', '<cmd>lua vim.lsp.buf.formatting()<CR>', opts)
    vim.keymap.set('n', '<leader>sq', '<cmd>lua vim.lsp.buf.code_action()<CR>', opts)
    vim.keymap.set('n', '<leader>sh', '<cmd>lua vim.lsp.buf.hover()<CR>', opts)
  end
})
```

### Telescope Integration

```lua
-- ~/.config/nvim/lua/telescope/_extensions/mcp_standards.lua
local telescope = require('telescope')
local actions = require('telescope.actions')
local pickers = require('telescope.pickers')
local finders = require('telescope.finders')
local conf = require('telescope.config').values

local function mcp_standards_picker(opts)
  opts = opts or {}
  
  pickers.new(opts, {
    prompt_title = 'MCP Standards',
    finder = finders.new_async_job({
      command_generator = function(prompt)
        return {
          'mcp-standards', 'query', 
          '--semantic', prompt,
          '--format', 'json'
        }
      end,
      entry_maker = function(entry)
        local parsed = vim.fn.json_decode(entry)
        return {
          value = parsed,
          display = parsed.title,
          ordinal = parsed.title .. ' ' .. table.concat(parsed.tags, ' ')
        }
      end
    }),
    sorter = conf.generic_sorter(opts),
    attach_mappings = function(prompt_bufnr, map)
      actions.select_default:replace(function()
        actions.close(prompt_bufnr)
        local selection = actions.get_selected_entry()
        -- Open standard in new buffer
        vim.cmd('vnew')
        vim.api.nvim_buf_set_lines(0, 0, -1, false, 
          vim.split(selection.value.content, '\n'))
      end)
      return true
    end
  }):find()
end

return telescope.register_extension({
  exports = {
    standards = mcp_standards_picker
  }
})
```

### Nvim-cmp Source

```lua
-- ~/.config/nvim/lua/cmp_mcp_standards.lua
local source = {}

source.new = function()
  return setmetatable({}, { __index = source })
end

source.is_available = function()
  return vim.fn.executable('mcp-standards') == 1
end

source.get_trigger_characters = function()
  return { '.', ':', '"', "'" }
end

source.complete = function(self, params, callback)
  local context = params.context
  local cursor = context.cursor
  
  -- Get completion from MCP server
  local items = {}
  local handle = io.popen('mcp-standards complete --position ' .. cursor.line .. ':' .. cursor.col)
  local result = handle:read("*a")
  handle:close()
  
  if result then
    local completions = vim.fn.json_decode(result)
    for _, completion in ipairs(completions) do
      table.insert(items, {
        label = completion.label,
        kind = completion.kind,
        detail = completion.detail,
        documentation = completion.documentation,
        insertText = completion.insertText
      })
    end
  end
  
  callback(items)
end

return source
```

## Sublime Text

### Package Configuration

```json
// Packages/User/MCP Standards.sublime-settings
{
  "mcp_standards": {
    "server_command": ["mcp-standards", "serve", "--stdio"],
    "enabled": true,
    "validation_on_save": true,
    "fix_on_save": false,
    "show_diagnostics_panel": true,
    "diagnostics_highlight_style": "box",
    "token_budget": 4000
  }
}
```

### Build System

```json
// Packages/User/MCP Standards.sublime-build
{
  "target": "mcp_standards_exec",
  "cancel": {"kill": true},
  "variants": [
    {
      "name": "Validate",
      "cmd": ["mcp-standards", "validate", "$file"],
      "file_regex": "^(.+?):(\\d+):(\\d+): (error|warning): (.+)$"
    },
    {
      "name": "Fix",
      "cmd": ["mcp-standards", "validate", "--fix", "$file"]
    },
    {
      "name": "Query",
      "cmd": ["mcp-standards", "query", "--context", "$file_path/.mcp-context.json"]
    }
  ]
}
```

### Plugin Script

```python
# Packages/MCP Standards/mcp_standards.py
import sublime
import sublime_plugin
import subprocess
import json

class McpStandardsValidateCommand(sublime_plugin.TextCommand):
    def run(self, edit):
        file_path = self.view.file_name()
        if not file_path:
            return
        
        # Run validation
        result = subprocess.run(
            ['mcp-standards', 'validate', file_path, '--format', 'json'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            sublime.status_message("MCP Standards: No issues found")
        else:
            issues = json.loads(result.stdout)
            self.show_issues(issues)
    
    def show_issues(self, issues):
        panel = self.view.window().create_output_panel("mcp_standards")
        panel.set_read_only(False)
        
        for issue in issues['issues']:
            panel.run_command('append', {
                'characters': f"{issue['file']}:{issue['line']}:{issue['column']}: "
                            f"{issue['severity']}: {issue['message']}\n"
            })
        
        panel.set_read_only(True)
        self.view.window().run_command("show_panel", {"panel": "output.mcp_standards"})

class McpStandardsEventListener(sublime_plugin.EventListener):
    def on_post_save_async(self, view):
        settings = sublime.load_settings("MCP Standards.sublime-settings")
        if settings.get("mcp_standards", {}).get("validation_on_save", True):
            view.run_command("mcp_standards_validate")
```

## Emacs

### Configuration with LSP-mode

```elisp
;; ~/.emacs.d/mcp-standards.el
(require 'lsp-mode)

;; Define MCP Standards LSP client
(defcustom lsp-mcp-standards-server-command
  '("mcp-standards" "serve" "--stdio")
  "Command to start MCP Standards LSP server."
  :group 'lsp-mcp-standards
  :type '(repeat string))

(lsp-register-client
 (make-lsp-client
  :new-connection (lsp-stdio-connection lsp-mcp-standards-server-command)
  :major-modes '(js-mode typescript-mode python-mode yaml-mode)
  :priority -1
  :server-id 'mcp-standards
  :initialization-options '((validation (onSave t) (onType nil))
                          (suggestions (enabled t))
                          (tokenBudget 4000))))

;; Keybindings
(defun mcp-standards-setup ()
  "Setup MCP Standards keybindings."
  (local-set-key (kbd "C-c m v") 'mcp-standards-validate)
  (local-set-key (kbd "C-c m f") 'mcp-standards-fix)
  (local-set-key (kbd "C-c m q") 'mcp-standards-query))

(add-hook 'js-mode-hook #'mcp-standards-setup)
(add-hook 'python-mode-hook #'mcp-standards-setup)

;; Interactive commands
(defun mcp-standards-validate ()
  "Validate current buffer with MCP Standards."
  (interactive)
  (compile (format "mcp-standards validate %s" (buffer-file-name))))

(defun mcp-standards-fix ()
  "Fix current buffer with MCP Standards."
  (interactive)
  (shell-command
   (format "mcp-standards validate --fix %s" (buffer-file-name)))
  (revert-buffer t t))

(defun mcp-standards-query (query)
  "Query MCP Standards."
  (interactive "sQuery: ")
  (let ((buf (get-buffer-create "*MCP Standards Query*")))
    (with-current-buffer buf
      (erase-buffer)
      (insert (shell-command-to-string
               (format "mcp-standards query --semantic '%s'" query)))
      (markdown-mode))
    (switch-to-buffer buf)))
```

### Flycheck Integration

```elisp
;; ~/.emacs.d/flycheck-mcp-standards.el
(require 'flycheck)

(flycheck-define-checker mcp-standards
  "MCP Standards validator."
  :command ("mcp-standards" "validate" source "--format" "json")
  :error-parser flycheck-parse-json
  :modes (js-mode typescript-mode python-mode)
  :next-checkers ((warning . javascript-eslint)))

(add-to-list 'flycheck-checkers 'mcp-standards)

;; Auto-fix function
(defun flycheck-mcp-standards-fix ()
  "Fix MCP Standards issues in current buffer."
  (interactive)
  (when (and flycheck-mode
             (eq flycheck-checker 'mcp-standards))
    (shell-command-on-region
     (point-min) (point-max)
     "mcp-standards validate --fix -"
     nil t)))
```

## Generic LSP Integration

For any editor supporting Language Server Protocol:

### LSP Server Wrapper

```bash
#!/bin/bash
# mcp-standards-lsp

# Start MCP Standards in LSP mode
exec mcp-standards serve --stdio --lsp-mode "$@"
```

### LSP Configuration

```json
{
  "languageserver": {
    "mcp-standards": {
      "command": "mcp-standards-lsp",
      "filetypes": ["javascript", "typescript", "python", "yaml"],
      "rootPatterns": [".mcp-standards.yaml", ".git"],
      "settings": {
        "mcp": {
          "validation": {
            "enabled": true,
            "onSave": true,
            "onType": false
          },
          "codeActions": {
            "enabled": true,
            "showDocumentation": true
          },
          "completion": {
            "enabled": true,
            "triggerCharacters": [".", ":", "\"", "'"]
          }
        }
      }
    }
  }
}
```

## Common Features Across IDEs

### 1. Real-time Validation
- Underline/highlight standards violations
- Show error details on hover
- Gutter icons for issue severity

### 2. Quick Fixes
- Apply automated fixes
- Bulk fix all issues in file
- Preview changes before applying

### 3. Code Completion
- Standards-compliant snippets
- Context-aware suggestions
- Documentation on hover

### 4. Code Actions
- "Fix with MCP Standards"
- "Query applicable standards"
- "Generate compliant code"

### 5. Integrated Documentation
- View standards inline
- Search standards database
- Context-sensitive help

## Troubleshooting IDE Integration

### Server Not Starting

```bash
# Check if server starts manually
mcp-standards serve --stdio < /dev/null

# Check logs
tail -f ~/.cache/mcp-standards/logs/server.log
```

### Performance Issues

```yaml
# .mcp-standards.yaml
server:
  ide_mode:
    debounce_ms: 1000  # Increase for less frequent validation
    max_file_size_kb: 500  # Skip large files
    incremental: true  # Only validate changes
```

### Communication Errors

```bash
# Test LSP communication
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | mcp-standards serve --stdio
```

## Best Practices

1. **Configure per-project**: Use `.mcp-standards.yaml` in project root
2. **Optimize for performance**: Adjust validation frequency
3. **Use quick fixes**: Let MCP fix issues automatically
4. **Learn keyboard shortcuts**: Speed up your workflow
5. **Customize rules**: Disable noisy rules per-project
6. **Keep server running**: Better performance than starting per-file