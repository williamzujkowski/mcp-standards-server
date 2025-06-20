"""
Tree-sitter utilities for language analyzers
@nist-controls: SA-11, SA-15
@evidence: Unified AST parsing infrastructure for security analysis
"""
import logging
from pathlib import Path
from typing import Any
import tree_sitter

logger = logging.getLogger(__name__)


class TreeSitterManager:
    """
    Manages tree-sitter parsers for different languages
    @nist-controls: SA-11, SA-15
    @evidence: Centralized parser management for consistent analysis
    """
    
    def __init__(self):
        self.parsers = {}
        self.languages = {}
        self._initialized = False
        
    def initialize(self):
        """Initialize tree-sitter languages"""
        if self._initialized:
            return
            
        # For now, use individual language setup until tree-sitter-languages issue is resolved
        self._setup_individual_languages()
            
    def _setup_individual_languages(self):
        """Setup individual language packages"""
        language_modules = {
            'python': 'tree_sitter_python',
            'javascript': 'tree_sitter_javascript',
            'typescript': 'tree_sitter_typescript',
            'go': 'tree_sitter_go',
            'java': 'tree_sitter_java',
            'ruby': 'tree_sitter_ruby',
            'php': 'tree_sitter_php',
            'rust': 'tree_sitter_rust',
            'cpp': 'tree_sitter_cpp',
            'csharp': 'tree_sitter_csharp'
        }
        
        for lang_name, module_name in language_modules.items():
            try:
                module = __import__(module_name)
                language = getattr(module, 'language', None)
                if language:
                    self.languages[lang_name] = language
                    parser = tree_sitter.Parser()
                    parser.set_language(language)
                    self.parsers[lang_name] = parser
                    logger.info(f"Loaded tree-sitter language: {lang_name}")
            except ImportError:
                logger.debug(f"tree-sitter language not available: {lang_name}")
            except Exception as e:
                logger.error(f"Error loading tree-sitter language {lang_name}: {e}")
                
        self._initialized = True
        
    def get_parser_for_language(self, language: str) -> tree_sitter.Parser | None:
        """Get parser for a specific language"""
        if not self._initialized:
            self.initialize()
            
        # Handle TypeScript as JavaScript
        if language == 'typescript':
            language = 'javascript'
            
        # If using bundled languages
        if hasattr(self, 'get_parser'):
            try:
                return self.get_parser(language)
            except Exception:
                pass
                
        # Fallback to individual parsers
        return self.parsers.get(language)
        
    def parse_code(self, code: str, language: str) -> tree_sitter.Tree | None:
        """Parse code and return AST tree"""
        parser = self.get_parser_for_language(language)
        if not parser:
            return None
            
        try:
            tree = parser.parse(bytes(code, 'utf8'))
            return tree
        except Exception as e:
            logger.error(f"Error parsing {language} code: {e}")
            return None
            
    def query_tree(self, tree: tree_sitter.Tree, query_string: str, language: str) -> list[tuple[Any, str]]:
        """Execute a tree-sitter query on the AST"""
        if not tree:
            return []
            
        try:
            # Get language object
            if hasattr(self, 'get_language'):
                language_obj = self.get_language(language)
            else:
                language_obj = self.languages.get(language)
                
            if not language_obj:
                return []
                
            query = language_obj.query(query_string)
            captures = query.captures(tree.root_node)
            
            results = []
            for node, capture_name in captures:
                text = node.text.decode('utf8') if node.text else ""
                results.append((node, text))
                
            return results
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []


# Singleton instance
tree_sitter_manager = TreeSitterManager()


def get_function_definitions(code: str, language: str) -> list[dict[str, Any]]:
    """Extract function definitions from code"""
    manager = tree_sitter_manager
    tree = manager.parse_code(code, language)
    if not tree:
        return []
        
    functions = []
    
    # Language-specific queries
    queries = {
        'python': """
            (function_definition
                name: (identifier) @function.name
                parameters: (parameters) @function.params
            ) @function
        """,
        'javascript': """
            [
                (function_declaration
                    name: (identifier) @function.name
                ) @function
                (method_definition
                    name: (property_identifier) @function.name
                ) @function
                (arrow_function
                    parameters: (formal_parameters) @function.params
                ) @function
            ]
        """,
        'go': """
            (function_declaration
                name: (identifier) @function.name
                parameters: (parameter_list) @function.params
            ) @function
        """,
        'java': """
            (method_declaration
                name: (identifier) @function.name
                parameters: (formal_parameters) @function.params
            ) @function
        """
    }
    
    query_string = queries.get(language)
    if not query_string:
        return functions
        
    results = manager.query_tree(tree, query_string, language)
    
    for node, text in results:
        if node.type in ['function_definition', 'function_declaration', 'method_definition', 'method_declaration', 'arrow_function']:
            func_info = {
                'name': None,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'text': text
            }
            
            # Extract function name
            for child in node.children:
                if child.type == 'identifier' or child.type == 'property_identifier':
                    func_info['name'] = child.text.decode('utf8')
                    break
                    
            functions.append(func_info)
            
    return functions


def get_imports(code: str, language: str) -> list[dict[str, Any]]:
    """Extract import statements from code"""
    manager = tree_sitter_manager
    tree = manager.parse_code(code, language)
    if not tree:
        return []
        
    imports = []
    
    # Language-specific queries
    queries = {
        'python': """
            [
                (import_statement) @import
                (import_from_statement) @import
            ]
        """,
        'javascript': """
            [
                (import_statement) @import
                (import_declaration) @import
            ]
        """,
        'go': """
            (import_declaration) @import
        """,
        'java': """
            (import_declaration) @import
        """
    }
    
    query_string = queries.get(language)
    if not query_string:
        return imports
        
    results = manager.query_tree(tree, query_string, language)
    
    for node, text in results:
        import_info = {
            'line': node.start_point[0] + 1,
            'text': text,
            'module': None
        }
        
        # Extract module name based on language
        if language == 'python':
            if 'import' in text:
                parts = text.split()
                if 'from' in parts:
                    idx = parts.index('from')
                    if idx + 1 < len(parts):
                        import_info['module'] = parts[idx + 1]
                elif 'import' in parts:
                    idx = parts.index('import')
                    if idx + 1 < len(parts):
                        import_info['module'] = parts[idx + 1].split('.')[0]
                        
        imports.append(import_info)
        
    return imports


def get_class_definitions(code: str, language: str) -> list[dict[str, Any]]:
    """Extract class definitions from code"""
    manager = tree_sitter_manager
    tree = manager.parse_code(code, language)
    if not tree:
        return []
        
    classes = []
    
    # Language-specific queries
    queries = {
        'python': """
            (class_definition
                name: (identifier) @class.name
            ) @class
        """,
        'javascript': """
            (class_declaration
                name: (identifier) @class.name
            ) @class
        """,
        'go': """
            (type_declaration
                (type_spec
                    name: (type_identifier) @class.name
                    type: (struct_type)
                )
            ) @class
        """,
        'java': """
            (class_declaration
                name: (identifier) @class.name
            ) @class
        """
    }
    
    query_string = queries.get(language)
    if not query_string:
        return classes
        
    results = manager.query_tree(tree, query_string, language)
    
    for node, text in results:
        if node.type in ['class_definition', 'class_declaration', 'type_declaration']:
            class_info = {
                'name': None,
                'start_line': node.start_point[0] + 1,
                'end_line': node.end_point[0] + 1,
                'text': text[:200] + '...' if len(text) > 200 else text
            }
            
            # Extract class name
            for child in node.children:
                if child.type == 'identifier' or child.type == 'type_identifier':
                    class_info['name'] = child.text.decode('utf8')
                    break
                    
            classes.append(class_info)
            
    return classes


def find_security_decorators(code: str, language: str) -> list[dict[str, Any]]:
    """Find security-related decorators/annotations"""
    manager = tree_sitter_manager
    tree = manager.parse_code(code, language)
    if not tree:
        return []
        
    decorators = []
    
    # Language-specific queries
    queries = {
        'python': """
            (decorator
                (identifier) @decorator.name
            ) @decorator
        """,
        'javascript': """
            (decorator
                (identifier) @decorator.name
            ) @decorator
        """,
        'java': """
            (annotation
                name: (identifier) @annotation.name
            ) @annotation
        """
    }
    
    query_string = queries.get(language)
    if not query_string:
        return decorators
        
    results = manager.query_tree(tree, query_string, language)
    
    # Security-related decorator patterns
    security_patterns = [
        'auth', 'permission', 'role', 'secure', 'protect',
        'validate', 'sanitize', 'rate_limit', 'csrf', 'cors'
    ]
    
    for node, text in results:
        text_lower = text.lower()
        if any(pattern in text_lower for pattern in security_patterns):
            decorator_info = {
                'line': node.start_point[0] + 1,
                'text': text,
                'type': 'security'
            }
            decorators.append(decorator_info)
            
    return decorators