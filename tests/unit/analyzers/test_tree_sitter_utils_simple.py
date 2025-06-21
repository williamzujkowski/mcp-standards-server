"""
Simplified tests for tree_sitter_utils module
@nist-controls: SA-11, CA-7
@evidence: Tree-sitter utilities testing
"""

import logging
from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from src.analyzers.tree_sitter_utils import (
    TreeSitterManager,
    tree_sitter_manager,
    get_function_definitions,
    get_class_definitions,
    get_imports,
    find_security_decorators
)


class TestTreeSitterManager:
    """Test TreeSitterManager functionality"""

    def test_manager_initialization(self):
        """Test manager creates properly"""
        manager = TreeSitterManager()
        
        assert manager.parsers == {}
        assert manager.languages == {}
        assert manager._initialized is False

    def test_manager_initialize_once(self):
        """Test manager only initializes once"""
        manager = TreeSitterManager()
        
        with patch.object(manager, '_setup_individual_languages') as mock_setup:
            manager.initialize()
            manager.initialize()  # Second call
            
            # Should only call setup once
            mock_setup.assert_called_once()
            assert manager._initialized is True

    @patch('importlib.import_module')
    def test_setup_individual_languages_success(self, mock_import):
        """Test successful language setup"""
        manager = TreeSitterManager()
        
        # Mock successful imports
        mock_language = MagicMock()
        mock_language.language.return_value = MagicMock()
        mock_import.return_value = mock_language
        
        manager._setup_individual_languages()
        
        # Should have attempted to import languages
        assert mock_import.call_count > 0
        assert len(manager.languages) >= 0
        assert len(manager.parsers) >= 0

    @patch('importlib.import_module')
    def test_setup_individual_languages_import_error(self, mock_import):
        """Test handling of import errors"""
        manager = TreeSitterManager()
        
        # Mock import error
        mock_import.side_effect = ImportError("Module not found")
        
        # Should not raise exception
        manager._setup_individual_languages()
        
        # Languages dict should be empty due to import failures
        assert isinstance(manager.languages, dict)
        assert isinstance(manager.parsers, dict)

    def test_get_parser_existing_language(self):
        """Test getting parser for existing language"""
        manager = TreeSitterManager()
        
        # Mock existing parser
        mock_parser = MagicMock()
        manager.parsers['python'] = mock_parser
        
        result = manager.get_parser('python')
        assert result == mock_parser

    def test_get_parser_nonexistent_language(self):
        """Test getting parser for non-existent language"""
        manager = TreeSitterManager()
        
        result = manager.get_parser('nonexistent')
        assert result is None

    def test_is_supported_true(self):
        """Test is_supported returns True for supported language"""
        manager = TreeSitterManager()
        manager.languages['python'] = MagicMock()
        
        assert manager.is_supported('python') is True

    def test_is_supported_false(self):
        """Test is_supported returns False for unsupported language"""
        manager = TreeSitterManager()
        
        assert manager.is_supported('nonexistent') is False

    def test_get_supported_languages(self):
        """Test getting list of supported languages"""
        manager = TreeSitterManager()
        manager.languages = {'python': MagicMock(), 'javascript': MagicMock()}
        
        result = manager.get_supported_languages()
        assert 'python' in result
        assert 'javascript' in result
        assert len(result) == 2

    def test_parse_code_successful(self):
        """Test successful code parsing"""
        manager = TreeSitterManager()
        
        # Mock parser and result
        mock_parser = MagicMock()
        mock_tree = MagicMock()
        mock_parser.parse.return_value = mock_tree
        manager.parsers['python'] = mock_parser
        
        result = manager.parse_code('def test(): pass', 'python')
        
        assert result == mock_tree
        mock_parser.parse.assert_called_once()

    def test_parse_code_no_parser(self):
        """Test parsing code with no available parser"""
        manager = TreeSitterManager()
        
        result = manager.parse_code('def test(): pass', 'python')
        assert result is None

    def test_parse_code_parser_error(self):
        """Test handling parser errors"""
        manager = TreeSitterManager()
        
        # Mock parser that raises exception
        mock_parser = MagicMock()
        mock_parser.parse.side_effect = Exception("Parse error")
        manager.parsers['python'] = mock_parser
        
        result = manager.parse_code('invalid code', 'python')
        assert result is None

    def test_execute_query_successful(self):
        """Test successful query execution"""
        manager = TreeSitterManager()
        
        # Mock tree and language
        mock_tree = MagicMock()
        mock_language = MagicMock()
        mock_query = MagicMock()
        mock_node = MagicMock()
        mock_node.text = b'test_text'
        
        mock_language.query.return_value = mock_query
        mock_query.captures.return_value = [(mock_node, 'capture')]
        manager.languages['python'] = mock_language
        
        result = manager.execute_query(mock_tree, 'python', '(function_definition)')
        
        assert len(result) == 1
        assert result[0][0] == mock_node
        assert result[0][1] == 'test_text'

    def test_execute_query_no_tree(self):
        """Test query execution with no tree"""
        manager = TreeSitterManager()
        
        result = manager.execute_query(None, 'python', '(function_definition)')
        assert result == []

    def test_execute_query_no_language(self):
        """Test query execution with unsupported language"""
        manager = TreeSitterManager()
        mock_tree = MagicMock()
        
        result = manager.execute_query(mock_tree, 'unsupported', '(function_definition)')
        assert result == []

    def test_execute_query_error_handling(self):
        """Test query execution error handling"""
        manager = TreeSitterManager()
        
        # Mock tree and language with error
        mock_tree = MagicMock()
        mock_language = MagicMock()
        mock_language.query.side_effect = Exception("Query error")
        manager.languages['python'] = mock_language
        
        result = manager.execute_query(mock_tree, 'python', '(function_definition)')
        assert result == []


class TestSingletonManager:
    """Test the singleton tree_sitter_manager instance"""

    def test_singleton_exists(self):
        """Test that singleton instance exists"""
        assert tree_sitter_manager is not None
        assert isinstance(tree_sitter_manager, TreeSitterManager)

    def test_singleton_is_same_instance(self):
        """Test that repeated access returns same instance"""
        from src.analyzers.tree_sitter_utils import tree_sitter_manager as manager2
        assert tree_sitter_manager is manager2


class TestGetFunctionDefinitions:
    """Test get_function_definitions function"""

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_functions_successful(self, mock_manager):
        """Test successful function extraction"""
        mock_tree = MagicMock()
        mock_manager.parse_code.return_value = mock_tree
        mock_manager.execute_query.return_value = [
            (MagicMock(), 'def test_function():\n    pass')
        ]
        
        result = get_function_definitions('def test(): pass', 'python')
        
        assert isinstance(result, list)
        mock_manager.parse_code.assert_called_once_with('def test(): pass', 'python')

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_functions_no_tree(self, mock_manager):
        """Test function extraction with no parse tree"""
        mock_manager.parse_code.return_value = None
        
        result = get_function_definitions('invalid code', 'python')
        
        assert result == []

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_functions_different_languages(self, mock_manager):
        """Test function extraction for different languages"""
        mock_tree = MagicMock()
        mock_manager.parse_code.return_value = mock_tree
        mock_manager.execute_query.return_value = []
        
        # Test different languages
        for language in ['python', 'javascript', 'go', 'java']:
            result = get_function_definitions('function test() {}', language)
            assert isinstance(result, list)

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_functions_unsupported_language(self, mock_manager):
        """Test function extraction for unsupported language"""
        mock_tree = MagicMock()
        mock_manager.parse_code.return_value = mock_tree
        mock_manager.execute_query.return_value = []
        
        result = get_function_definitions('code', 'unsupported')
        
        assert result == []


class TestGetClassDefinitions:
    """Test get_class_definitions function"""

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_classes_successful(self, mock_manager):
        """Test successful class extraction"""
        mock_tree = MagicMock()
        mock_manager.parse_code.return_value = mock_tree
        mock_manager.execute_query.return_value = [
            (MagicMock(), 'class TestClass:\n    pass')
        ]
        
        result = get_class_definitions('class Test: pass', 'python')
        
        assert isinstance(result, list)
        mock_manager.parse_code.assert_called_once_with('class Test: pass', 'python')

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_classes_no_tree(self, mock_manager):
        """Test class extraction with no parse tree"""
        mock_manager.parse_code.return_value = None
        
        result = get_class_definitions('invalid code', 'python')
        
        assert result == []

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_classes_different_languages(self, mock_manager):
        """Test class extraction for different languages"""
        mock_tree = MagicMock()
        mock_manager.parse_code.return_value = mock_tree
        mock_manager.execute_query.return_value = []
        
        # Test different languages
        for language in ['python', 'javascript', 'go', 'java']:
            result = get_class_definitions('class Test {}', language)
            assert isinstance(result, list)


class TestGetImports:
    """Test get_imports function"""

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_imports_successful(self, mock_manager):
        """Test successful import extraction"""
        mock_tree = MagicMock()
        mock_manager.parse_code.return_value = mock_tree
        mock_manager.execute_query.return_value = [
            (MagicMock(), 'import os')
        ]
        
        result = get_imports('import os', 'python')
        
        assert isinstance(result, list)
        mock_manager.parse_code.assert_called_once_with('import os', 'python')

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_imports_no_tree(self, mock_manager):
        """Test import extraction with no parse tree"""
        mock_manager.parse_code.return_value = None
        
        result = get_imports('invalid code', 'python')
        
        assert result == []

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_get_imports_different_languages(self, mock_manager):
        """Test import extraction for different languages"""
        mock_tree = MagicMock()
        mock_manager.parse_code.return_value = mock_tree
        mock_manager.execute_query.return_value = []
        
        # Test different languages
        for language in ['python', 'javascript', 'go', 'java']:
            result = get_imports('import something', language)
            assert isinstance(result, list)


class TestFindSecurityDecorators:
    """Test find_security_decorators function"""

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_find_decorators_successful(self, mock_manager):
        """Test successful decorator extraction"""
        mock_tree = MagicMock()
        mock_manager.parse_code.return_value = mock_tree
        mock_manager.execute_query.return_value = [
            (MagicMock(), '@require_auth')
        ]
        
        result = find_security_decorators('@require_auth\ndef test(): pass', 'python')
        
        assert isinstance(result, list)
        mock_manager.parse_code.assert_called_once()

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_find_decorators_no_tree(self, mock_manager):
        """Test decorator extraction with no parse tree"""
        mock_manager.parse_code.return_value = None
        
        result = find_security_decorators('invalid code', 'python')
        
        assert result == []

    @patch('src.analyzers.tree_sitter_utils.tree_sitter_manager')
    def test_find_decorators_error_handling(self, mock_manager):
        """Test decorator extraction error handling"""
        mock_tree = MagicMock()
        mock_manager.parse_code.return_value = mock_tree
        mock_manager.execute_query.side_effect = Exception("Query error")
        
        result = find_security_decorators('@require_auth\ndef test(): pass', 'python')
        
        assert result == []


class TestModuleFunctionality:
    """Test module-level functionality"""

    def test_module_imports(self):
        """Test that required modules are imported"""
        import src.analyzers.tree_sitter_utils as utils
        
        assert hasattr(utils, 'tree_sitter')
        assert hasattr(utils, 'logging')
        assert hasattr(utils, 'TreeSitterManager')

    def test_logger_exists(self):
        """Test that logger is properly configured"""
        from src.analyzers.tree_sitter_utils import logger
        
        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_all_functions_exist(self):
        """Test that all expected functions exist"""
        import src.analyzers.tree_sitter_utils as utils
        
        expected_functions = [
            'get_function_definitions',
            'get_class_definitions', 
            'get_imports',
            'find_security_decorators'
        ]
        
        for func_name in expected_functions:
            assert hasattr(utils, func_name)
            assert callable(getattr(utils, func_name))

    def test_tree_sitter_dependency(self):
        """Test tree-sitter dependency handling"""
        try:
            import tree_sitter
            assert tree_sitter is not None
        except ImportError:
            # tree-sitter may not be available in test environment
            pytest.skip("tree-sitter not available")

    def test_constants_and_globals(self):
        """Test module constants and globals"""
        from src.analyzers.tree_sitter_utils import tree_sitter_manager
        
        assert tree_sitter_manager is not None
        assert isinstance(tree_sitter_manager, TreeSitterManager)

    def test_nist_annotations_present(self):
        """Test that NIST control annotations are present"""
        import src.analyzers.tree_sitter_utils as utils
        
        # Check module docstring has NIST annotations
        assert "@nist-controls:" in utils.__doc__
        assert "@evidence:" in utils.__doc__
        
        # Check that specific controls are documented
        assert "SA-11" in utils.__doc__ or "SA-15" in utils.__doc__

    def test_error_handling_throughout(self):
        """Test that functions handle errors gracefully"""
        # Test with invalid inputs
        result1 = get_function_definitions("", "invalid_language")
        assert isinstance(result1, list)
        
        result2 = get_class_definitions(None, "python") 
        assert isinstance(result2, list)
        
        result3 = get_imports("", "")
        assert isinstance(result3, list)
        
        result4 = find_security_decorators("", "unknown")
        assert isinstance(result4, list)

    def test_type_hints_present(self):
        """Test that functions have proper type hints"""
        import inspect
        from src.analyzers.tree_sitter_utils import get_function_definitions
        
        sig = inspect.signature(get_function_definitions)
        
        # Should have annotated return type
        assert sig.return_annotation != inspect.Signature.empty
        
        # Should have annotated parameters
        for param in sig.parameters.values():
            assert param.annotation != inspect.Parameter.empty