"""
Code analyzers for different programming languages
@nist-controls: SA-11, SA-15
@evidence: Multi-language security analysis support
"""
from pathlib import Path
from typing import Optional

from .base import BaseAnalyzer, CodeAnnotation, SecurityPattern
from .go_analyzer import GoAnalyzer
from .java_analyzer import JavaAnalyzer
from .javascript_analyzer import JavaScriptAnalyzer
from .python_analyzer import PythonAnalyzer

__all__ = [
    'BaseAnalyzer',
    'CodeAnnotation',
    'SecurityPattern',
    'PythonAnalyzer',
    'JavaScriptAnalyzer',
    'GoAnalyzer',
    'JavaAnalyzer',
    'get_analyzer_for_file'
]

def get_analyzer_for_file(file_path: Path) -> Optional[BaseAnalyzer]:
    """Get appropriate analyzer for file type"""
    analyzers = {
        '.py': PythonAnalyzer(),
        '.js': JavaScriptAnalyzer(),
        '.jsx': JavaScriptAnalyzer(),
        '.ts': JavaScriptAnalyzer(),
        '.tsx': JavaScriptAnalyzer(),
        '.go': GoAnalyzer(),
        '.java': JavaAnalyzer()
    }
    return analyzers.get(file_path.suffix)