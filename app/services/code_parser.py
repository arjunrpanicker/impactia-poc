"""
Enhanced code parsing service with better language support and AST analysis
"""
import ast
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class Language(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CSHARP = "csharp"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"

@dataclass
class CodeElement:
    name: str
    type: str  # function, class, method, variable
    content: str
    start_line: int
    end_line: int
    parameters: List[str] = None
    return_type: str = None
    docstring: str = None
    complexity: int = 0
    dependencies: List[str] = None

class CodeParser:
    def __init__(self):
        self.language_patterns = {
            Language.PYTHON: {
                'function': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:',
                'class': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:',
                'import': r'(?:from\s+[\w.]+\s+)?import\s+([\w.,\s*]+)',
            },
            Language.JAVASCRIPT: {
                'function': r'(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
                'arrow_function': r'const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
                'class': r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?\s*{',
                'import': r'import\s+(?:{[^}]+}|\*\s+as\s+\w+|\w+)\s+from\s+[\'"][^\'"]+[\'"]',
            },
            Language.TYPESCRIPT: {
                'function': r'(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:\s*[^{]+',
                'method': r'(?:public|private|protected)?\s*(?:async\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:\s*[^{]+',
                'class': r'(?:export\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+[a-zA-Z_][a-zA-Z0-9_]*)?\s*{',
                'interface': r'interface\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*{',
            },
            Language.CSHARP: {
                'method': r'(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?[a-zA-Z_<>[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
                'class': r'(?:public|internal)?\s*(?:abstract\s+|sealed\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                'interface': r'(?:public|internal)?\s*interface\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            }
        }

    def detect_language(self, file_path: str, content: str) -> Language:
        """Detect programming language from file extension and content"""
        ext = os.path.splitext(file_path)[1].lower()
        
        extension_map = {
            '.py': Language.PYTHON,
            '.js': Language.JAVASCRIPT,
            '.jsx': Language.JAVASCRIPT,
            '.ts': Language.TYPESCRIPT,
            '.tsx': Language.TYPESCRIPT,
            '.cs': Language.CSHARP,
            '.java': Language.JAVA,
            '.go': Language.GO,
            '.rs': Language.RUST,
            '.cpp': Language.CPP,
            '.cc': Language.CPP,
            '.hpp': Language.CPP,
        }
        
        return extension_map.get(ext, Language.PYTHON)

    def parse_file(self, file_path: str, content: str) -> List[CodeElement]:
        """Parse a file and extract code elements"""
        language = self.detect_language(file_path, content)
        
        if language == Language.PYTHON:
            return self._parse_python(content)
        else:
            return self._parse_generic(content, language)

    def _parse_python(self, content: str) -> List[CodeElement]:
        """Parse Python code using AST"""
        elements = []
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    element = self._extract_python_function(node, lines)
                    elements.append(element)
                elif isinstance(node, ast.ClassDef):
                    element = self._extract_python_class(node, lines)
                    elements.append(element)
                    
        except SyntaxError as e:
            # Fallback to regex parsing
            return self._parse_generic(content, Language.PYTHON)
            
        return elements

    def _extract_python_function(self, node: ast.FunctionDef, lines: List[str]) -> CodeElement:
        """Extract Python function details"""
        start_line = node.lineno
        end_line = node.end_lineno or start_line + 10
        
        # Extract function content
        content = '\n'.join(lines[start_line-1:end_line])
        
        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]
        
        # Extract docstring
        docstring = ast.get_docstring(node) or ""
        
        # Calculate complexity (simplified)
        complexity = self._calculate_complexity(node)
        
        # Extract dependencies (function calls)
        dependencies = self._extract_dependencies(node)
        
        return CodeElement(
            name=node.name,
            type="function",
            content=content,
            start_line=start_line,
            end_line=end_line,
            parameters=parameters,
            docstring=docstring,
            complexity=complexity,
            dependencies=dependencies
        )

    def _extract_python_class(self, node: ast.ClassDef, lines: List[str]) -> CodeElement:
        """Extract Python class details"""
        start_line = node.lineno
        end_line = node.end_lineno or start_line + 20
        
        content = '\n'.join(lines[start_line-1:end_line])
        docstring = ast.get_docstring(node) or ""
        
        # Extract base classes
        base_classes = [base.id for base in node.bases if hasattr(base, 'id')]
        
        return CodeElement(
            name=node.name,
            type="class",
            content=content,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            dependencies=base_classes
        )

    def _parse_generic(self, content: str, language: Language) -> List[CodeElement]:
        """Parse code using regex patterns"""
        elements = []
        lines = content.split('\n')
        patterns = self.language_patterns.get(language, {})
        
        for pattern_type, pattern in patterns.items():
            if pattern_type == 'import':
                continue
                
            for match in re.finditer(pattern, content, re.MULTILINE):
                name = match.group(1)
                start_line = content[:match.start()].count('\n') + 1
                
                # Estimate end line (simple heuristic)
                end_line = min(start_line + 20, len(lines))
                element_content = '\n'.join(lines[start_line-1:end_line])
                
                elements.append(CodeElement(
                    name=name,
                    type=pattern_type,
                    content=element_content,
                    start_line=start_line,
                    end_line=end_line
                ))
                
        return elements

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
                
        return complexity

    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extract function calls and imports"""
        dependencies = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if hasattr(child.func, 'id'):
                    dependencies.append(child.func.id)
                elif hasattr(child.func, 'attr'):
                    dependencies.append(child.func.attr)
                    
        return list(set(dependencies))

    def extract_imports(self, content: str, language: Language) -> List[str]:
        """Extract import statements"""
        imports = []
        patterns = self.language_patterns.get(language, {})
        
        if 'import' in patterns:
            for match in re.finditer(patterns['import'], content, re.MULTILINE):
                imports.append(match.group(0))
                
        return imports