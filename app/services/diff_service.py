import ast
import os
import re
from typing import List, Optional, Tuple, Dict, Any
from openai import AzureOpenAI
from dataclasses import dataclass
from enum import Enum

class ChangeType(Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"

@dataclass
class FunctionChange:
    name: str
    change_type: ChangeType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    line_changes: List[Tuple[int, str]] = None  # (line_number, change_type)

class GitDiffParser:
    """Enhanced git diff parser with better AST integration"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    def parse_git_diff(self, diff_content: str) -> Dict[str, Any]:
        """Parse git diff and extract meaningful changes"""
        files_changed = {}
        current_file = None
        hunks = []
        
        lines = diff_content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # New file header
            if line.startswith('diff --git'):
                if current_file:
                    files_changed[current_file] = self._analyze_file_changes(hunks)
                
                # Extract file path
                match = re.search(r'diff --git a/(.*?) b/(.*)', line)
                if match:
                    current_file = match.group(2)
                    hunks = []
            
            # Hunk header
            elif line.startswith('@@'):
                hunk_info = self._parse_hunk_header(line)
                hunk_content = []
                i += 1
                
                # Collect hunk content
                while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('diff --git'):
                    hunk_content.append(lines[i])
                    i += 1
                
                hunks.append({
                    'header': hunk_info,
                    'content': hunk_content
                })
                continue
            
            i += 1
        
        # Process last file
        if current_file:
            files_changed[current_file] = self._analyze_file_changes(hunks)
        
        return files_changed

    def _parse_hunk_header(self, header: str) -> Dict[str, int]:
        """Parse hunk header to extract line numbers"""
        match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', header)
        if match:
            return {
                'old_start': int(match.group(1)),
                'old_count': int(match.group(2)) if match.group(2) else 1,
                'new_start': int(match.group(3)),
                'new_count': int(match.group(4)) if match.group(4) else 1
            }
        return {}

    def _analyze_file_changes(self, hunks: List[Dict]) -> Dict[str, Any]:
        """Analyze changes within a file"""
        old_content = []
        new_content = []
        line_changes = []
        
        for hunk in hunks:
            hunk_content = hunk['content']
            header = hunk['header']
            
            old_line_num = header.get('old_start', 1)
            new_line_num = header.get('new_start', 1)
            
            for line in hunk_content:
                if line.startswith(' '):  # Context line
                    old_content.append(line[1:])
                    new_content.append(line[1:])
                    old_line_num += 1
                    new_line_num += 1
                elif line.startswith('-'):  # Deleted line
                    old_content.append(line[1:])
                    line_changes.append((old_line_num, 'deleted'))
                    old_line_num += 1
                elif line.startswith('+'):  # Added line
                    new_content.append(line[1:])
                    line_changes.append((new_line_num, 'added'))
                    new_line_num += 1
        
        return {
            'old_content': '\n'.join(old_content),
            'new_content': '\n'.join(new_content),
            'line_changes': line_changes
        }

class EnhancedASTAnalyzer:
    """Enhanced AST analyzer with better diff support"""
    
    def __init__(self):
        self.supported_languages = {
            '.py': self._analyze_python,
            '.js': self._analyze_javascript,
            '.ts': self._analyze_typescript,
            '.jsx': self._analyze_javascript,
            '.tsx': self._analyze_typescript
        }

    def analyze_function_changes(self, file_path: str, old_content: str, new_content: str) -> List[FunctionChange]:
        """Analyze function-level changes between old and new content"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in self.supported_languages:
            return self.supported_languages[file_ext](old_content, new_content)
        else:
            # Fallback to regex-based analysis
            return self._analyze_generic(old_content, new_content)

    def _analyze_python(self, old_content: str, new_content: str) -> List[FunctionChange]:
        """Analyze Python code changes using AST"""
        changes = []
        
        try:
            old_functions = self._extract_python_functions(old_content)
            new_functions = self._extract_python_functions(new_content)
            
            # Find added functions
            for name, func_info in new_functions.items():
                if name not in old_functions:
                    changes.append(FunctionChange(
                        name=name,
                        change_type=ChangeType.ADDED,
                        new_content=func_info['content']
                    ))
            
            # Find deleted functions
            for name, func_info in old_functions.items():
                if name not in new_functions:
                    changes.append(FunctionChange(
                        name=name,
                        change_type=ChangeType.DELETED,
                        old_content=func_info['content']
                    ))
            
            # Find modified functions
            for name in old_functions:
                if name in new_functions:
                    old_func = old_functions[name]
                    new_func = new_functions[name]
                    
                    if self._functions_differ(old_func, new_func):
                        changes.append(FunctionChange(
                            name=name,
                            change_type=ChangeType.MODIFIED,
                            old_content=old_func['content'],
                            new_content=new_func['content']
                        ))
            
        except SyntaxError as e:
            print(f"[DEBUG] Syntax error in Python analysis: {e}")
            # Fallback to regex analysis
            return self._analyze_generic(old_content, new_content)
        
        return changes

    def _extract_python_functions(self, content: str) -> Dict[str, Dict]:
        """Extract Python functions using AST"""
        functions = {}
        
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    
                    func_content = '\n'.join(lines[start_line:end_line])
                    
                    functions[node.name] = {
                        'content': func_content,
                        'start_line': start_line,
                        'end_line': end_line,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                        'docstring': ast.get_docstring(node),
                        'complexity': self._calculate_complexity(node)
                    }
        except Exception as e:
            print(f"[DEBUG] Error extracting Python functions: {e}")
        
        return functions

    def _functions_differ(self, old_func: Dict, new_func: Dict) -> bool:
        """Check if two function definitions differ significantly"""
        # Compare normalized content (ignoring whitespace differences)
        old_normalized = re.sub(r'\s+', ' ', old_func['content']).strip()
        new_normalized = re.sub(r'\s+', ' ', new_func['content']).strip()
        
        return old_normalized != new_normalized

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity

    def _analyze_javascript(self, old_content: str, new_content: str) -> List[FunctionChange]:
        """Analyze JavaScript/JSX changes using regex patterns"""
        return self._analyze_generic(old_content, new_content, language='javascript')

    def _analyze_typescript(self, old_content: str, new_content: str) -> List[FunctionChange]:
        """Analyze TypeScript/TSX changes using regex patterns"""
        return self._analyze_generic(old_content, new_content, language='typescript')

    def _analyze_generic(self, old_content: str, new_content: str, language: str = 'generic') -> List[FunctionChange]:
        """Generic function analysis using regex patterns"""
        changes = []
        
        # Language-specific patterns
        patterns = {
            'python': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:',
            'javascript': r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)',
            'typescript': r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*:\s*[^=]*=>)',
            'generic': r'(?:def|function|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        }
        
        pattern = patterns.get(language, patterns['generic'])
        
        old_functions = self._extract_functions_regex(old_content, pattern)
        new_functions = self._extract_functions_regex(new_content, pattern)
        
        # Compare functions
        all_function_names = set(old_functions.keys()) | set(new_functions.keys())
        
        for name in all_function_names:
            if name in old_functions and name in new_functions:
                if old_functions[name] != new_functions[name]:
                    changes.append(FunctionChange(
                        name=name,
                        change_type=ChangeType.MODIFIED,
                        old_content=old_functions[name],
                        new_content=new_functions[name]
                    ))
            elif name in new_functions:
                changes.append(FunctionChange(
                    name=name,
                    change_type=ChangeType.ADDED,
                    new_content=new_functions[name]
                ))
            elif name in old_functions:
                changes.append(FunctionChange(
                    name=name,
                    change_type=ChangeType.DELETED,
                    old_content=old_functions[name]
                ))
        
        return changes

    def _extract_functions_regex(self, content: str, pattern: str) -> Dict[str, str]:
        """Extract functions using regex patterns"""
        functions = {}
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            match = re.search(pattern, line)
            if match:
                # Get the first non-None group (function name)
                func_name = next((group for group in match.groups() if group), None)
                if func_name:
                    # Extract function content (simple heuristic)
                    func_content = self._extract_function_content(lines, i)
                    functions[func_name] = func_content
        
        return functions

    def _extract_function_content(self, lines: List[str], start_line: int) -> str:
        """Extract function content using indentation heuristics"""
        content_lines = [lines[start_line]]
        
        # For Python, use indentation
        if 'def ' in lines[start_line]:
            base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
            
            for i in range(start_line + 1, min(start_line + 50, len(lines))):
                line = lines[i]
                if line.strip() == '':
                    content_lines.append(line)
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and line.strip():
                    break
                
                content_lines.append(line)
        else:
            # For other languages, use braces or limited context
            brace_count = 0
            found_opening = False
            
            for i in range(start_line, min(start_line + 50, len(lines))):
                line = lines[i]
                if i > start_line:
                    content_lines.append(line)
                
                brace_count += line.count('{') - line.count('}')
                if '{' in line:
                    found_opening = True
                
                if found_opening and brace_count == 0:
                    break
        
        return '\n'.join(content_lines)

class HybridDiffAnalyzer:
    """Hybrid analyzer that combines AST and LLM approaches"""
    
    def __init__(self):
        self.git_parser = GitDiffParser()
        self.ast_analyzer = EnhancedASTAnalyzer()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    def analyze_diff(self, diff_content: str, file_path: str) -> Dict[str, Any]:
        """Analyze diff using hybrid approach"""
        # Parse git diff
        file_changes = self.git_parser.parse_git_diff(diff_content)
        
        if file_path not in file_changes:
            return {"error": f"No changes found for {file_path}"}
        
        change_info = file_changes[file_path]
        old_content = change_info['old_content']
        new_content = change_info['new_content']
        
        # Try AST analysis first
        ast_changes = self.ast_analyzer.analyze_function_changes(file_path, old_content, new_content)
        
        # Determine if we should use LLM fallback
        should_use_llm = self._should_use_llm_fallback(ast_changes, change_info)
        
        if should_use_llm:
            llm_analysis = self._get_llm_analysis(old_content, new_content, file_path)
            return {
                "method": "hybrid",
                "ast_changes": [self._serialize_change(change) for change in ast_changes],
                "llm_analysis": llm_analysis,
                "recommendation": "LLM analysis recommended due to complex changes"
            }
        else:
            return {
                "method": "ast",
                "changes": [self._serialize_change(change) for change in ast_changes],
                "recommendation": "AST analysis sufficient for these changes"
            }

    def _should_use_llm_fallback(self, ast_changes: List[FunctionChange], change_info: Dict) -> bool:
        """Determine if LLM fallback is needed"""
        # Use LLM if:
        # 1. No functions detected by AST
        # 2. Large number of line changes
        # 3. Complex modifications detected
        
        if not ast_changes:
            return True
        
        line_changes = len(change_info.get('line_changes', []))
        if line_changes > 50:  # Large changes
            return True
        
        # Check for complex modifications
        modified_functions = [c for c in ast_changes if c.change_type == ChangeType.MODIFIED]
        if len(modified_functions) > 5:  # Many functions modified
            return True
        
        return False

    def _get_llm_analysis(self, old_content: str, new_content: str, file_path: str) -> str:
        """Get LLM analysis for complex changes"""
        prompt = f"""
Analyze the changes in this file: {file_path}

OLD VERSION:
```
{old_content[:2000]}  # Truncate for token limits
```

NEW VERSION:
```
{new_content[:2000]}  # Truncate for token limits
```

Provide a concise summary of:
1. What functions/methods were changed
2. The nature of the changes (logic, parameters, return values)
3. Potential impact on dependent code
4. Risk level (low/medium/high)

Keep the response under 300 words.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a code analysis expert. Provide concise, actionable insights about code changes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM analysis failed: {str(e)}"

    def _serialize_change(self, change: FunctionChange) -> Dict[str, Any]:
        """Serialize FunctionChange for JSON response"""
        return {
            "name": change.name,
            "change_type": change.change_type.value,
            "old_content": change.old_content,
            "new_content": change.new_content
        }

# Main function for backward compatibility
def generate_functional_diff(base_code: str, updated_code: str, function_name: str) -> str:
    """Generate functional diff summary for a function (backward compatible)"""
    analyzer = HybridDiffAnalyzer()
    
    # Create a mock diff for the analyzer
    mock_diff = f"""diff --git a/temp.py b/temp.py
index 1234567..abcdefg 100644
--- a/temp.py
+++ b/temp.py
@@ -1,{len(base_code.split())} +1,{len(updated_code.split())} @@
-{base_code}
+{updated_code}
"""
    
    try:
        result = analyzer.analyze_diff(mock_diff, "temp.py")
        
        if result.get("method") == "hybrid":
            return result.get("llm_analysis", "No analysis available")
        else:
            changes = result.get("changes", [])
            function_change = next((c for c in changes if c["name"] == function_name), None)
            
            if function_change:
                return f"Function '{function_name}' was {function_change['change_type']}"
            else:
                return f"No changes detected for function '{function_name}'"
                
    except Exception as e:
        return f"Analysis failed: {str(e)}"