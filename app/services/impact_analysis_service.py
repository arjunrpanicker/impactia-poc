"""
Impact Analysis Service - Focused on method-level change impact and dependency tracking
"""
import os
import re
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .azure_openai_service import AzureOpenAIService
from .enhanced_rag_service import EnhancedRAGService
from .diff_service import generate_functional_diff
from ..models.analysis import CodeChange
from ..utils.diff_utils import GitDiffExtractor, is_diff_format

@dataclass
class ChangedMethod:
    file_path: str
    method: str
    summary: str

@dataclass
class ImpactedMethod:
    file_path: str
    method: str
    impact_reason: str
    impact_description: str

@dataclass
class MethodImpactAnalysisResult:
    changed_methods: List[ChangedMethod]
    impacted_methods: List[ImpactedMethod]

class ImpactAnalysisService:
    """Service focused on method-level change impact analysis and dependency tracking"""
    
    def __init__(self):
        self.openai_service = AzureOpenAIService()
        self.rag_service = EnhancedRAGService()

    async def analyze_impact(self, changes: List[CodeChange]) -> MethodImpactAnalysisResult:
        """Analyze the impact of code changes at method level and track their ripple effects"""
        
        print(f"[DEBUG] Starting method-level impact analysis for {len(changes)} changes")
        
        # Step 1: Extract actual file changes from diffs and analyze changed methods
        changed_methods = []
        actual_file_changes = {}
        
        for change in changes:
            if change.diff and is_diff_format(change.diff):
                print(f"[DEBUG] Processing diff content from {change.file_path}")
                
                # Extract actual files from diff
                file_changes = GitDiffExtractor.extract_file_changes(change.diff)
                print(f"[DEBUG] Extracted {len(file_changes)} files from diff")
                
                for file_path, change_data in file_changes.items():
                    print(f"[DEBUG] Analyzing file: {file_path}")
                    actual_file_changes[file_path] = change_data
                    
                    # Analyze method-level changes in this file
                    methods = await self._extract_changed_methods(
                        file_path, 
                        change_data['old_content'], 
                        change_data['new_content']
                    )
                    changed_methods.extend(methods)
            
            elif change.content:
                # Handle regular file content
                print(f"[DEBUG] Processing regular file content: {change.file_path}")
                methods = await self._extract_methods_from_content(change.file_path, change.content)
                changed_methods.extend(methods)
                actual_file_changes[change.file_path] = {
                    'old_content': '',
                    'new_content': change.content
                }
        
        print(f"[DEBUG] Found {len(changed_methods)} changed methods across {len(actual_file_changes)} files")
        for method in changed_methods:
            print(f"[DEBUG] - {method.file_path}::{method.method} - {method.summary}")
        
        # Step 2: Find impacted methods through dependency analysis
        impacted_methods = await self._find_impacted_methods(changed_methods, actual_file_changes)
        
        print(f"[DEBUG] Found {len(impacted_methods)} impacted methods")
        for method in impacted_methods:
            print(f"[DEBUG] - {method.file_path}::{method.method} - {method.impact_reason}")
        
        return MethodImpactAnalysisResult(
            changed_methods=changed_methods,
            impacted_methods=impacted_methods
        )

    async def _extract_changed_methods(self, file_path: str, old_content: str, new_content: str) -> List[ChangedMethod]:
        """Extract methods that were changed between old and new content"""
        
        print(f"[DEBUG] Extracting changed methods from {file_path}")
        print(f"[DEBUG] Old content: {len(old_content)} chars, New content: {len(new_content)} chars")
        
        changed_methods = []
        
        # Extract methods from both versions
        old_methods = self._extract_methods_from_code(old_content, file_path)
        new_methods = self._extract_methods_from_code(new_content, file_path)
        
        print(f"[DEBUG] Old methods: {list(old_methods.keys())}")
        print(f"[DEBUG] New methods: {list(new_methods.keys())}")
        
        # Find all method names
        all_method_names = set(old_methods.keys()) | set(new_methods.keys())
        
        for method_name in all_method_names:
            old_method_content = old_methods.get(method_name, "")
            new_method_content = new_methods.get(method_name, "")
            
            # Determine change type and summary
            if method_name in old_methods and method_name in new_methods:
                # Method exists in both - check if modified
                if self._normalize_content(old_method_content) != self._normalize_content(new_method_content):
                    summary = await self._analyze_method_change(
                        method_name, old_method_content, new_method_content, file_path
                    )
                    changed_methods.append(ChangedMethod(
                        file_path=file_path,
                        method=method_name,
                        summary=f"Modified: {summary}"
                    ))
                    print(f"[DEBUG] Method modified: {method_name}")
            
            elif method_name in new_methods:
                # Method added
                summary = await self._analyze_new_method(method_name, new_method_content, file_path)
                changed_methods.append(ChangedMethod(
                    file_path=file_path,
                    method=method_name,
                    summary=f"Added: {summary}"
                ))
                print(f"[DEBUG] Method added: {method_name}")
            
            elif method_name in old_methods:
                # Method deleted
                changed_methods.append(ChangedMethod(
                    file_path=file_path,
                    method=method_name,
                    summary="Deleted: Method removed from codebase"
                ))
                print(f"[DEBUG] Method deleted: {method_name}")
        
        # If no methods detected but content changed, analyze at file level
        if not changed_methods and old_content != new_content:
            print(f"[DEBUG] No methods detected, analyzing file-level changes")
            file_summary = await self._analyze_file_level_changes(old_content, new_content, file_path)
            changed_methods.append(ChangedMethod(
                file_path=file_path,
                method="file_level_changes",
                summary=file_summary
            ))
        
        return changed_methods

    def _extract_methods_from_code(self, content: str, file_path: str) -> Dict[str, str]:
        """Extract methods from code content with their full implementation"""
        
        if not content or not content.strip():
            return {}
        
        methods = {}
        lines = content.split('\n')
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Language-specific patterns for method detection
        patterns = {
            '.py': r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:',
            '.js': r'^\s*(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)|([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*function\s*\([^)]*\)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)',
            '.ts': r'^\s*(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)|([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:\s*[^{]*{|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^=]*)?=>)',
            '.jsx': r'^\s*(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)',
            '.tsx': r'^\s*(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^=]*)?=>)',
            '.cs': r'^\s*(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?[a-zA-Z_<>[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
            '.java': r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?[a-zA-Z_<>[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
        }
        
        pattern = patterns.get(file_ext, r'^\s*(?:def|function|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        
        for i, line in enumerate(lines):
            match = re.search(pattern, line)
            if match:
                # Get the first non-None group (method name)
                method_name = next((group for group in match.groups() if group), None)
                if method_name:
                    # Extract method content
                    method_content = self._extract_method_content(lines, i, file_ext)
                    methods[method_name] = method_content
                    print(f"[DEBUG] Extracted method: {method_name} ({len(method_content)} chars)")
        
        return methods

    def _extract_method_content(self, lines: List[str], start_line: int, file_ext: str) -> str:
        """Extract the full content of a method"""
        
        content_lines = [lines[start_line]]
        
        if file_ext == '.py':
            # Python: use indentation
            if start_line >= len(lines):
                return content_lines[0]
                
            base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
            
            for i in range(start_line + 1, min(start_line + 50, len(lines))):
                if i >= len(lines):
                    break
                line = lines[i]
                
                if line.strip() == '':
                    content_lines.append(line)
                    continue
                
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= base_indent and line.strip():
                    break
                
                content_lines.append(line)
        else:
            # Other languages: use braces or limited context
            brace_count = 0
            found_opening = False
            
            for i in range(start_line, min(start_line + 50, len(lines))):
                if i >= len(lines):
                    break
                line = lines[i]
                if i > start_line:
                    content_lines.append(line)
                
                brace_count += line.count('{') - line.count('}')
                if '{' in line:
                    found_opening = True
                
                if found_opening and brace_count == 0:
                    break
        
        return '\n'.join(content_lines)

    async def _extract_methods_from_content(self, file_path: str, content: str) -> List[ChangedMethod]:
        """Extract methods from new file content"""
        
        methods = self._extract_methods_from_code(content, file_path)
        changed_methods = []
        
        for method_name, method_content in methods.items():
            summary = await self._analyze_new_method(method_name, method_content, file_path)
            changed_methods.append(ChangedMethod(
                file_path=file_path,
                method=method_name,
                summary=f"Added: {summary}"
            ))
        
        return changed_methods

    async def _analyze_method_change(self, method_name: str, old_content: str, new_content: str, file_path: str) -> str:
        """Analyze what changed in a specific method"""
        
        print(f"[DEBUG] Analyzing method change: {method_name}")
        
        # Try functional diff first
        try:
            functional_summary = generate_functional_diff(old_content, new_content, method_name)
            if functional_summary and "Analysis failed" not in functional_summary and len(functional_summary) > 10:
                return functional_summary
        except Exception as e:
            print(f"[DEBUG] Functional diff failed for {method_name}: {e}")
        
        # Fallback to pattern analysis
        return await self._analyze_method_patterns(old_content, new_content, method_name)

    async def _analyze_method_patterns(self, old_content: str, new_content: str, method_name: str) -> str:
        """Analyze method changes using pattern detection"""
        
        # Detect specific change patterns
        old_lines = set(line.strip() for line in old_content.split('\n') if line.strip())
        new_lines = set(line.strip() for line in new_content.split('\n') if line.strip())
        
        added_lines = new_lines - old_lines
        removed_lines = old_lines - new_lines
        
        patterns = []
        
        # Check for specific patterns in added lines
        added_content = ' '.join(added_lines).lower()
        removed_content = ' '.join(removed_lines).lower()
        
        if any(keyword in added_content for keyword in ['validate', 'check', 'verify']):
            patterns.append("added validation")
        if any(keyword in added_content for keyword in ['log', 'logger', 'debug', 'info']):
            patterns.append("added logging")
        if any(keyword in added_content for keyword in ['try:', 'except', 'catch', 'error']):
            patterns.append("added error handling")
        if any(keyword in added_content for keyword in ['return', 'yield']):
            patterns.append("modified return logic")
        if any(keyword in added_content for keyword in ['if', 'else', 'elif', 'switch']):
            patterns.append("added conditional logic")
        
        if any(keyword in removed_content for keyword in ['validate', 'check', 'verify']):
            patterns.append("removed validation")
        if any(keyword in removed_content for keyword in ['log', 'logger', 'debug']):
            patterns.append("removed logging")
        
        # Check for parameter changes
        old_params = re.findall(r'\([^)]*\)', old_content)
        new_params = re.findall(r'\([^)]*\)', new_content)
        if old_params != new_params:
            patterns.append("modified parameters")
        
        if patterns:
            return ', '.join(patterns)
        
        # Generic change description
        lines_added = len(added_lines)
        lines_removed = len(removed_lines)
        
        if lines_added > lines_removed:
            return f"expanded functionality ({lines_added - lines_removed} net lines added)"
        elif lines_removed > lines_added:
            return f"simplified logic ({lines_removed - lines_added} net lines removed)"
        else:
            return "modified implementation"

    async def _analyze_new_method(self, method_name: str, content: str, file_path: str) -> str:
        """Analyze what a new method does"""
        
        # Detect patterns in the method
        content_lower = content.lower()
        patterns = []
        
        if any(keyword in content_lower for keyword in ['validate', 'check', 'verify']):
            patterns.append("validation")
        if any(keyword in content_lower for keyword in ['calculate', 'compute', 'process']):
            patterns.append("computation")
        if any(keyword in content_lower for keyword in ['get', 'fetch', 'retrieve', 'find']):
            patterns.append("data retrieval")
        if any(keyword in content_lower for keyword in ['save', 'store', 'insert', 'update']):
            patterns.append("data persistence")
        if any(keyword in content_lower for keyword in ['send', 'post', 'request', 'api']):
            patterns.append("API communication")
        if any(keyword in content_lower for keyword in ['render', 'display', 'show']):
            patterns.append("UI rendering")
        if any(keyword in content_lower for keyword in ['auth', 'login', 'permission']):
            patterns.append("authentication")
        
        if patterns:
            return f"implements {', '.join(patterns)}"
        
        # Count lines to estimate complexity
        lines = [line for line in content.split('\n') if line.strip()]
        if len(lines) > 20:
            return "complex business logic implementation"
        elif len(lines) > 10:
            return "moderate functionality implementation"
        else:
            return "simple utility function"

    async def _analyze_file_level_changes(self, old_content: str, new_content: str, file_path: str) -> str:
        """Analyze changes when no specific methods are detected"""
        
        # Check for imports/dependencies changes
        old_imports = re.findall(r'(?:import|from|require|include)\s+[^\n]+', old_content)
        new_imports = re.findall(r'(?:import|from|require|include)\s+[^\n]+', new_content)
        
        if set(old_imports) != set(new_imports):
            return "modified imports and dependencies"
        
        # Check for configuration changes
        if any(keyword in file_path.lower() for keyword in ['config', 'setting', 'env']):
            return "configuration changes"
        
        # Check for test file changes
        if any(keyword in file_path.lower() for keyword in ['test', 'spec']):
            return "test case modifications"
        
        # Generic file changes
        old_lines = len([line for line in old_content.split('\n') if line.strip()])
        new_lines = len([line for line in new_content.split('\n') if line.strip()])
        
        if new_lines > old_lines:
            return f"file expanded with {new_lines - old_lines} additional lines"
        elif old_lines > new_lines:
            return f"file simplified by removing {old_lines - new_lines} lines"
        else:
            return "file content restructured"

    async def _find_impacted_methods(self, changed_methods: List[ChangedMethod], file_changes: Dict[str, Dict]) -> List[ImpactedMethod]:
        """Find methods impacted by the changed methods through dependency analysis"""
        
        print(f"[DEBUG] Finding impacted methods for {len(changed_methods)} changed methods")
        
        impacted_methods = []
        
        try:
            # Create CodeChange objects for RAG analysis
            rag_changes = []
            for file_path, change_data in file_changes.items():
                rag_changes.append(CodeChange(
                    file_path=file_path,
                    content=change_data['new_content'],
                    diff=None,
                    change_type="modified"
                ))
            
            if not rag_changes:
                print(f"[DEBUG] No RAG changes to analyze")
                return impacted_methods
            
            # Use RAG service to find related code
            related_code = await self.rag_service.get_related_code(rag_changes)
            
            print(f"[DEBUG] RAG analysis completed")
            
            # Process similar methods from vector search
            similar_methods = related_code.get("similar_code", {}).get("methods", [])
            print(f"[DEBUG] Found {len(similar_methods)} similar methods")
            
            for similar_method in similar_methods:
                method_name = similar_method.get("name", "")
                method_file = similar_method.get("file_path", "")
                similarity = similar_method.get("similarity", 0)
                
                if method_name and method_file and similarity > 0.7:
                    # Check if this method calls any of our changed methods
                    impact_reason, impact_description = await self._analyze_method_dependency(
                        method_name, method_file, changed_methods, similar_method
                    )
                    
                    if impact_reason:
                        impacted_methods.append(ImpactedMethod(
                            file_path=method_file,
                            method=method_name,
                            impact_reason=impact_reason,
                            impact_description=impact_description
                        ))
            
            # Process dependency chains
            dependency_chains = related_code.get("dependency_chains", [])
            print(f"[DEBUG] Found {len(dependency_chains)} dependency chains")
            
            for chain in dependency_chains:
                source_file = chain.get("file_path", "")
                dependent_files = chain.get("dependent_files", [])
                
                for dep_file in dependent_files:
                    dep_file_path = dep_file.get("file_path", "")
                    methods = dep_file.get("methods", [])
                    
                    for method in methods:
                        method_name = method.get("name", "")
                        method_summary = method.get("summary", "")
                        
                        if method_name and dep_file_path:
                            # Find which changed method this depends on
                            related_changed_method = self._find_related_changed_method(
                                source_file, changed_methods
                            )
                            
                            if related_changed_method:
                                impact_reason = f"calls {related_changed_method.method}()"
                                impact_description = await self._generate_impact_description(
                                    method_name, related_changed_method, method_summary
                                )
                                
                                impacted_methods.append(ImpactedMethod(
                                    file_path=dep_file_path,
                                    method=method_name,
                                    impact_reason=impact_reason,
                                    impact_description=impact_description
                                ))
            
        except Exception as e:
            print(f"[DEBUG] Error in dependency analysis: {e}")
            import traceback
            traceback.print_exc()
        
        # Remove duplicates
        unique_impacted = []
        seen = set()
        
        for imp_method in impacted_methods:
            key = f"{imp_method.file_path}::{imp_method.method}"
            if key not in seen:
                unique_impacted.append(imp_method)
                seen.add(key)
        
        print(f"[DEBUG] Found {len(unique_impacted)} unique impacted methods")
        return unique_impacted

    async def _analyze_method_dependency(self, method_name: str, method_file: str, 
                                       changed_methods: List[ChangedMethod], 
                                       similar_method: Dict) -> Tuple[str, str]:
        """Analyze how a method depends on changed methods"""
        
        method_content = similar_method.get("content", "").lower()
        
        # Check if this method calls any of the changed methods
        for changed_method in changed_methods:
            changed_method_name = changed_method.method.lower()
            changed_file = changed_method.file_path
            
            # Look for method calls
            if changed_method_name in method_content:
                # Check for actual method call patterns
                call_patterns = [
                    f"{changed_method_name}(",
                    f".{changed_method_name}(",
                    f" {changed_method_name} ",
                ]
                
                if any(pattern in method_content for pattern in call_patterns):
                    impact_reason = f"calls {changed_method.method}()"
                    impact_description = await self._generate_impact_description(
                        method_name, changed_method, ""
                    )
                    return impact_reason, impact_description
        
        # Check for file-level dependencies
        for changed_method in changed_methods:
            changed_file = changed_method.file_path
            file_name = os.path.basename(changed_file).lower()
            
            if file_name.replace('.py', '').replace('.js', '').replace('.ts', '') in method_content:
                impact_reason = f"imports from {os.path.basename(changed_file)}"
                impact_description = f"May be affected by changes in imported {changed_method.method} method"
                return impact_reason, impact_description
        
        return "", ""

    def _find_related_changed_method(self, source_file: str, changed_methods: List[ChangedMethod]) -> Optional[ChangedMethod]:
        """Find which changed method is related to the source file"""
        
        for changed_method in changed_methods:
            if changed_method.file_path == source_file:
                return changed_method
        
        return None

    async def _generate_impact_description(self, impacted_method: str, changed_method: ChangedMethod, method_summary: str) -> str:
        """Generate a description of how the impacted method might be affected"""
        
        change_summary = changed_method.summary.lower()
        
        # Generate specific impact descriptions based on change type
        if "added validation" in change_summary:
            return f"May now receive validation errors when calling {changed_method.method}"
        elif "added logging" in change_summary:
            return f"Will generate additional log entries when {changed_method.method} is called"
        elif "added error handling" in change_summary:
            return f"May receive different error responses from {changed_method.method}"
        elif "modified parameters" in change_summary:
            return f"May need parameter updates to match new {changed_method.method} signature"
        elif "modified return logic" in change_summary:
            return f"May receive different return values from {changed_method.method}"
        elif "deleted" in change_summary:
            return f"Will fail when trying to call removed {changed_method.method} method"
        elif "added" in change_summary:
            return f"Can now utilize new {changed_method.method} functionality"
        else:
            return f"Behavior may change due to modifications in {changed_method.method}"

    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison"""
        # Remove extra whitespace and normalize line endings
        return re.sub(r'\s+', ' ', content).strip()