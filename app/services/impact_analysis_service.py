"""
Impact Analysis Service - Focused on change impact and dependency tracking
"""
import os
import re
import json
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .azure_openai_service import AzureOpenAIService
from .enhanced_rag_service import EnhancedRAGService
from .diff_service import generate_functional_diff
from ..models.analysis import CodeChange
from ..utils.diff_utils import GitDiffExtractor, is_diff_format

@dataclass
class ChangedFile:
    file_path: str
    functional_summary: str

@dataclass
class ImpactedFile:
    file_path: str
    impact_reason: str
    impact_description: str

@dataclass
class ImpactAnalysisResult:
    summary: str
    changed_files: List[ChangedFile]
    impacted_files: List[ImpactedFile]

class ImpactAnalysisService:
    """Service focused on change impact analysis and dependency tracking"""
    
    def __init__(self):
        self.openai_service = AzureOpenAIService()
        self.rag_service = EnhancedRAGService()

    async def analyze_impact(self, changes: List[CodeChange]) -> ImpactAnalysisResult:
        """Analyze the impact of code changes and track their ripple effects"""
        
        print(f"[DEBUG] Starting impact analysis for {len(changes)} changes")
        
        # Step 1: Analyze each changed file
        changed_files = []
        for change in changes:
            functional_summary = await self._analyze_changed_file(change)
            changed_files.append(ChangedFile(
                file_path=change.file_path,
                functional_summary=functional_summary
            ))
        
        print(f"[DEBUG] Analyzed {len(changed_files)} changed files")
        
        # Step 2: Find impacted files through dependency analysis
        impacted_files = await self._find_impacted_files(changes)
        
        print(f"[DEBUG] Found {len(impacted_files)} impacted files")
        
        # Step 3: Generate overall summary
        summary = await self._generate_overall_summary(changed_files, impacted_files)
        
        return ImpactAnalysisResult(
            summary=summary,
            changed_files=changed_files,
            impacted_files=impacted_files
        )

    async def _analyze_changed_file(self, change: CodeChange) -> str:
        """Analyze a single changed file to understand what changed functionally"""
        
        print(f"[DEBUG] Analyzing changed file: {change.file_path}")
        
        if change.diff:
            # Extract old and new content from diff
            file_changes = GitDiffExtractor.extract_file_changes(change.diff)
            
            if change.file_path in file_changes:
                file_change = file_changes[change.file_path]
                old_content = file_change['old_content']
                new_content = file_change['new_content']
                
                # Use functional diff analysis
                functional_summary = generate_functional_diff(old_content, new_content, "file_analysis")
                
                if functional_summary and "Analysis failed" not in functional_summary:
                    return functional_summary
            
            # Fallback: analyze diff directly
            return await self._analyze_diff_content(change.diff, change.file_path)
        
        elif change.content:
            # New file or full content provided
            if change.change_type == "added":
                return await self._analyze_new_file_content(change.content, change.file_path)
            else:
                return await self._analyze_file_content(change.content, change.file_path)
        
        return f"Changes detected in {change.file_path} but unable to determine specific functional impact"

    async def _analyze_diff_content(self, diff_content: str, file_path: str) -> str:
        """Analyze diff content to understand functional changes"""
        
        # Extract key changes from diff
        lines = diff_content.split('\n')
        added_lines = []
        removed_lines = []
        modified_functions = []
        
        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                clean_line = line[1:].strip()
                if clean_line and not clean_line.startswith(('//','#','/*')):
                    added_lines.append(clean_line)
                    
                    # Check for function definitions
                    func_match = re.search(r'(def|function|class|method)\s+([a-zA-Z_][a-zA-Z0-9_]*)', clean_line)
                    if func_match:
                        modified_functions.append(f"Added {func_match.group(1)} '{func_match.group(2)}'")
            
            elif line.startswith('-') and not line.startswith('---'):
                clean_line = line[1:].strip()
                if clean_line and not clean_line.startswith(('//','#','/*')):
                    removed_lines.append(clean_line)
                    
                    # Check for function definitions
                    func_match = re.search(r'(def|function|class|method)\s+([a-zA-Z_][a-zA-Z0-9_]*)', clean_line)
                    if func_match:
                        modified_functions.append(f"Removed {func_match.group(1)} '{func_match.group(2)}'")
        
        # Generate summary
        summary_parts = []
        
        if modified_functions:
            summary_parts.append(f"Function changes: {'; '.join(modified_functions[:3])}")
        
        # Detect patterns in changes
        change_patterns = self._detect_change_patterns(added_lines + removed_lines)
        if change_patterns:
            summary_parts.append(f"Key changes: {', '.join(change_patterns)}")
        
        if not summary_parts:
            line_changes = len(added_lines) + len(removed_lines)
            summary_parts.append(f"Modified {line_changes} lines of code")
        
        return '. '.join(summary_parts)

    async def _analyze_new_file_content(self, content: str, file_path: str) -> str:
        """Analyze new file content"""
        
        # Extract functions and key patterns
        functions = re.findall(r'(?:def|function|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        patterns = self._detect_change_patterns(content.split('\n'))
        
        summary_parts = []
        
        if functions:
            summary_parts.append(f"New file with {len(functions)} functions/classes: {', '.join(functions[:3])}")
        
        if patterns:
            summary_parts.append(f"Implements: {', '.join(patterns)}")
        
        if not summary_parts:
            summary_parts.append(f"New file added: {os.path.basename(file_path)}")
        
        return '. '.join(summary_parts)

    async def _analyze_file_content(self, content: str, file_path: str) -> str:
        """Analyze general file content for changes"""
        
        # Use LLM for content analysis if it's complex
        if len(content) > 500:
            return await self._llm_analyze_content(content, file_path)
        
        # Simple pattern analysis for smaller files
        patterns = self._detect_change_patterns(content.split('\n'))
        functions = re.findall(r'(?:def|function|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        
        summary_parts = []
        
        if functions:
            summary_parts.append(f"Contains {len(functions)} functions/classes")
        
        if patterns:
            summary_parts.append(f"Implements: {', '.join(patterns)}")
        
        return '. '.join(summary_parts) if summary_parts else f"File content updated: {os.path.basename(file_path)}"

    def _detect_change_patterns(self, lines: List[str]) -> List[str]:
        """Detect common patterns in code changes"""
        
        patterns = []
        content = '\n'.join(lines).lower()
        
        pattern_map = {
            'validation': ['validate', 'check', 'verify', 'assert'],
            'authentication': ['auth', 'login', 'password', 'token'],
            'database operations': ['query', 'insert', 'update', 'delete', 'select'],
            'API endpoints': ['route', 'endpoint', 'api', '@app.', '@router.'],
            'error handling': ['try:', 'except', 'catch', 'error', 'exception'],
            'logging': ['log', 'logger', 'debug', 'info', 'warn', 'error'],
            'configuration': ['config', 'setting', 'env', 'constant'],
            'business logic': ['calculate', 'process', 'transform', 'generate']
        }
        
        for pattern_name, keywords in pattern_map.items():
            if any(keyword in content for keyword in keywords):
                patterns.append(pattern_name)
        
        return patterns[:3]  # Limit to top 3 patterns

    async def _llm_analyze_content(self, content: str, file_path: str) -> str:
        """Use LLM to analyze complex content changes"""
        
        prompt = f"""
Analyze this code file and provide a brief functional summary of what it does:

File: {file_path}
Content (first 1000 chars):
{content[:1000]}

Provide a concise summary in 1-2 sentences focusing on:
1. What functionality this code implements
2. Key operations or business logic
3. Any notable patterns (API, database, validation, etc.)

Keep it under 100 words and focus on functional impact.
"""
        
        try:
            response = self.openai_service.client.chat.completions.create(
                model=self.openai_service.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a code analyst. Provide concise functional summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[DEBUG] LLM content analysis failed: {e}")
            return f"Code changes in {os.path.basename(file_path)}"

    async def _find_impacted_files(self, changes: List[CodeChange]) -> List[ImpactedFile]:
        """Find files impacted by the changes through dependency analysis"""
        
        print(f"[DEBUG] Finding impacted files for {len(changes)} changes")
        
        impacted_files = []
        
        try:
            # Use RAG service to find related code
            related_code = await self.rag_service.get_related_code(changes)
            
            print(f"[DEBUG] RAG analysis completed")
            
            # Process dependency chains
            dependency_chains = related_code.get("dependency_chains", [])
            for chain in dependency_chains:
                source_file = chain.get("file_path", "")
                dependent_files = chain.get("dependent_files", [])
                
                for dep_file in dependent_files:
                    dep_file_path = dep_file.get("file_path", "")
                    methods = dep_file.get("methods", [])
                    
                    if dep_file_path and dep_file_path != source_file:
                        # Determine impact reason and description
                        impact_reason, impact_description = self._analyze_dependency_impact(
                            source_file, dep_file_path, methods
                        )
                        
                        impacted_files.append(ImpactedFile(
                            file_path=dep_file_path,
                            impact_reason=impact_reason,
                            impact_description=impact_description
                        ))
            
            # Process direct dependencies
            direct_deps = related_code.get("direct_dependencies", {})
            incoming_refs = direct_deps.get("incoming", [])
            
            for ref_file in incoming_refs:
                if not any(imp.file_path == ref_file for imp in impacted_files):
                    impacted_files.append(ImpactedFile(
                        file_path=ref_file,
                        impact_reason="Direct dependency reference",
                        impact_description=f"File references changed code and may be affected by modifications"
                    ))
            
            # Process similar files (potential indirect impact)
            similar_files = related_code.get("similar_code", {}).get("files", [])
            for similar_file in similar_files[:5]:  # Limit to top 5
                file_path = similar_file.get("path", "")
                similarity = similar_file.get("similarity", 0)
                
                if file_path and similarity > 0.8:  # High similarity threshold
                    if not any(imp.file_path == file_path for imp in impacted_files):
                        impacted_files.append(ImpactedFile(
                            file_path=file_path,
                            impact_reason="Similar code patterns",
                            impact_description=f"Contains similar code patterns (similarity: {similarity:.2f}) and may need consistent updates"
                        ))
            
        except Exception as e:
            print(f"[DEBUG] Error in dependency analysis: {e}")
            # Continue with empty impacted files list
        
        # Remove duplicates
        unique_impacted = []
        seen_paths = set()
        
        for imp_file in impacted_files:
            if imp_file.file_path not in seen_paths:
                unique_impacted.append(imp_file)
                seen_paths.add(imp_file.file_path)
        
        print(f"[DEBUG] Found {len(unique_impacted)} unique impacted files")
        return unique_impacted

    def _analyze_dependency_impact(self, source_file: str, dependent_file: str, methods: List[Dict]) -> tuple[str, str]:
        """Analyze the specific impact relationship between files"""
        
        if not methods:
            return "File dependency", f"Depends on changes in {os.path.basename(source_file)}"
        
        # Analyze method relationships
        method_summaries = []
        impact_types = []
        
        for method in methods:
            method_name = method.get("name", "")
            method_summary = method.get("summary", "")
            
            if method_name:
                method_summaries.append(method_name)
                
                # Determine impact type from summary
                if any(keyword in method_summary.lower() for keyword in ["calls", "invokes", "uses"]):
                    impact_types.append("method_call")
                elif any(keyword in method_summary.lower() for keyword in ["inherits", "extends"]):
                    impact_types.append("inheritance")
                elif any(keyword in method_summary.lower() for keyword in ["imports", "references"]):
                    impact_types.append("import_reference")
        
        # Generate impact reason
        if "method_call" in impact_types:
            impact_reason = f"Calls methods: {', '.join(method_summaries[:3])}"
        elif "inheritance" in impact_types:
            impact_reason = f"Inherits from modified classes"
        elif "import_reference" in impact_types:
            impact_reason = f"Imports/references modified code"
        else:
            impact_reason = f"Uses methods: {', '.join(method_summaries[:3])}"
        
        # Generate impact description
        if len(method_summaries) == 1:
            impact_description = f"Behavior may change due to modifications in the '{method_summaries[0]}' method"
        elif len(method_summaries) > 1:
            impact_description = f"Multiple dependencies on changed methods may affect functionality"
        else:
            impact_description = f"Dependent on changes in {os.path.basename(source_file)} and may require updates"
        
        return impact_reason, impact_description

    async def _generate_overall_summary(self, changed_files: List[ChangedFile], impacted_files: List[ImpactedFile]) -> str:
        """Generate an overall summary of the impact analysis"""
        
        # Extract key information
        total_changed = len(changed_files)
        total_impacted = len(impacted_files)
        
        # Analyze change types
        change_summaries = [cf.functional_summary for cf in changed_files]
        combined_changes = ' '.join(change_summaries)
        
        # Detect major change categories
        categories = []
        if any(keyword in combined_changes.lower() for keyword in ['function', 'method', 'class']):
            categories.append("code structure")
        if any(keyword in combined_changes.lower() for keyword in ['api', 'endpoint', 'route']):
            categories.append("API endpoints")
        if any(keyword in combined_changes.lower() for keyword in ['database', 'query', 'insert', 'update']):
            categories.append("database operations")
        if any(keyword in combined_changes.lower() for keyword in ['validation', 'check', 'verify']):
            categories.append("validation logic")
        if any(keyword in combined_changes.lower() for keyword in ['auth', 'login', 'permission']):
            categories.append("authentication")
        
        # Generate summary
        summary_parts = []
        
        if total_changed == 1:
            summary_parts.append(f"Modified {changed_files[0].file_path}")
        else:
            summary_parts.append(f"Modified {total_changed} files")
        
        if categories:
            summary_parts.append(f"affecting {', '.join(categories)}")
        
        if total_impacted > 0:
            summary_parts.append(f"with {total_impacted} dependent files potentially impacted")
        
        # Add key change highlights
        key_changes = []
        for cf in changed_files:
            if len(cf.functional_summary) < 100:  # Include short, clear summaries
                key_changes.append(cf.functional_summary)
        
        if key_changes:
            summary_parts.append(f"Key changes: {'; '.join(key_changes[:2])}")
        
        return '. '.join(summary_parts) + '.'