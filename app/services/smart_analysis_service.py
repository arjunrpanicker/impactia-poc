"""
Smart analysis service that decides between AST and LLM analysis
"""
import os
import re
import ast
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .azure_openai_service import AzureOpenAIService

class AnalysisMethod(Enum):
    AST_ONLY = "ast_only"
    LLM_ONLY = "llm_only"
    HYBRID = "hybrid"
    AUTO = "auto"

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

@dataclass
class AnalysisResult:
    method_used: AnalysisMethod
    function_changes: List[FunctionChange]
    summary: str
    confidence_score: float
    recommendations: List[str]
    performance_impact: str
    risk_level: str

class SmartAnalysisService:
    """Service that intelligently chooses between AST and LLM analysis"""
    
    def __init__(self):
        self.openai_service = AzureOpenAIService()
        
        # Configuration thresholds
        self.config = {
            'max_lines_for_ast': 200,
            'max_functions_for_ast': 10,
            'min_confidence_threshold': 0.7,
            'complex_change_threshold': 50
        }

    async def analyze_code_changes(
        self, 
        file_path: str, 
        old_content: str, 
        new_content: str,
        method: AnalysisMethod = AnalysisMethod.AUTO
    ) -> AnalysisResult:
        """Analyze code changes using the most appropriate method"""
        
        print(f"[DEBUG] Analyzing {file_path} with method {method}")
        print(f"[DEBUG] Old content length: {len(old_content)}")
        print(f"[DEBUG] New content length: {len(new_content)}")
        
        # Handle empty content cases
        if not old_content and not new_content:
            print(f"[DEBUG] Both contents empty for {file_path}")
            return AnalysisResult(
                method_used=method,
                function_changes=[],
                summary="No content to analyze",
                confidence_score=0.0,
                recommendations=["No changes detected"],
                performance_impact="none",
                risk_level="low"
            )
        
        # If only new content (file addition)
        if not old_content and new_content:
            print(f"[DEBUG] New file detected: {file_path}")
            return await self._analyze_new_file(file_path, new_content, method)
        
        # If only old content (file deletion)
        if old_content and not new_content:
            print(f"[DEBUG] File deletion detected: {file_path}")
            return await self._analyze_deleted_file(file_path, old_content, method)
        
        # Determine the best analysis method
        if method == AnalysisMethod.AUTO:
            method = self._determine_best_method(file_path, old_content, new_content)
        
        print(f"[DEBUG] Using analysis method: {method}")
        
        # Perform analysis based on chosen method
        try:
            if method == AnalysisMethod.AST_ONLY:
                return await self._ast_analysis(file_path, old_content, new_content)
            elif method == AnalysisMethod.LLM_ONLY:
                return await self._llm_analysis(file_path, old_content, new_content)
            else:  # HYBRID
                return await self._hybrid_analysis(file_path, old_content, new_content)
        except Exception as e:
            print(f"[DEBUG] Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return fallback result
            return AnalysisResult(
                method_used=method,
                function_changes=[],
                summary=f"Analysis failed: {str(e)}",
                confidence_score=0.0,
                recommendations=["Manual review required due to analysis failure"],
                performance_impact="unknown",
                risk_level="medium"
            )

    async def _analyze_new_file(self, file_path: str, content: str, method: AnalysisMethod) -> AnalysisResult:
        """Analyze a newly added file"""
        print(f"[DEBUG] Analyzing new file: {file_path}")
        
        functions = self._extract_functions_from_content(content, file_path)
        print(f"[DEBUG] Found {len(functions)} functions in new file")
        
        function_changes = [
            FunctionChange(
                name=func_name,
                change_type=ChangeType.ADDED,
                new_content=func_content[:200] + "..." if len(func_content) > 200 else func_content
            )
            for func_name, func_content in functions.items()
        ]
        
        # If no functions found, analyze content type
        if not functions:
            # Check if it's a code file with other content
            if self._is_code_file(file_path):
                summary = f"New code file added: {file_path} (no functions detected)"
                recommendations = ["Review file structure", "Consider adding functions if needed"]
            else:
                summary = f"New file added: {file_path}"
                recommendations = ["Review file content and purpose"]
        else:
            summary = f"New file added with {len(functions)} function(s): {', '.join(functions.keys())}"
            recommendations = ["Add test coverage for new functions", "Review code for best practices"]
        
        return AnalysisResult(
            method_used=method,
            function_changes=function_changes,
            summary=summary,
            confidence_score=0.9,
            recommendations=recommendations,
            performance_impact="low" if len(functions) < 5 else "medium",
            risk_level="low" if len(functions) < 3 else "medium"
        )

    async def _analyze_deleted_file(self, file_path: str, content: str, method: AnalysisMethod) -> AnalysisResult:
        """Analyze a deleted file"""
        print(f"[DEBUG] Analyzing deleted file: {file_path}")
        
        functions = self._extract_functions_from_content(content, file_path)
        print(f"[DEBUG] Found {len(functions)} functions in deleted file")
        
        function_changes = [
            FunctionChange(
                name=func_name,
                change_type=ChangeType.DELETED,
                old_content=func_content[:200] + "..." if len(func_content) > 200 else func_content
            )
            for func_name, func_content in functions.items()
        ]
        
        summary = f"File deleted with {len(functions)} function(s): {', '.join(functions.keys())}" if functions else f"File deleted: {file_path}"
        
        return AnalysisResult(
            method_used=method,
            function_changes=function_changes,
            summary=summary,
            confidence_score=0.9,
            recommendations=["Verify no breaking changes", "Update dependent code", "Remove related tests"],
            performance_impact="low",
            risk_level="high" if len(functions) > 0 else "medium"
        )

    def _is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file"""
        code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.cs', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.rb', '.php'}
        return os.path.splitext(file_path)[1].lower() in code_extensions

    def _extract_functions_from_content(self, content: str, file_path: str) -> Dict[str, str]:
        """Extract functions from content using regex patterns"""
        print(f"[DEBUG] Extracting functions from {file_path}")
        
        functions = {}
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Language-specific patterns
        patterns = {
            '.py': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:',
            '.js': r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)',
            '.ts': r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^=]*)?=>)',
            '.jsx': r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)',
            '.tsx': r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)|const\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^=]*)?=>)',
            '.cs': r'(?:public|private|protected|internal)?\s*(?:static\s+)?(?:async\s+)?[a-zA-Z_<>[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
            '.java': r'(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?[a-zA-Z_<>[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)',
        }
        
        pattern = patterns.get(file_ext, r'(?:def|function|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)')
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            matches = re.finditer(pattern, line)
            for match in matches:
                # Get the first non-None group (function name)
                func_name = next((group for group in match.groups() if group), None)
                if func_name:
                    print(f"[DEBUG] Found function: {func_name} at line {i+1}")
                    # Extract function content (simple heuristic)
                    func_content = self._extract_function_content(lines, i, file_ext)
                    functions[func_name] = func_content
        
        print(f"[DEBUG] Extracted {len(functions)} functions: {list(functions.keys())}")
        return functions

    def _extract_function_content(self, lines: List[str], start_line: int, file_ext: str) -> str:
        """Extract function content using language-specific heuristics"""
        content_lines = [lines[start_line]]
        
        if file_ext == '.py':
            # Python: use indentation
            base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
            
            for i in range(start_line + 1, min(start_line + 30, len(lines))):
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
            
            for i in range(start_line, min(start_line + 30, len(lines))):
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

    def _determine_best_method(self, file_path: str, old_content: str, new_content: str) -> AnalysisMethod:
        """Determine the best analysis method based on content characteristics"""
        
        # File type considerations
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Languages with good AST support
        ast_supported = {'.py', '.js', '.ts', '.jsx', '.tsx'}
        
        if file_ext not in ast_supported:
            print(f"[DEBUG] File extension {file_ext} not supported by AST, using LLM")
            return AnalysisMethod.LLM_ONLY
        
        # Content size considerations
        old_lines = len(old_content.split('\n'))
        new_lines = len(new_content.split('\n'))
        total_lines = max(old_lines, new_lines)
        
        if total_lines > self.config['max_lines_for_ast']:
            print(f"[DEBUG] File too large ({total_lines} lines), using LLM")
            return AnalysisMethod.LLM_ONLY
        
        # Complexity considerations
        change_complexity = self._assess_change_complexity(old_content, new_content)
        
        if change_complexity > self.config['complex_change_threshold']:
            print(f"[DEBUG] Complex changes detected ({change_complexity}), using hybrid")
            return AnalysisMethod.HYBRID
        
        # Default to AST for supported languages with simple changes
        print(f"[DEBUG] Using AST for simple changes")
        return AnalysisMethod.AST_ONLY

    def _assess_change_complexity(self, old_content: str, new_content: str) -> int:
        """Assess the complexity of changes between two versions"""
        complexity_score = 0
        
        # Line-based complexity
        old_lines = set(old_content.split('\n'))
        new_lines = set(new_content.split('\n'))
        
        added_lines = new_lines - old_lines
        removed_lines = old_lines - new_lines
        
        complexity_score += len(added_lines) + len(removed_lines)
        
        # Structural complexity indicators
        structural_indicators = [
            'class ', 'def ', 'function ', 'import ', 'from ',
            'if ', 'for ', 'while ', 'try:', 'except:', 'async ',
            'await ', 'yield ', 'return ', 'raise ', 'with '
        ]
        
        for indicator in structural_indicators:
            old_count = old_content.count(indicator)
            new_count = new_content.count(indicator)
            complexity_score += abs(new_count - old_count) * 2
        
        return complexity_score

    async def _ast_analysis(self, file_path: str, old_content: str, new_content: str) -> AnalysisResult:
        """Perform AST-based analysis"""
        try:
            print(f"[DEBUG] Starting AST analysis for {file_path}")
            
            # Extract functions from both versions
            old_functions = self._extract_functions_from_content(old_content, file_path)
            new_functions = self._extract_functions_from_content(new_content, file_path)
            
            print(f"[DEBUG] Old functions: {list(old_functions.keys())}")
            print(f"[DEBUG] New functions: {list(new_functions.keys())}")
            
            changes = []
            
            # Find all function names
            all_function_names = set(old_functions.keys()) | set(new_functions.keys())
            
            for name in all_function_names:
                if name in old_functions and name in new_functions:
                    # Function exists in both - check if modified
                    if self._normalize_content(old_functions[name]) != self._normalize_content(new_functions[name]):
                        changes.append(FunctionChange(
                            name=name,
                            change_type=ChangeType.MODIFIED,
                            old_content=old_functions[name][:200] + "..." if len(old_functions[name]) > 200 else old_functions[name],
                            new_content=new_functions[name][:200] + "..." if len(new_functions[name]) > 200 else new_functions[name]
                        ))
                elif name in new_functions:
                    # Function added
                    changes.append(FunctionChange(
                        name=name,
                        change_type=ChangeType.ADDED,
                        new_content=new_functions[name][:200] + "..." if len(new_functions[name]) > 200 else new_functions[name]
                    ))
                elif name in old_functions:
                    # Function deleted
                    changes.append(FunctionChange(
                        name=name,
                        change_type=ChangeType.DELETED,
                        old_content=old_functions[name][:200] + "..." if len(old_functions[name]) > 200 else old_functions[name]
                    ))
            
            print(f"[DEBUG] AST found {len(changes)} changes")
            
            summary = self._generate_ast_summary(changes)
            confidence = self._calculate_ast_confidence(changes, old_content, new_content)
            recommendations = self._generate_ast_recommendations(changes)
            
            return AnalysisResult(
                method_used=AnalysisMethod.AST_ONLY,
                function_changes=changes,
                summary=summary,
                confidence_score=confidence,
                recommendations=recommendations,
                performance_impact=self._assess_performance_impact(changes),
                risk_level=self._assess_risk_level(changes)
            )
            
        except Exception as e:
            print(f"[DEBUG] AST analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to LLM if AST fails
            return await self._llm_analysis(file_path, old_content, new_content)

    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison"""
        # Remove extra whitespace and normalize line endings
        return re.sub(r'\s+', ' ', content).strip()

    async def _llm_analysis(self, file_path: str, old_content: str, new_content: str) -> AnalysisResult:
        """Perform LLM-based analysis"""
        try:
            print(f"[DEBUG] Starting LLM analysis for {file_path}")
            
            # Prepare prompt for LLM
            prompt = f"""
Analyze the changes in this file: {file_path}

OLD VERSION:
```
{old_content[:1500] if old_content else "No previous content (new file)"}
```

NEW VERSION:
```
{new_content[:1500] if new_content else "No content (file deleted)"}
```

Please provide:
1. A summary of what changed
2. List of functions/methods that were added, modified, or deleted
3. Risk level (low/medium/high)
4. Performance impact (low/medium/high)
5. Recommendations

Keep the response concise and structured.
"""
            
            # Get LLM analysis
            response = await self.openai_service.client.chat.completions.create(
                model=self.openai_service.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a code analysis expert. Provide concise, actionable insights about code changes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            llm_result = response.choices[0].message.content.strip()
            print(f"[DEBUG] LLM result: {llm_result[:200]}...")
            
            # Parse LLM result to extract structured information
            function_changes = self._parse_llm_result(llm_result, old_content, new_content, file_path)
            
            return AnalysisResult(
                method_used=AnalysisMethod.LLM_ONLY,
                function_changes=function_changes,
                summary=llm_result,
                confidence_score=0.8,  # LLM generally provides good insights
                recommendations=self._extract_recommendations_from_llm(llm_result),
                performance_impact=self._extract_performance_impact_from_llm(llm_result),
                risk_level=self._extract_risk_level_from_llm(llm_result)
            )
            
        except Exception as e:
            print(f"[DEBUG] LLM analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return minimal result if LLM fails
            return AnalysisResult(
                method_used=AnalysisMethod.LLM_ONLY,
                function_changes=[],
                summary=f"LLM analysis failed: {str(e)}. Manual review recommended.",
                confidence_score=0.0,
                recommendations=["Manual review recommended due to analysis failure"],
                performance_impact="unknown",
                risk_level="medium"
            )

    async def _hybrid_analysis(self, file_path: str, old_content: str, new_content: str) -> AnalysisResult:
        """Perform hybrid AST + LLM analysis"""
        print(f"[DEBUG] Starting hybrid analysis for {file_path}")
        
        # Get AST analysis
        ast_result = await self._ast_analysis(file_path, old_content, new_content)
        
        # Get LLM analysis for additional insights
        llm_result = await self._llm_analysis(file_path, old_content, new_content)
        
        # Combine results
        combined_summary = f"AST Analysis: {ast_result.summary}\n\nLLM Insights: {llm_result.summary}"
        combined_recommendations = list(set(ast_result.recommendations + llm_result.recommendations))
        
        # Use higher confidence score
        confidence = max(ast_result.confidence_score, llm_result.confidence_score)
        
        return AnalysisResult(
            method_used=AnalysisMethod.HYBRID,
            function_changes=ast_result.function_changes,  # Prefer AST for structured data
            summary=combined_summary,
            confidence_score=confidence,
            recommendations=combined_recommendations,
            performance_impact=ast_result.performance_impact or llm_result.performance_impact,
            risk_level=self._combine_risk_levels(ast_result.risk_level, llm_result.risk_level)
        )

    def _generate_ast_summary(self, changes: List[FunctionChange]) -> str:
        """Generate summary from AST changes"""
        if not changes:
            return "No function-level changes detected"
        
        added = len([c for c in changes if c.change_type == ChangeType.ADDED])
        modified = len([c for c in changes if c.change_type == ChangeType.MODIFIED])
        deleted = len([c for c in changes if c.change_type == ChangeType.DELETED])
        
        summary_parts = []
        if added:
            added_names = [c.name for c in changes if c.change_type == ChangeType.ADDED]
            summary_parts.append(f"{added} function(s) added: {', '.join(added_names)}")
        if modified:
            modified_names = [c.name for c in changes if c.change_type == ChangeType.MODIFIED]
            summary_parts.append(f"{modified} function(s) modified: {', '.join(modified_names)}")
        if deleted:
            deleted_names = [c.name for c in changes if c.change_type == ChangeType.DELETED]
            summary_parts.append(f"{deleted} function(s) deleted: {', '.join(deleted_names)}")
        
        return "; ".join(summary_parts) if summary_parts else "Functions analyzed but no significant changes detected"

    def _calculate_ast_confidence(self, changes: List[FunctionChange], old_content: str, new_content: str) -> float:
        """Calculate confidence score for AST analysis"""
        base_confidence = 0.9  # AST is generally reliable
        
        # Reduce confidence for complex changes
        total_lines = max(len(old_content.split('\n')), len(new_content.split('\n')))
        if total_lines > 100:
            base_confidence -= 0.1
        
        # Reduce confidence if many functions changed
        if len(changes) > 5:
            base_confidence -= 0.1
        
        return max(0.5, base_confidence)

    def _generate_ast_recommendations(self, changes: List[FunctionChange]) -> List[str]:
        """Generate recommendations based on AST changes"""
        recommendations = []
        
        if len(changes) > 5:
            recommendations.append("Large number of function changes detected - consider breaking into smaller commits")
        
        added_functions = [c for c in changes if c.change_type == ChangeType.ADDED]
        if len(added_functions) > 3:
            recommendations.append("Multiple new functions added - ensure adequate test coverage")
        
        deleted_functions = [c for c in changes if c.change_type == ChangeType.DELETED]
        if deleted_functions:
            recommendations.append("Functions deleted - verify no breaking changes for dependent code")
        
        modified_functions = [c for c in changes if c.change_type == ChangeType.MODIFIED]
        if modified_functions:
            recommendations.append("Functions modified - review changes and update tests")
        
        if not recommendations:
            recommendations.append("Review changes for correctness and test coverage")
        
        return recommendations

    def _assess_performance_impact(self, changes: List[FunctionChange]) -> str:
        """Assess performance impact from function changes"""
        if not changes:
            return "minimal"
        
        if len(changes) > 10:
            return "high"
        elif len(changes) > 5:
            return "medium"
        else:
            return "low"

    def _assess_risk_level(self, changes: List[FunctionChange]) -> str:
        """Assess risk level from function changes"""
        if not changes:
            return "low"
        
        deleted_count = len([c for c in changes if c.change_type == ChangeType.DELETED])
        if deleted_count > 0:
            return "high"
        
        if len(changes) > 8:
            return "high"
        elif len(changes) > 3:
            return "medium"
        else:
            return "low"

    def _parse_llm_result(self, llm_result: str, old_content: str, new_content: str, file_path: str) -> List[FunctionChange]:
        """Parse LLM result to extract function changes"""
        changes = []
        
        # Try to extract function names from the LLM result
        function_patterns = [
            r'function\s+`?(\w+)`?',
            r'method\s+`?(\w+)`?',
            r'`(\w+)`\s+function',
            r'def\s+`?(\w+)`?',
            r'`(\w+)\(\)`',
        ]
        
        mentioned_functions = set()
        for pattern in function_patterns:
            matches = re.findall(pattern, llm_result, re.IGNORECASE)
            mentioned_functions.update(matches)
        
        # If no functions mentioned in LLM result, extract from content
        if not mentioned_functions:
            old_functions = self._extract_functions_from_content(old_content, file_path)
            new_functions = self._extract_functions_from_content(new_content, file_path)
            mentioned_functions = set(old_functions.keys()) | set(new_functions.keys())
        
        # Create function changes based on mentioned functions
        for func_name in mentioned_functions:
            # Determine change type based on LLM result content
            if "added" in llm_result.lower() or "new" in llm_result.lower():
                change_type = ChangeType.ADDED
            elif "deleted" in llm_result.lower() or "removed" in llm_result.lower():
                change_type = ChangeType.DELETED
            else:
                change_type = ChangeType.MODIFIED
            
            changes.append(FunctionChange(
                name=func_name,
                change_type=change_type,
                new_content="Content analyzed by LLM"
            ))
        
        return changes

    def _extract_recommendations_from_llm(self, llm_result: str) -> List[str]:
        """Extract recommendations from LLM result"""
        recommendations = []
        
        result_lower = llm_result.lower()
        
        if "test" in result_lower:
            recommendations.append("Update test cases to reflect changes")
        
        if "breaking" in result_lower:
            recommendations.append("Review for potential breaking changes")
        
        if "performance" in result_lower:
            recommendations.append("Consider performance implications")
        
        if "security" in result_lower:
            recommendations.append("Review security implications")
        
        if "documentation" in result_lower or "comment" in result_lower:
            recommendations.append("Update documentation and comments")
        
        if not recommendations:
            recommendations.append("Review changes thoroughly before deployment")
        
        return recommendations

    def _extract_performance_impact_from_llm(self, llm_result: str) -> str:
        """Extract performance impact from LLM result"""
        result_lower = llm_result.lower()
        
        if "high" in result_lower and "performance" in result_lower:
            return "high"
        elif "performance" in result_lower or "optimization" in result_lower:
            return "medium"
        else:
            return "low"

    def _extract_risk_level_from_llm(self, llm_result: str) -> str:
        """Extract risk level from LLM result"""
        result_lower = llm_result.lower()
        
        if "high risk" in result_lower or "critical" in result_lower:
            return "high"
        elif "medium risk" in result_lower or "moderate" in result_lower:
            return "medium"
        elif "low risk" in result_lower:
            return "low"
        else:
            return "medium"  # Default

    def _combine_risk_levels(self, ast_risk: str, llm_risk: str) -> str:
        """Combine risk levels from AST and LLM analysis"""
        risk_hierarchy = {"low": 1, "medium": 2, "high": 3}
        
        ast_level = risk_hierarchy.get(ast_risk, 2)
        llm_level = risk_hierarchy.get(llm_risk, 2)
        
        # Take the higher risk level
        combined_level = max(ast_level, llm_level)
        
        for level, value in risk_hierarchy.items():
            if value == combined_level:
                return level
        
        return "medium"  # Default fallback