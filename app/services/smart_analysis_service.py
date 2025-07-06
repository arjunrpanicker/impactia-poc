"""
Smart analysis service that decides between AST and LLM analysis
"""
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .diff_service import HybridDiffAnalyzer, FunctionChange, ChangeType
from .azure_openai_service import AzureOpenAIService
from ..utils.diff_utils import GitDiffExtractor

class AnalysisMethod(Enum):
    AST_ONLY = "ast_only"
    LLM_ONLY = "llm_only"
    HYBRID = "hybrid"
    AUTO = "auto"

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
        self.hybrid_analyzer = HybridDiffAnalyzer()
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
        
        # Determine the best analysis method
        if method == AnalysisMethod.AUTO:
            method = self._determine_best_method(file_path, old_content, new_content)
        
        # Perform analysis based on chosen method
        if method == AnalysisMethod.AST_ONLY:
            return await self._ast_analysis(file_path, old_content, new_content)
        elif method == AnalysisMethod.LLM_ONLY:
            return await self._llm_analysis(file_path, old_content, new_content)
        else:  # HYBRID
            return await self._hybrid_analysis(file_path, old_content, new_content)

    def _determine_best_method(self, file_path: str, old_content: str, new_content: str) -> AnalysisMethod:
        """Determine the best analysis method based on content characteristics"""
        
        # File type considerations
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Languages with good AST support
        ast_supported = {'.py', '.js', '.ts', '.jsx', '.tsx'}
        
        if file_ext not in ast_supported:
            return AnalysisMethod.LLM_ONLY
        
        # Content size considerations
        old_lines = len(old_content.split('\n'))
        new_lines = len(new_content.split('\n'))
        total_lines = max(old_lines, new_lines)
        
        if total_lines > self.config['max_lines_for_ast']:
            return AnalysisMethod.LLM_ONLY
        
        # Complexity considerations
        change_complexity = self._assess_change_complexity(old_content, new_content)
        
        if change_complexity > self.config['complex_change_threshold']:
            return AnalysisMethod.HYBRID
        
        # Default to AST for supported languages with simple changes
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
            changes = self.hybrid_analyzer.ast_analyzer.analyze_function_changes(
                file_path, old_content, new_content
            )
            
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
            # Fallback to LLM if AST fails
            return await self._llm_analysis(file_path, old_content, new_content)

    async def _llm_analysis(self, file_path: str, old_content: str, new_content: str) -> AnalysisResult:
        """Perform LLM-based analysis"""
        try:
            llm_result = self.hybrid_analyzer._get_llm_analysis(old_content, new_content, file_path)
            
            # Parse LLM result to extract structured information
            function_changes = self._parse_llm_result(llm_result)
            
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
            # Return minimal result if LLM fails
            return AnalysisResult(
                method_used=AnalysisMethod.LLM_ONLY,
                function_changes=[],
                summary=f"Analysis failed: {str(e)}",
                confidence_score=0.0,
                recommendations=["Manual review recommended"],
                performance_impact="unknown",
                risk_level="medium"
            )

    async def _hybrid_analysis(self, file_path: str, old_content: str, new_content: str) -> AnalysisResult:
        """Perform hybrid AST + LLM analysis"""
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
            summary_parts.append(f"{added} function(s) added")
        if modified:
            summary_parts.append(f"{modified} function(s) modified")
        if deleted:
            summary_parts.append(f"{deleted} function(s) deleted")
        
        return "; ".join(summary_parts)

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

    def _parse_llm_result(self, llm_result: str) -> List[FunctionChange]:
        """Parse LLM result to extract function changes"""
        # This is a simplified parser - could be enhanced with more sophisticated NLP
        changes = []
        
        # Look for function mentions in the LLM result
        import re
        function_patterns = [
            r'function\s+(\w+)',
            r'method\s+(\w+)',
            r'(\w+)\s+function',
            r'def\s+(\w+)'
        ]
        
        for pattern in function_patterns:
            matches = re.findall(pattern, llm_result, re.IGNORECASE)
            for match in matches:
                changes.append(FunctionChange(
                    name=match,
                    change_type=ChangeType.MODIFIED,  # Default assumption
                    new_content="Content extracted from LLM analysis"
                ))
        
        return changes

    def _extract_recommendations_from_llm(self, llm_result: str) -> List[str]:
        """Extract recommendations from LLM result"""
        # Simple extraction - could be enhanced
        recommendations = []
        
        if "test" in llm_result.lower():
            recommendations.append("Update test cases to reflect changes")
        
        if "breaking" in llm_result.lower():
            recommendations.append("Review for potential breaking changes")
        
        if "performance" in llm_result.lower():
            recommendations.append("Consider performance implications")
        
        return recommendations

    def _extract_performance_impact_from_llm(self, llm_result: str) -> str:
        """Extract performance impact from LLM result"""
        result_lower = llm_result.lower()
        
        if "high" in result_lower and "performance" in result_lower:
            return "high"
        elif "performance" in result_lower:
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
        else:
            return "low"

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