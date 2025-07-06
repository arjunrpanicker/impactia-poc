"""
Performance impact analysis service
"""
import re
import ast
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..services.code_parser import CodeParser, CodeElement

@dataclass
class PerformanceIssue:
    type: str
    severity: str
    description: str
    file_path: str
    line_number: int
    suggestion: str

class PerformanceAnalyzer:
    def __init__(self):
        self.parser = CodeParser()
        
        # Performance anti-patterns
        self.anti_patterns = {
            'python': {
                'nested_loops': r'for\s+\w+\s+in\s+.*:\s*\n\s*for\s+\w+\s+in\s+.*:',
                'string_concatenation': r'\w+\s*\+=\s*["\'].*["\']',
                'global_variables': r'global\s+\w+',
                'recursive_without_memoization': r'def\s+(\w+).*:\s*.*\1\(',
            },
            'javascript': {
                'nested_loops': r'for\s*\([^)]*\)\s*{[^}]*for\s*\([^)]*\)',
                'dom_queries_in_loop': r'for\s*\([^)]*\)\s*{[^}]*document\.',
                'synchronous_requests': r'XMLHttpRequest.*\.open\([^,]*,\s*[^,]*,\s*false',
            },
            'csharp': {
                'string_concatenation': r'\w+\s*\+=\s*["\'].*["\']',
                'linq_multiple_enumeration': r'\.Where\(.*\)\..*\.Where\(',
                'boxing_unboxing': r'object\s+\w+\s*=\s*\d+',
            }
        }

    def analyze_performance_impact(self, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance implications of code changes"""
        performance_issues = []
        complexity_changes = []
        memory_impact = []
        
        for change in changes:
            file_path = change.get('file_path', '')
            content = change.get('content', '')
            
            if not content:
                continue
                
            # Detect language
            language = self._detect_language(file_path)
            
            # Analyze performance issues
            issues = self._detect_performance_issues(content, file_path, language)
            performance_issues.extend(issues)
            
            # Analyze complexity changes
            complexity = self._analyze_complexity_changes(content, file_path)
            if complexity:
                complexity_changes.append(complexity)
                
            # Analyze memory impact
            memory = self._analyze_memory_impact(content, file_path, language)
            if memory:
                memory_impact.extend(memory)
        
        return {
            "performance_issues": performance_issues,
            "complexity_changes": complexity_changes,
            "memory_impact": memory_impact,
            "recommendations": self._generate_performance_recommendations(performance_issues),
            "overall_impact": self._calculate_overall_impact(performance_issues, complexity_changes)
        }

    def _detect_performance_issues(self, content: str, file_path: str, language: str) -> List[PerformanceIssue]:
        """Detect performance anti-patterns in code"""
        issues = []
        lines = content.split('\n')
        
        patterns = self.anti_patterns.get(language, {})
        
        for pattern_name, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                
                issue = PerformanceIssue(
                    type=pattern_name,
                    severity=self._get_severity(pattern_name),
                    description=self._get_issue_description(pattern_name),
                    file_path=file_path,
                    line_number=line_number,
                    suggestion=self._get_suggestion(pattern_name, language)
                )
                issues.append(issue)
                
        return issues

    def _analyze_complexity_changes(self, content: str, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze algorithmic complexity changes"""
        try:
            elements = self.parser.parse_file(file_path, content)
            
            high_complexity_functions = []
            for element in elements:
                if element.type == "function" and element.complexity > 10:
                    high_complexity_functions.append({
                        "name": element.name,
                        "complexity": element.complexity,
                        "risk_level": "high" if element.complexity > 15 else "medium"
                    })
            
            if high_complexity_functions:
                return {
                    "file_path": file_path,
                    "high_complexity_functions": high_complexity_functions,
                    "average_complexity": sum(f["complexity"] for f in high_complexity_functions) / len(high_complexity_functions)
                }
                
        except Exception as e:
            print(f"Error analyzing complexity for {file_path}: {str(e)}")
            
        return None

    def _analyze_memory_impact(self, content: str, file_path: str, language: str) -> List[Dict[str, Any]]:
        """Analyze potential memory impact"""
        memory_issues = []
        
        # Language-specific memory patterns
        memory_patterns = {
            'python': {
                'large_list_comprehension': r'\[.*for.*in.*for.*in.*\]',
                'global_collections': r'global\s+\w+.*=\s*\[|\{',
                'memory_leak_potential': r'while\s+True:.*append\(',
            },
            'javascript': {
                'closure_memory_leak': r'function.*{.*var.*=.*function',
                'dom_references': r'var\s+\w+\s*=\s*document\.',
                'large_arrays': r'new\s+Array\(\d{4,}\)',
            },
            'csharp': {
                'large_collections': r'new\s+List<.*>\(\d{4,}\)',
                'event_handler_leak': r'\w+\s*\+=\s*\w+',
                'disposable_not_disposed': r'new\s+\w+\(.*\)\s*;(?!.*using|.*Dispose)',
            }
        }
        
        patterns = memory_patterns.get(language, {})
        
        for pattern_name, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.MULTILINE)
            
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                
                memory_issues.append({
                    "type": pattern_name,
                    "file_path": file_path,
                    "line_number": line_number,
                    "severity": self._get_memory_severity(pattern_name),
                    "description": self._get_memory_description(pattern_name)
                })
                
        return memory_issues

    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = file_path.lower().split('.')[-1]
        
        language_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'javascript',
            'jsx': 'javascript',
            'tsx': 'javascript',
            'cs': 'csharp',
            'java': 'java',
            'go': 'go',
            'rs': 'rust',
            'cpp': 'cpp',
            'cc': 'cpp',
            'hpp': 'cpp',
        }
        
        return language_map.get(ext, 'unknown')

    def _get_severity(self, pattern_name: str) -> str:
        """Get severity level for performance issue"""
        severity_map = {
            'nested_loops': 'high',
            'string_concatenation': 'medium',
            'global_variables': 'low',
            'recursive_without_memoization': 'high',
            'dom_queries_in_loop': 'high',
            'synchronous_requests': 'critical',
            'linq_multiple_enumeration': 'medium',
            'boxing_unboxing': 'low',
        }
        
        return severity_map.get(pattern_name, 'medium')

    def _get_issue_description(self, pattern_name: str) -> str:
        """Get description for performance issue"""
        descriptions = {
            'nested_loops': 'Nested loops can lead to O(n²) or worse time complexity',
            'string_concatenation': 'String concatenation in loops is inefficient',
            'global_variables': 'Global variables can impact performance and maintainability',
            'recursive_without_memoization': 'Recursive function without memoization may cause exponential time complexity',
            'dom_queries_in_loop': 'DOM queries inside loops cause performance bottlenecks',
            'synchronous_requests': 'Synchronous AJAX requests block the UI thread',
            'linq_multiple_enumeration': 'Multiple LINQ enumerations can impact performance',
            'boxing_unboxing': 'Boxing/unboxing operations add overhead',
        }
        
        return descriptions.get(pattern_name, 'Potential performance issue detected')

    def _get_suggestion(self, pattern_name: str, language: str) -> str:
        """Get performance improvement suggestion"""
        suggestions = {
            'nested_loops': 'Consider using more efficient algorithms or data structures',
            'string_concatenation': 'Use StringBuilder (C#) or join() method (Python) for string concatenation',
            'global_variables': 'Consider using local variables or class members instead',
            'recursive_without_memoization': 'Add memoization or convert to iterative approach',
            'dom_queries_in_loop': 'Cache DOM queries outside the loop',
            'synchronous_requests': 'Use asynchronous requests with async/await or promises',
            'linq_multiple_enumeration': 'Use ToList() or ToArray() to materialize the query once',
            'boxing_unboxing': 'Use generic collections to avoid boxing/unboxing',
        }
        
        return suggestions.get(pattern_name, 'Review code for performance optimization opportunities')

    def _get_memory_severity(self, pattern_name: str) -> str:
        """Get severity for memory issues"""
        severity_map = {
            'large_list_comprehension': 'medium',
            'global_collections': 'high',
            'memory_leak_potential': 'critical',
            'closure_memory_leak': 'high',
            'dom_references': 'medium',
            'large_arrays': 'high',
            'large_collections': 'medium',
            'event_handler_leak': 'high',
            'disposable_not_disposed': 'high',
        }
        
        return severity_map.get(pattern_name, 'medium')

    def _get_memory_description(self, pattern_name: str) -> str:
        """Get description for memory issues"""
        descriptions = {
            'large_list_comprehension': 'Large list comprehensions can consume significant memory',
            'global_collections': 'Global collections may lead to memory leaks',
            'memory_leak_potential': 'Infinite loop with growing collections can cause memory leaks',
            'closure_memory_leak': 'Closures can prevent garbage collection',
            'dom_references': 'Storing DOM references can prevent cleanup',
            'large_arrays': 'Large array allocation may impact memory usage',
            'large_collections': 'Large collection initialization uses significant memory',
            'event_handler_leak': 'Event handlers may prevent garbage collection',
            'disposable_not_disposed': 'IDisposable objects should be properly disposed',
        }
        
        return descriptions.get(pattern_name, 'Potential memory issue detected')

    def _generate_performance_recommendations(self, issues: List[PerformanceIssue]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Group issues by type
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.type] = issue_counts.get(issue.type, 0) + 1
        
        # Generate recommendations based on issue patterns
        if issue_counts.get('nested_loops', 0) > 2:
            recommendations.append("Consider reviewing algorithm complexity - multiple nested loops detected")
            
        if issue_counts.get('string_concatenation', 0) > 3:
            recommendations.append("Optimize string operations - use efficient concatenation methods")
            
        if issue_counts.get('dom_queries_in_loop', 0) > 0:
            recommendations.append("Cache DOM queries to improve frontend performance")
            
        if not recommendations:
            recommendations.append("No major performance issues detected")
            
        return recommendations

    def _calculate_overall_impact(self, issues: List[PerformanceIssue], complexity_changes: List[Dict]) -> str:
        """Calculate overall performance impact"""
        critical_count = sum(1 for issue in issues if issue.severity == 'critical')
        high_count = sum(1 for issue in issues if issue.severity == 'high')
        
        high_complexity_count = sum(1 for change in complexity_changes 
                                  for func in change.get('high_complexity_functions', [])
                                  if func['risk_level'] == 'high')
        
        if critical_count > 0 or high_complexity_count > 2:
            return "high"
        elif high_count > 2 or high_complexity_count > 0:
            return "medium"
        else:
            return "low"