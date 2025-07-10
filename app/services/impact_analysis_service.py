"""
Comprehensive Impact Analysis Service
Focuses on generating meaningful, plain-English impact summaries for test case generation
"""
import os
import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .azure_openai_service import AzureOpenAIService
from ..models.analysis import CodeChange

class ImpactCategory(Enum):
    BUSINESS_LOGIC = "business_logic"
    DATA_PROCESSING = "data_processing"
    USER_INTERFACE = "user_interface"
    API_ENDPOINTS = "api_endpoints"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    ERROR_HANDLING = "error_handling"

@dataclass
class ImpactArea:
    category: ImpactCategory
    description: str
    affected_functionality: List[str]
    test_scenarios: List[str]
    risk_level: str
    user_impact: str

@dataclass
class ComprehensiveImpactAnalysis:
    summary: str
    impact_areas: List[ImpactArea]
    overall_risk: str
    testing_priority: str
    recommended_test_types: List[str]
    potential_side_effects: List[str]
    rollback_considerations: List[str]

class ImpactAnalysisService:
    """Service focused on comprehensive impact analysis for test generation"""
    
    def __init__(self):
        self.openai_service = AzureOpenAIService()

    async def analyze_comprehensive_impact(self, changes: List[CodeChange]) -> ComprehensiveImpactAnalysis:
        """Generate comprehensive impact analysis focused on testing needs"""
        
        print(f"[DEBUG] Starting comprehensive impact analysis for {len(changes)} changes")
        
        # Prepare change context
        change_context = self._prepare_change_context(changes)
        
        # Generate impact analysis using LLM
        impact_analysis = await self._generate_impact_analysis(change_context)
        
        # Parse and structure the analysis
        structured_analysis = self._structure_analysis(impact_analysis, changes)
        
        return structured_analysis

    def _prepare_change_context(self, changes: List[CodeChange]) -> str:
        """Prepare a comprehensive context of all changes"""
        context_parts = []
        
        for i, change in enumerate(changes, 1):
            context_parts.append(f"\n=== CHANGE {i}: {change.file_path} ===")
            context_parts.append(f"Change Type: {change.change_type}")
            
            if change.diff:
                # Extract meaningful parts from diff
                diff_summary = self._extract_diff_summary(change.diff)
                context_parts.append(f"Changes Summary:\n{diff_summary}")
            elif change.content:
                # Analyze content for key patterns
                content_analysis = self._analyze_content_patterns(change.content, change.file_path)
                context_parts.append(f"Content Analysis:\n{content_analysis}")
            
            context_parts.append("---")
        
        return "\n".join(context_parts)

    def _extract_diff_summary(self, diff_content: str) -> str:
        """Extract meaningful summary from diff content"""
        lines = diff_content.split('\n')
        
        added_lines = []
        removed_lines = []
        modified_functions = []
        
        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                clean_line = line[1:].strip()
                if clean_line and not clean_line.startswith('//') and not clean_line.startswith('#'):
                    added_lines.append(clean_line)
                    
                    # Check for function definitions
                    if any(keyword in clean_line.lower() for keyword in ['def ', 'function ', 'class ', 'method']):
                        func_match = re.search(r'(def|function|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', clean_line)
                        if func_match:
                            modified_functions.append(f"Added {func_match.group(1)}: {func_match.group(2)}")
            
            elif line.startswith('-') and not line.startswith('---'):
                clean_line = line[1:].strip()
                if clean_line and not clean_line.startswith('//') and not clean_line.startswith('#'):
                    removed_lines.append(clean_line)
                    
                    # Check for function definitions
                    if any(keyword in clean_line.lower() for keyword in ['def ', 'function ', 'class ', 'method']):
                        func_match = re.search(r'(def|function|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', clean_line)
                        if func_match:
                            modified_functions.append(f"Removed {func_match.group(1)}: {func_match.group(2)}")
        
        summary_parts = []
        
        if modified_functions:
            summary_parts.append(f"Function Changes: {'; '.join(modified_functions[:5])}")
        
        if added_lines:
            summary_parts.append(f"Key Additions ({len(added_lines)} lines): {'; '.join(added_lines[:3])}")
        
        if removed_lines:
            summary_parts.append(f"Key Removals ({len(removed_lines)} lines): {'; '.join(removed_lines[:3])}")
        
        return '\n'.join(summary_parts) if summary_parts else "Minor changes detected"

    def _analyze_content_patterns(self, content: str, file_path: str) -> str:
        """Analyze content for key patterns and functionality"""
        patterns = {
            'api_endpoints': [r'@app\.route', r'@router\.', r'app\.get|app\.post|app\.put|app\.delete', r'endpoint\s*=', r'path\s*='],
            'database_operations': [r'\.query\(', r'\.insert\(', r'\.update\(', r'\.delete\(', r'SELECT\s+', r'INSERT\s+', r'UPDATE\s+', r'DELETE\s+'],
            'authentication': [r'auth', r'login', r'password', r'token', r'session', r'permission', r'role'],
            'validation': [r'validate', r'check', r'verify', r'assert', r'raise\s+\w*Error', r'if\s+not\s+'],
            'business_logic': [r'calculate', r'process', r'transform', r'generate', r'compute', r'algorithm'],
            'ui_components': [r'render', r'component', r'template', r'html', r'css', r'style', r'onclick', r'onchange'],
            'error_handling': [r'try:', r'except', r'catch', r'error', r'exception', r'finally'],
            'configuration': [r'config', r'setting', r'environment', r'env', r'constant', r'CONST'],
            'integration': [r'api', r'service', r'client', r'request', r'response', r'webhook', r'callback']
        }
        
        detected_patterns = {}
        content_lower = content.lower()
        
        for category, pattern_list in patterns.items():
            matches = []
            for pattern in pattern_list:
                if re.search(pattern, content_lower):
                    matches.append(pattern)
            
            if matches:
                detected_patterns[category] = len(matches)
        
        # Extract function names
        functions = re.findall(r'(?:def|function|class)\s+([a-zA-Z_][a-zA-Z0-9_]*)', content)
        
        analysis_parts = []
        
        if functions:
            analysis_parts.append(f"Functions/Classes: {', '.join(functions[:5])}")
        
        if detected_patterns:
            pattern_summary = ', '.join([f"{cat}: {count}" for cat, count in detected_patterns.items()])
            analysis_parts.append(f"Detected Patterns: {pattern_summary}")
        
        # File type analysis
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in ['.html', '.css', '.js', '.jsx', '.tsx', '.vue']:
            analysis_parts.append("File Type: Frontend/UI component")
        elif file_ext in ['.py', '.java', '.cs', '.go']:
            analysis_parts.append("File Type: Backend/Business logic")
        elif file_ext in ['.sql', '.db']:
            analysis_parts.append("File Type: Database/Data layer")
        
        return '\n'.join(analysis_parts) if analysis_parts else "Standard code file"

    async def _generate_impact_analysis(self, change_context: str) -> str:
        """Generate comprehensive impact analysis using LLM"""
        
        prompt = f"""
You are a senior QA engineer and system analyst. Analyze the following code changes and provide a comprehensive impact analysis that will help generate test cases.

CHANGES TO ANALYZE:
{change_context}

Please provide a detailed analysis in the following JSON format:

{{
    "summary": "A clear, plain-English summary of what changed and the overall impact",
    "impact_areas": [
        {{
            "category": "business_logic|data_processing|user_interface|api_endpoints|authentication|validation|configuration|integration|performance|error_handling",
            "description": "Plain English description of what's impacted in this area",
            "affected_functionality": ["List of specific features/functions affected"],
            "test_scenarios": ["Specific test scenarios to validate this impact"],
            "risk_level": "low|medium|high|critical",
            "user_impact": "How this affects end users"
        }}
    ],
    "overall_risk": "low|medium|high|critical",
    "testing_priority": "low|medium|high|critical",
    "recommended_test_types": ["unit", "integration", "e2e", "performance", "security", "regression"],
    "potential_side_effects": ["List of potential unintended consequences"],
    "rollback_considerations": ["What to consider if rollback is needed"]
}}

FOCUS ON:
1. **Business Impact**: How do these changes affect what users can do?
2. **Data Flow**: How is data processing, storage, or retrieval affected?
3. **User Experience**: What changes in user interactions or interface?
4. **System Integration**: How do these changes affect connections between components?
5. **Risk Assessment**: What could go wrong and how likely is it?
6. **Test Strategy**: What specific tests are needed to validate these changes?

PROVIDE ACTIONABLE INSIGHTS that help create comprehensive test plans.
"""

        try:
            response = self.openai_service.client.chat.completions.create(
                model=self.openai_service.deployment_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a senior QA engineer and system analyst. Provide comprehensive, actionable impact analysis for test planning. Always respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"[DEBUG] LLM analysis failed: {str(e)}")
            # Return fallback analysis
            return self._generate_fallback_analysis(change_context)

    def _generate_fallback_analysis(self, change_context: str) -> str:
        """Generate fallback analysis when LLM fails"""
        
        # Count changes and extract basic info
        change_count = change_context.count("=== CHANGE")
        has_functions = bool(re.search(r'(def|function|class)', change_context, re.IGNORECASE))
        has_api = bool(re.search(r'(route|endpoint|api)', change_context, re.IGNORECASE))
        has_database = bool(re.search(r'(query|insert|update|delete|database)', change_context, re.IGNORECASE))
        
        fallback_analysis = {
            "summary": f"Analysis of {change_count} code changes. {'Functions/classes modified. ' if has_functions else ''}{'API endpoints affected. ' if has_api else ''}{'Database operations involved.' if has_database else ''}",
            "impact_areas": [],
            "overall_risk": "medium",
            "testing_priority": "medium",
            "recommended_test_types": ["unit", "integration"],
            "potential_side_effects": ["Potential breaking changes", "Data consistency issues"],
            "rollback_considerations": ["Backup current state", "Test rollback procedure"]
        }
        
        # Add impact areas based on detected patterns
        if has_functions:
            fallback_analysis["impact_areas"].append({
                "category": "business_logic",
                "description": "Business logic functions have been modified",
                "affected_functionality": ["Core application logic"],
                "test_scenarios": ["Test modified functions with various inputs", "Verify business rules still apply"],
                "risk_level": "medium",
                "user_impact": "May affect core functionality"
            })
        
        if has_api:
            fallback_analysis["impact_areas"].append({
                "category": "api_endpoints",
                "description": "API endpoints have been modified",
                "affected_functionality": ["API responses", "Request handling"],
                "test_scenarios": ["Test API endpoints with various payloads", "Verify response formats"],
                "risk_level": "high",
                "user_impact": "May affect client applications"
            })
        
        if has_database:
            fallback_analysis["impact_areas"].append({
                "category": "data_processing",
                "description": "Database operations have been modified",
                "affected_functionality": ["Data storage", "Data retrieval"],
                "test_scenarios": ["Test data operations", "Verify data integrity"],
                "risk_level": "high",
                "user_impact": "May affect data consistency"
            })
        
        return json.dumps(fallback_analysis)

    def _structure_analysis(self, analysis_json: str, changes: List[CodeChange]) -> ComprehensiveImpactAnalysis:
        """Structure the analysis into the final format"""
        
        try:
            analysis_data = json.loads(analysis_json)
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Failed to parse analysis JSON: {e}")
            # Create minimal analysis
            analysis_data = {
                "summary": f"Analysis of {len(changes)} code changes",
                "impact_areas": [],
                "overall_risk": "medium",
                "testing_priority": "medium",
                "recommended_test_types": ["unit", "integration"],
                "potential_side_effects": ["Manual review required"],
                "rollback_considerations": ["Backup current state"]
            }
        
        # Convert to structured format
        impact_areas = []
        for area_data in analysis_data.get("impact_areas", []):
            try:
                impact_area = ImpactArea(
                    category=ImpactCategory(area_data.get("category", "business_logic")),
                    description=area_data.get("description", ""),
                    affected_functionality=area_data.get("affected_functionality", []),
                    test_scenarios=area_data.get("test_scenarios", []),
                    risk_level=area_data.get("risk_level", "medium"),
                    user_impact=area_data.get("user_impact", "")
                )
                impact_areas.append(impact_area)
            except (ValueError, KeyError) as e:
                print(f"[DEBUG] Error processing impact area: {e}")
                continue
        
        return ComprehensiveImpactAnalysis(
            summary=analysis_data.get("summary", "Code changes analyzed"),
            impact_areas=impact_areas,
            overall_risk=analysis_data.get("overall_risk", "medium"),
            testing_priority=analysis_data.get("testing_priority", "medium"),
            recommended_test_types=analysis_data.get("recommended_test_types", ["unit", "integration"]),
            potential_side_effects=analysis_data.get("potential_side_effects", []),
            rollback_considerations=analysis_data.get("rollback_considerations", [])
        )