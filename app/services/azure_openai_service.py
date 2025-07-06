import os
import json
from typing import List, Dict, Any
from openai import AzureOpenAI
from ..models.analysis import RiskLevel, ChangeAnalysisResponse, ChangedComponent, CodeChange
from ..models.analysis import (
    RiskLevel, ChangeAnalysisResponseWithCode, ChangedComponentWithCode, MethodWithCode,
    DependencyChainWithCode, DependentFileWithCode, DependentMethodWithSummary
)
from .diff_service import generate_functional_diff
from ..utils.diff_utils import is_diff_format

class AzureOpenAIService:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")

    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text using Azure OpenAI"""
        response = self.client.embeddings.create(
            model=self.embeddings_deployment,
            input=text
        )
        return response.data[0].embedding

    def _extract_methods(self, content: str) -> List[Dict[str, str]]:
        """Extract methods from code content"""
        import re
        
        methods = []
        lines = content.split('\n')
        
        # Match common method patterns
        patterns = [
            # Python
            (r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:', 1),
            # JavaScript/TypeScript
            (r'(async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)', 2),
            # C#/Java
            (r'(public|private|protected|internal)?\s+[a-zA-Z_<>[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)', 2),
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern, group_idx in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    method_name = match.group(group_idx)
                    
                    # Extract complete function using regex
                    if pattern.startswith('def'):  # Python function
                        # Find the complete function definition
                        func_pattern = rf'def\s+{re.escape(method_name)}\s*\([^)]*\)\s*:.*?(?=\n\s*def|\n\s*class|\Z)'
                        func_match = re.search(func_pattern, content, re.DOTALL | re.MULTILINE)
                        if func_match:
                            method_content = func_match.group(0)
                        else:
                            # Fallback to context window
                            start_line = max(0, i - 5)
                            end_line = min(len(lines), i + 50)  # Increased context
                            method_content = '\n'.join(lines[start_line:end_line])
                    else:
                        # For non-Python, use larger context window
                        start_line = max(0, i - 5)
                        end_line = min(len(lines), i + 50)
                        method_content = '\n'.join(lines[start_line:end_line])
                    
                    methods.append({
                        "name": method_name,
                        "content": method_content,
                        "start_line": i
                    })
        
        return methods

    async def analyze_impact(self, changes: List[CodeChange], related_code: Dict[str, Any]) -> ChangeAnalysisResponseWithCode:
        """Analyze the impact of code changes using Azure OpenAI and include full method code and impacted file content."""
        import re
        def normalize(name):
            return re.sub(r'[^a-zA-Z0-9]', '', name).lower()
        
        # Detect UI components
        ui_files = [change for change in changes if any(change.file_path.endswith(ext) for ext in ['.jsx', '.tsx', '.vue', '.svelte', '.html', '.css'])]
        is_ui_change = len(ui_files) > 0
        
        # Format changes into a readable structure
        formatted_changes = []
        for change in changes:
            change_text = f"\nFile: {change.file_path}\n"
            change_text += f"Type: {change.change_type}\n"
            if change.diff:
                change_text += f"Diff:\n{change.diff}\n"
            elif change.content:
                change_text += f"Content:\n{change.content}\n"
            formatted_changes.append(change_text)

        # Add UI-specific context if needed
        ui_context = ""
        if is_ui_change:
            ui_context = """
            For UI components, analyze:
            1. Component Structure:
               - Component hierarchy
               - Props and state management
               - Event handlers
               - Conditional rendering
               - Styling changes
            
            2. User Interactions:
               - Click events
               - Form submissions
               - Input changes
               - Navigation
               - Modal/overlay interactions
            
            3. Visual Elements:
               - Layout changes
               - Styling modifications
               - Responsive design
               - Accessibility attributes
            
            4. State Management:
               - Local state changes
               - Global state updates
               - Context usage
               - Props drilling
            
            5. Integration Points:
               - API calls
               - Event propagation
               - Parent-child communication
               - Route changes
            """

        # Format dependencies information
        dependencies_text = "\nDependency Analysis:\n"
        if "direct_dependencies" in related_code:
            deps = related_code["direct_dependencies"]
            dependencies_text += "\nIncoming References (files that depend on the changed files):\n"
            for ref in deps.get("incoming", []):
                dependencies_text += f"- {ref}\n"
            
            dependencies_text += "\nOutgoing References (files that the changed files depend on):\n"
            for ref in deps.get("outgoing", []):
                dependencies_text += f"- {ref}\n"

        # Add dependency chain information
        if "dependency_chains" in related_code:
            dependencies_text += "\nDetailed Dependency Chains:\n"
            for chain in related_code["dependency_chains"]:
                dependencies_text += f"\nFile: {chain['file_path']}\n"
                dependencies_text += "Dependent Files:\n"
                for dep in chain.get("dependent_files", []):
                    dependencies_text += f"- {dep['file_path']}\n"
                    for method in dep.get("methods", []):
                        dependencies_text += f"  - Method {method['name']}: {method['summary']}\n"

        if "dependency_visualization" in related_code:
            dependencies_text += "\nDependency Flow:\n"
            for viz in related_code["dependency_visualization"]:
                dependencies_text += f"- {viz}\n"

        # Format similar code information
        similar_code_text = "\nSimilar Code Analysis:\n"
        if "similar_code" in related_code:
            similar = related_code["similar_code"]
            
            similar_code_text += "\nSimilar Files:\n"
            for file in similar.get("files", []):
                similar_code_text += f"- {file['path']} (similarity: {file['similarity']:.2f})\n"
                for method in file.get("methods", []):
                    similar_code_text += f"  - Method: {method.get('name', 'unknown')}\n"
            
            similar_code_text += "\nSimilar Methods:\n"
            for method in similar.get("methods", []):
                similar_code_text += f"- {method['name']} in {method.get('file_path', 'unknown')} (similarity: {method['similarity']:.2f})\n"

        # --- Functional Diff Summaries ---
        # Build a map of file_path -> {method_name: summary}
        functional_summaries = {}
        for change in changes:
            file_path = change.file_path
            base_code = change.content or ''  # For now, use uploaded content as both base and updated (extend as needed)
            updated_code = change.content or ''
            
            # Debug output to see what content we're working with
            print(f"[DEBUG] Processing file: {file_path}")
            print(f"[DEBUG] Content type: {'diff' if is_diff_format(base_code) else 'regular'}")
            print(f"[DEBUG] Content preview (first 200 chars):\n{base_code[:200]}...")
            
            # If you have access to both base and updated code, use them here
            # For each method in the file, generate a summary
            method_summaries = {}
            methods = self._extract_methods(updated_code)
            for m in methods:
                summary = generate_functional_diff(base_code, updated_code, m['name'])
                method_summaries[m['name']] = summary
            functional_summaries[file_path] = method_summaries
        # --- End Functional Diff Summaries ---

        # When building the LLM prompt, include the functional summaries for each changed/impacted method
        # Instead of sending full code or raw diff, send the summary
        # Example: Add to the prompt for each changed method:
        # "Functional change summary for {file_path}.{method_name}: {summary}"
        # You can concatenate these summaries and add to the prompt before the rest of the context
        diff_summaries_text = ""
        for file_path, methods in functional_summaries.items():
            for method_name, summary in methods.items():
                diff_summaries_text += f"\nFunctional change summary for {file_path}.{method_name}:\n{summary}\n"

        # Prepare the prompt
        prompt = f"""{diff_summaries_text}\n\nAnalyze these code changes and their dependencies to provide a comprehensive impact analysis in the required JSON format:

CHANGES:
{''.join(formatted_changes)}

{ui_context}

DEPENDENCY INFORMATION:
{dependencies_text}

SIMILAR CODE PATTERNS:
{similar_code_text}

Analyze and include in your response:
1. A clear summary of the changes and their impact
2. Detailed analysis of each changed file, including:
   - Changed methods (use exact names as in code, case and underscores must match) and their new behavior
   - Dependent methods in other files that may be affected
   - For UI components:
     * Component rendering and behavior changes
     * User interaction modifications
     * Visual and layout changes
     * State management updates
     * Integration point changes
3. Complete dependency chains showing how changes propagate through the codebase
4. Visualization of dependencies between files

IMPORTANT: Respond with ONLY a valid JSON object matching the structure specified. No additional text or explanations."""
        
        # Update prompt to instruct LLM to use exact method names as in code, not to guess, and only include top-level functions
        prompt = prompt.replace(
            "- Changed methods (use exact names as in code, case and underscores must match, and ONLY include methods that actually exist in the provided code for each file; do NOT guess or hallucinate method names) and their new behavior",
            "- Changed methods (use exact names as in code, case and underscores must match, and ONLY include top-level function or method names that actually exist in the provided code for each file; do NOT guess, hallucinate, or include variables/classes/inner blocks) and their new behavior"
        )
        
        # Get completion from Azure OpenAI
        response = self.client.chat.completions.create(
            model=self.deployment_name,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a code analysis expert. Analyze the impact of code changes and their dependencies to provide detailed insights.
IMPORTANT: Your response must be ONLY a valid JSON object with no additional text or explanation.

Required JSON structure:
{
    "summary": "Brief summary of changes and their impact",
    "changed_components": [
        {
            "file_path": "path/to/changed/file",
            "methods": ["methodName1", "methodName2"],
            "impact_description": "Description of how these methods are impacted",
            "risk_level": "low|medium|high|critical",
            "associated_unit_tests": ["tests/UnitTests/path/to/test1.cs", "tests/UnitTests/path/to/test2.cs"]
        }
    ],
    "dependency_chains": [
        {
            "file_path": "path/to/changed/file",
            "methods": [
                {
                    "name": "methodName",
                    "summary": "Description of how this method is impacted"
                }
            ],
            "impacted_files": [
                {
                    "file_path": "path/to/dependent/file",
                    "methods": [
                        {
                            "name": "methodName",
                            "summary": "Description of how this dependent method is affected"
                        }
                    ]
                }
            ],
            "associated_unit_tests": ["tests/UnitTests/path/to/test1.cs", "tests/UnitTests/path/to/test2.cs"]
        }
    ],
    "dependency_chain_visualization": ["file1.cs->file2.cs"]
}

Rules:
1. Response must be ONLY the JSON object, no other text
2. All arrays must have at least one item
3. All fields are required except dependency_chains and dependency_chain_visualization
4. Use proper JSON formatting with double quotes for strings
5. Focus on method-level changes and their impact
6. Include both direct changes and dependency impacts
7. For dependency chains:
   - Show how changes propagate through the codebase
   - Include all affected methods in each file
   - Provide clear summaries of impact at each level
   - Consider both direct and indirect dependencies
   - Include methods that call or are called by changed methods
   - Include methods that use or are used by changed methods
8. For risk levels:
   - low: Minor changes with no significant impact
   - medium: Changes that affect specific functionality
   - high: Changes that affect multiple components
   - critical: Changes that affect core functionality or security
9. For associated_unit_tests:
   - Include full paths to unit test files only
   - Focus on tests that directly verify the changed functionality
   - Include tests for dependent components that might be affected
   - All test paths should be under tests/UnitTests/ directory
10. For UI components:
    - Treat component methods as regular methods
    - Include component lifecycle methods
    - Consider event handlers as methods
    - Include state management methods
    - Consider UI-specific dependencies"""
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={ "type": "json_object" }
        )
        
        try:
            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()
            if not response_text:
                raise Exception("Empty response from GPT")
                
            analysis_json = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["summary", "changed_components"]
            missing_fields = [field for field in required_fields if field not in analysis_json]
            if missing_fields:
                raise KeyError(f"Missing required fields: {', '.join(missing_fields)}")
            
            # After parsing the LLM response (analysis_json):
            # 1. For each changed file, extract all methods and their code from uploaded content
            file_method_map = {}
            uploaded_content_map = {change.file_path: change.content or '' for change in changes}
            for change in changes:
                file_path = change.file_path
                file_content = change.content or ''
                methods = self._extract_methods(file_content)
                # Normalize method names for robust matching
                file_method_map[file_path] = {normalize(m['name']): m['content'] for m in methods}
            # 2. Build changed_components with robust method code matching
            changed_components = []
            for comp in analysis_json["changed_components"]:
                file_path = comp["file_path"]
                method_objs = []
                method_code_map = file_method_map.get(file_path, {})
                valid_method_names = set(method_code_map.keys())
                for method_name in comp["methods"]:
                    norm_name = normalize(method_name)
                    code = method_code_map.get(norm_name)
                    # Fallback: try to find in similar_code.methods from related_code
                    if code is None:
                        found_candidates = []
                        for m in related_code.get('similar_code', {}).get('methods', []):
                            m_name = normalize(m.get('name', ''))
                            m_path = m.get('file_path')
                            if m_name == norm_name:
                                found_candidates.append((m_path, m.get('content', '')[:60]))
                                if m_path == file_path:
                                    code = m.get('content')
                                    break
                        # If not found with exact file path, use any candidate with matching name
                        if code is None and found_candidates:
                            code = found_candidates[0][1]  # Use the first candidate's code
                            print(f"[DEBUG] Used fallback code for method '{method_name}' from file '{found_candidates[0][0]}'")
                        if not found_candidates:
                            print(f"[DEBUG] No candidates found in vector DB for method '{method_name}' (file: {file_path})")
                    if code is None:
                        code = f"Not available: method '{method_name}' not found in uploaded file or vector index"
                    method_objs.append(MethodWithCode(name=method_name, code=code))
                changed_components.append(ChangedComponentWithCode(
                    file_path=file_path,
                    methods=method_objs,
                    impact_description=comp["impact_description"],
                    risk_level=comp["risk_level"],
                    associated_unit_tests=comp["associated_unit_tests"]
                ))
            # 3. For dependency_chains, add full file content to each impacted file
            dependency_chains = []
            for chain in (analysis_json.get("dependency_chains") or []):
                impacted_files = []
                for dep in chain.get("impacted_files", []):
                    dep_file_path = dep["file_path"]
                    # Try to use uploaded content if available
                    dep_code = uploaded_content_map.get(dep_file_path)
                    if dep_code is None:
                        # Try to read from disk
                        try:
                            with open(dep_file_path, 'r', encoding='utf-8') as f:
                                dep_code = f.read()
                        except Exception:
                            dep_code = 'Not available'
                    methods = [
                        DependentMethodWithSummary(name=m["name"], summary=m["summary"]) for m in dep.get("methods", [])
                    ]
                    impacted_files.append(DependentFileWithCode(
                        file_path=dep_file_path,
                        methods=methods,
                        code=dep_code
                    ))
                methods = [
                    DependentMethodWithSummary(name=m["name"], summary=m["summary"]) for m in chain.get("methods", [])
                ]
                dependency_chains.append(DependencyChainWithCode(
                    file_path=chain["file_path"],
                    methods=methods,
                    impacted_files=impacted_files,
                    associated_unit_tests=chain.get("associated_unit_tests", [])
                ))
            return ChangeAnalysisResponseWithCode(
                summary=analysis_json["summary"],
                changed_components=changed_components,
                dependency_chains=dependency_chains,
                dependency_chain_visualization=analysis_json.get("dependency_chain_visualization"),
                risk_level=analysis_json.get("risk_level")
            )
            
        except json.JSONDecodeError as e:
            print(f"Raw GPT response: {response.choices[0].message.content}")
            raise Exception(f"Failed to parse GPT response as JSON: {str(e)}")
        except KeyError as e:
            raise Exception(f"Missing required field in GPT response: {str(e)}")
        except ValueError as e:
            raise Exception(f"Invalid value in GPT response: {str(e)}")