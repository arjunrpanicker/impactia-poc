import ast
import os
import re
from typing import List, Optional, Tuple
from openai import AzureOpenAI

def is_valid_python_code(code: str) -> bool:
    """Check if the code is valid Python syntax"""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def extract_complete_function(code: str, function_name: str) -> Optional[str]:
    """Extract a complete function definition from code using regex"""
    # Pattern to match function definitions
    pattern = rf'def\s+{re.escape(function_name)}\s*\([^)]*\)\s*:.*?(?=\n\s*def|\n\s*class|\Z)'
    
    match = re.search(pattern, code, re.DOTALL | re.MULTILINE)
    if match:
        function_code = match.group(0)
        # Try to parse it to ensure it's valid
        if is_valid_python_code(function_code):
            return function_code
    
    return None

def list_top_level_functions(code: str) -> List[str]:
    """Extract top-level function names from code"""
    try:
        tree = ast.parse(code)
        return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    except SyntaxError as e:
        print(f"[DEBUG] SyntaxError in list_top_level_functions: {e}")
        print(f"[DEBUG] Offending code:\n{code[:500]}...")
        # Fallback: try to extract function names using regex
        function_names = []
        pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, code)
        function_names.extend(matches)
        print(f"[DEBUG] Fallback regex found functions: {function_names}")
        return function_names

def extract_function_ast(code: str, function_name: str) -> Optional[ast.FunctionDef]:
    """Extract function AST node from code"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None
    except SyntaxError as e:
        print(f"[DEBUG] SyntaxError in extract_function_ast: {e}")
        print(f"[DEBUG] Offending code:\n{code[:500]}...")
        return None

def extract_calls_and_conditionals(func_node) -> dict:
    """Extract function calls, conditionals, returns, and raises from AST node"""
    calls = set()
    conditionals = set()
    returns = set()
    raises = set()
    
    try:
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
                calls.add(node.func.id)
            if isinstance(node, ast.If):
                try:
                    conditionals.add(ast.unparse(node.test))
                except Exception:
                    conditionals.add(str(node.test))
            if isinstance(node, ast.Return):
                try:
                    returns.add(ast.unparse(node.value) if node.value else 'return')
                except Exception:
                    returns.add('return')
            if isinstance(node, ast.Raise):
                try:
                    raises.add(ast.unparse(node.exc) if node.exc else 'raise')
                except Exception:
                    raises.add('raise')
    except Exception as e:
        print(f"[DEBUG] Error extracting calls and conditionals: {e}")
    
    return {
        'calls': calls,
        'conditionals': conditionals,
        'returns': returns,
        'raises': raises
    }

def rule_based_summary(base_code: str, updated_code: str, function_name: str) -> List[str]:
    """Generate rule-based summary of function changes"""
    # Try to extract complete functions first
    base_func_code = extract_complete_function(base_code, function_name)
    updated_func_code = extract_complete_function(updated_code, function_name)
    
    if not base_func_code or not updated_func_code:
        print(f"[DEBUG] Could not extract complete functions for '{function_name}'")
        return []
    
    # Parse the complete functions
    base_func = extract_function_ast(base_func_code, function_name)
    updated_func = extract_function_ast(updated_func_code, function_name)
    
    if not base_func or not updated_func:
        print(f"[DEBUG] Could not parse AST for function '{function_name}'")
        return []

    base_info = extract_calls_and_conditionals(base_func)
    updated_info = extract_calls_and_conditionals(updated_func)

    summary = []
    # Function calls
    added_calls = updated_info['calls'] - base_info['calls']
    removed_calls = base_info['calls'] - updated_info['calls']
    for call in added_calls:
        summary.append(f"Added call to `{call}`")
    for call in removed_calls:
        summary.append(f"Removed call to `{call}`")
    # Conditionals
    added_conds = updated_info['conditionals'] - base_info['conditionals']
    removed_conds = base_info['conditionals'] - updated_info['conditionals']
    for cond in added_conds:
        summary.append(f"Added conditional: `{cond}`")
    for cond in removed_conds:
        summary.append(f"Removed conditional: `{cond}`")
    # Returns
    if updated_info['returns'] != base_info['returns']:
        summary.append("Changed return statement(s)")
    # Raises
    if updated_info['raises'] != base_info['raises']:
        summary.append("Changed raised exception(s)")
    return summary

def llm_fallback(base_code: str, updated_code: str, function_name: str) -> str:
    """LLM fallback for generating function change summaries"""
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    # Extract complete functions for LLM
    base_func_code = extract_complete_function(base_code, function_name)
    updated_func_code = extract_complete_function(updated_code, function_name)
    
    if not base_func_code or not updated_func_code:
        return f"Could not extract complete function '{function_name}' for analysis"
    
    prompt = f"""
Here is the original function `{function_name}`:

{base_func_code}

Here is the updated function `{function_name}`:

{updated_func_code}

Summarise what changed in behaviour or logic in 1-5 bullet points.
"""
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a code review assistant. Summarise code changes for test case generation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        print("[DEBUG] LLM summary output:")
        print(response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[DEBUG] LLM fallback error: {e}")
        return f"Error generating summary for function '{function_name}': {str(e)}"

def generate_functional_diff(base_code: str, updated_code: str, function_name: str) -> str:
    """Generate functional diff summary for a function"""
    print(f"[DEBUG] Generating functional diff for function: {function_name}")
    
    # Only operate on real, top-level functions
    base_funcs = set(list_top_level_functions(base_code))
    updated_funcs = set(list_top_level_functions(updated_code))
    
    if function_name not in base_funcs and function_name not in updated_funcs:
        print(f"[DEBUG] Function '{function_name}' not found in either version. No summary generated.")
        return f"Function '{function_name}' not found in either version. No summary generated."
    
    # Try rule-based summary first
    summary = rule_based_summary(base_code, updated_code, function_name)
    print("[DEBUG] AST summary output:")
    for line in summary:
        print(f"- {line}")
    
    if len(summary) < 2:
        print("[DEBUG] Using LLM fallback.")
        return llm_fallback(base_code, updated_code, function_name)
    
    print("[DEBUG] Using AST rule-based summary.")
    return "\n".join(f"- {line}" for line in summary) 