import re

def is_diff_format(content: str) -> bool:
    """Check if the content is in diff format"""
    return content.startswith('diff --git') or content.startswith('--- a/') or '@@' in content

def extract_file_content_from_diff(diff_content: str) -> str:
    """Extract the actual file content from a diff format"""
    lines = diff_content.split('\n')
    file_content = []
    in_hunk = False
    
    for line in lines:
        # Skip diff headers
        if (line.startswith('diff --git') or 
            line.startswith('--- a/') or 
            line.startswith('+++ b/') or 
            line.startswith('index ') or
            line.startswith('new file mode') or
            line.startswith('deleted file mode')):
            continue
        
        # Handle hunk headers (@@ ... @@)
        if line.startswith('@@'):
            in_hunk = True
            continue
        
        if in_hunk:
            # Include context lines (unchanged lines starting with space)
            if line.startswith(' '):
                file_content.append(line[1:])  # Remove the leading space
            # Include added lines (starting with +)
            elif line.startswith('+'):
                file_content.append(line[1:])  # Remove the + prefix
            # Skip removed lines (starting with -)
            elif line.startswith('-'):
                continue
            # Handle lines that don't start with +, -, or space (end of hunk)
            else:
                # If we encounter a line that doesn't match diff format, 
                # it might be the end of the diff or a new section
                if not line.startswith('diff --git') and not line.startswith('--- a/') and not line.startswith('+++ b/'):
                    file_content.append(line)
    
    result = '\n'.join(file_content)
    
    # Debug output
    print(f"[DEBUG] Extracted content from diff (first 200 chars):\n{result[:200]}...")
    
    return result 