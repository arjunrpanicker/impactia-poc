import re
from typing import Dict, List, Tuple, Optional

class GitDiffExtractor:
    """Enhanced git diff content extractor"""
    
    @staticmethod
    def is_diff_format(content: str) -> bool:
        """Check if the content is in diff format"""
        diff_indicators = [
            content.startswith('diff --git'),
            content.startswith('--- a/'),
            content.startswith('+++ b/'),
            '@@' in content,
            re.search(r'^[+-]', content, re.MULTILINE) is not None
        ]
        return any(diff_indicators)

    @staticmethod
    def extract_file_changes(diff_content: str) -> Dict[str, Dict[str, str]]:
        """Extract file changes from git diff"""
        print(f"[DEBUG] Starting diff extraction from content length: {len(diff_content)}")
        
        files = {}
        lines = diff_content.split('\n')
        
        current_file = None
        current_old_content = []
        current_new_content = []
        in_hunk = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # File header patterns
            if line.startswith('diff --git'):
                print(f"[DEBUG] Found diff header: {line}")
                # Save previous file if exists
                if current_file:
                    files[current_file] = {
                        'old_content': '\n'.join(current_old_content),
                        'new_content': '\n'.join(current_new_content)
                    }
                    print(f"[DEBUG] Saved file {current_file}: old={len(current_old_content)} lines, new={len(current_new_content)} lines")
                
                # Extract file path - try multiple patterns
                file_path = None
                
                # Pattern 1: diff --git a/path b/path
                match = re.search(r'diff --git a/(.*?) b/(.*)', line)
                if match:
                    file_path = match.group(2)
                    print(f"[DEBUG] Extracted file path (pattern 1): {file_path}")
                else:
                    # Pattern 2: diff --git a/path b/path (same path)
                    match = re.search(r'diff --git a/(.*?) b/\1', line)
                    if match:
                        file_path = match.group(1)
                        print(f"[DEBUG] Extracted file path (pattern 2): {file_path}")
                    else:
                        # Pattern 3: Just extract anything after a/ and b/
                        match = re.search(r'b/(.+?)(?:\s|$)', line)
                        if match:
                            file_path = match.group(1)
                            print(f"[DEBUG] Extracted file path (pattern 3): {file_path}")
                
                if file_path:
                    current_file = file_path
                    current_old_content = []
                    current_new_content = []
                    in_hunk = False
                else:
                    print(f"[DEBUG] Could not extract file path from: {line}")
            
            # Alternative file header patterns
            elif line.startswith('--- a/') and current_file is None:
                match = re.search(r'--- a/(.*)', line)
                if match:
                    current_file = match.group(1)
                    current_old_content = []
                    current_new_content = []
                    in_hunk = False
                    print(f"[DEBUG] Extracted file path from ---: {current_file}")
            
            elif line.startswith('+++ b/') and current_file is None:
                match = re.search(r'\+\+\+ b/(.*)', line)
                if match:
                    current_file = match.group(1)
                    current_old_content = []
                    current_new_content = []
                    in_hunk = False
                    print(f"[DEBUG] Extracted file path from +++: {current_file}")
            
            # Skip other metadata lines
            elif (line.startswith('index ') or 
                  line.startswith('--- a/') or 
                  line.startswith('+++ b/') or
                  line.startswith('new file mode') or
                  line.startswith('deleted file mode') or
                  line.startswith('similarity index') or
                  line.startswith('rename from') or
                  line.startswith('rename to')):
                print(f"[DEBUG] Skipping metadata line: {line[:50]}...")
            
            # Hunk header
            elif line.startswith('@@'):
                print(f"[DEBUG] Found hunk header: {line}")
                in_hunk = True
            
            # Hunk content
            elif in_hunk and current_file:
                if line.startswith(' '):  # Context line
                    current_old_content.append(line[1:])
                    current_new_content.append(line[1:])
                elif line.startswith('-'):  # Deleted line
                    current_old_content.append(line[1:])
                elif line.startswith('+'):  # Added line
                    current_new_content.append(line[1:])
                elif line.startswith('\\'):  # "No newline at end of file" etc.
                    pass  # Skip these lines
                else:
                    # End of hunk or unknown line
                    if line.strip() == '':
                        # Empty line in hunk
                        current_old_content.append('')
                        current_new_content.append('')
                    else:
                        # Might be end of hunk
                        in_hunk = False
            
            # Handle cases where diff doesn't have proper headers
            elif not current_file and (line.startswith('+') or line.startswith('-')):
                # Try to infer a filename or use a default
                current_file = "unknown_file"
                current_old_content = []
                current_new_content = []
                in_hunk = True
                print(f"[DEBUG] Inferred file for orphaned diff content: {current_file}")
                
                # Process this line
                if line.startswith('-'):
                    current_old_content.append(line[1:])
                elif line.startswith('+'):
                    current_new_content.append(line[1:])
            
            i += 1
        
        # Save last file
        if current_file:
            files[current_file] = {
                'old_content': '\n'.join(current_old_content),
                'new_content': '\n'.join(current_new_content)
            }
            print(f"[DEBUG] Saved final file {current_file}: old={len(current_old_content)} lines, new={len(current_new_content)} lines")
        
        print(f"[DEBUG] Total files extracted: {len(files)}")
        for file_path, content in files.items():
            print(f"[DEBUG] File: {file_path}")
            print(f"[DEBUG]   Old content preview: {content['old_content'][:100]}...")
            print(f"[DEBUG]   New content preview: {content['new_content'][:100]}...")
        
        return files

    @staticmethod
    def extract_file_content_from_diff(diff_content: str, prefer_new: bool = True) -> str:
        """Extract file content from diff (backward compatible)"""
        files = GitDiffExtractor.extract_file_changes(diff_content)
        
        if not files:
            print("[DEBUG] No files extracted, returning original content")
            return diff_content  # Return original if parsing fails
        
        # Get the first file's content
        first_file = next(iter(files.values()))
        
        if prefer_new:
            result = first_file.get('new_content', '')
            print(f"[DEBUG] Returning new content: {len(result)} chars")
            return result
        else:
            result = first_file.get('old_content', '')
            print(f"[DEBUG] Returning old content: {len(result)} chars")
            return result

    @staticmethod
    def get_change_statistics(diff_content: str) -> Dict[str, int]:
        """Get statistics about the changes"""
        stats = {
            'files_changed': 0,
            'lines_added': 0,
            'lines_deleted': 0,
            'lines_modified': 0
        }
        
        files = GitDiffExtractor.extract_file_changes(diff_content)
        stats['files_changed'] = len(files)
        
        lines = diff_content.split('\n')
        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                stats['lines_added'] += 1
            elif line.startswith('-') and not line.startswith('---'):
                stats['lines_deleted'] += 1
        
        return stats

    @staticmethod
    def parse_unified_diff(diff_content: str) -> Dict[str, Dict[str, str]]:
        """Parse unified diff format (alternative parser)"""
        print(f"[DEBUG] Trying unified diff parser")
        
        files = {}
        current_file = None
        old_content = []
        new_content = []
        
        lines = diff_content.split('\n')
        
        for line in lines:
            # Look for file indicators
            if line.startswith('---'):
                # Old file
                match = re.search(r'---\s+(.+?)(?:\s+\d{4}-\d{2}-\d{2}|\s+\w{3}\s+\w{3}|\s*$)', line)
                if match:
                    old_file = match.group(1)
                    if old_file.startswith('a/'):
                        old_file = old_file[2:]
                    print(f"[DEBUG] Found old file: {old_file}")
            
            elif line.startswith('+++'):
                # New file
                match = re.search(r'\+\+\+\s+(.+?)(?:\s+\d{4}-\d{2}-\d{2}|\s+\w{3}\s+\w{3}|\s*$)', line)
                if match:
                    new_file = match.group(1)
                    if new_file.startswith('b/'):
                        new_file = new_file[2:]
                    current_file = new_file
                    old_content = []
                    new_content = []
                    print(f"[DEBUG] Found new file: {new_file}")
            
            elif line.startswith('@@'):
                # Hunk header - continue processing
                continue
            
            elif current_file:
                if line.startswith(' '):
                    # Context line
                    old_content.append(line[1:])
                    new_content.append(line[1:])
                elif line.startswith('-'):
                    # Deleted line
                    old_content.append(line[1:])
                elif line.startswith('+'):
                    # Added line
                    new_content.append(line[1:])
                elif line.startswith('\\'):
                    # Metadata line like "\ No newline at end of file"
                    continue
                else:
                    # End of current file or new file starting
                    if current_file and (old_content or new_content):
                        files[current_file] = {
                            'old_content': '\n'.join(old_content),
                            'new_content': '\n'.join(new_content)
                        }
                        print(f"[DEBUG] Saved file {current_file} via unified parser")
                    current_file = None
                    old_content = []
                    new_content = []
        
        # Save last file
        if current_file and (old_content or new_content):
            files[current_file] = {
                'old_content': '\n'.join(old_content),
                'new_content': '\n'.join(new_content)
            }
            print(f"[DEBUG] Saved final file {current_file} via unified parser")
        
        return files

# Backward compatibility functions
def is_diff_format(content: str) -> bool:
    """Backward compatible function"""
    return GitDiffExtractor.is_diff_format(content)

def extract_file_content_from_diff(diff_content: str) -> str:
    """Backward compatible function"""
    return GitDiffExtractor.extract_file_content_from_diff(diff_content)