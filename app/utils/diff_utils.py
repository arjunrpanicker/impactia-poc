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
        files = {}
        current_file = None
        current_old_content = []
        current_new_content = []
        
        lines = diff_content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # File header
            if line.startswith('diff --git'):
                # Save previous file
                if current_file:
                    files[current_file] = {
                        'old_content': '\n'.join(current_old_content),
                        'new_content': '\n'.join(current_new_content)
                    }
                
                # Extract new file path
                match = re.search(r'diff --git a/(.*?) b/(.*)', line)
                if match:
                    current_file = match.group(2)
                    current_old_content = []
                    current_new_content = []
            
            # Skip metadata lines
            elif (line.startswith('index ') or 
                  line.startswith('--- a/') or 
                  line.startswith('+++ b/') or
                  line.startswith('new file mode') or
                  line.startswith('deleted file mode')):
                pass
            
            # Hunk header
            elif line.startswith('@@'):
                # Process hunk content
                i += 1
                while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('diff --git'):
                    hunk_line = lines[i]
                    
                    if hunk_line.startswith(' '):  # Context line
                        current_old_content.append(hunk_line[1:])
                        current_new_content.append(hunk_line[1:])
                    elif hunk_line.startswith('-'):  # Deleted line
                        current_old_content.append(hunk_line[1:])
                    elif hunk_line.startswith('+'):  # Added line
                        current_new_content.append(hunk_line[1:])
                    else:  # Handle edge cases
                        current_old_content.append(hunk_line)
                        current_new_content.append(hunk_line)
                    
                    i += 1
                continue
            
            i += 1
        
        # Save last file
        if current_file:
            files[current_file] = {
                'old_content': '\n'.join(current_old_content),
                'new_content': '\n'.join(current_new_content)
            }
        
        return files

    @staticmethod
    def extract_file_content_from_diff(diff_content: str, prefer_new: bool = True) -> str:
        """Extract file content from diff (backward compatible)"""
        files = GitDiffExtractor.extract_file_changes(diff_content)
        
        if not files:
            return diff_content  # Return original if parsing fails
        
        # Get the first file's content
        first_file = next(iter(files.values()))
        
        if prefer_new:
            return first_file.get('new_content', '')
        else:
            return first_file.get('old_content', '')

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

# Backward compatibility functions
def is_diff_format(content: str) -> bool:
    """Backward compatible function"""
    return GitDiffExtractor.is_diff_format(content)

def extract_file_content_from_diff(diff_content: str) -> str:
    """Backward compatible function"""
    return GitDiffExtractor.extract_file_content_from_diff(diff_content)