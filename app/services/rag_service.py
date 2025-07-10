import os
import zipfile
import tempfile
import asyncio
from typing import Dict, Any, List, Tuple
from supabase import create_client, Client
from fastapi import UploadFile
from ..models.analysis import IndexingResult, CodeChange
from ..services.azure_openai_service import AzureOpenAIService

class RAGService:
    def __init__(self):
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL", ""),
            os.getenv("SUPABASE_KEY", "")
        )
        self.openai_service = AzureOpenAIService()
        self.batch_size = 20  # Increased batch size
        self.skip_tests = True

    async def index_repository(self, file: UploadFile) -> IndexingResult:
        """Index repository code into Supabase vector store"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save and extract zip
                zip_path = os.path.join(temp_dir, "repo.zip")
                with open(zip_path, "wb") as f:
                    content = await file.read()
                    f.write(content)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Get all code files
                code_files = []
                for root, _, files in os.walk(temp_dir):
                    for filename in files:
                        if self._should_skip_file(filename, root):
                            continue
                        file_path = os.path.join(root, filename)
                        try:
                            # Quick check if file is readable and not empty
                            if os.path.getsize(file_path) > 0:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    f.readline()  # Try reading first line
                                code_files.append((file_path, os.path.relpath(file_path, temp_dir)))
                        except (IOError, UnicodeDecodeError):
                            continue

                total_files = len(code_files)
                print(f"Found {total_files} code files to process...")

                # Process files in batches
                indexed_files = 0
                total_methods = 0
                embedding_count = 0

                for i in range(0, len(code_files), self.batch_size):
                    batch = code_files[i:i + self.batch_size]
                    results = await asyncio.gather(
                        *[self._process_file(file_path, relative_path) 
                          for file_path, relative_path in batch],
                        return_exceptions=True
                    )

                    # Process results
                    for result in results:
                        if isinstance(result, Exception):
                            continue
                        if result:
                            indexed_files += 1
                            total_methods += result[0]
                            embedding_count += result[1]

                    print(f"Processed {indexed_files}/{total_files} files...")

                return IndexingResult(
                    indexed_files=indexed_files,
                    total_methods=total_methods,
                    embedding_count=embedding_count
                )

        except Exception as e:
            raise Exception(f"Failed to index repository: {str(e)}")

    def _should_skip_file(self, filename: str, root: str) -> bool:
        """Determine if a file should be skipped"""
        # Skip hidden files
        if filename.startswith('.'):
            return True

        # Skip test files if configured, but be more precise
        if self.skip_tests and any(pattern in filename.lower() for pattern in [
            'test.', '.test.', '.spec.', '.tests.',
            'mock.', '.mock.', '.fixture.'
        ]):
            return True

        # Skip binary and large files
        skip_extensions = {
            # Binary files
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
            '.woff', '.woff2', '.ttf', '.eot', '.otf',
            '.mp4', '.mp3', '.wav', '.avi', '.mov',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.rar', '.7z', '.tar', '.gz',
            '.exe', '.dll', '.so', '.dylib',
            # Generated files
            '.min.js', '.min.css', '.map',
            # Large data files
            '.csv', '.dat', '.log', '.dump'
        }
        
        if any(filename.lower().endswith(ext) for ext in skip_extensions):
            return True

        # Skip only specific binary and generated file patterns
        skip_patterns = {
            'node_modules/',
            '/dist/',
            '/build/',
            '/bin/',
            '/obj/',
            '/vendor/',
            '/third_party/',
            '/.git/',
            '/.vs/',
            '/.idea/',
            '/wwwroot/fonts/',
            '/wwwroot/images/',
            '/wwwroot/css/',
            '/wwwroot/js/'
        }
        
        normalized_path = root.replace('\\', '/').lower()
        return any(pattern in normalized_path + '/' for pattern in skip_patterns)

    async def _process_file(self, file_path: str, relative_path: str) -> Tuple[int, int]:
        """Process a single file and return (methods_count, embeddings_count)"""
        try:
            # Check file size (skip if too large)
            file_size = os.path.getsize(file_path)
            if file_size > 100 * 1024:  # Skip files larger than 100KB
                print(f"Skipping large file {relative_path} ({file_size/1024:.1f}KB)")
                return 0, 0

            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                        # Check if content is binary (contains null bytes or too many non-printable chars)
                        if '\x00' in content or sum(not c.isprintable() for c in content) > len(content) * 0.3:
                            print(f"Skipping binary file {relative_path}")
                            return 0, 0
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error reading file {relative_path} with {encoding}: {str(e)}")
                    continue
            
            if content is None:
                print(f"Could not read file {relative_path} with any encoding")
                return 0, 0

            # Extract methods if it's a source code file
            methods = []
            if any(relative_path.lower().endswith(ext) for ext in ['.py', '.js', '.ts', '.cs', '.java', '.cpp', '.go']):
                methods = self._extract_methods(content)
            
            # Generate embedding for the file content
            try:
                # Truncate content if too long (approximately 6000 tokens)
                if len(content) > 24000:  # Rough estimate: 1 token ≈ 4 characters
                    content = content[:24000] + "\n... (content truncated due to length)"
                
                file_embedding = await self.openai_service.get_embeddings(content)
            except Exception as e:
                print(f"Error generating embedding for {relative_path}: {str(e)}")
                return 0, 0

            # Store file embedding with methods in metadata
            await self._store_embeddings(
                embeddings=file_embedding,
                metadata={
                    "type": "file",
                    "path": relative_path,
                    "size": len(content),
                    "methods": [
                        {
                            "name": method["name"],
                            "content": method["content"],
                            "start_line": method.get("start_line", 0)
                        }
                        for method in methods
                    ],
                    "file_type": os.path.splitext(relative_path)[1].lower()
                },
                content=content,
                file_path=relative_path,
                code_type="file"
            )

            return len(methods), 1

        except Exception as e:
            print(f"Error processing file {relative_path}: {str(e)}")
            return 0, 0

    def _is_code_file(self, filename: str) -> bool:
        """Check if the file is a code file based on extension"""
        code_extensions = {
            # Source code files
            '.py', '.pyx', '.pyi',  # Python
            '.js', '.ts', '.jsx', '.tsx',  # JavaScript/TypeScript
            '.cs', '.vb', '.fs', '.fsx', '.fsi',  # .NET
            '.java', '.scala', '.kt', '.groovy',  # JVM
            '.cpp', '.hpp', '.cc', '.hh',  # C++
            '.c', '.h',  # C
            '.go',  # Go
            '.rs',  # Rust
            '.rb',  # Ruby
            '.php',  # PHP
            '.swift',  # Swift
            '.m', '.mm',  # Objective-C
            
            # Web files
            '.html', '.htm', '.css', '.scss', '.sass', '.less',
            '.vue', '.svelte', '.astro',
            '.xaml', '.razor', '.cshtml', '.aspx', '.ascx',
            
            # Shell and scripts
            '.sh', '.bash', '.zsh', '.fish',
            '.ps1', '.psm1', '.psd1',  # PowerShell
            '.bat', '.cmd',  # Windows batch
            
            # Data and config files
            '.json', '.yaml', '.yml', '.toml', '.ini',
            '.xml', '.xsd', '.dtd',
            '.proto',  # Protocol Buffers
            '.graphql', '.gql',  # GraphQL
            
            # Project files
            '.csproj', '.vbproj', '.fsproj',  # .NET projects
            '.sln',  # Solution files
            '.proj', '.targets', '.props',  # MSBuild files
            '.config', '.settings',  # Configuration
            '.dockerfile', 'dockerfile',  # Docker
            '.cmake', '.cmake.in',  # CMake
            
            # Documentation
            '.md', '.rst', '.txt',  # Documentation
            
            # Template files
            '.template', '.tmpl', '.j2', '.jinja2',
            
            # Database
            '.sql', '.psql', '.pgsql', '.mysql',
            '.mongo', '.mongodb',
            
            # Other
            '.gradle',  # Gradle scripts
            '.rake',  # Rake files
            '.r', '.rmd',  # R
            '.dart',  # Dart
            '.elm',  # Elm
            '.ex', '.exs',  # Elixir
            '.erl',  # Erlang
            '.hs',  # Haskell
            '.lua',  # Lua
            '.pl', '.pm',  # Perl
        }
        return any(filename.lower().endswith(ext) for ext in code_extensions)

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

    async def _store_embeddings(self, embeddings: List[float], metadata: Dict[str, Any], content: str, file_path: str, code_type: str):
        """Store embeddings and metadata in Supabase"""
        try:
            data = {
                "embedding": embeddings,
                "metadata": metadata,
                "content": content,
                "file_path": file_path,
                "code_type": code_type,
                "repository": "main"  # TODO: Make this configurable
            }
            
            result = self.supabase.table("code_embeddings").insert(data).execute()
            return result
            
        except Exception as e:
            raise Exception(f"Failed to store embeddings: {str(e)}")

    async def get_related_code(self, changes: List[CodeChange]) -> Dict[str, Any]:
        """Get related code for the given changes"""
        try:
            # Step 1: Get direct file changes
            changed_files = set()
            combined_text = ""
            for change in changes:
                changed_files.add(change.file_path)
                if change.diff:
                    combined_text += f"File: {change.file_path}\nDiff:\n{change.diff}\n\n"
                elif change.content:
                    combined_text += f"File: {change.file_path}\nContent:\n{change.content}\n\n"

            if not combined_text:
                raise ValueError("No valid content or diff found in changes")

            # Step 2: Generate embedding for the changes
            change_embedding = self.openai_service.get_embeddings(combined_text)
            
            # Step 3: Search for semantically similar code with a lower threshold
            similar_code = await self._search_similar(change_embedding, limit=10, threshold=0.6)
            
            # Step 4: Search for direct references and dependencies
            reference_results = await self._search_references(list(changed_files))
            
            # Step 5: Combine and format the results
            return {
                "changed_files": list(changed_files),
                "direct_dependencies": {
                    "incoming": reference_results.get("incoming_refs", []),
                    "outgoing": reference_results.get("outgoing_refs", [])
                },
                "dependency_chains": reference_results.get("dependency_chains", []),
                "dependency_visualization": reference_results.get("dependency_visualization", []),
                "similar_code": {
                    "files": [
                        {
                            "path": item["metadata"].get("path", ""),
                            "content": item["content"],
                            "similarity": item["similarity"],
                            "methods": item["metadata"].get("methods", [])
                        }
                        for item in similar_code
                        if item["metadata"].get("type") == "file"
                    ],
                    "methods": [
                        {
                            "name": item["metadata"].get("name", ""),
                            "file_path": item["metadata"].get("path", ""),
                            "content": item["content"],
                            "similarity": item["similarity"]
                        }
                        for item in similar_code
                        if item["metadata"].get("type") == "method"
                    ]
                }
            }
            
        except Exception as e:
            raise Exception(f"Failed to get related code: {str(e)}")

    async def _search_similar(self, query_embedding: List[float], limit: int = 5, threshold: float = 0.7):
        """Search for similar code using vector similarity"""
        try:
            result = self.supabase.rpc(
                "match_code_embeddings",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": threshold,
                    "match_count": limit
                }
            ).execute()
            
            return result.data
            
        except Exception as e:
            raise Exception(f"Failed to search similar code: {str(e)}")

    async def _search_references(self, file_paths: List[str]) -> Dict[str, Any]:
        """Search for direct references to and from the changed files"""
        try:
            # Get all code embeddings that might contain references
            result = self.supabase.table("code_embeddings").select("*").execute()
            
            incoming_refs = []  # Files that reference the changed files
            outgoing_refs = []  # Files that are referenced by the changed files
            
            for item in result.data:
                content = item.get("content", "").lower()
                current_path = item.get("file_path", "")
                methods = item.get("metadata", {}).get("methods", [])
                
                # Skip the changed files themselves when looking for incoming references
                if current_path not in file_paths:
                    # Look for incoming references to changed files
                    for changed_file in file_paths:
                        if changed_file.lower() in content:
                            # Find the specific methods that reference the changed file
                            referencing_methods = []
                            for method in methods:
                                method_content = method.get("content", "").lower()
                                if changed_file.lower() in method_content:
                                    referencing_methods.append({
                                        "name": method.get("name", ""),
                                        "summary": f"References {changed_file}"
                                    })
                            
                            if referencing_methods:
                                incoming_refs.append({
                                    "file_path": current_path,
                                    "methods": referencing_methods
                                })
                            break
                else:
                    # For changed files, look for outgoing references
                    for other_item in result.data:
                        other_path = other_item.get("file_path", "")
                        if other_path != current_path and other_path.lower() in content:
                            # Find the specific methods that make the reference
                            referencing_methods = []
                            for method in methods:
                                method_content = method.get("content", "").lower()
                                if other_path.lower() in method_content:
                                    referencing_methods.append({
                                        "name": method.get("name", ""),
                                        "summary": f"References {other_path}"
                                    })
                            
                            if referencing_methods:
                                outgoing_refs.append({
                                    "file_path": other_path,
                                    "methods": referencing_methods
                                })
            
            # Create dependency chains
            dependency_chains = []
            dependency_visualization = []
            
            # Build chains from both incoming and outgoing references
            for changed_file in file_paths:
                chain = {
                    "file_path": changed_file,
                    "methods": [],  # Will be populated by the analyzer
                    "dependent_files": []
                }
                
                # Add incoming references to the chain
                for ref in incoming_refs:
                    if any(method.get("summary", "").lower().find(changed_file.lower()) != -1 
                          for method in ref.get("methods", [])):
                        chain["dependent_files"].append(ref)
                        dependency_visualization.append(f"{ref['file_path']}->{changed_file}")
                
                # Add outgoing references to the chain
                outgoing_for_file = [ref for ref in outgoing_refs 
                                   if changed_file.lower() in ref.get("file_path", "").lower()]
                for ref in outgoing_for_file:
                    chain["dependent_files"].append(ref)
                    dependency_visualization.append(f"{changed_file}->{ref['file_path']}")
                
                if chain["dependent_files"]:
                    dependency_chains.append(chain)
            
            return {
                "incoming_refs": [ref["file_path"] for ref in incoming_refs],
                "outgoing_refs": [ref["file_path"] for ref in outgoing_refs],
                "dependency_chains": dependency_chains,
                "dependency_visualization": dependency_visualization
            }
            
        except Exception as e:
            raise Exception(f"Failed to search references: {str(e)}") 