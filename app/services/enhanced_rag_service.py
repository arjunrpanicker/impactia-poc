"""
Enhanced RAG service with better similarity search and caching
"""
import os
import hashlib
import json
import zipfile
import tempfile
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from supabase import create_client, Client
from fastapi import UploadFile
from ..services.azure_openai_service import AzureOpenAIService
from ..services.code_parser import CodeParser
from ..services.dependency_analyzer import DependencyAnalyzer
from ..models.analysis import CodeChange, IndexingResult
import networkx as nx

class EnhancedRAGService:
    def __init__(self):
        self.supabase: Client = create_client(
            os.getenv("SUPABASE_URL", ""),
            os.getenv("SUPABASE_KEY", "")
        )
        self.openai_service = AzureOpenAIService()
        self.parser = CodeParser()
        self.dependency_analyzer = DependencyAnalyzer()
        self.cache_ttl = timedelta(hours=1)
        self.batch_size = 20
        self.skip_tests = True

    def _generate_cache_key(self, changes: List[CodeChange]) -> str:
        """Generate a cache key for the given changes"""
        try:
            # Create a deterministic key based on file paths and content hashes
            key_parts = []
            for change in changes:
                file_part = change.file_path
                content_hash = hashlib.md5((change.content or "").encode()).hexdigest()[:8]
                diff_hash = hashlib.md5((change.diff or "").encode()).hexdigest()[:8]
                key_parts.append(f"{file_part}:{content_hash}:{diff_hash}")
            
            combined = "|".join(sorted(key_parts))
            return hashlib.sha256(combined.encode()).hexdigest()[:32]
        except Exception as e:
            print(f"[DEBUG] Error generating cache key: {e}")
            # Fallback to timestamp-based key
            return f"fallback_{int(datetime.now().timestamp())}"

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired"""
        try:
            result = self.supabase.table("analysis_cache").select("*").eq("cache_key", cache_key).execute()
            
            if result.data:
                cache_entry = result.data[0]
                expires_at = datetime.fromisoformat(cache_entry["expires_at"].replace('Z', '+00:00'))
                
                if datetime.now(expires_at.tzinfo) < expires_at:
                    print(f"[DEBUG] Cache hit for key: {cache_key}")
                    return cache_entry["result"]
                else:
                    print(f"[DEBUG] Cache expired for key: {cache_key}")
                    # Clean up expired entry
                    self.supabase.table("analysis_cache").delete().eq("cache_key", cache_key).execute()
            
            print(f"[DEBUG] Cache miss for key: {cache_key}")
            return None
            
        except Exception as e:
            print(f"[DEBUG] Error getting cached result: {e}")
            return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any]) -> None:
        """Cache the analysis result"""
        try:
            expires_at = datetime.now() + self.cache_ttl
            
            cache_data = {
                "cache_key": cache_key,
                "result": result,
                "expires_at": expires_at.isoformat()
            }
            
            # Use upsert to handle key conflicts
            self.supabase.table("analysis_cache").upsert(cache_data).execute()
            print(f"[DEBUG] Cached result for key: {cache_key}")
            
        except Exception as e:
            print(f"[DEBUG] Error caching result: {e}")
            # Don't fail the main operation if caching fails
            pass

    async def _find_related_test_files(self, changed_files: List[str]) -> Dict[str, List[str]]:
        """Find test files related to changed files"""
        test_files = {}
        
        for file_path in changed_files:
            # Generate potential test file patterns
            test_patterns = self._generate_test_patterns(file_path)
            
            related_tests = []
            for pattern in test_patterns:
                try:
                    # Search for test files
                    result = self.supabase.table("code_embeddings").select("file_path").ilike(
                        "file_path", pattern
                    ).execute()
                    
                    for item in result.data:
                        test_file = item.get("file_path", "")
                        if test_file and test_file not in related_tests:
                            related_tests.append(test_file)
                except Exception as e:
                    print(f"[DEBUG] Error searching for test pattern {pattern}: {e}")
                    continue
                    
            test_files[file_path] = related_tests
            
        return test_files

    def _generate_test_patterns(self, file_path: str) -> List[str]:
        """Generate potential test file patterns for a given file"""
        patterns = []
        
        # Extract file info
        file_name = os.path.basename(file_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        file_dir = os.path.dirname(file_path)
        
        # Common test patterns
        test_patterns = [
            f"%test%{file_name_no_ext}%",
            f"%{file_name_no_ext}%test%",
            f"%{file_name_no_ext}.test.%",
            f"%{file_name_no_ext}.spec.%",
            f"%test_{file_name_no_ext}%",
            f"%{file_name_no_ext}_test%",
            f"%tests%{file_name_no_ext}%",
            f"%{file_name_no_ext}%tests%",
        ]
        
        # Add directory-based patterns
        if file_dir:
            test_patterns.extend([
                f"%test%{file_dir}%{file_name_no_ext}%",
                f"%tests%{file_dir}%{file_name_no_ext}%",
                f"%{file_dir}%test%{file_name_no_ext}%",
                f"%{file_dir}%tests%{file_name_no_ext}%",
            ])
        
        return test_patterns
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

    async def _process_file(self, file_path: str, relative_path: str) -> tuple[int, int]:
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
                
                file_embedding = self.openai_service.get_embeddings(content)
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

    async def get_enhanced_related_code(self, changes: List[CodeChange]) -> Dict[str, Any]:
        """Get enhanced related code analysis with dependency graph"""
        try:
            print(f"[DEBUG] Enhanced RAG: Processing {len(changes)} changes")
            
            # Generate cache key
            cache_key = self._generate_cache_key(changes)
            
            # Check cache first
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result

            # Also run legacy analysis for comparison and fallback
            try:
                from .rag_service import RAGService
                legacy_rag = RAGService()
                legacy_result = await legacy_rag.get_related_code(changes)
                print(f"[DEBUG] Legacy RAG found: {len(legacy_result.get('similar_code', {}).get('files', []))} similar files")
            except Exception as e:
                print(f"[DEBUG] Legacy RAG failed: {e}")
                legacy_result = {}

            # Build comprehensive analysis
            result = await self._build_comprehensive_analysis(changes)
            
            # Merge with legacy results for better coverage
            if legacy_result:
                result["legacy_analysis"] = legacy_result
                
                # Enhance with legacy findings
                legacy_similar = legacy_result.get("similar_code", {})
                if legacy_similar.get("files"):
                    result.setdefault("similar_files_legacy", legacy_similar["files"])
                if legacy_similar.get("methods"):
                    result.setdefault("similar_methods_legacy", legacy_similar["methods"])
                
                # Add legacy dependency chains
                legacy_deps = legacy_result.get("dependency_chains", [])
                if legacy_deps:
                    result.setdefault("dependency_chains_legacy", legacy_deps)
            
            # Cache the result
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to get enhanced related code: {str(e)}")

    async def get_related_code(self, changes: List[CodeChange]) -> Dict[str, Any]:
        """Legacy compatibility method - delegates to enhanced analysis"""
        try:
            print(f"[DEBUG] Legacy get_related_code called with {len(changes)} changes")
            
            # Use the enhanced method but format for legacy compatibility
            enhanced_result = await self.get_enhanced_related_code(changes)
            
            # Convert enhanced format to legacy format
            legacy_result = {
                "changed_files": enhanced_result.get("changed_files", []),
                "direct_dependencies": {
                    "incoming": [],
                    "outgoing": []
                },
                "dependency_chains": [],
                "dependency_visualization": [],
                "similar_code": {
                    "files": [],
                    "methods": []
                }
            }
            
            # Extract similar files from enhanced result
            related_files = enhanced_result.get("related_files", {})
            for file_path, content in related_files.items():
                legacy_result["similar_code"]["files"].append({
                    "path": file_path,
                    "content": content,
                    "similarity": 0.8,  # Default similarity
                    "methods": []
                })
            
            # Extract semantic similarities
            semantic_similarities = enhanced_result.get("semantic_similarities", {})
            for func_list in semantic_similarities.get("by_function_signature", []):
                legacy_result["similar_code"]["methods"].append({
                    "name": func_list.get("function_name", "unknown"),
                    "file_path": func_list.get("file_path", ""),
                    "content": func_list.get("content", ""),
                    "similarity": 0.8
                })
            
            # Create dependency chains from dependency analysis
            dependency_analysis = enhanced_result.get("dependency_analysis", {})
            impact_chains = dependency_analysis.get("impact_chains", [])
            
            for chain in impact_chains:
                legacy_chain = {
                    "file_path": chain.get("source_file", ""),
                    "methods": [],
                    "dependent_files": []
                }
                
                # Add impacted files
                impacted_files = chain.get("impacted_files", {})
                for file_path, elements in impacted_files.items():
                    dependent_file = {
                        "file_path": file_path,
                        "methods": []
                    }
                    
                    for element in elements:
                        dependent_file["methods"].append({
                            "name": element.get("name", "unknown"),
                            "summary": f"Impacted by changes (depth: {element.get('depth', 0)})"
                        })
                    
                    legacy_chain["dependent_files"].append(dependent_file)
                
                legacy_result["dependency_chains"].append(legacy_chain)
            
            # Add dependency visualization
            visualization = enhanced_result.get("dependency_analysis", {}).get("visualization", [])
            legacy_result["dependency_visualization"] = visualization
            
            # Add direct dependencies
            if impact_chains:
                # Extract incoming and outgoing references
                incoming_refs = set()
                outgoing_refs = set()
                
                for chain in impact_chains:
                    source_file = chain.get("source_file", "")
                    impacted_files = chain.get("impacted_files", {})
                    
                    for file_path in impacted_files.keys():
                        if file_path != source_file:
                            incoming_refs.add(file_path)
                
                legacy_result["direct_dependencies"]["incoming"] = list(incoming_refs)
                legacy_result["direct_dependencies"]["outgoing"] = list(outgoing_refs)
            
            print(f"[DEBUG] Legacy format conversion completed")
            print(f"[DEBUG] Similar files: {len(legacy_result['similar_code']['files'])}")
            print(f"[DEBUG] Similar methods: {len(legacy_result['similar_code']['methods'])}")
            print(f"[DEBUG] Dependency chains: {len(legacy_result['dependency_chains'])}")
            
            return legacy_result
            
        except Exception as e:
            print(f"[DEBUG] Legacy get_related_code failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return minimal legacy format on error
            return {
                "changed_files": [change.file_path for change in changes],
                "direct_dependencies": {"incoming": [], "outgoing": []},
                "dependency_chains": [],
                "dependency_visualization": [],
                "similar_code": {"files": [], "methods": []},
                "error": str(e)
            }

    async def _build_comprehensive_analysis(self, changes: List[CodeChange]) -> Dict[str, Any]:
        """Build comprehensive code analysis"""
        # Step 1: Parse changed files
        changed_files = {}
        for change in changes:
            if change.content:
                changed_files[change.file_path] = change.content

        # Step 2: Get related files from vector search
        related_files = await self._get_related_files_from_vector_search(changes)
        
        print(f"[DEBUG] Found {len(related_files)} related files from vector search")
        
        # Step 3: Build dependency graph
        all_files = {**changed_files, **related_files}
        
        try:
            dependency_graph = self.dependency_analyzer.build_dependency_graph(all_files)
            print(f"[DEBUG] Built dependency graph with {dependency_graph.number_of_nodes()} nodes")
        except Exception as e:
            print(f"[DEBUG] Dependency graph building failed: {e}")
            dependency_graph = None
        
        # Step 4: Analyze impact
        impact_analysis = {}
        if dependency_graph:
            try:
                impact_analysis = self.dependency_analyzer.analyze_change_impact(list(changed_files.keys()))
                print(f"[DEBUG] Impact analysis found {len(impact_analysis.get('impacted_nodes', []))} impacted nodes")
            except Exception as e:
                print(f"[DEBUG] Impact analysis failed: {e}")
        
        # Step 5: Get semantic similarities
        semantic_analysis = await self._get_semantic_similarities(changes)
        
        # Step 6: Find test files
        test_files = await self._find_related_test_files(list(changed_files.keys()))
        
        return {
            "changed_files": list(changed_files.keys()),
            "dependency_analysis": impact_analysis,
            "semantic_similarities": semantic_analysis,
            "related_files": related_files,
            "test_files": test_files,
            "graph_metrics": {
                "total_nodes": dependency_graph.number_of_nodes() if dependency_graph else 0,
                "total_edges": dependency_graph.number_of_edges() if dependency_graph else 0,
                "density": nx.density(dependency_graph) if dependency_graph and dependency_graph.number_of_nodes() > 0 else 0
            }
        }

    async def _get_related_files_from_vector_search(self, changes: List[CodeChange]) -> Dict[str, str]:
        """Get related files using vector similarity search"""
        related_files = {}
        
        for change in changes:
            if not change.content:
                continue
                
            # Generate embedding for the change
            embedding = self.openai_service.get_embeddings(change.content)
            
            # Search for similar files
            similar_results = await self._search_similar_enhanced(
                embedding, 
                limit=15, 
                threshold=0.65,
                exclude_files=[change.file_path]
            )
            
            for result in similar_results:
                file_path = result["metadata"].get("path", "")
                if file_path and file_path not in related_files:
                    related_files[file_path] = result["content"]
                    
        return related_files

    async def _search_similar_enhanced(self, query_embedding: List[float], limit: int = 10, 
                                     threshold: float = 0.7, exclude_files: List[str] = None):
        """Enhanced similarity search with filtering"""
        try:
            # Build exclusion filter
            exclusion_filter = ""
            if exclude_files:
                file_list = "', '".join(exclude_files)
                exclusion_filter = f"AND metadata->>'path' NOT IN ('{file_list}')"
            
            # Use custom query for better filtering
            query = f"""
            SELECT id, content, metadata, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM code_embeddings
            WHERE 1 - (embedding <=> %s::vector) > %s
            {exclusion_filter}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            
            result = self.supabase.rpc(
                "execute_sql",
                {
                    "query": query,
                    "params": [query_embedding, query_embedding, threshold, query_embedding, limit]
                }
            ).execute()
            
            return result.data
            
        except Exception as e:
            # Fallback to original method
            return await self._search_similar_fallback(query_embedding, limit, threshold)

    async def _search_similar_fallback(self, query_embedding: List[float], limit: int, threshold: float):
        """Fallback similarity search"""
        result = self.supabase.rpc(
            "match_code_embeddings",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": limit
            }
        ).execute()
        
        return result.data

    async def _get_semantic_similarities(self, changes: List[CodeChange]) -> Dict[str, Any]:
        """Get semantic similarities using multiple strategies"""
        similarities = {
            "by_function_signature": [],
            "by_variable_usage": [],
            "by_import_patterns": [],
            "by_code_structure": []
        }
        
        for change in changes:
            if not change.content:
                continue
                
            # Parse the changed file
            elements = self.parser.parse_file(change.file_path, change.content)
            
            # Find similar function signatures
            for element in elements:
                if element.type == "function" and element.parameters:
                    similar_functions = await self._find_similar_function_signatures(
                        element.name, element.parameters
                    )
                    similarities["by_function_signature"].extend(similar_functions)
                    
            # Find similar import patterns
            language = self.parser.detect_language(change.file_path, change.content)
            imports = self.parser.extract_imports(change.content, language)
            if imports:
                similar_imports = await self._find_similar_import_patterns(imports)
                similarities["by_import_patterns"].extend(similar_imports)
                
        return similarities

    async def _find_similar_function_signatures(self, func_name: str, parameters: List[str]) -> List[Dict]:
        """Find functions with similar signatures"""
        try:
            # Search for functions with similar names or parameter patterns
            param_pattern = ", ".join(parameters) if parameters else ""
            
            result = self.supabase.table("code_embeddings").select("*").ilike(
                "content", f"%def {func_name}%"
            ).limit(10).execute()
            
            similar_functions = []
            for item in result.data:
                metadata = item.get("metadata", {})
                methods = metadata.get("methods", [])
                
                for method in methods:
                    if method.get("name") == func_name:
                        similar_functions.append({
                            "file_path": metadata.get("path", ""),
                            "function_name": func_name,
                            "content": method.get("content", ""),
                            "similarity_reason": "function_signature"
                        })
                        
            return similar_functions
            
        except Exception as e:
            print(f"Error finding similar function signatures: {str(e)}")
            return []

    async def _find_similar_import_patterns(self, imports: List[str]) -> List[Dict]:
        """Find files with similar import patterns"""
        try:
            similar_files = []
            
            for import_stmt in imports[:5]:  # Limit to first 5 imports
                result = self.supabase.table("code_embeddings").select("*").ilike(
                    "content", f"%{import_stmt}%"
                ).limit(5).execute()
                
                for item in result.data:
                    similar_files.append({
                        "file_path": item.get("file_path", ""),
                        "import_statement": import_stmt,
                        "similarity_reason": "import_pattern"
                    })
                    
            return similar_files
            
        except Exception as e:
            print(f"Error finding similar import patterns: {str(e)}")
            return []

    async def _find_related_test_files(self, changed_files: List[str]) -> Dict[str, List[str]]:
        """Find test files related to changed files"""
        test_files = {}
        
        for file_path in changed_files:
            # Generate potential test file patterns
            test_patterns = self._generate_test_patterns(file_path)
            
            related_tests = []
            for pattern in test_patterns:
                # Search for test files
                result = self.supabase.table("code_embeddings").select("file_path").ilike(
                    "file_path", pattern
                ).execute()
                
                for item in result.data:
                    test_file = item.get("file_path", "")
                    if test_file and test_file not in related_tests:
                        related_tests.append(test_file)
                        