"""
Enhanced RAG service with better similarity search and caching
"""
import os
import hashlib
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from supabase import create_client, Client
from ..services.azure_openai_service import AzureOpenAIService
from ..services.code_parser import CodeParser
from ..services.dependency_analyzer import DependencyAnalyzer
from ..models.analysis import CodeChange

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

    async def get_enhanced_related_code(self, changes: List[CodeChange]) -> Dict[str, Any]:
        """Get enhanced related code analysis with dependency graph"""
        try:
            # Generate cache key
            cache_key = self._generate_cache_key(changes)
            
            # Check cache first
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result

            # Build comprehensive analysis
            result = await self._build_comprehensive_analysis(changes)
            
            # Cache the result
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to get enhanced related code: {str(e)}")

    async def _build_comprehensive_analysis(self, changes: List[CodeChange]) -> Dict[str, Any]:
        """Build comprehensive code analysis"""
        # Step 1: Parse changed files
        changed_files = {}
        for change in changes:
            if change.content:
                changed_files[change.file_path] = change.content

        # Step 2: Get related files from vector search
        related_files = await self._get_related_files_from_vector_search(changes)
        
        # Step 3: Build dependency graph
        all_files = {**changed_files, **related_files}
        dependency_graph = self.dependency_analyzer.build_dependency_graph(all_files)
        
        # Step 4: Analyze impact
        impact_analysis = self.dependency_analyzer.analyze_change_impact(list(changed_files.keys()))
        
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
                "total_nodes": dependency_graph.number_of_nodes(),
                "total_edges": dependency_graph.number_of_edges(),
                "density": nx.density(dependency_graph) if dependency_graph.number_of_nodes() > 0 else 0
            }
        }

    async def _get_related_files_from_vector_search(self, changes: List[CodeChange]) -> Dict[str, str]:
        """Get related files using vector similarity search"""
        related_files = {}
        
        for change in changes:
            if not change.content:
                continue
                
            # Generate embedding for the change
            embedding = await self.openai_service.get_embeddings(change.content)
            
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
                        
            # Also search for files that reference the changed file
            result = self.supabase.table("code_embeddings").select("file_path").ilike(
                "content", f"%{os.path.basename(file_path)}%"
            ).ilike("file_path", "%test%").execute()
            
            for item in result.data:
                test_file = item.get("file_path", "")
                if test_file and test_file not in related_tests:
                    related_tests.append(test_file)
                    
            test_files[file_path] = related_tests
            
        return test_files

    def _generate_test_patterns(self, file_path: str) -> List[str]:
        """Generate potential test file patterns"""
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        dir_name = os.path.dirname(file_path)
        
        patterns = [
            f"%test_{base_name}%",
            f"%{base_name}_test%",
            f"%{base_name}.test%",
            f"%{base_name}.spec%",
            f"%Test{base_name}%",
            f"%{base_name}Test%",
            f"%tests/{base_name}%",
            f"%test/{base_name}%",
            f"%__tests__/{base_name}%"
        ]
        
        return patterns

    def _generate_cache_key(self, changes: List[CodeChange]) -> str:
        """Generate cache key for changes"""
        content_hash = hashlib.md5()
        
        for change in changes:
            content_hash.update(f"{change.file_path}:{change.change_type}".encode())
            if change.content:
                content_hash.update(change.content.encode())
            if change.diff:
                content_hash.update(change.diff.encode())
                
        return f"rag_analysis_{content_hash.hexdigest()}"

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result"""
        try:
            result = self.supabase.table("analysis_cache").select("*").eq(
                "cache_key", cache_key
            ).gte(
                "created_at", (datetime.utcnow() - self.cache_ttl).isoformat()
            ).execute()
            
            if result.data:
                return json.loads(result.data[0]["result"])
                
        except Exception as e:
            print(f"Cache retrieval error: {str(e)}")
            
        return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache analysis result"""
        try:
            data = {
                "cache_key": cache_key,
                "result": json.dumps(result),
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.supabase.table("analysis_cache").upsert(data).execute()
            
        except Exception as e:
            print(f"Cache storage error: {str(e)}")