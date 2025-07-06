"""
Enhanced dependency analysis service
"""
import networkx as nx
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from ..services.code_parser import CodeParser, CodeElement

@dataclass
class DependencyNode:
    file_path: str
    element_name: str
    element_type: str
    dependencies: List[str]
    dependents: List[str]
    risk_score: float = 0.0

class DependencyAnalyzer:
    def __init__(self):
        self.parser = CodeParser()
        self.dependency_graph = nx.DiGraph()
        self.file_elements: Dict[str, List[CodeElement]] = {}

    def build_dependency_graph(self, files_content: Dict[str, str]) -> nx.DiGraph:
        """Build a comprehensive dependency graph"""
        # Parse all files
        for file_path, content in files_content.items():
            elements = self.parser.parse_file(file_path, content)
            self.file_elements[file_path] = elements
            
            # Add nodes to graph
            for element in elements:
                node_id = f"{file_path}::{element.name}"
                self.dependency_graph.add_node(
                    node_id,
                    file_path=file_path,
                    element_name=element.name,
                    element_type=element.type,
                    complexity=element.complexity,
                    content=element.content
                )

        # Build edges based on dependencies
        self._build_edges()
        
        return self.dependency_graph

    def _build_edges(self):
        """Build edges between dependent elements"""
        for file_path, elements in self.file_elements.items():
            for element in elements:
                if not element.dependencies:
                    continue
                    
                source_id = f"{file_path}::{element.name}"
                
                # Find matching dependencies in other files
                for dep_name in element.dependencies:
                    target_nodes = self._find_dependency_targets(dep_name, file_path)
                    
                    for target_id in target_nodes:
                        self.dependency_graph.add_edge(source_id, target_id)

    def _find_dependency_targets(self, dep_name: str, source_file: str) -> List[str]:
        """Find potential dependency targets"""
        targets = []
        
        for file_path, elements in self.file_elements.items():
            for element in elements:
                if element.name == dep_name:
                    target_id = f"{file_path}::{element.name}"
                    targets.append(target_id)
                    
        return targets

    def analyze_change_impact(self, changed_files: List[str]) -> Dict[str, Any]:
        """Analyze the impact of changes on the dependency graph"""
        impacted_nodes = set()
        risk_analysis = {}
        
        # Find all nodes in changed files
        changed_nodes = []
        for file_path in changed_files:
            if file_path in self.file_elements:
                for element in self.file_elements[file_path]:
                    node_id = f"{file_path}::{element.name}"
                    changed_nodes.append(node_id)
                    impacted_nodes.add(node_id)

        # Find downstream dependencies
        for node_id in changed_nodes:
            if node_id in self.dependency_graph:
                # Get all nodes that depend on this node (reverse direction)
                dependents = list(self.dependency_graph.predecessors(node_id))
                impacted_nodes.update(dependents)
                
                # Calculate risk scores
                risk_analysis[node_id] = self._calculate_risk_score(node_id)

        # Build impact chains
        impact_chains = self._build_impact_chains(changed_nodes)
        
        # Generate visualization data
        visualization = self._generate_visualization(impacted_nodes)
        
        return {
            "impacted_nodes": list(impacted_nodes),
            "impact_chains": impact_chains,
            "risk_analysis": risk_analysis,
            "visualization": visualization,
            "metrics": self._calculate_metrics(impacted_nodes)
        }

    def _calculate_risk_score(self, node_id: str) -> float:
        """Calculate risk score for a node"""
        if node_id not in self.dependency_graph:
            return 0.0
            
        node_data = self.dependency_graph.nodes[node_id]
        
        # Factors for risk calculation
        complexity_score = min(node_data.get('complexity', 1) / 10.0, 1.0)
        
        # Number of dependents (how many things depend on this)
        dependents_count = len(list(self.dependency_graph.predecessors(node_id)))
        dependents_score = min(dependents_count / 10.0, 1.0)
        
        # Number of dependencies (how many things this depends on)
        dependencies_count = len(list(self.dependency_graph.successors(node_id)))
        dependencies_score = min(dependencies_count / 10.0, 1.0)
        
        # Weighted risk score
        risk_score = (
            complexity_score * 0.3 +
            dependents_score * 0.5 +
            dependencies_score * 0.2
        )
        
        return min(risk_score, 1.0)

    def _build_impact_chains(self, changed_nodes: List[str]) -> List[Dict[str, Any]]:
        """Build detailed impact chains"""
        chains = []
        
        for node_id in changed_nodes:
            if node_id not in self.dependency_graph:
                continue
                
            node_data = self.dependency_graph.nodes[node_id]
            
            # Get all paths from this node (BFS)
            impacted_files = {}
            visited = set()
            queue = [(node_id, 0)]  # (node_id, depth)
            
            while queue:
                current_node, depth = queue.pop(0)
                if current_node in visited or depth > 3:  # Limit depth
                    continue
                    
                visited.add(current_node)
                current_data = self.dependency_graph.nodes[current_node]
                file_path = current_data['file_path']
                
                if file_path not in impacted_files:
                    impacted_files[file_path] = []
                    
                impacted_files[file_path].append({
                    "name": current_data['element_name'],
                    "type": current_data['element_type'],
                    "depth": depth,
                    "risk_score": self._calculate_risk_score(current_node)
                })
                
                # Add predecessors to queue
                for predecessor in self.dependency_graph.predecessors(current_node):
                    queue.append((predecessor, depth + 1))
            
            chains.append({
                "source_file": node_data['file_path'],
                "source_element": node_data['element_name'],
                "impacted_files": impacted_files
            })
            
        return chains

    def _generate_visualization(self, impacted_nodes: Set[str]) -> List[str]:
        """Generate visualization data for dependency chains"""
        visualization = []
        
        for node_id in impacted_nodes:
            if node_id not in self.dependency_graph:
                continue
                
            node_data = self.dependency_graph.nodes[node_id]
            source_file = node_data['file_path']
            
            # Get direct dependencies
            for successor in self.dependency_graph.successors(node_id):
                successor_data = self.dependency_graph.nodes[successor]
                target_file = successor_data['file_path']
                
                if source_file != target_file:
                    viz_entry = f"{source_file}->{target_file}"
                    if viz_entry not in visualization:
                        visualization.append(viz_entry)
                        
        return visualization

    def _calculate_metrics(self, impacted_nodes: Set[str]) -> Dict[str, Any]:
        """Calculate impact metrics"""
        if not impacted_nodes:
            return {}
            
        total_complexity = 0
        file_count = set()
        element_types = {}
        
        for node_id in impacted_nodes:
            if node_id not in self.dependency_graph:
                continue
                
            node_data = self.dependency_graph.nodes[node_id]
            total_complexity += node_data.get('complexity', 0)
            file_count.add(node_data['file_path'])
            
            element_type = node_data['element_type']
            element_types[element_type] = element_types.get(element_type, 0) + 1
            
        return {
            "total_impacted_files": len(file_count),
            "total_impacted_elements": len(impacted_nodes),
            "average_complexity": total_complexity / len(impacted_nodes) if impacted_nodes else 0,
            "element_type_distribution": element_types
        }

    def get_critical_paths(self, source_files: List[str], max_depth: int = 5) -> List[List[str]]:
        """Find critical dependency paths from source files"""
        critical_paths = []
        
        for file_path in source_files:
            if file_path not in self.file_elements:
                continue
                
            for element in self.file_elements[file_path]:
                source_node = f"{file_path}::{element.name}"
                
                if source_node not in self.dependency_graph:
                    continue
                    
                # Find all simple paths from this node
                for target_node in self.dependency_graph.nodes():
                    if source_node == target_node:
                        continue
                        
                    try:
                        paths = list(nx.all_simple_paths(
                            self.dependency_graph, 
                            source_node, 
                            target_node, 
                            cutoff=max_depth
                        ))
                        
                        for path in paths:
                            # Convert node IDs back to file paths
                            file_path_sequence = []
                            for node_id in path:
                                node_data = self.dependency_graph.nodes[node_id]
                                file_path_sequence.append(node_data['file_path'])
                            
                            if len(set(file_path_sequence)) > 1:  # Only multi-file paths
                                critical_paths.append(file_path_sequence)
                                
                    except nx.NetworkXNoPath:
                        continue
                        
        return critical_paths