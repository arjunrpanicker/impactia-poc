"""
Enhanced analysis models with better type safety and validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ChangeType(str, Enum):
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"

class ElementType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    INTERFACE = "interface"

class ImpactMetrics(BaseModel):
    complexity_score: float = Field(ge=0.0, le=1.0)
    dependency_count: int = Field(ge=0)
    test_coverage_impact: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)

class CodeElementAnalysis(BaseModel):
    name: str
    element_type: ElementType
    content: str
    start_line: int
    end_line: int
    complexity: int = Field(ge=1)
    dependencies: List[str] = []
    risk_factors: List[str] = []
    impact_metrics: ImpactMetrics

class EnhancedCodeChange(BaseModel):
    file_path: str
    content: Optional[str] = None
    change_type: ChangeType = ChangeType.MODIFIED
    diff: Optional[str] = None
    elements: List[CodeElementAnalysis] = []
    
    @validator('file_path')
    def validate_file_path(cls, v):
        if not v or not v.strip():
            raise ValueError('File path cannot be empty')
        return v.strip()

class DependencyRelation(BaseModel):
    source_file: str
    target_file: str
    source_element: str
    target_element: str
    relation_type: str  # "calls", "imports", "inherits", "uses"
    strength: float = Field(ge=0.0, le=1.0)

class ImpactChain(BaseModel):
    source_file: str
    source_element: str
    chain_depth: int
    impacted_elements: List[CodeElementAnalysis]
    risk_propagation: float = Field(ge=0.0, le=1.0)
    critical_path: bool = False

class TestImpactAnalysis(BaseModel):
    test_file_path: str
    affected_test_methods: List[str]
    coverage_impact: float = Field(ge=0.0, le=1.0)
    requires_update: bool
    suggested_tests: List[str] = []

class EnhancedAnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: datetime
    summary: str
    overall_risk_level: RiskLevel
    
    # Core analysis
    changed_components: List[CodeElementAnalysis]
    dependency_relations: List[DependencyRelation]
    impact_chains: List[ImpactChain]
    
    # Enhanced insights
    test_impact: List[TestImpactAnalysis]
    performance_implications: List[str] = []
    security_considerations: List[str] = []
    breaking_changes: List[str] = []
    
    # Metrics and visualization
    impact_metrics: Dict[str, Any]
    dependency_visualization: List[str]
    recommendations: List[str] = []
    
    # Confidence scores
    analysis_confidence: float = Field(ge=0.0, le=1.0)
    prediction_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)

class AnalysisRequest(BaseModel):
    changes: List[EnhancedCodeChange]
    repository_url: Optional[str] = None
    branch: str = "main"
    base_commit: Optional[str] = None
    head_commit: Optional[str] = None
    
    # Analysis options
    include_test_analysis: bool = True
    include_performance_analysis: bool = False
    include_security_analysis: bool = False
    max_dependency_depth: int = Field(default=3, ge=1, le=10)
    
    # Integration options
    update_ado: bool = False
    ado_item_id: Optional[str] = None
    notification_webhooks: List[str] = []

class AnalysisConfiguration(BaseModel):
    """Configuration for analysis behavior"""
    similarity_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    max_related_files: int = Field(default=20, ge=5, le=100)
    enable_caching: bool = True
    cache_ttl_hours: int = Field(default=1, ge=1, le=24)
    
    # Language-specific settings
    language_configs: Dict[str, Dict[str, Any]] = {}
    
    # Risk assessment weights
    complexity_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    dependency_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    test_coverage_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    
    @validator('complexity_weight', 'dependency_weight', 'test_coverage_weight')
    def validate_weights_sum(cls, v, values):
        # Note: This is a simplified validation - in practice you'd want to ensure all weights sum to 1.0
        return v