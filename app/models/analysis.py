from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum
from fastapi import UploadFile, Form

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MethodChange(BaseModel):
    name: str
    summary: str

class DependentMethod(BaseModel):
    name: str
    summary: str

class DependentFile(BaseModel):
    file_path: str
    methods: List[DependentMethod]

class CodeChange(BaseModel):
    file_path: str
    content: Optional[str] = None
    change_type: str = "modified"  # "added", "modified", "deleted"
    diff: Optional[str] = None

class ChangeAnalysisRequestForm:
    def __init__(
        self,
        files: List[UploadFile],
        change_types: List[str] = Form(default=["modified"]),
        repository_url: Optional[str] = Form(default=None),
        branch: Optional[str] = Form(default=None),
        update_ado: bool = Form(default=False),
        ado_item_id: Optional[str] = Form(default=None),
        base_commit: Optional[str] = Form(default=None),
        head_commit: Optional[str] = Form(default=None)
    ):
        self.files = files
        self.change_types = change_types
        self.repository_url = repository_url
        self.branch = branch
        self.update_ado = update_ado
        self.ado_item_id = ado_item_id
        self.base_commit = base_commit
        self.head_commit = head_commit

class ChangeAnalysisRequest(BaseModel):
    changes: List[CodeChange]  # Now accepts multiple file changes
    repository_url: Optional[str] = None
    branch: Optional[str] = None
    update_ado: bool = False
    ado_item_id: Optional[str] = None
    base_commit: Optional[str] = None  # For git diff comparison
    head_commit: Optional[str] = None  # For git diff comparison

class MethodWithCode(BaseModel):
    name: str
    code: str

class ChangedComponentWithCode(BaseModel):
    file_path: str
    methods: List[MethodWithCode]
    impact_description: str
    risk_level: RiskLevel
    associated_unit_tests: List[str]

class DependentMethodWithSummary(BaseModel):
    name: str
    summary: str

class DependentFileWithCode(BaseModel):
    file_path: str
    methods: List[DependentMethodWithSummary]
    code: str

class DependencyChainWithCode(BaseModel):
    file_path: str
    methods: List[DependentMethodWithSummary]
    impacted_files: List[DependentFileWithCode]
    associated_unit_tests: List[str]

class ChangeAnalysisResponseWithCode(BaseModel):
    summary: str
    changed_components: List[ChangedComponentWithCode]
    dependency_chains: Optional[List[DependencyChainWithCode]] = None
    dependency_chain_visualization: Optional[List[str]] = None
    risk_level: Optional[RiskLevel] = None

class ChangedComponent(BaseModel):
    file_path: str
    methods: List[str]
    impact_description: str
    risk_level: RiskLevel
    associated_unit_tests: List[str]  # Full paths to associated unit test files

class DependencyChain(BaseModel):
    file_path: str
    methods: List[DependentMethod]
    impacted_files: List[DependentFile]
    associated_unit_tests: List[str]  # Full paths to associated unit test files

class ChangeAnalysisResponse(BaseModel):
    summary: str
    changed_components: List[ChangedComponent]
    dependency_chains: Optional[List[DependencyChain]] = None
    dependency_chain_visualization: Optional[List[str]] = None  # Format: ["file1.cs->file2.cs"]
    risk_level: Optional[RiskLevel] = None

class IndexingResult(BaseModel):
    indexed_files: int
    total_methods: int
    embedding_count: int 