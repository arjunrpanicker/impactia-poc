from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import re
from dotenv import load_dotenv

from .services.enhanced_rag_service import EnhancedRAGService
from .services.azure_openai_service import AzureOpenAIService
from .services.ado_service import AzureDevOpsService
from .services.performance_analyzer import PerformanceAnalyzer
from .models.analysis import ChangeAnalysisRequest, ChangeAnalysisResponse, ChangeAnalysisRequestForm, CodeChange, ChangeAnalysisResponseWithCode
from .models.enhanced_analysis import AnalysisRequest, EnhancedAnalysisResponse, AnalysisConfiguration
from .utils.diff_utils import is_diff_format, extract_file_content_from_diff

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Enhanced Code Change Impact Analysis API",
    description="Advanced API for analyzing code changes and their impact using RAG, dependency analysis, and Azure OpenAI",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
enhanced_rag_service = EnhancedRAGService()
rag_service = enhanced_rag_service  # Backward compatibility
openai_service = AzureOpenAIService()
performance_analyzer = PerformanceAnalyzer()

# Conditionally initialize ADO service
ENABLE_ADO_INTEGRATION = os.getenv("ENABLE_ADO_INTEGRATION", "false").lower() == "true"
ado_service = AzureDevOpsService() if ENABLE_ADO_INTEGRATION else None

# Global configuration
analysis_config = AnalysisConfiguration()

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features": {
            "ado_integration": ENABLE_ADO_INTEGRATION,
            "enhanced_rag": True,
            "dependency_analysis": True,
            "performance_analysis": True,
            "caching": analysis_config.enable_caching
        }
    }

@app.post("/analyze/enhanced", response_model=EnhancedAnalysisResponse)
async def analyze_changes_enhanced(
    request: AnalysisRequest
):
    """
    Enhanced analysis endpoint with comprehensive impact analysis
    """
    try:
        # Validate ADO integration request
        if request.update_ado and not ENABLE_ADO_INTEGRATION:
            raise HTTPException(
                status_code=400,
                detail="Azure DevOps integration is disabled. Set ENABLE_ADO_INTEGRATION=true to enable it."
            )

        # Convert to legacy format for processing
        legacy_changes = []
        for change in request.changes:
            legacy_changes.append(CodeChange(
                file_path=change.file_path,
                content=change.content,
                diff=change.diff,
                change_type=change.change_type.value
            ))

        # 1. Get enhanced related code analysis
        related_code = await enhanced_rag_service.get_enhanced_related_code(legacy_changes)
        
        # 2. Analyze impact using Azure OpenAI
        analysis_result = await openai_service.analyze_impact(
            legacy_changes,
            related_code
        )
        
        # 3. Performance analysis (if enabled)
        performance_impact = []
        if request.include_performance_analysis:
            perf_analysis = performance_analyzer.analyze_performance_impact([
                {"file_path": change.file_path, "content": change.content}
                for change in request.changes if change.content
            ])
            performance_impact = perf_analysis.get("recommendations", [])
        
        # 4. Build enhanced response
        from datetime import datetime
        import uuid
        
        response = EnhancedAnalysisResponse(
            analysis_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            summary=analysis_result.summary,
            overall_risk_level=analysis_result.risk_level or "medium",
            changed_components=analysis_result.changed_components,
            dependency_relations=[],  # TODO: Convert from dependency_chains
            impact_chains=[],  # TODO: Convert from dependency_chains
            test_impact=[],  # TODO: Extract from related_code
            performance_implications=performance_impact,
            security_considerations=[],  # TODO: Add security analysis
            breaking_changes=[],  # TODO: Detect breaking changes
            impact_metrics=related_code.get("graph_metrics", {}),
            dependency_visualization=analysis_result.dependency_chain_visualization or [],
            recommendations=[],  # TODO: Generate recommendations
            analysis_confidence=0.85  # TODO: Calculate based on data quality
        )
        
        # 5. Update ADO if requested and enabled
        if request.update_ado and ENABLE_ADO_INTEGRATION and request.ado_item_id:
            # Convert to legacy format for ADO update
            legacy_response = ChangeAnalysisResponse(
                summary=analysis_result.summary,
                changed_components=[],  # TODO: Convert
                dependency_chains=analysis_result.dependency_chains,
                dependency_chain_visualization=analysis_result.dependency_chain_visualization,
                risk_level=analysis_result.risk_level
            )
            await ado_service.update_work_item(request.ado_item_id, legacy_response)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=ChangeAnalysisResponseWithCode)
async def analyze_changes(
    request: ChangeAnalysisRequestForm = Depends()
):
    """
    Legacy analysis endpoint (maintained for backward compatibility)
    """
    try:
        # Validate ADO integration request
        if request.update_ado and not ENABLE_ADO_INTEGRATION:
            raise HTTPException(
                status_code=400,
                detail="Azure DevOps integration is disabled. Set ENABLE_ADO_INTEGRATION=true to enable it."
            )

        # Process uploaded files
        changes = []
        for i, file in enumerate(request.files):
            content = await file.read()
            change_type = request.change_types[i] if i < len(request.change_types) else "modified"
            
            # Try to decode as text
            try:
                text_content = content.decode('utf-8')
                
                # Check if it's a diff format
                if is_diff_format(text_content):
                    # Extract actual file content from diff
                    file_content = extract_file_content_from_diff(text_content)
                    changes.append(CodeChange(
                        file_path=file.filename,
                        content=file_content,
                        diff=text_content,
                        change_type=change_type
                    ))
                else:
                    # Regular file content
                    changes.append(CodeChange(
                        file_path=file.filename,
                        content=text_content,
                        diff=None,
                        change_type=change_type
                    ))
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a valid text file"
                )

        # Use enhanced RAG service
        related_code = await enhanced_rag_service.get_enhanced_related_code(changes)
        
        # Analyze impact using Azure OpenAI
        analysis_result = await openai_service.analyze_impact(
            changes,
            related_code
        )
        
        # Format response
        response = ChangeAnalysisResponseWithCode(
            summary=analysis_result.summary,
            changed_components=analysis_result.changed_components,
            dependency_chains=analysis_result.dependency_chains,
            dependency_chain_visualization=analysis_result.dependency_chain_visualization,
            risk_level=analysis_result.risk_level,
            code=related_code
        )
        
        # Update ADO if requested and enabled
        if request.update_ado and ENABLE_ADO_INTEGRATION and request.ado_item_id:
            # Convert to legacy format for ADO
            legacy_response = ChangeAnalysisResponse(
                summary=analysis_result.summary,
                changed_components=[],  # TODO: Convert from enhanced format
                dependency_chains=analysis_result.dependency_chains,
                dependency_chain_visualization=analysis_result.dependency_chain_visualization,
                risk_level=analysis_result.risk_level
            )
            await ado_service.update_work_item(request.ado_item_id, legacy_response)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_repository(file: UploadFile = File(...)):
    """
    Index repository code into Supabase (enhanced with better parsing)
    """
    try:
        # Use enhanced RAG service for indexing
        result = await enhanced_rag_service.index_repository(file)
        return {"status": "success", "indexed_files": result.indexed_files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/performance")
async def analyze_performance(
    files: List[UploadFile] = File(...)
):
    """
    Dedicated performance analysis endpoint
    """
    try:
        changes = []
        for file in files:
            content = await file.read()
            text_content = content.decode('utf-8')
            changes.append({
                "file_path": file.filename,
                "content": text_content
            })
        
        result = performance_analyzer.analyze_performance_impact(changes)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_configuration():
    """Get current analysis configuration"""
    return analysis_config.dict()

@app.post("/config")
async def update_configuration(config: AnalysisConfiguration):
    """Update analysis configuration"""
    global analysis_config
    analysis_config = config
    return {"status": "Configuration updated successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)