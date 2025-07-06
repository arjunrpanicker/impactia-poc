from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import re
from dotenv import load_dotenv

from .services.rag_service import RAGService
from .services.azure_openai_service import AzureOpenAIService
from .services.ado_service import AzureDevOpsService
from .models.analysis import ChangeAnalysisRequest, ChangeAnalysisResponse, ChangeAnalysisRequestForm, CodeChange, ChangeAnalysisResponseWithCode
from .utils.diff_utils import is_diff_format, extract_file_content_from_diff

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Code Change Impact Analysis API",
    description="API for analyzing code changes and their impact using RAG and Azure OpenAI",
    version="1.0.0"
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
rag_service = RAGService()
openai_service = AzureOpenAIService()

# Conditionally initialize ADO service
ENABLE_ADO_INTEGRATION = os.getenv("ENABLE_ADO_INTEGRATION", "false").lower() == "true"
ado_service = AzureDevOpsService() if ENABLE_ADO_INTEGRATION else None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "features": {
            "ado_integration": ENABLE_ADO_INTEGRATION
        }
    }

@app.post("/analyze", response_model=ChangeAnalysisResponseWithCode)
async def analyze_changes(
    request: ChangeAnalysisRequestForm = Depends()
):
    """
    Analyze code changes and their impact using multipart form data
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

        # 1. Get related code from RAG
        related_code = await rag_service.get_related_code(changes)
        
        # 2. Analyze impact using Azure OpenAI
        analysis_result = await openai_service.analyze_impact(
            changes,
            related_code
        )
        
        # 3. Format response
        response = ChangeAnalysisResponseWithCode(
            summary=analysis_result.summary,
            changed_components=analysis_result.changed_components,
            dependency_chains=analysis_result.dependency_chains,
            dependency_chain_visualization=analysis_result.dependency_chain_visualization,
            risk_level=analysis_result.risk_level,
            code=related_code
        )
        
        # 4. Update ADO if requested and enabled
        if request.update_ado and ENABLE_ADO_INTEGRATION and request.ado_item_id:
            await ado_service.update_work_item(
                request.ado_item_id,
                response
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_repository(file: UploadFile = File(...)):
    """
    Index repository code into Supabase
    """
    try:
        # Process and index the repository code
        result = await rag_service.index_repository(file)
        return {"status": "success", "indexed_files": result.indexed_files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 