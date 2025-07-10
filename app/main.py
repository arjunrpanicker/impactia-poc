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
from .services.smart_analysis_service import SmartAnalysisService, AnalysisMethod
from .services.impact_analysis_service import ImpactAnalysisService
from .models.analysis import ChangeAnalysisRequest, ChangeAnalysisResponse, ChangeAnalysisRequestForm, CodeChange, ChangeAnalysisResponseWithCode
from .models.enhanced_analysis import AnalysisRequest, EnhancedAnalysisResponse, AnalysisConfiguration
from .utils.diff_utils import is_diff_format, extract_file_content_from_diff, GitDiffExtractor

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Enhanced Code Change Impact Analysis API",
    description="Advanced API for analyzing code changes and their impact using RAG, dependency analysis, and Azure OpenAI",
    version="2.1.0"
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
smart_analysis_service = SmartAnalysisService()
impact_analysis_service = ImpactAnalysisService()

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
        "version": "2.1.0",
        "features": {
            "ado_integration": ENABLE_ADO_INTEGRATION,
            "enhanced_rag": True,
            "dependency_analysis": True,
            "performance_analysis": True,
            "smart_analysis": True,
            "hybrid_diff_analysis": True,
            "caching": analysis_config.enable_caching
        }
    }

@app.post("/analyze/smart")
async def analyze_changes_smart(
    files: List[UploadFile] = File(...),
    analysis_method: str = Form(default="auto"),  # auto, ast_only, llm_only, hybrid
    include_performance: bool = Form(default=False),
    update_ado: bool = Form(default=False),
    ado_item_id: str = Form(default=None)
):
    """
    Smart analysis endpoint that chooses the best analysis method
    """
    try:
        print(f"[DEBUG] Smart analysis started with {len(files)} files")
        print(f"[DEBUG] Analysis method: {analysis_method}")
        
        # Validate ADO integration request
        if update_ado and not ENABLE_ADO_INTEGRATION:
            raise HTTPException(
                status_code=400,
                detail="Azure DevOps integration is disabled. Set ENABLE_ADO_INTEGRATION=true to enable it."
            )

        # Parse analysis method
        try:
            method = AnalysisMethod(analysis_method.lower())
        except ValueError:
            method = AnalysisMethod.AUTO

        results = []
        all_changes = []  # Collect all changes for RAG analysis
        
        for file in files:
            print(f"[DEBUG] Processing file: {file.filename}")
            content = await file.read()
            
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                print(f"[DEBUG] Failed to decode {file.filename} as UTF-8")
                results.append({
                    "file_path": file.filename,
                    "analysis_method": "error",
                    "summary": f"Failed to decode file {file.filename} as UTF-8",
                    "confidence_score": 0.0,
                    "function_changes": [],
                    "recommendations": ["File encoding issue - manual review required"],
                    "performance_impact": "unknown",
                    "risk_level": "medium"
                })
                continue
            
            print(f"[DEBUG] File content length: {len(text_content)}")
            
            # Check if it's a diff format
            if is_diff_format(text_content):
                print(f"[DEBUG] File {file.filename} is in diff format")
                
                # Try primary extraction method
                file_changes = GitDiffExtractor.extract_file_changes(text_content)
                
                # If primary method fails, try unified diff parser
                if not file_changes:
                    print(f"[DEBUG] Primary extraction failed, trying unified diff parser")
                    file_changes = GitDiffExtractor.parse_unified_diff(text_content)
                
                # If still no results, try to extract content manually
                if not file_changes:
                    print(f"[DEBUG] All extraction methods failed, trying manual extraction")
                    # Look for any content that looks like code
                    lines = text_content.split('\n')
                    old_content = []
                    new_content = []
                    
                    for line in lines:
                        if line.startswith(' '):
                            old_content.append(line[1:])
                            new_content.append(line[1:])
                        elif line.startswith('-'):
                            old_content.append(line[1:])
                        elif line.startswith('+'):
                            new_content.append(line[1:])
                    
                    if old_content or new_content:
                        file_changes = {
                            file.filename: {
                                'old_content': '\n'.join(old_content),
                                'new_content': '\n'.join(new_content)
                            }
                        }
                        print(f"[DEBUG] Manual extraction found content: old={len(old_content)} lines, new={len(new_content)} lines")
                
                print(f"[DEBUG] Extracted {len(file_changes)} file changes from diff")
                
                if not file_changes:
                    # If we still can't extract anything, create a fallback analysis
                    print(f"[DEBUG] No file changes extracted, creating fallback analysis")
                    results.append({
                        "file_path": file.filename,
                        "analysis_method": "fallback",
                        "summary": f"Diff file detected but could not extract specific file changes. Manual review recommended.",
                        "confidence_score": 0.3,
                        "function_changes": [],
                        "recommendations": [
                            "Manual review of diff file required",
                            "Check diff format compatibility",
                            "Verify file contains valid changes"
                        ],
                        "performance_impact": "unknown",
                        "risk_level": "medium"
                    })
                    continue
                
                for file_path, changes in file_changes.items():
                    old_content = changes['old_content']
                    new_content = changes['new_content']
                    
                    print(f"[DEBUG] Analyzing {file_path}: old={len(old_content)} chars, new={len(new_content)} chars")
                    
                    # Add to changes collection for RAG analysis
                    from .models.analysis import CodeChange
                    all_changes.append(CodeChange(
                        file_path=file_path,
                        content=new_content,
                        diff=text_content,
                        change_type="modified"
                    ))
                    
                    # Perform smart analysis
                    analysis_result = await smart_analysis_service.analyze_code_changes(
                        file_path, old_content, new_content, method
                    )
                    
                    print(f"[DEBUG] Analysis result for {file_path}: {len(analysis_result.function_changes)} function changes")
                    
                    results.append({
                        "file_path": file_path,
                        "analysis_method": analysis_result.method_used.value,
                        "summary": analysis_result.summary,
                        "confidence_score": analysis_result.confidence_score,
                        "function_changes": [
                            {
                                "name": change.name,
                                "change_type": change.change_type.value,
                                "has_old_content": change.old_content is not None,
                                "has_new_content": change.new_content is not None,
                                "old_content_preview": change.old_content[:100] + "..." if change.old_content and len(change.old_content) > 100 else change.old_content,
                                "new_content_preview": change.new_content[:100] + "..." if change.new_content and len(change.new_content) > 100 else change.new_content
                            }
                            for change in analysis_result.function_changes
                        ],
                        "recommendations": analysis_result.recommendations,
                        "performance_impact": analysis_result.performance_impact,
                        "risk_level": analysis_result.risk_level
                    })
            else:
                print(f"[DEBUG] File {file.filename} is regular content")
                
                # Add to changes collection for RAG analysis
                from .models.analysis import CodeChange
                all_changes.append(CodeChange(
                    file_path=file.filename,
                    content=text_content,
                    change_type="added"
                ))
                
                # Regular file content - treat as new file
                analysis_result = await smart_analysis_service.analyze_code_changes(
                    file.filename, "", text_content, method
                )
                
                print(f"[DEBUG] Analysis result for {file.filename}: {len(analysis_result.function_changes)} function changes")
                
                results.append({
                    "file_path": file.filename,
                    "analysis_method": analysis_result.method_used.value,
                    "summary": analysis_result.summary,
                    "confidence_score": analysis_result.confidence_score,
                    "function_changes": [
                        {
                            "name": change.name,
                            "change_type": change.change_type.value,
                            "has_old_content": change.old_content is not None,
                            "has_new_content": change.new_content is not None,
                            "old_content_preview": change.old_content[:100] + "..." if change.old_content and len(change.old_content) > 100 else change.old_content,
                            "new_content_preview": change.new_content[:100] + "..." if change.new_content and len(change.new_content) > 100 else change.new_content
                        }
                        for change in analysis_result.function_changes
                    ],
                    "recommendations": analysis_result.recommendations,
                    "performance_impact": analysis_result.performance_impact,
                    "risk_level": analysis_result.risk_level
                })

        print(f"[DEBUG] Analysis completed with {len(results)} results")
        
        # Performance analysis if requested
        performance_results = []
        if include_performance:
            print(f"[DEBUG] Running performance analysis")
            perf_changes = []
            for file in files:
                content = await file.read()
                try:
                    text_content = content.decode('utf-8')
                    perf_changes.append({
                        "file_path": file.filename,
                        "content": text_content
                    })
                except UnicodeDecodeError:
                    continue
            
            if perf_changes:
                performance_results = performance_analyzer.analyze_performance_impact(perf_changes)

        # Calculate correct metrics
        files_with_changes = 0
        total_function_changes = 0
        
        for result in results:
            func_changes = result.get("function_changes", [])
            if func_changes and len(func_changes) > 0:
                files_with_changes += 1
                total_function_changes += len(func_changes)
                print(f"[DEBUG] File {result['file_path']} has {len(func_changes)} function changes")
            else:
                print(f"[DEBUG] File {result['file_path']} has no function changes")
        
        print(f"[DEBUG] Final metrics: files_with_changes={files_with_changes}, total_function_changes={total_function_changes}")
        
        response = {
            "analysis_results": results,
            "performance_analysis": performance_results if include_performance else None,
            "overall_risk": _calculate_overall_risk(results),
            "recommendations": _generate_overall_recommendations(results),
            "total_files_analyzed": len(results),
            "analysis_summary": {
                "files_with_changes": files_with_changes,
                "total_function_changes": total_function_changes,
                "methods_used": list(set(r["analysis_method"] for r in results))
            }
        }
        
        print(f"[DEBUG] Returning response with {len(results)} analysis results")
        for i, result in enumerate(results):
            print(f"[DEBUG] Result {i+1}: {result['file_path']} - {len(result['function_changes'])} function changes")
            if result['function_changes']:
                for j, func_change in enumerate(result['function_changes']):
                    print(f"[DEBUG]   Function {j+1}: {func_change['name']} ({func_change['change_type']})")
        
        return response

    except Exception as e:
        print(f"[DEBUG] Error in smart analysis: {str(e)}")
        import traceback
        traceback.print_exc()
@app.post("/analyze/impact")
async def analyze_impact(
    files: List[UploadFile] = File(...),
    update_ado: bool = Form(default=False),
    ado_item_id: str = Form(default=None)
):
    """
    Impact analysis focused on change impact and dependency tracking
    """
    try:
        print(f"[DEBUG] Impact analysis started with {len(files)} files")
        
        # Validate ADO integration request
        if update_ado and not ENABLE_ADO_INTEGRATION:
            raise HTTPException(
                status_code=400,
                detail="Azure DevOps integration is disabled. Set ENABLE_ADO_INTEGRATION=true to enable it."
            )

        # Process uploaded files into CodeChange objects
        changes = []
        all_extracted_files = {}  # Track all files extracted from diffs
        
        for file in files:
            print(f"[DEBUG] Processing file: {file.filename}")
            content = await file.read()
            
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                print(f"[DEBUG] Failed to decode {file.filename} as UTF-8")
                continue
            
            print(f"[DEBUG] File content length: {len(text_content)}")
            
            # Create CodeChange object
            if is_diff_format(text_content):
                print(f"[DEBUG] File {file.filename} is in diff format")
                
                # Extract actual files from the diff
                from .utils.diff_utils import GitDiffExtractor
                file_changes = GitDiffExtractor.extract_file_changes(text_content)
                
                print(f"[DEBUG] Extracted {len(file_changes)} files from diff {file.filename}")
                
                if file_changes:
                    # Create CodeChange objects for each file in the diff
                    for file_path, change_data in file_changes.items():
                        print(f"[DEBUG] Creating change for extracted file: {file_path}")
                        changes.append(CodeChange(
                            file_path=file_path,
                            content=None,
                            diff=text_content,  # Keep original diff for context
                            change_type="modified"
                        ))
                        all_extracted_files[file_path] = change_data
                else:
                    # Fallback: treat the diff file itself as changed
                    print(f"[DEBUG] No files extracted from diff, treating {file.filename} as changed file")
                    changes.append(CodeChange(
                        file_path=file.filename,
                        content=None,
                        diff=text_content,
                        change_type="modified"
                    ))
            else:
                print(f"[DEBUG] File {file.filename} is regular content")
                changes.append(CodeChange(
                    file_path=file.filename,
                    content=text_content,
                    change_type="added"
                ))

        if not changes:
            raise HTTPException(status_code=400, detail="No valid changes found in uploaded files")

        print(f"[DEBUG] Created {len(changes)} CodeChange objects")
        
        # Perform impact analysis
        impact_analysis = await impact_analysis_service.analyze_impact(changes)
        
        # Convert to response format - handle extracted files properly
        changed_files_response = []
        
        # Group by actual file paths (in case multiple changes refer to same file)
        file_summaries = {}
        for cf in impact_analysis.changed_files:
            if cf.file_path in file_summaries:
                # Combine summaries for the same file
                file_summaries[cf.file_path] += f"; {cf.functional_summary}"
            else:
                file_summaries[cf.file_path] = cf.functional_summary
        
        # Create response objects
        for file_path, summary in file_summaries.items():
            changed_files_response.append({
                "file_path": file_path,
                "functional_summary": summary
            })
        
        response = {
            "summary": impact_analysis.summary,
            "changed_files": changed_files_response,
            "impacted_files": [
                {
                    "file_path": imp.file_path,
                    "impact_reason": imp.impact_reason,
                    "impact_description": imp.impact_description
                }
                for imp in impact_analysis.impacted_files
            ]
        }
        
        print(f"[DEBUG] Impact analysis completed")
        print(f"[DEBUG] Changed files: {len(impact_analysis.changed_files)}")
        print(f"[DEBUG] Impacted files: {len(impact_analysis.impacted_files)}")
        
        return response
        
    except Exception as e:
        print(f"[DEBUG] Error in impact analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Impact analysis failed: {str(e)}")

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

def _calculate_overall_risk(results: List[Dict]) -> str:
    """Calculate overall risk from analysis results"""
    risk_scores = {"low": 1, "medium": 2, "high": 3}
    
    if not results:
        return "low"
    
    max_risk = max(risk_scores.get(result.get("risk_level", "low"), 1) for result in results)
    
    for level, score in risk_scores.items():
        if score == max_risk:
            return level
    
    return "medium"

def _generate_overall_recommendations(results: List[Dict]) -> List[str]:
    """Generate overall recommendations from analysis results"""
    all_recommendations = []
    
    for result in results:
        all_recommendations.extend(result.get("recommendations", []))
    
    # Remove duplicates while preserving order
    unique_recommendations = []
    seen = set()
    
    for rec in all_recommendations:
        if rec not in seen:
            unique_recommendations.append(rec)
            seen.add(rec)
    
    # If no recommendations, add default ones
    if not unique_recommendations:
        unique_recommendations = [
            "Review all changes thoroughly",
            "Ensure adequate test coverage",
            "Consider impact on dependent systems"
        ]
    
    return unique_recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)