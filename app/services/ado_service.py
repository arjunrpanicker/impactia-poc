import os
from typing import Dict, Any
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication
from ..models.analysis import ChangeAnalysisResponse

class AzureDevOpsService:
    def __init__(self):
        # Initialize Azure DevOps client
        personal_access_token = os.getenv("AZURE_DEVOPS_PAT")
        organization_url = f"https://dev.azure.com/{os.getenv('AZURE_DEVOPS_ORG')}"
        
        # Create a connection to Azure DevOps
        credentials = BasicAuthentication('', personal_access_token)
        self.connection = Connection(base_url=organization_url, creds=credentials)
        
        # Get clients
        self.git_client = self.connection.clients.get_git_client()
        self.work_item_client = self.connection.clients.get_work_item_tracking_client()
        
        # Store project name
        self.project = os.getenv("AZURE_DEVOPS_PROJECT")

    async def update_work_item(self, work_item_id: str, analysis: ChangeAnalysisResponse):
        """
        Update Azure DevOps work item with analysis results
        """
        try:
            # Format the analysis results as markdown
            markdown_content = self._format_analysis_as_markdown(analysis)
            
            # Create patch document
            patch_document = [
                {
                    "op": "add",
                    "path": "/fields/System.Description",
                    "value": markdown_content
                },
                {
                    "op": "add",
                    "path": "/fields/Custom.ImpactAnalysis",
                    "value": str(analysis.dict())
                }
            ]
            
            # Update work item
            result = self.work_item_client.update_work_item(
                document=patch_document,
                id=work_item_id,
                project=self.project
            )
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to update work item: {str(e)}")

    def _format_analysis_as_markdown(self, analysis: ChangeAnalysisResponse) -> str:
        """
        Format the analysis results as markdown for Azure DevOps
        """
        markdown = f"""
# Code Change Impact Analysis

## Summary
{analysis.summary}
"""

        if analysis.risk_level:
            markdown += f"\n## Risk Level: {analysis.risk_level.value.upper()}\n"

        markdown += "\n## Changed Components\n"
        
        for component in analysis.changed_components:
            markdown += f"""
### {component.file_path}
- **Methods**: {', '.join(component.methods)}
- **Impact**: {component.impact_description}
- **Risk Level**: {component.risk_level.value}
- **Associated Unit Tests**: {', '.join(component.associated_unit_tests)}
"""

        if analysis.dependency_chains:
            markdown += "\n## Dependency Chains\n"
            
            for chain in analysis.dependency_chains:
                markdown += f"""
### {chain.file_path}
#### Changed Methods:
"""
                for method in chain.methods:
                    markdown += f"- **{method.name}**: {method.summary}\n"
                
                markdown += "\n#### Impacted Files:\n"
                for imp_file in chain.impacted_files:
                    markdown += f"\n##### {imp_file.file_path}\n"
                    for method in imp_file.methods:
                        markdown += f"- **{method.name}**: {method.summary}\n"
                
                if chain.associated_unit_tests:
                    markdown += f"\n#### Associated Unit Tests:\n"
                    for test in chain.associated_unit_tests:
                        markdown += f"- {test}\n"

        if analysis.dependency_chain_visualization:
            markdown += "\n## Dependency Chain Visualization\n"
            markdown += "```\n"
            for chain in analysis.dependency_chain_visualization:
                markdown += f"{chain}\n"
            markdown += "```\n"
            
        return markdown 