# 🚀 Code Change Impact Analysis Backend

## Overview
This backend system analyzes code changes from Pull Requests, retrieves impacted code from Supabase (RAG), and identifies the impact of changes using Azure OpenAI. The response is structured as JSON for easy integration with Azure DevOps (ADO).

## Features
- Uses Supabase as a vector database to store repository embeddings
- Extracts code changes from Pull Requests
- Fetches impacted and dependent code from the repository
- Uses Azure OpenAI for impact analysis
- Generates structured JSON output for ADO automation
- Pluggable design for different project types

## Setup

### Prerequisites
- Python 3.9+
- Azure OpenAI API access
- Supabase account
- Azure DevOps account

### Installation
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your configuration:
   ```bash
   cp .env.example .env
   ```

### Configuration
Update the `.env` file with your credentials:
- Azure OpenAI API configuration
- Supabase credentials
- Azure DevOps PAT and organization details

## Usage
1. Start the server:
   ```bash
   uvicorn app.main:app --reload
   ```
2. The API will be available at `http://localhost:8000`
3. API documentation is available at `http://localhost:8000/docs`

## API Endpoints
- `POST /analyze`: Analyze code changes from a PR
- `POST /index`: Index repository code into Supabase
- `GET /health`: Health check endpoint

## Architecture
1. **Repository Indexing**
   - Code is indexed and stored in Supabase using vector embeddings
   - Each file and method is embedded using Azure OpenAI

2. **Change Analysis**
   - PR changes are extracted and analyzed
   - Related code is retrieved using vector similarity search
   - Impact analysis is performed using Azure OpenAI

3. **Integration**
   - Results are formatted as JSON
   - Automated integration with Azure DevOps workflows

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 