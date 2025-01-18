import azure.functions as func
import json
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import OpenAI
import os
from typing import List, Dict, Optional
from loguru import logger

app = func.FunctionApp()

def parse_field_query(query: str) -> Dict[str, Optional[str]]:
    """
    Parse field-specific context from the query.
    
    Args:
        query: Natural language query from user
        
    Returns:
        Dictionary of extracted context (location, equipment, requirement)
    """
    # Common electrical terms to look for
    contexts = {
        'location': ['in', 'at', 'inside', 'outside', 'within', 'through'],
        'equipment': [
            'transformer', 'conduit', 'busway', 'panel', 'switchgear',
            'motor', 'generator', 'disconnect', 'receptacle', 'outlet',
            'luminaire', 'fixture', 'raceway', 'cable', 'wire'
        ],
        'requirement': [
            'clearance', 'spacing', 'distance', 'rating', 'size', 'grounding',
            'support', 'mounting', 'installation', 'protection', 'classification'
        ]
    }
    
    parsed = {
        'location': None,
        'equipment': None,
        'requirement': None
    }
    
    words = query.lower().split()
    for i, word in enumerate(words):
        # Look for location context
        if word in contexts['location'] and i + 1 < len(words):
            parsed['location'] = words[i + 1]
            
        # Look for equipment
        if word in contexts['equipment']:
            parsed['equipment'] = word
            
        # Look for requirement type
        if word in contexts['requirement']:
            parsed['requirement'] = word
            
    return parsed

def format_response(results: List[Dict]) -> Dict:
    """
    Format search results for field use.
    
    Args:
        results: Raw search results
        
    Returns:
        Formatted results with code references and context
    """
    formatted_results = []
    
    for result in results:
        formatted_result = {
            'code_reference': f"Section {result.get('section_number', 'N/A')}",
            'page_number': result.get('page_number', 'N/A'),
            'requirement': result.get('content', '').strip(),
            'score': result.get('@search.score', 0),
            'related_sections': result.get('related_sections', []),
            'context_tags': result.get('context_tags', [])
        }
        formatted_results.append(formatted_result)
    
    return {
        'results': formatted_results,
        'total_results': len(formatted_results)
    }

@app.route(route="search")
def search_nfpa70(req: func.HttpRequest) -> func.HttpResponse:
    """
    Enhanced search endpoint for field use.
    
    Args:
        req: HTTP request containing search query
        
    Returns:
        HTTP response with search results
    """
    try:
        # Get search parameters
        query = req.params.get('q')
        if not query:
            return func.HttpResponse(
                "Please provide a search query",
                status_code=400
            )

        # Parse field context from query
        field_context = parse_field_query(query)
        
        # Generate embedding for semantic search
        try:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            response = client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            search_vector = response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "Error generating query embedding"}),
                status_code=500,
                mimetype="application/json"
            )

        # Set up search client
        try:
            credential = AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"])
            search_client = SearchClient(
                os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"],
                os.environ["AZURE_SEARCH_INDEX_NAME"],
                credential
            )
        except Exception as e:
            logger.error(f"Error setting up search client: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "Error connecting to search service"}),
                status_code=500,
                mimetype="application/json"
            )

        # Build filter based on context
        filter_conditions = []
        for context_type, value in field_context.items():
            if value:
                filter_conditions.append(f"context_tags/any(t: t eq '{value}')")
        
        filter_expression = " and ".join(filter_conditions) if filter_conditions else None

        # Perform hybrid search
        try:
            results = search_client.search(
                search_text=query,  # For keyword matching
                vector=search_vector,  # For semantic matching
                vector_fields="content_vector",
                filter=filter_expression,
                select=[
                    "section_number",
                    "content",
                    "page_number",
                    "related_sections",
                    "context_tags"
                ],
                top=5,
                semantic_configuration_name="my-semantic-config"
            )

            # Format results for field use
            formatted_results = format_response(list(results))
            
            return func.HttpResponse(
                json.dumps(formatted_results),
                mimetype="application/json"
            )

        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": "Error performing search"}),
                status_code=500,
                mimetype="application/json"
            )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": "Internal server error"}),
            status_code=500,
            mimetype="application/json"
        ) 