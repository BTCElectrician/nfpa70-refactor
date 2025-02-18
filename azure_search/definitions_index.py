from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswParameters,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticSearch,
    SemanticPrioritizedFields
)
from loguru import logger

def create_definitions_index(service_endpoint: str, admin_key: str, index_name: str) -> None:
    """
    Create or update an Azure Cognitive Search index specifically for Article 100 definitions.
    
    Args:
        service_endpoint: The URL of your Azure Search service
        admin_key: The admin API key for your search service
        index_name: The name to give your search index
    """
    try:
        # Set up the client with admin credentials
        credential = AzureKeyCredential(admin_key)
        index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

        # Delete existing index if it exists
        try:
            index_client.delete_index(index_name)
            logger.info(f"Deleted existing index '{index_name}'")
        except Exception:
            logger.info(f"Index '{index_name}' does not exist yet")

        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine"
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="hnsw-profile",
                    algorithm_configuration_name="hnsw-config"
                )
            ]
        )

        # Define fields for the definitions index
        fields = [
            # Unique identifier for each definition
            SimpleField(
                name="id", 
                type=SearchFieldDataType.String, 
                key=True
            ),
            
            # The term being defined
            SearchableField(
                name="term",
                type=SearchFieldDataType.String,
                analyzer_name="standard.lucene"
            ),
            
            # Any parenthetical context
            SearchableField(
                name="context",
                type=SearchFieldDataType.String,
                analyzer_name="standard.lucene"
            ),
            
            # The actual definition text
            SearchableField(
                name="definition",
                type=SearchFieldDataType.String,
                analyzer_name="standard.lucene"
            ),
            
            # Page number for reference
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.Int32,
                filterable=True
            ),
            
            # Lists of references and notes
            SearchField(
                name="cross_references",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True
            ),
            SearchField(
                name="info_notes",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String)
            ),
            SearchField(
                name="committee_refs",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True
            ),
            SearchField(
                name="section_refs",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True
            ),
            
            # Vector field for semantic search
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="hnsw-profile"
            )
        ]

        # Configure semantic search
        semantic_config = SemanticConfiguration(
            name="definitions-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[
                    SemanticField(field_name="definition"),
                    SemanticField(field_name="term")
                ],
                title_field=SemanticField(field_name="term")
            )
        )
        semantic_search = SemanticSearch(configurations=[semantic_config])

        # Create the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )

        # Create or update the index
        logger.info(f"Creating definitions index '{index_name}'...")
        index_client.create_or_update_index(index)
        logger.info(f"Index '{index_name}' created successfully.")

    except Exception as e:
        logger.error(f"Error creating/updating definitions index: {str(e)}")
        raise