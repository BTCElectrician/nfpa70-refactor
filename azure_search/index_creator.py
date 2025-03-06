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

def create_search_index(service_endpoint: str, admin_key: str, index_name: str) -> None:
    """
    Create or update an Azure Cognitive Search index with vector search enabled.
    Updated for azure-search-documents==11.5.2.
    
    If the index already exists, we just update it rather than deleting it.
    """
    try:
        # Set up the client with admin credentials
        credential = AzureKeyCredential(admin_key)
        index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

        # ---------------------------------------------------------------------
        # Remove forced delete_index. We no longer nuke the old index each time.
        #
        # If we want to update the existing index, just call create_or_update_index.
        # ---------------------------------------------------------------------

        # Configure vector search with HNSW algorithm
        # HNSW (Hierarchical Navigable Small World) is efficient for similarity search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters=HnswParameters(
                        m=4,                    # Number of bi-directional links created for every new element
                        ef_construction=400,     # Size of dynamic candidate list for construction
                        ef_search=500,          # Size of dynamic candidate list for search
                        metric="cosine"         # Distance metric for vector comparison
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

        # Define all fields for the search index
        fields = [
            # Unique identifier for each document
            SimpleField(
                name="id", 
                type=SearchFieldDataType.String, 
                key=True
            ),
            
            # Main content field - searchable for full-text search
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                analyzer_name="standard.lucene"
            ),
            
            # Page number - useful for filtering and sorting
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.Int32,
                filterable=True
            ),
            
            # Article number - searchable and filterable
            SearchableField(
                name="article_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            
            # Section number - searchable and filterable
            SearchableField(
                name="section_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            
            # Article and section titles - searchable
            SearchableField(
                name="article_title",
                type=SearchFieldDataType.String
            ),
            SearchableField(
                name="section_title",
                type=SearchFieldDataType.String
            ),
            
            # Lists of tags and related sections
            SearchField(
                name="context_tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True
            ),
            SearchField(
                name="related_sections",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True
            ),
            
            # Vector field for semantic search
            # Must match OpenAI's embedding dimension (1536)
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="hnsw-profile"
            )
        ]

        # Configure semantic search capabilities
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
                title_field=SemanticField(field_name="section_title")
            )
        )
        semantic_search = SemanticSearch(configurations=[semantic_config])

        # Create the index with all configurations
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )

        logger.info(f"Creating or updating search index '{index_name}' ...")
        index_client.create_or_update_index(index)
        logger.info(f"Index '{index_name}' created or updated successfully.")

    except Exception as e:
        logger.error(f"Error creating/updating index: {str(e)}")
        raise