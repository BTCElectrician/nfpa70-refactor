from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    PrioritizedFields
)
from loguru import logger

def create_search_index(service_endpoint: str, admin_key: str, index_name: str) -> None:
    """
    Create an enhanced search index for electrical code content.
    
    Args:
        service_endpoint: Azure Search service endpoint
        admin_key: Azure Search admin key
        index_name: Name for the search index
    """
    try:
        credential = AzureKeyCredential(admin_key)
        index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

        # Define fields for the index
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="content", 
                type=SearchFieldDataType.String,
                analyzer_name="en.microsoft"
            ),
            SimpleField(
                name="page_number", 
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True
            ),
            SearchableField(
                name="article_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="article_title",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SearchableField(
                name="section_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="section_title",
                type=SearchFieldDataType.String,
                filterable=True
            ),
            SearchableField(
                name="context_tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
                searchable=True
            ),
            SearchableField(
                name="related_sections",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True
            ),
            # Vector field for semantic search
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=1536,
                vector_search_profile_name="my-vector-config",
            ),
        ]

        # Configure vector search
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-vector-config",
                    kind="hnsw",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine"
                    )
                )
            ]
        )

        # Configure semantic search for field-oriented queries
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=PrioritizedFields(
                title_field=SemanticField(field_name="section_title"),
                keywords_fields=[
                    SemanticField(field_name="context_tags"),
                    SemanticField(field_name="article_title")
                ],
                content_fields=[SemanticField(field_name="content")]
            )
        )

        semantic_settings = SemanticSettings(
            configurations=[semantic_config]
        )

        # Create the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=semantic_settings
        )

        logger.info(f"Creating {index_name} search index...")
        index_client.create_or_update_index(index)
        logger.info(f"Index {index_name} created successfully.")

    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise