import os
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
    VectorSearchAlgorithmMetric,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    PrioritizedFields
)
from loguru import logger

def create_search_index(service_endpoint: str, admin_key: str, index_name: str) -> None:
    """
    Create or update an Azure Cognitive Search index with vector search enabled.
    """
    try:
        credential = AzureKeyCredential(admin_key)
        index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

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
            # Example additional fields
            SearchableField(
                name="article_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="section_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True
            ),
            SearchableField(
                name="context_tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
                searchable=True
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=1536,       # Align w/ embedding model
                vector_search_configuration="myHnsw" # Name must match config
            ),
        ]

        # Basic VectorSearch config
        vector_search = VectorSearch(
            # A list of algorithm configurations
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
                    # If your azure-search-documents version supports metric
                    parameters={
                        "m": 16,
                        "efConstruction": 200,
                        "metric": VectorSearchAlgorithmMetric.COSINE
                    }
                )
            ]
        )

        # Optional semantic config
        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=PrioritizedFields(
                title_field=SemanticField(field_name="section_number"),
                content_fields=[SemanticField(field_name="content")]
            )
        )
        semantic_settings = SemanticSettings(configurations=[semantic_config])

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=semantic_settings
        )

        logger.info(f"Creating or updating search index '{index_name}' ...")
        index_client.create_or_update_index(index)
        logger.info(f"Index '{index_name}' created/updated successfully.")

    except Exception as e:
        logger.error(f"Error creating/updating index: {str(e)}")
        raise
