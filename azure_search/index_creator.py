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
    Includes fields like article_title, section_title, and gpt_analysis.
    Updated for azure-search-documents==11.5.2 with correct vector profile configuration.
    """
    try:
        credential = AzureKeyCredential(admin_key)
        index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)

        # Delete existing index if it exists
        try:
            index_client.delete_index(index_name)
            logger.info(f"Deleted existing index '{index_name}'")
        except Exception as e:
            logger.info(f"Index '{index_name}' does not exist yet")

        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, 
                       filterable=True, retrievable=True),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
                analyzer_name="standard.lucene"
            ),
            SimpleField(
                name="page_number",
                type=SearchFieldDataType.Int32,
                filterable=True,
                retrievable=True
            ),
            SearchableField(
                name="article_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
                retrievable=True
            ),
            SearchableField(
                name="section_number",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
                retrievable=True
            ),
            SearchableField(
                name="article_title",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True
            ),
            SearchableField(
                name="section_title",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True
            ),
            SearchableField(
                name="related_sections",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
                retrievable=True
            ),
            SearchableField(
                name="context_tags",
                type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                filterable=True,
                facetable=True,
                retrievable=True
            ),
            SearchableField(
                name="gpt_analysis",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="myHnswProfile",
                retrievable=True,
                stored=True
            )
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="myHnsw",
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
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw"
                )
            ]
        )

        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
                title_field=SemanticField(field_name="section_title")
            )
        )
        semantic_search = SemanticSearch(configurations=[semantic_config])

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search
        )

        logger.info(f"Creating search index '{index_name}' ...")
        index_client.create_or_update_index(index)
        logger.info(f"Index '{index_name}' created successfully.")

    except Exception as e:
        logger.error(f"Error creating/updating index: {str(e)}")
        raise