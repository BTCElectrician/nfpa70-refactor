import os
import json
import logging
from typing import Dict, Optional

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, HttpResponseError
from azure.storage.blob import BlobServiceClient, ContentSettings


class BlobStorageManager:
    """Manages Azure blob storage operations for JSON data processing.
    
    Provides a simple interface for uploading and downloading JSON data to/from
    Azure Blob Storage. Requires AZURE_STORAGE_CONNECTION_STRING environment variable.
    
    Attributes:
        container_name (str): Name of the blob container
        blob_name (str): Name of the blob file
    """
    
    def __init__(
        self, 
        container_name: str = "processed-data",
        blob_name: str = "processed_data.json"
    ) -> None:
        """Initialize the blob storage manager.
        
        Args:
            container_name: Name of the container to store blobs
            blob_name: Name of the blob file for data storage
            
        Raises:
            ValueError: If AZURE_STORAGE_CONNECTION_STRING is not set
        """
        self._initialize_connection()
        self._setup_clients(container_name, blob_name)
        self._ensure_container_exists()

    def _initialize_connection(self) -> None:
        """Set up the base connection using environment variables."""
        self.connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not self.connect_str:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING environment variable is required"
            )

    def _setup_clients(self, container_name: str, blob_name: str) -> None:
        """Initialize blob service, container, and blob clients."""
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connect_str,
            api_version="2025-01-05"
        )
        self.container_client = self.blob_service_client.get_container_client(
            container_name
        )
        self.blob_client = self.container_client.get_blob_client(blob_name)

    def _ensure_container_exists(self) -> None:
        """Create the container if it doesn't exist."""
        try:
            self.container_client.create_container(
                metadata={"purpose": "processed-data-storage"}
            )
            logging.info(f"Container {self.container_client.container_name} created")
        except ResourceExistsError:
            logging.debug(f"Container {self.container_client.container_name} already exists")

    def save_processed_data(self, data: Dict) -> None:
        """Save a Python dictionary to blob storage as JSON.
        
        Args:
            data: Dictionary containing the data to be stored
            
        Raises:
            HttpResponseError: If there's an HTTP-related error during upload
            ResourceExistsError: If blob exists and cannot be overwritten
            Exception: For other unexpected errors
        """
        try:
            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            
            self.blob_client.upload_blob(
                data=json_data,
                overwrite=True,
                content_settings=ContentSettings(
                    content_type='application/json; charset=utf-8',
                    cache_control='max-age=0, no-cache'
                ),
                standard_blob_tier="Hot"
            )
            
            logging.info(f"Data successfully saved to {self.blob_client.blob_name}")

        except HttpResponseError as http_err:
            logging.error(
                f"HTTP error uploading data: {http_err.status_code} - {http_err.reason}"
            )
            raise
        except ResourceExistsError as exists_err:
            logging.error(
                f"Blob exists or cannot be overwritten: {exists_err.error_code}"
            )
            raise
        except Exception as e:
            logging.error(f"Unexpected error uploading data: {str(e)}")
            raise

    def load_processed_data(self) -> Dict:
        """Load data from blob storage as a Python dictionary.
        
        Returns:
            Dictionary containing the loaded data or empty dict if blob not found
            
        Note:
            Returns empty dict on error to allow for graceful handling of missing data
        """
        try:
            download_stream = self.blob_client.download_blob(
                max_concurrency=4,
                validate_content=True
            )
            return json.loads(download_stream.readall())

        except ResourceNotFoundError:
            logging.warning(
                f"Blob {self.blob_client.blob_name} not found, returning empty dict"
            )
            return {}
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON data: {str(e)}")
            return {}