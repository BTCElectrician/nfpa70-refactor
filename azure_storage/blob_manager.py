import os
import json
import logging

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

class BlobStorageManager:
    """
    Basic blob manager for uploading/downloading JSON data.
    Relies on the AzureWebJobsStorage connection string in environment variables.
    """
    def __init__(self, container_name: str = "processed-data", blob_name: str = "processed_data.json"):
        self.connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not self.connect_str:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in environment")

        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connect_str,
            api_version="2025-01-05"  # Explicit service version [1][2]
        )
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.blob_client = self.container_client.get_blob_client(blob_name)

        # Modern container creation with metadata
        try:
            self.container_client.create_container(metadata={"purpose": "processed-data-storage"})
        except ResourceExistsError:
            pass

    def save_processed_data(self, data: dict) -> None:
        """Save dictionary to blob storage with optimized upload"""
        try:
            json_data = json.dumps(data, ensure_ascii=False)
            
            self.blob_client.upload_blob(
                data=json_data,
                overwrite=True,
                content_settings=ContentSettings(
                    content_type='application/json; charset=utf-8',  # Explicit UTF-8
                    cache_control='max-age=0, no-cache'
                ),
                standard_blob_tier="Hot"  # Explicit tier setting [1]
            )
            logging.info(f"Data committed to {self.blob_client.blob_name}")

        except Exception as e:
            logging.error(f"Upload failed: {str(e)}")
            raise

    def load_processed_data(self) -> dict:
        """Load data with modern download pattern"""
        try:
            download_stream = self.blob_client.download_blob(
                max_concurrency=4,  # Parallel downloads [2]
                validate_content=True  # CRC64 validation
            )
            return json.loads(download_stream.readall())

        except ResourceNotFoundError:
            logging.warning(f"Blob {self.blob_client.blob_name} not found")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON: {str(e)}")
            return {}
