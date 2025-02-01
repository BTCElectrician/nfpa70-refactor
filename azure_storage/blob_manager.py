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
        self.connect_str = os.getenv('AzureWebJobsStorage')
        if not self.connect_str:
            raise ValueError("Missing AzureWebJobsStorage connection string in environment.")

        self.blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
        self.container_name = container_name
        self.blob_name = blob_name

        # Ensure container exists (create if needed)
        try:
            self.blob_service_client.create_container(self.container_name)
        except ResourceExistsError:
            pass  # container already exists

    def save_processed_data(self, data: dict) -> None:
        """Save a Python dictionary to Blob Storage as JSON."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)

            blob_client.upload_blob(
                json.dumps(data),
                overwrite=True,
                content_settings=ContentSettings(
                    content_type='application/json',
                    cache_control='no-cache'
                )
            )
            logging.info(f"Data saved to blob: {self.blob_name}")

        except Exception as e:
            logging.error(f"Error saving data to blob: {str(e)}")
            raise

    def load_processed_data(self) -> dict:
        """Load data from Blob Storage and return as a Python dictionary."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(self.blob_name)

            downloaded = blob_client.download_blob().readall()
            return json.loads(downloaded)

        except ResourceNotFoundError:
            logging.warning(f"Blob {self.blob_name} not found in container {self.container_name}")
            return {}
        except Exception as e:
            logging.error(f"Error loading data from blob: {str(e)}")
            return {}
