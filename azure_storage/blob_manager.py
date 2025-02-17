import os
import json
import logging
from typing import Dict, Optional, Any, Union

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, HttpResponseError
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.blob._models import BlobProperties

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
        """Initialize the blob storage manager."""
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Store the container name
        self.container_name = container_name
        self.blob_name = blob_name
        
        # Initialize components
        self._initialize_connection()
        self._setup_clients(container_name, blob_name)
        self._ensure_container_exists()

    @property
    def blob_name(self) -> str:
        """Get the current blob name."""
        return self._blob_name

    @blob_name.setter
    def blob_name(self, value: str) -> None:
        """
        Set a new blob name and reinitialize the blob client.
        
        Args:
            value: New blob name to use
        """
        self._blob_name = value
        if hasattr(self, 'container_client'):
            self.blob_client = self.container_client.get_blob_client(value)
            self.logger.debug(f"Updated blob client to use blob: {value}")

    def _log_exception(self, e: Exception, context: str) -> None:
        """Helper to consistently log exception details."""
        self.logger.error(f"Exception in {context}:")
        self.logger.error(f"  Type: {type(e)}")
        self.logger.error(f"  Str representation: {str(e)}")
        self.logger.error(f"  Dir contents: {dir(e)}")
        
        # Log specific attributes if they exist
        if hasattr(e, 'message'):
            self.logger.error(f"  Message: {e.message}")
        if hasattr(e, 'status_code'):
            self.logger.error(f"  Status code: {e.status_code}")
        if hasattr(e, 'error_code'):
            self.logger.error(f"  Error code: {e.error_code}")
        if hasattr(e, 'response'):
            self.logger.error(f"  Response: {e.response}")

    def _initialize_connection(self) -> None:
        """Set up the base connection using environment variables."""
        self.connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not self.connect_str:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING environment variable is required"
            )

    def _setup_clients(self, container_name: str, blob_name: str) -> None:
        """Initialize blob service, container, and blob clients."""
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connect_str,
                api_version="2025-01-05"
            )
            self.container_client = self.blob_service_client.get_container_client(
                container_name
            )
            self.blob_client = self.container_client.get_blob_client(blob_name)
        except Exception as e:
            self._log_exception(e, "_setup_clients")
            raise

    def _ensure_container_exists(self) -> None:
        """Create the container if it doesn't exist."""
        try:
            self.logger.debug(
                f"Attempting to create container: {self.container_client.container_name}"
            )
            self.container_client.create_container(
                metadata={"purpose": "processed-data-storage"}
            )
            self.logger.info(f"Container {self.container_client.container_name} created")
            
        except ResourceExistsError:
            self.logger.debug(
                f"Container {self.container_client.container_name} already exists"
            )
            
        except HttpResponseError as e:
            self._log_exception(e, "_ensure_container_exists")
            self.logger.error(f"HTTP error creating container: {e.status_code} - {e.message}")
            raise
            
        except Exception as e:
            self._log_exception(e, "_ensure_container_exists")
            self.logger.error(f"Unexpected error creating container: {type(e)} - {str(e)}")
            raise

    def _convert_to_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable format with error handling.
        
        Handles various object types including Pydantic models, custom objects,
        and Azure SDK specific types.
        """
        try:
            # Handle None explicitly
            if obj is None:
                return None
                
            # Handle Azure SDK specific types
            if isinstance(obj, BlobProperties):
                return vars(obj)
                
            # Handle objects with to_dict method (like Pydantic models)
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
                
            # Handle objects with model_dump (newer Pydantic versions)
            if hasattr(obj, 'model_dump'):
                return obj.model_dump()
                
            # Handle general Python objects
            if hasattr(obj, '__dict__'):
                return vars(obj)
                
            # If it's already a basic type, return as is
            if isinstance(obj, (str, int, float, bool, list, dict)):
                return obj
                
            # Last resort: convert to string
            return str(obj)
            
        except Exception as e:
            self.logger.warning(f"Serialization fallback for {type(obj)}: {str(e)}")
            return str(obj)

    def _validate_data_structure(self, data: Union[Dict, str]) -> None:
        """Validate data structure before serialization."""
        # Handle both dict and string inputs
        if isinstance(data, str):
            try:
                # Validate it's actually JSON
                json.loads(data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON string: {str(e)}")
        elif not isinstance(data, dict):
            raise ValueError(f"Expected dict or JSON string, got {type(data)}")
        
        # If it's a dict, check for problematic values
        if isinstance(data, dict):
            def check_value(obj):
                if isinstance(obj, str) and hasattr(obj, 'value'):
                    raise ValueError("String values cannot have 'value' attribute")
                if isinstance(obj, dict):
                    for v in obj.values():
                        check_value(v)
                elif isinstance(obj, list):
                    for item in obj:
                        check_value(item)
                
            check_value(data)

    def save_processed_data(self, data: Dict) -> None:
        """
        Save a Python dictionary to blob storage as JSON.
        
        Args:
            data: Dictionary containing the data to store
            
        Raises:
            ValueError: If input data is not a dictionary
            HttpResponseError: If there's an HTTP-related error during upload
        """
        try:
            # Type validation
            if not isinstance(data, dict):
                raise ValueError(f"Expected dictionary, got {type(data)}")

            # Convert dict to JSON string
            json_data = json.dumps(data, ensure_ascii=False, indent=2)
            
            # Debug logging
            self.logger.debug(f"Data size: {len(json_data)} bytes")
            self.logger.debug(f"Data sample: {json_data[:500]}...")
            
            self.logger.debug(
                f"Uploading JSON data to blob '{self.blob_client.blob_name}' "
                f"in container '{self.container_client.container_name}'"
            )

            # Direct upload of string data without accessing .value
            self.blob_client.upload_blob(
                data=json_data,  # Pass the string directly
                overwrite=True,
                content_type='application/json'  # Simple content type setting
            )
            
            self.logger.info(f"Data successfully saved to {self.blob_client.blob_name}")

        except json.JSONDecodeError as e:
            self._log_exception(e, "save_processed_data")
            raise ValueError(f"Failed to serialize data to JSON: {str(e)}")
        except Exception as e:
            self._log_exception(e, "save_processed_data")
            raise

    def load_processed_data(self) -> Dict:
        """
        Load data from blob storage as a Python dictionary.
        
        Returns:
            Dictionary containing the loaded data or empty dict if blob not found
        """
        try:
            self.logger.debug(
                f"Attempting to download blob '{self.blob_client.blob_name}' "
                f"from container '{self.container_client.container_name}'"
            )
            
            # Download and decode the blob data
            download_stream = self.blob_client.download_blob()
            json_data = download_stream.readall()
            
            # Parse JSON string to dict
            data = json.loads(json_data)
            self.logger.debug(f"Successfully loaded {len(str(data))} bytes of data")
            return data

        except ResourceNotFoundError:
            self.logger.warning(f"Blob {self.blob_client.container_name} not found, returning empty dict")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON data: {str(e)}")
            return {}
        except Exception as e:
            self._log_exception(e, "load_processed_data")
            return {}