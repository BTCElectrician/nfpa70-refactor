import unittest
from unittest.mock import patch, MagicMock, ANY
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, BlobClient, ContentSettings
from azure_storage.blob_manager import BlobStorageManager

# Mock the CodeChunk structure from text_chunker.py
@dataclass
class MockCodeChunk:
    content: str
    page_number: int
    article_number: Optional[str]
    section_number: Optional[str]
    article_title: Optional[str]
    section_title: Optional[str]
    context_tags: List[str]
    related_sections: List[str]

@patch('azure.storage.blob.BlobServiceClient')
class TestBlobStorageManager(unittest.TestCase):
    """Test suite for BlobStorageManager serialization and data handling."""

    def setUp(self):
        """Set up test environment before each test."""
        # Configure logging
        logging.basicConfig(level=logging.DEBUG)
        
        # Create mock blob clients
        self.mock_blob_client = MagicMock()
        self.mock_container_client = MagicMock()
        
        # Set up environment variables
        self.env_patcher = patch.dict('os.environ', {
            'AZURE_STORAGE_CONNECTION_STRING': 'DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net'
        })
        self.env_patcher.start()
        
        # Patch Azure client initialization
        self.service_patcher = patch('azure.storage.blob.BlobServiceClient.from_connection_string')
        self.mock_service = self.service_patcher.start()
        
        # Configure mock returns
        self.mock_service.return_value.get_container_client.return_value = self.mock_container_client
        self.mock_container_client.get_blob_client.return_value = self.mock_blob_client

    def tearDown(self):
        """Clean up after each test."""
        self.env_patcher.stop()
        self.service_patcher.stop()

    def test_serialize_basic_data(self, mock_blob_service):
        """Test serialization of basic Python types."""
        manager = BlobStorageManager("test-container", "test.json")
        
        test_data = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        # Test serialization
        serialized = json.dumps(test_data, default=manager._convert_to_serializable)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized, test_data)

    def test_serialize_code_chunks(self, mock_blob_service):
        """Test serialization of CodeChunk-like objects."""
        manager = BlobStorageManager("test-container", "test.json")
        
        # Create mock chunks
        chunks = [
            MockCodeChunk(
                content="Test content 1",
                page_number=1,
                article_number="90",
                section_number="90.1",
                article_title="Test Article",
                section_title="Test Section",
                context_tags=["tag1", "tag2"],
                related_sections=["90.2", "90.3"]
            ),
            MockCodeChunk(
                content="Test content 2",
                page_number=2,
                article_number="91",
                section_number="91.1",
                article_title="Another Article",
                section_title="Another Section",
                context_tags=["tag3"],
                related_sections=[]
            )
        ]
        
        # Convert chunks to dictionary format (similar to main.py)
        chunk_dicts = []
        for c in chunks:
            chunk_dict = {
                "content": c.content,
                "page_number": c.page_number,
                "article_number": c.article_number,
                "section_number": c.section_number,
                "article_title": c.article_title or "",
                "section_title": c.section_title or "",
                "context_tags": c.context_tags,
                "related_sections": c.related_sections
            }
            chunk_dicts.append(chunk_dict)
        
        data = {"chunks": chunk_dicts}
        
        # Test serialization
        try:
            serialized = json.dumps(data, default=manager._convert_to_serializable)
            deserialized = json.loads(serialized)
            
            # Verify structure maintained
            self.assertIn("chunks", deserialized)
            self.assertEqual(len(deserialized["chunks"]), 2)
            self.assertEqual(deserialized["chunks"][0]["content"], "Test content 1")
            self.assertEqual(deserialized["chunks"][1]["article_number"], "91")
            
        except Exception as e:
            self.fail(f"Serialization failed: {str(e)}")

    def test_save_processed_data(self, mock_blob_service):
        """Test save_processed_data with mocked Azure client."""
        manager = BlobStorageManager("test-container", "test.json")
        
        # Create test data similar to actual chunk data
        test_data = {
            "chunks": [
                {
                    "content": "Test content",
                    "page_number": 1,
                    "article_number": "90",
                    "section_number": "90.1",
                    "article_title": "Test Article",
                    "section_title": "Test Section",
                    "context_tags": ["tag1", "tag2"],
                    "related_sections": ["90.2"]
                }
            ]
        }
        
        # Test the save operation
        try:
            manager.save_processed_data(test_data)
            
            # Verify upload_blob was called
            self.mock_blob_client.upload_blob.assert_called_once()
            
            # Get the data that would have been uploaded
            call_args = self.mock_blob_client.upload_blob.call_args
            uploaded_data = call_args[1]['data']
            
            # Verify it's valid JSON
            parsed_data = json.loads(uploaded_data)
            self.assertIn("chunks", parsed_data)
            self.assertEqual(len(parsed_data["chunks"]), 1)
            
        except Exception as e:
            self.fail(f"save_processed_data failed: {str(e)}")

    def test_handle_problematic_string_values(self, mock_blob_service):
        """Test handling of strings that might cause .value attribute errors."""
        manager = BlobStorageManager("test-container", "test.json")
        
        class ProblemString(str):
            @property
            def value(self):
                return self
        
        test_data = {
            "normal_string": "test",
            "problem_string": ProblemString("test")
        }
        
        # This should handle the ProblemString without trying to access .value
        serialized = json.dumps(test_data, default=manager._convert_to_serializable)
        deserialized = json.loads(serialized)
        
        self.assertEqual(deserialized["normal_string"], "test")
        self.assertEqual(deserialized["problem_string"], "test")

    def test_validation_catches_value_attribute(self, mock_blob_service):
        """Test that validation catches strings with value attributes."""
        manager = BlobStorageManager("test-container", "test.json")
        
        class StringWithValue(str):
            @property
            def value(self):
                return self
        
        problematic_data = {
            "key": StringWithValue("test")
        }
        
        with self.assertRaises(ValueError):
            manager._validate_data_structure(problematic_data)

if __name__ == '__main__':
    unittest.main()