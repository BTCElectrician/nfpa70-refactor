from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

# Use DefaultAzureCredential for authentication
credential = DefaultAzureCredential()
account_name = "aisearchohmniv1"

# Connect to Blob Storage
blob_service_client = BlobServiceClient(
    account_url=f"https://{account_name}.blob.core.windows.net",
    credential=credential
)

# List containers to verify access
for container in blob_service_client.list_containers():
    print(container["name"])
