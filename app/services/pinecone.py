from pinecone import Pinecone, ServerlessSpec
from ..config import settings
import asyncio

class PineconeService:
    def __init__(self):
        self.pc = Pinecone(
            api_key=settings.pinecone_api_key
        )

    def create_index(self, index_name: str):
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            
            if index_name in existing_indexes.names():
                print(f"Index {index_name} already exists")
                return
            
            print(f"Creating new index {index_name}")
            self.pc.create_index(
                name=index_name,
                dimension=1536,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            
            # Wait for index to be ready
            while True:
                if index_name in self.pc.list_indexes().names():
                    break
                asyncio.sleep(1)
                
            print(f"Index {index_name} ready")
            
        except Exception as e:
            print(f"Error in create_index: {str(e)}")
            raise

    def get_index(self, index_name: str):
        return self.pc.Index(index_name)

    async def upsert_vectors_async(self, index_name: str, vectors: list):
        index = self.get_index(index_name)
        # Use batching for vector uploads
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
            await asyncio.sleep(0)  # Allow other tasks to run