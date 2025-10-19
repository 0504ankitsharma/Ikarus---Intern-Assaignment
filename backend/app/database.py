from pinecone import Pinecone, ServerlessSpec
from app.config import settings
import logging
from typing import List, Dict, Any
import time

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index_name = settings.PINECONE_INDEX_NAME
            self.index = None
            self._initialize_index()
        except Exception as e:
            logger.error(f"Error initializing Pinecone client: {str(e)}")
            raise

    def _initialize_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            # Get list of indexes
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes]
            
            # Create index if not found
            if self.index_name not in index_names:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.PINECONE_DIMENSION,  # e.g., 768 or 1536
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_REGION)
                )
                time.sleep(10)  # Give time for index to initialize

            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Error creating or connecting to Pinecone index: {str(e)}")
            raise

    def upsert_vectors(self, vectors: List[tuple]):
        """
        Upsert vectors into Pinecone.
        Format: [(id, embedding, metadata)]
        """
        try:
            if not vectors:
                logger.warning("No vectors provided for upsert.")
                return

            self.index.upsert(vectors=vectors)
            logger.info(f"✅ Successfully upserted {len(vectors)} vectors to Pinecone.")
        except Exception as e:
            logger.error(f"❌ Error upserting vectors to Pinecone: {str(e)}")
            raise

    def query_vectors(self, query_vector: List[float], top_k: int = 5, filter_dict: Dict = None):
        """Query the Pinecone index for similar vectors."""
        try:
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict or {}
            )
            return response.matches
        except Exception as e:
            logger.error(f"❌ Error querying Pinecone: {str(e)}")
            raise

    def delete_all(self):
        """Delete all vectors in the index."""
        try:
            self.index.delete(delete_all=True)
            logger.info("🗑️ All vectors deleted from Pinecone index.")
        except Exception as e:
            logger.error(f"Error deleting vectors: {str(e)}")
            raise

    def get_stats(self):
        """Fetch index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            raise


# Create global instance
vector_db = VectorDatabase()
