"""
Embedding service using sentence-transformers (free, local).
Falls back to hash-based embeddings if model loading fails.
OpenAI embeddings are optional and not used by default.
"""
import os
import logging
import hashlib
from typing import List
from app.config import settings
from app.models import Product

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

class EmbeddingService:
    def __init__(self):
        self.sentence_model = None
        self.openai_client = None
        self.use_openai = False
        
        # Try to load sentence-transformers first (FREE, no API needed)
        if _HAS_SENTENCE_TRANSFORMERS:
            try:
                model_name = getattr(settings, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                logger.info(f"Loading sentence-transformers model: {model_name}")
                self.sentence_model = SentenceTransformer(model_name)
                logger.info("✅ Sentence-transformers embedding configured (FREE, no quota limits!)")
            except Exception as e:
                logger.warning("Could not load sentence-transformers: %s", e)
        
        # OpenAI is optional (only if sentence-transformers fails)
        if not self.sentence_model and _HAS_OPENAI and getattr(settings, "OPENAI_API_KEY", None):
            try:
                self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
                self.use_openai = True
                logger.info("OpenAI embedding configured as fallback")
            except Exception as e:
                logger.warning("Could not initialize OpenAI: %s", e)
        
        if not self.sentence_model and not self.openai_client:
            logger.info("Using hash-based embeddings as final fallback")
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text query."""
        try:
            # Try sentence-transformers first (FREE)
            if self.sentence_model:
                try:
                    embedding = self.sentence_model.encode(text, convert_to_tensor=False)
                    embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                    
                    # Pad or truncate to match Pinecone dimension
                    if len(embedding) < settings.PINECONE_DIMENSION:
                        embedding.extend([0.0] * (settings.PINECONE_DIMENSION - len(embedding)))
                    elif len(embedding) > settings.PINECONE_DIMENSION:
                        embedding = embedding[:settings.PINECONE_DIMENSION]
                    return embedding
                except Exception as e:
                    logger.warning("Sentence-transformers failed: %s", e)
            
            # Try OpenAI if sentence-transformers failed
            if self.openai_client and self.use_openai:
                try:
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=text
                    )
                    embedding = response.data[0].embedding
                    # Pad or truncate to match Pinecone dimension
                    if len(embedding) < settings.PINECONE_DIMENSION:
                        embedding.extend([0.0] * (settings.PINECONE_DIMENSION - len(embedding)))
                    elif len(embedding) > settings.PINECONE_DIMENSION:
                        embedding = embedding[:settings.PINECONE_DIMENSION]
                    return embedding
                except Exception as e:
                    logger.warning("OpenAI embedding failed: %s", e)
            
            # Final fallback to hash-based
            return _text_to_vector(text, dim=settings.PINECONE_DIMENSION)
        except Exception as e:
            logger.exception("Error embedding text: %s", e)
            raise
    
    def encode_product(self, product) -> List[float]:
        """Encode product for embedding. (Synchronous wrapper)"""
        from app.models import Product
        text = " ".join(filter(None, [
            getattr(product, "title", "") or "",
            getattr(product, "brand", "") or "",
            getattr(product, "description", "") or "",
            getattr(product, "material", "") or "",
            getattr(product, "color", "") or "",
            ",".join(getattr(product, "categories", []) if getattr(product, "categories", None) else []),
        ]))
        
        # Try sentence-transformers first (FREE)
        if self.sentence_model:
            try:
                embedding = self.sentence_model.encode(text, convert_to_tensor=False)
                embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                
                # Pad or truncate to match Pinecone dimension
                if len(embedding) < settings.PINECONE_DIMENSION:
                    embedding.extend([0.0] * (settings.PINECONE_DIMENSION - len(embedding)))
                elif len(embedding) > settings.PINECONE_DIMENSION:
                    embedding = embedding[:settings.PINECONE_DIMENSION]
                return embedding
            except Exception as e:
                logger.warning("Sentence-transformers failed: %s", e)
        
        # Try OpenAI if sentence-transformers failed
        if self.openai_client and self.use_openai:
            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = response.data[0].embedding
                # Pad or truncate to match Pinecone dimension
                if len(embedding) < settings.PINECONE_DIMENSION:
                    embedding.extend([0.0] * (settings.PINECONE_DIMENSION - len(embedding)))
                elif len(embedding) > settings.PINECONE_DIMENSION:
                    embedding = embedding[:settings.PINECONE_DIMENSION]
                return embedding
            except Exception as e:
                logger.warning("OpenAI embedding failed: %s", e)
        
        # Final fallback
        return _text_to_vector(text, dim=settings.PINECONE_DIMENSION)

def _text_to_vector(text: str, dim: int = 1536):
    """Deterministic pseudo-embedding using SHA256 hashed chunks -> vector of floats in [-1,1]."""
    if not text:
        return [0.0] * dim
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # expand to dim floats by hashing with counters
    vec = []
    i = 0
    while len(vec) < dim:
        chunk = hashlib.sha256(h + i.to_bytes(2, "big")).digest()
        for b in chunk:
            vec.append((b / 127.5) - 1.0)  # map 0-255 -> -1..1
            if len(vec) >= dim:
                break
        i += 1
    return vec

class ProductEncoderError(Exception):
    pass

class ProductEncoder:
    def __init__(self, service: EmbeddingService):
        self.service = service

    def encode_product(self, product: Product) -> List[float]:
        """Return embedding vector for a product. Uses external API if configured else fallback."""
        try:
            text = " ".join(filter(None, [
                getattr(product, "title", "") or "",
                getattr(product, "brand", "") or "",
                getattr(product, "description", "") or "",
                getattr(product, "material", "") or "",
                getattr(product, "color", "") or "",
                ",".join(getattr(product, "categories", []) if getattr(product, "categories", None) else []),
            ]))
            if self.service.model:
                # attempt to call gemini embed (guarded)
                try:
                    response = genai.embed_content(model=self.service.model, content=text, task_type="retrieval_document")
                    # response may be dict-like with 'embedding' key or attribute
                    emb = None
                    if isinstance(response, dict) and "embedding" in response:
                        emb = response["embedding"]
                    else:
                        emb = getattr(response, "embedding", None) or getattr(response, "embeddings", None)
                    if not emb:
                        raise ProductEncoderError("No embedding returned by provider")
                    return list(emb)
                except Exception as e:
                    logger.warning("Provider embedding failed, falling back to local: %s", e)
                    return _text_to_vector(text)
            else:
                return _text_to_vector(text)
        except Exception as e:
            logger.exception("Error encoding product %s: %s", getattr(product, "uniq_id", "<unknown>"), e)
            raise

# Global instances for import convenience
embedding_service = EmbeddingService()
product_encoder = ProductEncoder(embedding_service)
