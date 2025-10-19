"""
GenAI service that provides conversational responses and product descriptions.
If OpenAI is available it will use it; otherwise, it produces deterministic safe text
suitable for local testing and development.
"""
import logging
from typing import List
from app.config import settings
from app.models import Product, GeneratedDescription, ChatMessage

logger = logging.getLogger(__name__)
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

class GenAIService:
    def __init__(self):
        self.client = None
        self.model_name = None
        
        # OpenAI is OPTIONAL - system works fine without it
        if _HAS_OPENAI and getattr(settings, "OPENAI_API_KEY", None):
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                self.model_name = getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
                logger.info("✅ OpenAI provider initialized with model %s (OPTIONAL)", self.model_name)
            except Exception as e:
                logger.warning("⚠️ OpenAI not available, using free local fallbacks: %s", e)
                self.client = None
        else:
            logger.info("🆓 OpenAI not configured; using FREE local text generation")

    def generate_product_description(self, product: Product) -> GeneratedDescription:
        """Return a GeneratedDescription dataclass-like dict or model instance.
        If an external generative API is available, attempt to use it; otherwise, create a safe description.
        """
        try:
            title = getattr(product, "title", "") or "Unknown product"
            brand = getattr(product, "brand", "") or ""
            desc = getattr(product, "description", "") or ""
            material = getattr(product, "material", "") or ""
            color = getattr(product, "color", "") or ""
            generated = f"{title} by {brand}. {desc} Material: {material}. Color: {color}."
            # Try provider if available
            if self.client:
                try:
                    prompt = f"Write a short product description for the following product:\n\nTitle: {title}\nBrand: {brand}\nDescription: {desc}\nMaterial: {material}\nColor: {color}\n\nKeep it under 80 words."
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150,
                        temperature=0.7
                    )
                    generated = response.choices[0].message.content
                except Exception as e:
                    logger.warning("OpenAI generation failed, using fallback: %s", e)
            # Return a simple GeneratedDescription-like object (dict) to avoid tight coupling
            return GeneratedDescription(text=generated, original=desc)
        except Exception as e:
            logger.exception("Error generating product description: %s", e)
            return GeneratedDescription(text=str(e))
    
    async def generate_description(self, product: Product) -> str:
        """Generate a product description and return just the text."""
        try:
            desc = self.generate_product_description(product)
            return desc.text
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return f"{product.title} - {product.description or 'No description available'}"

    def conversational_response(self, user_query: str, products: List[Product]) -> str:
        """Return a conversational assistant message. Fallback to simple template if provider unavailable."""
        try:
            if self.client:
                # Try provider chat style if available
                try:
                    prompt = f"User asked: {user_query}\nReturn a short helpful message and summarize top {min(3,len(products))} product titles."
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=150
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.warning("OpenAI conversational failed: %s", e)
            # Local fallback
            titles = [getattr(p, "title", "product") for p in (products or [])][:3]
            if titles:
                return f"I found {len(products)} products. Top picks: " + ", ".join(titles) + "."
            return "I couldn't find anything matching your query — try different keywords."
        except Exception as e:
            logger.exception("Error building conversational response: %s", e)
            return "Sorry, something went wrong."
    
    async def chat_response(self, message: str, conversation_history: List[ChatMessage]) -> str:
        """Generate a chat response based on user message and conversation history."""
        try:
            if self.client:
                try:
                    # Build messages from conversation history
                    messages = [{"role": "system", "content": "You are a helpful furniture shopping assistant."}]
                    for msg in conversation_history[-5:]:
                        messages.append({"role": msg.role, "content": msg.content})
                    messages.append({"role": "user", "content": message})
                    
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=200,
                        temperature=0.8
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logger.warning(f"OpenAI chat failed, using fallback: {e}")
            
            # Fallback response
            return f"I understand you're looking for furniture. Let me find some recommendations for you based on: {message}"
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            return "I'm here to help you find furniture. What are you looking for?"
    
    def enhance_query(self, query: str) -> str:
        """Enhance user query for better semantic search."""
        try:
            if self.client:
                try:
                    prompt = f"""Expand this furniture search query to include related terms and synonyms for better search results. Keep it concise (under 50 words).

Query: {query}

Expanded query:"""
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100
                    )
                    enhanced = response.choices[0].message.content.strip()
                    if enhanced and len(enhanced) < 200:
                        return enhanced
                except Exception as e:
                    logger.warning(f"Query enhancement failed: {e}")
            # Fallback: return original query
            return query
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query

# Global instance
genai_service = GenAIService()
