from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse, ChatMessage, RecommendedProduct, Product
from app.services.embedding_service import embedding_service
from app.services.genai_service import genai_service
from app.database import vector_db
import time
import logging

logger = logging.getLogger(__name__)

# Base path for chat under recommendation namespace
router = APIRouter(prefix="/api/recommendations", tags=["Chat"])

@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(req: ChatRequest):
    """
    Chat endpoint that provides AI-generated responses and product recommendations.
    """
    try:
        start_time = time.time()
        logger.info(f"🗣️ Chat request received: {req.message}")

        # Step 1: Get embedding for query
        query_vector = await embedding_service.embed_text(req.message)

        # Step 2: Query vector DB for recommendations
        results = vector_db.query_vectors(query_vector, top_k=req.top_k)
        if not results or len(results) == 0:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        # Step 3: Build recommendations
        recommendations = []
        for match in results:
            # Handle both dict and object types from Pinecone
            metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
            
            # Parse categories if string
            categories = metadata.get("categories", [])
            if isinstance(categories, str):
                categories = categories.split(",") if categories else []
            
            product = Product(
                uniq_id=metadata.get("uniq_id", ""),
                title=metadata.get("title", "Unknown Product"),
                brand=metadata.get("brand"),
                description=metadata.get("description"),
                price=metadata.get("price"),
                categories=categories,
                images=metadata.get("images", []),
                manufacturer=metadata.get("manufacturer"),
                package_dimensions=metadata.get("package_dimensions"),
                country_of_origin=metadata.get("country_of_origin"),
                material=metadata.get("material"),
                color=metadata.get("color"),
            )
            recommendations.append(RecommendedProduct(product=product, score=float(score)))

        # Step 4: Generate AI assistant reply
        ai_reply = await genai_service.chat_response(req.message, req.conversation_history)

        # Step 5: Update conversation history
        updated_history = req.conversation_history + [
            ChatMessage(role="user", content=req.message),
            ChatMessage(role="assistant", content=ai_reply),
        ]

        total_time = round(time.time() - start_time, 3)
        logger.info(f"💬 Chat completed in {total_time}s")

        return ChatResponse(
            message=ai_reply,
            recommendations=recommendations,
            conversation_history=updated_history,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
