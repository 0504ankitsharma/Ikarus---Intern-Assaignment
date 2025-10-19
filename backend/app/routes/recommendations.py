import time
import logging
from fastapi import APIRouter, HTTPException
from app.models import (
    RecommendationRequest,
    RecommendationResponse,
    Product,
    RecommendedProduct,
    GeneratedDescription,
)
from app.services.embedding_service import embedding_service
from app.services.genai_service import genai_service
from app.database import vector_db
from app.utils.data_loader import data_loader

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["Recommendations"])


@router.post("/search", response_model=RecommendationResponse)
async def get_recommendations(req: RecommendationRequest):
    """
    Generate product recommendations based on a user's query.
    """
    try:
        start_time = time.time()
        logger.info(f"🟢 Received recommendation request for query: {req.query}")

        # Step 1: Generate query embedding
        query_vector = await embedding_service.embed_text(req.query)

        # Step 2: Query the Pinecone vector database
        results = vector_db.query_vectors(query_vector, top_k=req.top_k)
        if not results or len(results) == 0:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        recommended_products = []
        for match in results:
            metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})

            product = Product(
                uniq_id=metadata.get("uniq_id", ""),
                title=metadata.get("title", "Unknown Product"),
                brand=metadata.get("brand", None),
                description=metadata.get("description", None),
                price=metadata.get("price", None),
                categories=metadata.get("categories", "").split(",") if isinstance(metadata.get("categories"), str) else [],
                images=metadata.get("images", []),
                manufacturer=metadata.get("manufacturer", None),
                package_dimensions=metadata.get("package_dimensions", None),
                country_of_origin=metadata.get("country_of_origin", None),
                material=metadata.get("material", None),
                color=metadata.get("color", None),
            )

            # Step 3: Optionally generate AI-enhanced description
            generated_description = None
            if req.include_description:
                try:
                    gen_text = await genai_service.generate_description(product)
                    generated_description = GeneratedDescription(
                        text=gen_text,  # Fixed: 'text' not 'generated'
                        original=product.description,
                    )
                except Exception as gen_err:
                    logger.warning(f"⚠️ Failed to generate description: {gen_err}")
                    generated_description = None

            recommended_products.append(
                RecommendedProduct(
                    product=product,
                    score=float(match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)),
                    generated_description=generated_description,
                )
            )

        total_time = round(time.time() - start_time, 3)
        response = RecommendationResponse(
            query=req.query,
            recommendations=recommended_products,
            total_results=len(recommended_products),
            processing_time=total_time,
        )

        logger.info(f"✅ Recommendations completed for query: '{req.query}' in {total_time}s")
        return response

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ Error during recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/similar/{product_id}")
async def get_similar_products(product_id: str, top_k: int = 5):
    """
    Get similar products based on a product ID.
    """
    try:
        logger.info(f"🔍 Finding similar products for: {product_id}")
        
        # Get the product
        product = data_loader.get_product_by_id(product_id)
        if not product:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
        
        # Get product embedding
        product_vector = embedding_service.encode_product(product)
        
        # Query for similar products (top_k + 1 to exclude the product itself)
        results = vector_db.query_vectors(product_vector, top_k=top_k + 1)
        
        # Build similar products list (skip first result as it's the same product)
        similar_products = []
        for match in results[1:top_k + 1]:
            metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
            
            similar_product = Product(
                uniq_id=metadata.get("uniq_id", ""),
                title=metadata.get("title", "Unknown Product"),
                brand=metadata.get("brand", None),
                description=metadata.get("description", None),
                price=metadata.get("price", None),
                categories=metadata.get("categories", "").split(",") if isinstance(metadata.get("categories"), str) else [],
                images=metadata.get("images", []),
                manufacturer=metadata.get("manufacturer", None),
                package_dimensions=metadata.get("package_dimensions", None),
                country_of_origin=metadata.get("country_of_origin", None),
                material=metadata.get("material", None),
                color=metadata.get("color", None),
            )
            
            similar_products.append(
                RecommendedProduct(
                    product=similar_product,
                    score=float(match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)),
                    generated_description=None,
                )
            )
        
        logger.info(f"✅ Found {len(similar_products)} similar products")
        return {
            "product_id": product_id,
            "similar_products": similar_products,
            "total": len(similar_products)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error finding similar products: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
