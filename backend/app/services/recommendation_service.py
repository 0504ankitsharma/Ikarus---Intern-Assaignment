from typing import List
import logging
import asyncio
from app.models import Product, RecommendedProduct
from app.services.embedding_service import embedding_service
from app.services.genai_service import genai_service
from app.database import vector_db
from app.utils.data_loader import data_loader

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(self):
        self.products_indexed = False

    async def index_products(self):
        """Index all products in vector database"""
        try:
            if self.products_indexed:
                logger.info("Products already indexed")
                return

            logger.info("Starting product indexing...")
            products = data_loader.get_products()
            vectors = []

            for product in products:
                try:
                    # Use synchronous encode_product method
                    embedding = embedding_service.encode_product(product)
                    metadata = {
                        "uniq_id": product.uniq_id,
                        "title": product.title,
                        "brand": product.brand or "",
                        "description": product.description or "",
                        "price": product.price or "",
                        "categories": ",".join(product.categories) if product.categories else "",
                        "images": product.images or [],
                        "material": product.material or "",
                        "color": product.color or "",
                        "manufacturer": product.manufacturer or "",
                        "package_dimensions": product.package_dimensions or "",
                        "country_of_origin": product.country_of_origin or "",
                    }
                    vectors.append((product.uniq_id, embedding, metadata))
                except Exception as e:
                    logger.error(f"Error processing product {product.uniq_id}: {str(e)}")
                    continue

            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                vector_db.upsert_vectors(batch)
                logger.info(f"Indexed {i + len(batch)}/{len(vectors)} products")

            self.products_indexed = True
            logger.info(f"✅ Successfully indexed {len(vectors)} products")

        except Exception as e:
            logger.error(f"Error indexing products: {str(e)}")
            raise

    async def get_recommendations(self, query: str, top_k: int = 5,
                                  include_description: bool = True) -> List[RecommendedProduct]:
        """Get product recommendations based on query"""
        try:
            if not self.products_indexed:
                await self.index_products()

            enhanced_query = genai_service.enhance_query(query)
            logger.info(f"Enhanced query: {enhanced_query}")

            query_embedding = await embedding_service.embed_text(enhanced_query)
            results = vector_db.query_vectors(query_embedding, top_k=top_k)

            # Retrieve matched products
            products = []
            for match in results:
                p = data_loader.get_product_by_id(match.id)
                if p:
                    products.append(p)

            if not products:
                logger.warning("No matching products found.")
                return []

            # Generate product descriptions concurrently
            async def generate_all_descriptions():
                tasks = [
                    asyncio.to_thread(genai_service.generate_product_description, product)
                    for product in products
                ]
                return await asyncio.gather(*tasks)

            generated_descriptions = []
            if include_description:
                try:
                    generated_descriptions = await generate_all_descriptions()
                except Exception as e:
                    logger.error(f"Error generating descriptions: {e}")
                    generated_descriptions = [None] * len(products)
            else:
                generated_descriptions = [None] * len(products)

            recommendations = [
                RecommendedProduct(
                    product=product,
                    score=float(match.score),
                    generated_description=desc
                )
                for product, match, desc in zip(products, results, generated_descriptions)
            ]

            logger.info(f"Found {len(recommendations)} recommendations for query: {query}")
            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

    async def get_similar_products(self, product_id: str, top_k: int = 5) -> List[RecommendedProduct]:
        """Get similar products"""
        try:
            product = data_loader.get_product_by_id(product_id)
            if not product:
                logger.warning(f"Product not found: {product_id}")
                return []

            # Use synchronous encode_product method
            product_embedding = embedding_service.encode_product(product)
            results = vector_db.query_vectors(product_embedding, top_k=top_k + 1)

            recommendations = []
            for match in results[1:]:
                similar_product = data_loader.get_product_by_id(match.id)
                if similar_product:
                    recommendations.append(RecommendedProduct(
                        product=similar_product,
                        score=float(match.score),
                        generated_description=None
                    ))

            return recommendations

        except Exception as e:
            logger.error(f"Error getting similar products: {str(e)}")
            raise


# Global instance
recommendation_service = RecommendationService()
