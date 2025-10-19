import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import recommendations, analytics, chat

# Setup logging

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("app.main")

app = FastAPI(title="Ikarus Furniture Recommendation API 🚀")

# CORS configuration

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# Register routers
app.include_router(recommendations.router)
app.include_router(chat.router)
app.include_router(analytics.router)

@app.on_event("startup")
async def startup_event():
    """Run background indexing on startup."""
    logger.info("🚀 Starting up application...")
    try:
        from app.services.recommendation_service import recommendation_service
        logger.info("🔄 Starting background product indexing...")
        asyncio.create_task(recommendation_service.index_products())
    except Exception as e:
        logger.error(f"❌ Error during background indexing: {e}")

@app.get("/")
async def root():
    return {"message": "✅ Ikarus Furniture Recommendation API is running!"}

@app.get("/health")
async def health():
    return {"status": "ok"}
