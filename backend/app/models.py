from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Product(BaseModel):
    uniq_id: str
    title: str
    brand: Optional[str] = None
    description: Optional[str] = None
    price: Optional[str] = None
    categories: Optional[List[str]] = None
    images: Optional[List[str]] = None
    manufacturer: Optional[str] = None
    package_dimensions: Optional[str] = None
    country_of_origin: Optional[str] = None
    material: Optional[str] = None
    color: Optional[str] = None

class RecommendationRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    include_description: bool = True

class GeneratedDescription(BaseModel):
    text: str
    original: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class RecommendedProduct(BaseModel):
    product: Product
    score: float
    generated_description: Optional[GeneratedDescription] = None

class RecommendationResponse(BaseModel):
    query: str
    recommendations: List[RecommendedProduct]
    total_results: int
    processing_time: float

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []
    top_k: int = Field(default=5, ge=1, le=20)

class ChatResponse(BaseModel):
    message: str
    recommendations: List[RecommendedProduct]
    conversation_history: List[ChatMessage]

class AnalyticsResponse(BaseModel):
    total_products: int
    categories_distribution: Dict[str, int]
    brand_distribution: Dict[str, int]
    price_statistics: Dict[str, Any]
    material_distribution: Dict[str, int]
    color_distribution: Dict[str, int]
    country_distribution: Dict[str, int]
    top_brands: List[Dict[str, Any]]
    price_ranges: Dict[str, int]