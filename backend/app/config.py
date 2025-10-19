from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # OpenAI Configuration (OPTIONAL - uses free alternatives if not provided)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Pinecone Configuration
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str
    PINECONE_DIMENSION: int = 1024
    PINECONE_ENVIRONMENT: str = "us-east-1-aws"
    PINECONE_REGION: str = "us-east-1"
    
    # Data Configuration
    DATA_PATH: str = "data/dataset.csv"
    
    # Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CV_MODEL: str = "efficientnet_b0"
    
    # API Configuration
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()