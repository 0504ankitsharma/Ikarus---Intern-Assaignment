import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import logging
from typing import List, Optional
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)

class ImageService:
    def __init__(self):
        self.model = None
        self.transform = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained CV model"""
        try:
            logger.info(f"Loading CV model: {settings.CV_MODEL}")
            
            # Load EfficientNet model
            self.model = models.efficientnet_b0(pretrained=True)
            self.model.eval()
            
            # Define image transformations
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("CV model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading CV model: {str(e)}")
            raise
    
    def load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Load image from URL"""
        try:
            # Clean URL
            url = url.strip()
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
        except Exception as e:
            logger.warning(f"Error loading image from {url}: {str(e)}")
            return None
    
    def extract_features(self, image: Image.Image) -> Optional[List[float]]:
        """Extract features from image using CV model"""
        try:
            # Transform image
            img_tensor = self.transform(image).unsqueeze(0)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Convert to numpy and then to list
            features_np = features.squeeze().numpy()
            
            # Normalize features
            features_normalized = features_np / (np.linalg.norm(features_np) + 1e-8)
            
            return features_normalized.tolist()
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None
    
    def get_image_embedding(self, image_url: str) -> Optional[List[float]]:
        """Get embedding for an image from URL"""
        try:
            img = self.load_image_from_url(image_url)
            if img is None:
                return None
            
            features = self.extract_features(img)
            
            # Adjust dimension to match Pinecone
            if features and len(features) < settings.PINECONE_DIMENSION:
                padding = [0.0] * (settings.PINECONE_DIMENSION - len(features))
                features.extend(padding)
            elif features and len(features) > settings.PINECONE_DIMENSION:
                features = features[:settings.PINECONE_DIMENSION]
            
            return features
        except Exception as e:
            logger.error(f"Error getting image embedding: {str(e)}")
            return None
    
    def classify_image(self, image_url: str) -> Optional[str]:
        """Classify image (basic implementation)"""
        try:
            # This is a simplified version
            # In production, you'd use a fine-tuned model for furniture classification
            features = self.get_image_embedding(image_url)
            if features:
                return "furniture"  # Placeholder
            return None
        except Exception as e:
            logger.error(f"Error classifying image: {str(e)}")
            return None

# Global instance
image_service = ImageService()