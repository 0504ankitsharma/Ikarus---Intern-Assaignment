import pandas as pd
import ast
import logging
from typing import List, Dict
from app.config import settings
from app.models import Product

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.df = None
        self.products = []
    
    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV"""
        try:
            self.df = pd.read_csv(settings.DATA_PATH)
            logger.info(f"Loaded {len(self.df)} products from dataset")
            self._preprocess_data()
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _preprocess_data(self):
        """Preprocess the data"""
        # Parse categories from string to list
        if 'categories' in self.df.columns:
            self.df['categories'] = self.df['categories'].apply(self._parse_list)
        
        # Parse images from string to list
        if 'images' in self.df.columns:
            self.df['images'] = self.df['images'].apply(self._parse_list)
        
        # Fill NaN values
        self.df = self.df.fillna("")
        
        # Convert to Product models
        self.products = [self._row_to_product(row) for _, row in self.df.iterrows()]
        logger.info(f"Preprocessed {len(self.products)} products")
    
    def _parse_list(self, value):
        """Parse string representation of list"""
        if pd.isna(value) or value == "":
            return []
        try:
            if isinstance(value, str):
                # Try to parse as Python literal
                return ast.literal_eval(value)
            return value if isinstance(value, list) else []
        except:
            # If parsing fails, split by comma
            return [item.strip() for item in str(value).split(',')]
    
    def _row_to_product(self, row) -> Product:
        """Convert DataFrame row to Product model"""
        return Product(
            uniq_id=str(row.get('uniq_id', '')),
            title=str(row.get('title', '')),
            brand=str(row.get('brand', '')) if row.get('brand') else None,
            description=str(row.get('description', '')) if row.get('description') else None,
            price=str(row.get('price', '')) if row.get('price') else None,
            categories=row.get('categories', []) if isinstance(row.get('categories'), list) else [],
            images=row.get('images', []) if isinstance(row.get('images'), list) else [],
            manufacturer=str(row.get('manufacturer', '')) if row.get('manufacturer') else None,
            package_dimensions=str(row.get('package_dimensions', '')) if row.get('package_dimensions') else None,
            country_of_origin=str(row.get('country_of_origin', '')) if row.get('country_of_origin') else None,
            material=str(row.get('material', '')) if row.get('material') else None,
            color=str(row.get('color', '')) if row.get('color') else None
        )
    
    def get_products(self) -> List[Product]:
        """Get all products as Product models"""
        if not self.products:
            self.load_data()
        return self.products
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get products as DataFrame"""
        if self.df is None:
            self.load_data()
        return self.df
    
    def get_product_by_id(self, uniq_id: str) -> Product:
        """Get a specific product by ID"""
        for product in self.get_products():
            if product.uniq_id == uniq_id:
                return product
        return None

# Global instance
data_loader = DataLoader()