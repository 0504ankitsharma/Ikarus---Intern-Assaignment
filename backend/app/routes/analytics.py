from fastapi import APIRouter, HTTPException
import logging
import pandas as pd
from collections import Counter
from app.models import AnalyticsResponse
from app.utils.data_loader import data_loader
import re

router = APIRouter(prefix="/api/analytics", tags=["analytics"])
logger = logging.getLogger(__name__)

def extract_price_value(price_str: str) -> float:
    """Extract numeric value from price string"""
    if not price_str:
        return 0.0
    try:
        # Remove any non-numeric or decimal characters
        numeric_str = re.sub(r'[^\d.]', '', str(price_str))
        return float(numeric_str) if numeric_str else 0.0
    except Exception as e:
        logger.warning(f"Price conversion failed for {price_str}: {e}")
        return 0.0

@router.get("/", response_model=AnalyticsResponse)
async def get_analytics():
    """
    Get comprehensive analytics about the product dataset.
    """
    try:
        df = data_loader.get_dataframe()

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found in dataset")

        # Convert relevant columns to lists or default empty lists if missing
        for col in ['categories', 'brand', 'price', 'material', 'color', 'country_of_origin']:
            if col not in df.columns:
                df[col] = []

        # Total products
        total_products = len(df)

        # Categories distribution
        all_categories = []
        for cats in df['categories']:
            if isinstance(cats, list):
                all_categories.extend(cats)
            elif isinstance(cats, str) and cats.strip():
                all_categories.append(cats.strip())
        categories_distribution = dict(Counter(all_categories).most_common(20))

        # Brand distribution
        brands = df['brand'].dropna().astype(str).str.strip()
        brand_distribution = dict(Counter(brands).most_common(20))

        # Price statistics
        df['price_numeric'] = df['price'].apply(extract_price_value)
        valid_prices = df[df['price_numeric'] > 0]['price_numeric']
        price_stats = {
            "min": float(valid_prices.min()) if not valid_prices.empty else 0,
            "max": float(valid_prices.max()) if not valid_prices.empty else 0,
            "mean": float(valid_prices.mean()) if not valid_prices.empty else 0,
            "median": float(valid_prices.median()) if not valid_prices.empty else 0,
        }

        # Material distribution
        materials = df['material'].dropna().astype(str).str.strip()
        material_distribution = dict(Counter(materials).most_common(15))

        # Color distribution
        colors = df['color'].dropna().astype(str).str.strip()
        color_distribution = dict(Counter(colors).most_common(15))

        # Country distribution
        countries = df['country_of_origin'].dropna().astype(str).str.strip()
        country_distribution = dict(Counter(countries).most_common(10))

        # Top brands by product count
        top_brands = [{"brand": brand, "count": int(count)} for brand, count in Counter(brands).most_common(10)]

        # Price ranges
        price_ranges = {
            "Under $25": int((df['price_numeric'] < 25).sum()),
            "$25-$50": int(((df['price_numeric'] >= 25) & (df['price_numeric'] < 50)).sum()),
            "$50-$100": int(((df['price_numeric'] >= 50) & (df['price_numeric'] < 100)).sum()),
            "$100-$200": int(((df['price_numeric'] >= 100) & (df['price_numeric'] < 200)).sum()),
            "$200+": int((df['price_numeric'] >= 200).sum())
        }

        return AnalyticsResponse(
            total_products=total_products,
            categories_distribution=categories_distribution,
            brand_distribution=brand_distribution,
            price_statistics=price_stats,
            material_distribution=material_distribution,
            color_distribution=color_distribution,
            country_distribution=country_distribution,
            top_brands=top_brands,
            price_ranges=price_ranges
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_analytics: {e}")
        raise HTTPException(status_code=500, detail="Error generating analytics data")

@router.get("/products")
async def get_all_products():
    """
    Get all products (for frontend display)
    """
    try:
        products = data_loader.get_products()
        if not products:
            raise HTTPException(status_code=404, detail="No products found")
        return {"products": products, "total": len(products)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_all_products: {e}")
        raise HTTPException(status_code=500, detail="Error fetching product list")
