export type ImageURL = string;

// Product interface matching backend API
export interface Product {
  uniq_id: string;
  title: string;
  brand: string | null;
  description: string | null;
  price: string | null;
  categories: string[];
  images: string[];
  manufacturer: string | null;
  package_dimensions: string | null;
  country_of_origin: string | null;
  material: string | null;
  color: string | null;
}

// Generated description with timestamp
export interface GeneratedDescription {
  text: string;
  original: string | null;
  timestamp: string;
}

// Recommended product with score and description
export interface RecommendedProduct {
  product: Product;
  score: number;
  generated_description: GeneratedDescription | null;
}

// Search response structure
export interface SearchResponse {
  query: string;
  recommendations: RecommendedProduct[];
  total_results: number;
  processing_time: number;
}

// Similar products response
export interface SimilarProductsResponse {
  product_id: string;
  similar_products: RecommendedProduct[];
  total: number;
}

// Chat message with timestamp
export type Role = 'user' | 'assistant';

export interface ChatMessage {
  role: Role;
  content: string;
  timestamp: string;
}

// Chat request
export interface ChatRequest {
  message: string;
  conversation_history?: ChatMessage[];
  top_k?: number;
}

// Chat response
export interface ChatResponse {
  message: string;
  recommendations: RecommendedProduct[];
  conversation_history: ChatMessage[];
}

// Analytics response structure
export interface Analytics {
  total_products: number;
  categories_distribution: Record<string, number>;
  brand_distribution: Record<string, number>;
  price_statistics: {
    min: number;
    max: number;
    mean: number;
    median: number;
  };
  material_distribution: Record<string, number>;
  color_distribution: Record<string, number>;
  country_distribution: Record<string, number>;
  top_brands: Array<{ brand: string; count: number }>;
  price_ranges: Record<string, number>;
}

// Products list response
export interface ProductsResponse {
  products: Product[];
  total: number;
}

// Health check response
export interface HealthResponse {
  status: string;
}