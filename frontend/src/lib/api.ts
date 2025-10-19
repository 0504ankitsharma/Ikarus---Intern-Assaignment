import type {
  Analytics,
  ChatMessage,
  ChatRequest,
  ChatResponse,
  HealthResponse,
  Product,
  ProductsResponse,
  RecommendedProduct,
  SearchResponse,
  SimilarProductsResponse,
} from '@/types';

// API Base URL from environment or default to production
const DEFAULT_BASE = 'https://0504ankitsharma-ikarus.hf.space';
const API_BASE = (import.meta.env.VITE_API_URL as string) || DEFAULT_BASE;

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText} ${text}`);
  }
  return res.json() as Promise<T>;
}

// Health check endpoint
export async function health(): Promise<HealthResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    return handle(res);
  } finally {
    clearTimeout(timeout);
  }
}

// Product search with natural language
export async function searchProducts(query: string, topK = 5, includeDescription = true): Promise<SearchResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const res = await fetch(`${API_BASE}/api/recommendations/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        query, 
        top_k: topK, 
        include_description: includeDescription 
      }),
      signal: controller.signal,
    });
    return handle(res);
  } finally {
    clearTimeout(timeout);
  }
}

// Chat-based recommendations with conversation history
export async function chatRecommendations(
  message: string, 
  conversationHistory: ChatMessage[] = [], 
  topK = 5
): Promise<ChatResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);

  try {
    const res = await fetch(`${API_BASE}/api/recommendations/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        message, 
        conversation_history: conversationHistory,
        top_k: topK 
      }),
      signal: controller.signal,
    });
    return handle(res);
  } finally {
    clearTimeout(timeout);
  }
}


// Get similar products
export async function similarProducts(productId: string, topK = 5): Promise<SimilarProductsResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);

  try {
    const res = await fetch(
      `${API_BASE}/api/recommendations/similar/${encodeURIComponent(productId)}?top_k=${topK}`,
      { signal: controller.signal }
    );
    return handle(res);
  } finally {
    clearTimeout(timeout);
  }
}

// Get analytics dashboard data
export async function analytics(): Promise<Analytics> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const res = await fetch(`${API_BASE}/api/analytics/`, { signal: controller.signal });
    return handle(res);
  } finally {
    clearTimeout(timeout);
  }
}

// Get all products
export async function getAllProducts(): Promise<ProductsResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const res = await fetch(`${API_BASE}/api/analytics/products`, { signal: controller.signal });
    return handle(res);
  } finally {
    clearTimeout(timeout);
  }
}


// Helper functions for data normalization
export function normalizeImages(images: string[] | string | null): string[] {
  if (!images) return [];
  if (Array.isArray(images)) return images;
  try {
    const parsed = JSON.parse(images);
    if (Array.isArray(parsed)) return parsed;
  } catch {}
  return String(images)
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
}

export function normalizeCategories(categories: string[] | string | null): string[] {
  if (!categories) return [];
  if (Array.isArray(categories)) return categories;
  try {
    const parsed = JSON.parse(categories);
    if (Array.isArray(parsed)) return parsed;
  } catch {}
  return String(categories)
    .split('>')
    .flatMap((c) => c.split(','))
    .map((s) => s.trim())
    .filter(Boolean);
}

export function extractGenText(gen: RecommendedProduct['generated_description']): string | null {
  if (!gen) return null;
  return gen.text;
}

export function parsePrice(price: string | null): number | null {
  if (!price) return null;
  const cleaned = price.replace(/[^\d.]/g, '');
  const parsed = parseFloat(cleaned);
  return isNaN(parsed) ? null : parsed;
}
