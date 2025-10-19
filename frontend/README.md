# Furniture Recommendation Frontend (React)

Concise React + Vite + TypeScript frontend for your FastAPI backend:
- Chat-style product recommendations with generated descriptions
- Show recommended items with images and attributes
- Fetch similar products
- Analytics page with KPIs and charts

Backend: [FastAPI on HF Spaces](https://0504ankitsharma-ikarus.hf.space)

## Features

- Recommendation chat
  - POST `/api/recommendations/chat`
  - Accumulates messages and shows recommendations
- Similar products
  - GET `/api/recommendations/similar/{product_id}`
- Analytics
  - GET `/api/analytics/` (summary)
  - GET `/api/analytics/products` (full list)
- Simple, clean UI and routing

## Quickstart

1. Prerequisites
   - Node.js 18+
   - pnpm / npm / yarn

2. Install
   ```bash
   pnpm install
   # or: npm install
   ```

3. Configure (optional)
   Copy `.env.example` to `.env` and adjust:
   ```bash
   VITE_API_BASE_URL=https://0504ankitsharma-ikarus.hf.space
   ```

4. Run
   ```bash
   pnpm dev
   # open http://localhost:5173
   ```

5. Build
   ```bash
   pnpm build
   pnpm preview
   ```

## Project Structure

- `src/pages/ChatPage.tsx` — Chat interface calling `/api/recommendations/chat`
- `src/components/ProductCard.tsx` — Product display with "Find similar"
- `src/pages/AnalyticsPage.tsx` — KPIs and charts from analytics endpoints
- `src/lib/api.ts` — API client and data normalizers
- `src/types.ts` — TypeScript interfaces matching your API schemas
- `src/styles.css` — Minimal modern UI styling

## Notes

- The frontend defaults to the provided backend URL. You can override via `.env`.
- Images and categories are normalized (array or JSON string).
- If CORS is restricted at the backend, you may need to enable it there.
- This frontend assumes the following response shapes:
  - `POST /api/recommendations/chat`: `{ reply?: string, recommendations?: RecommendedProduct[] }`
  - `GET /api/recommendations/similar/{id}`: `{ items: RecommendedProduct[] }`
  - `GET /api/analytics/`: summary metrics (see `src/types.ts`)
  - `GET /api/analytics/products`: `Product[]`

If your API returns slightly different field names, update `src/types.ts` and `src/lib/api.ts` mappings accordingly.

## License

MIT