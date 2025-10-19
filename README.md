````{"variant":"standard","id":"19023","title":"Ikarus Furniture Recommender AI – Full Project README (Updated Env Vars)"}
# 🪑 Ikarus Intern Assaignment

AI-powered product discovery platform that recommends furniture products through conversational chat, semantic search, and intelligent analytics — combining **Machine Learning, NLP, Computer Vision, and Generative AI**.

Deployed at:
- 🌐 **Frontend**: [www.ikarus-ai.vercel.app](https://www.ikarus-ai.vercel.app)
- 🤗 **Hugging Face Space**: [Ikarus – Furniture Recommender](https://huggingface.co/spaces/0504ankitsharma/ikarus)

---

## 🚀 Project Overview

**Ikarus** is an end-to-end AI system that enables users to explore furniture products interactively using natural language queries. It integrates:
- Conversational AI for personalized recommendations
- Vector-based semantic search (via Pinecone)
- Analytics dashboards for insights
- AI models for embeddings, CV classification, and description generation

This project fulfills the **AI-ML Intern Assignment** requirements: ML, NLP, CV, GenAI, Vector DB, React Frontend, FastAPI Backend, and integrated analytics.

---

## 🧩 Architecture

```
Frontend (React + TypeScript + Vite)
       │
       ▼
Backend API (FastAPI + LangChain + Pinecone)
       │
       ▼
Vector Database + ML Models + Analytics Layer
```

---

## 🖥️ Frontend Overview

### ⚙️ Tech Stack
- **React + TypeScript + Vite**
- **Chart.js + react-chartjs-2** for visual analytics
- **react-router-dom** for routing

### 🎯 Purpose
Provides the UI for Ikarus Furniture Recommender AI and connects with the FastAPI backend deployed on Hugging Face Spaces.

### 🧠 Core Features
| Feature | File | Description |
|----------|------|-------------|
| 💬 Chat Interface | `ChatPage.tsx` | Conversational recommendations with persistent message history |
| 🪑 Product Display | `ProductCard.tsx` | Shows product images, attributes, and “Find Similar” options |
| 📊 Analytics Dashboard | `AnalyticsPage.tsx` | Displays KPIs and visual charts via Chart.js |

### 🔗 API Integration
Consumes three main backend endpoints:
- `POST /api/recommendations/chat` — Conversational recommendations
- `GET /api/recommendations/similar/{id}` — Fetch similar products
- `GET /api/analytics/` & `/api/analytics/products` — Analytics data retrieval

### ⚙️ Configuration
Backend base URL managed via:
```bash
VITE_API_URL=[<backend_url>](https://0504ankitsharma-ikarus.hf.space)
```
Defaults to Hugging Face Space backend if not provided.

---

## 🧠 Backend Description

### 📌 Overview
**Ikarus Furniture Recommendation API** — An AI-powered FastAPI backend that powers semantic search, conversational recommendations, and analytics.

### ⚙️ Core Functionality
#### 🧭 Semantic Product Search
- Natural language queries for furniture discovery
- Pinecone-based vector similarity search
- Conversational product discovery (via LangChain)

#### 🤖 AI Services
| Task | Model/Tool |
|------|-------------|
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Generative AI** | OpenAI GPT-4o-mini (optional) |
| **Computer Vision** | EfficientNet-B0 |
| **Conversational Flow** | LangChain |

#### 📊 Analytics Dashboard
- Product insights (brand, category, pricing trends)
- Product listings and filters

---

## 🔗 API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/api/recommendations/search` | POST | Semantic product search |
| `/api/recommendations/chat` | POST | Conversational recommendations |
| `/api/recommendations/similar/{product_id}` | GET | Find similar products |
| `/api/analytics/` | GET | Analytics metrics |
| `/api/analytics/products` | GET | Product listings |
| `/health` | GET | Health check |

---

## 🧠 Tech Stack
**FastAPI** + **Pinecone** + **Sentence Transformers** + **LangChain** + **PyTorch**

Optional integrations:
- **OpenAI GPT-4o-mini** for product description generation
- **Docker** for containerized deployment

### 🔑 Key Features
✅ Fully FREE mode (no OpenAI key required)  
✅ Automatic product indexing on startup  
✅ Interactive Swagger docs at `/docs`  
✅ CORS-enabled for frontend integration  
✅ Hugging Face + Vercel deployment ready  

---

## 📊 Complete Project Components

### Part 1: Data Analytics (`Data_Analytics.ipynb`)
**Purpose:** Clean and explore the dataset for recommendation modeling.

#### 🧮 Pipeline
1. **Load:** 312 furniture products (12 columns)
2. **Clean:** Fill missing values, remove duplicates
3. **Analyze:** Visualize distributions & correlations
4. **Engineer:** Create `clean_description`, `desc_word_count`
5. **Output:** Cleaned dataset for ML model training

#### 📦 Libraries
`pandas`, `numpy`, `matplotlib`, `seaborn`

---

### Part 2: Model Training (`Model_Training.ipynb`)
**Purpose:** Build a content-based recommendation model.

#### 🔍 Workflow
1. **Vectorize:** Product descriptions via TF-IDF (max 10k features)
2. **Train:** Nearest Neighbors (cosine similarity)
3. **Functions:**
   - `get_recommendations(index)` → Similar items
   - `recommend_from_text(query)` → Query-based matches
4. **Evaluate:** `Precision@5` relevance metric
5. **Visualize:** Cosine similarity heatmap
6. **Save:** `tfidf_vectorizer.pkl`, `recommender_model.pkl`

#### ⚙️ Tech Stack
`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 📦 Project Setup

### 🔧 Frontend
```bash
cd frontend
npm install
npm run dev
```

### 🔧 Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 🧠 Environment Variables
```bash
# Production API URL (FastAPI backend)
VITE_API_URL=https://0504ankitsharma-ikarus.hf.space

# For local development, uncomment:
# VITE_API_URL=http://localhost:8000

PINECONE_API_KEY=<your_key>
OPENAI_API_KEY=<optional_key>
```

---

## 📁 Repository Structure
```
ikarus/
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── ChatPage.tsx
│   │   │   └── AnalyticsPage.tsx
│   │   ├── components/
│   │   │   └── ProductCard.tsx
│   │   └── lib/api.ts
│   └── vite.config.ts
│
├── backend/
│   ├── main.py
│   ├── routers/
│   ├── models/
│   ├── utils/
│   └── requirements.txt
│
├── notebooks/
│   ├── Data_Analytics.ipynb
│   └── Model_Training.ipynb
│
└── README.md
```

---

## 🧾 Credits

Developed by **Ankit Sharma (Bhavya Verma)**  
AI/ML Intern Assignment – **Ikarus Furniture Recommendation Web App**

> “Combining ML, NLP, CV, and GenAI into a seamless product discovery experience.”
````
