# 🪑 Ikarus AI/ML Intern Assignment

AI-powered product discovery platform that recommends furniture products through conversational chat, semantic search, and intelligent analytics — combining **Machine Learning, NLP, Computer Vision, and Generative AI**.

---

## 🌐 Deployment

**Frontend:** [ikarus-ai.vercel.app](https://ikarus-ai.vercel.app)
* **Hugging Face Space**: [Ikarus – Furniture Recommender](https://huggingface.co/spaces/0504ankitsharma/ikarus)

---

## 🚀 Project Overview

**Ikarus** is an end-to-end AI system that enables users to explore furniture products interactively using natural language queries.
It integrates:

* Conversational AI for personalized recommendations
* Vector-based semantic search (via Pinecone)
* Analytics dashboards for insights
* AI models for embeddings, computer vision classification, and description generation

This project fulfills the **AI-ML Intern Assignment** requirements by implementing:

> ML, NLP, CV, GenAI, Vector DB, React Frontend, FastAPI Backend, and integrated analytics.

---

## 🧩 System Architecture

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

* **React + TypeScript + Vite**
* **Chart.js + react-chartjs-2** for visual analytics
* **react-router-dom** for routing

### 🎯 Purpose

Provides the UI for Ikarus Furniture Recommender AI and connects with the FastAPI backend deployed on Hugging Face Spaces.

### 🧠 Core Features

| Feature                | File                | Description                                                     |
| ---------------------- | ------------------- | --------------------------------------------------------------- |
| 💬 Chat Interface      | `ChatPage.tsx`      | Conversational recommendations with persistent message history  |
| 🪑 Product Display     | `ProductCard.tsx`   | Displays product images, attributes, and “Find Similar” options |
| 📊 Analytics Dashboard | `AnalyticsPage.tsx` | Displays KPIs and visual charts via Chart.js                    |

### 🔗 API Integration

Consumes three main backend endpoints:

* `POST /api/recommendations/chat` — Conversational recommendations
* `GET /api/recommendations/similar/{id}` — Fetch similar products
* `GET /api/analytics/` & `/api/analytics/products` — Analytics data retrieval

### ⚙️ Configuration

Backend base URL managed via environment variable:

```bash
VITE_API_URL=https://0504ankitsharma-ikarus.hf.space
```

Defaults to Hugging Face Space backend if not provided.

---

## 🧠 Backend Overview

### 📌 Description

**Ikarus Furniture Recommendation API** — A FastAPI backend powering semantic search, conversational recommendations, and analytics.

### ⚙️ Core Functionality

#### 🧭 Semantic Product Search

* Natural language queries for furniture discovery
* Pinecone-based vector similarity search
* Conversational product discovery (via LangChain)

#### 🤖 AI & ML Services

| Task                    | Model/Tool                                 |
| ----------------------- | ------------------------------------------ |
| **Embeddings**          | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Generative AI**       | OpenAI GPT-4o-mini *(optional)*            |
| **Computer Vision**     | EfficientNet-B0                            |
| **Conversational Flow** | LangChain                                  |

#### 📊 Analytics Dashboard

* Product insights (brand, category, pricing trends)
* Product listings and filters

---

## 🔗 API Endpoints

| Endpoint                                    | Method | Description                    |
| ------------------------------------------- | ------ | ------------------------------ |
| `/api/recommendations/search`               | POST   | Semantic product search        |
| `/api/recommendations/chat`                 | POST   | Conversational recommendations |
| `/api/recommendations/similar/{product_id}` | GET    | Find similar products          |
| `/api/analytics/`                           | GET    | Retrieve product analytics     |
| `/api/analytics/products`                   | GET    | Fetch product listings         |
| `/health`                                   | GET    | Health check endpoint          |

---

## 🧠 Backend Tech Stack

**FastAPI** + **Pinecone** + **Sentence Transformers** + **LangChain** + **PyTorch**

Optional integrations:

* **OpenAI GPT-4o-mini** for generative descriptions
* **Docker** for containerized deployment

### 🔑 Key Features

✅ Works in fully FREE mode (no OpenAI key required)
✅ Automatic product indexing on startup
✅ Interactive Swagger docs at `/docs`
✅ CORS-enabled for frontend integration
✅ Hugging Face + Vercel deployment ready

---

## 📊 Project Notebooks

### 📘 Part 1: Data Analytics (`Data_Analytics.ipynb`)

**Purpose:** Clean and explore the dataset for ML model training.

#### 🧮 Pipeline

1. **Load:** 312 furniture products (12 columns)
2. **Clean:** Fill missing values, remove duplicates
3. **Analyze:** Visualize distributions & correlations
4. **Engineer:** Add `clean_description` & `desc_word_count`
5. **Output:** Cleaned dataset for model training

#### 📦 Libraries

`pandas`, `numpy`, `matplotlib`, `seaborn`

---

### 📗 Part 2: Model Training (`Model_Training.ipynb`)

**Purpose:** Build a content-based recommendation model.

#### 🔍 Workflow

1. **Vectorize:** Product descriptions via TF-IDF (max 10k features)
2. **Train:** Nearest Neighbors model using cosine similarity
3. **Functions:**

   * `get_recommendations(index)` → Finds similar items
   * `recommend_from_text(query)` → Retrieves matches from text query
4. **Evaluate:** `Precision@5` metric for recommendation quality
5. **Visualize:** Cosine similarity heatmap
6. **Save:** `tfidf_vectorizer.pkl`, `recommender_model.pkl`

#### ⚙️ Libraries

`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

---

## 🧾 Installation & Setup

### 🧩 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 🧩 Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 🌍 Environment Variables

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

## 👨‍💻 Developer

**Ankit Sharma**
AI/ML Intern Assignment

> “Combining ML, NLP, CV, and GenAI into a seamless product discovery experience.”
