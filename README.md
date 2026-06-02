# 📈 Tradebulls Financial Intelligence RAG System

Production-grade **Agentic RAG** system built for [Tradebulls Securities](https://www.tradebulls.in/) — a SEBI-registered stock broking firm. The system autonomously routes, retrieves, grades, and self-corrects using a LangGraph StateGraph with 10 specialized nodes.

> Built during internship at Tradebulls Securities (P) Ltd, then rebuilt from scratch with production-grade Advanced RAG architecture.

## 🔗 Links

- **Live Demo:** [Coming Soon]
- **GitHub:** [This Repo]

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│ Input Guardrail  │ ─── Block unsafe financial advice requests
└────────┬────────┘
         ▼
┌─────────────────┐
│  Route Query     │ ─── LLM classifies: direct / retrieve / fact-check
└────────┬────────┘
         │
    ┌────┴────────────────────┐
    ▼                         ▼
┌────────┐          ┌─────────────────┐
│ Direct │          │ Advanced Retrieve│
│ Answer │          │                 │
└────┬───┘          │ 1. HyDE         │
     │              │ 2. RRF Fusion   │
     │              │    (Dense+BM25  │
     │              │    +MultiQuery) │
     │              │ 3. Cohere       │
     │              │    Reranking    │
     │              └────────┬────────┘
     │                       ▼
     │              ┌─────────────────┐
     │              │ Grade Documents  │ ─── LLM grades each doc relevant/not
     │              └────────┬────────┘
     │                       │
     │              ┌────────┴────────┐
     │              ▼                 ▼
     │        [Relevant]        [Not Relevant]
     │              │                 │
     │              │         ┌───────▼───────┐
     │              │         │  Web Search    │ ─── CRAG fallback via Tavily
     │              │         └───────┬───────┘
     │              │                 │
     │              └────────┬────────┘
     │                       ▼
     │              ┌─────────────────┐
     │              │    Generate      │
     │              └────────┬────────┘
     │                       ▼
     │              ┌─────────────────┐
     │              │ Grade Generation │ ─── Self-RAG: grounded? answers query?
     │              └────────┬────────┘
     │                       │
     │              ┌────────┴────────┐
     │              ▼                 ▼
     │          [Pass]           [Fail]
     │              │                 │
     │              │         ┌───────▼───────┐
     │              │         │ Rewrite Query  │ ─── Loop back (max 2x)
     │              │         └───────┬───────┘
     │              │                 │
     │              │                 └──► back to Retrieve
     │              │
     └──────────────┤
                    ▼
           ┌─────────────────┐
           │ Output Guardrail │ ─── Add disclaimers, block unsafe claims
           └────────┬────────┘
                    ▼
                  [END]
```

---

## ✨ Key Features

### Advanced Retrieval Pipeline
- **HyDE** (Hypothetical Document Embeddings) — generates a financial analyst's hypothetical answer, embeds that for better retrieval
- **RRF Fusion** — 3 parallel retrievers (Dense Qdrant + BM25 + MultiQuery) fused with Reciprocal Rank Fusion
- **Cohere Cross-Encoder Reranking** — reranks top 12 candidates down to best 4 using `rerank-english-v3.0`

### Agentic RAG with LangGraph
- **Autonomous Routing** — LLM classifies queries into 3 pipelines
- **Document Grading** — each retrieved chunk graded for relevance
- **CRAG** (Corrective RAG) — automatic web search fallback when documents are insufficient
- **Self-RAG** — grades its own generation for groundedness and relevance
- **Query Rewriting** — rewrites and retries with reformulated query (max 2 loops)

### Financial Domain
- **4 Heterogeneous Sources** — YouTube transcripts, web scraping, PDF financial reports, structured FAQs
- **Smart Chunking** — SemanticChunker for PDFs (topic boundaries), RecursiveCharacterTextSplitter for others
- **Financial Guardrails** — blocks personalized investment advice, adds SEBI compliance disclaimers
- **Session-Isolated Qdrant** — each conversation gets its own vector store collection

### Production Engineering
- **Dual Evaluation** — RAGAS + DeepEval with documented scores
- **LangSmith Observability** — every chain traced end-to-end
- **Docker + Docker Compose** — containerized with Qdrant
- **AWS EC2 Deployment** — production deployment guide included

---

## 📊 Evaluation Results

### RAGAS Metrics
| Metric | Score |
|--------|-------|
| Faithfulness | 0.87 |
| Answer Relevancy | 0.91 |
| Context Precision | 0.85 |
| Context Recall | 0.83 |

### DeepEval Metrics
| Metric | Score |
|--------|-------|
| Answer Relevancy | 0.89 |
| Faithfulness | 0.86 |
| Contextual Relevancy | 0.84 |
| Hallucination | 0.12 (lower is better) |

> *Scores from evaluation on 20 golden QA pairs from actual Tradebulls research PDF dated 14 May 2026.*

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph StateGraph |
| LLM | GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | Qdrant (session-isolated) |
| Sparse Retrieval | BM25 |
| Reranking | Cohere rerank-english-v3.0 |
| Web Search | Tavily (CRAG fallback) |
| Evaluation | RAGAS + DeepEval |
| Observability | LangSmith |
| UI | Streamlit |
| Deployment | Docker + AWS EC2 |

---

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/Rajneel-Chavan/tradebulls-financial-rag.git
cd tradebulls-financial-rag

cp .env.example .env
# Fill in your API keys in .env
```

### 2. Run with Docker Compose (Recommended)
```bash
docker-compose up --build
```
Open http://localhost:8501

### 3. Run Locally
```bash
pip install -r requirements.txt

# Start Qdrant (requires Docker)
docker run -p 6333:6333 qdrant/qdrant

# Run Streamlit
streamlit run app.py
```

### 4. Add Your Documents
Place Tradebulls PDF reports in the `documents/` folder, then click "Load Tradebulls Data" in the sidebar.

---

## 📋 Run Evaluation

```bash
# RAGAS evaluation
python -m evaluation.evaluate_ragas

# DeepEval evaluation
python -m evaluation.evaluate_deepeval
```

Results are saved to `evaluation/ragas_results.json` and `evaluation/deepeval_results.json`.

---

## 🌐 AWS EC2 Deployment

```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
sudo apt update
sudo apt install docker.io docker-compose -y
sudo usermod -aG docker $USER

# Clone and deploy
git clone https://github.com/Rajneel-Chavan/tradebulls-financial-rag.git
cd tradebulls-financial-rag
cp .env.example .env
# Edit .env with your API keys

docker-compose up -d --build
```

Access at `http://your-ec2-ip:8501`

---

## 📂 Project Structure

```
tradebulls-financial-rag/
├── backend/
│   ├── config.py              # Environment & model configuration
│   ├── data_loader.py         # 4-source data ingestion with metadata
│   ├── chunker.py             # Semantic + Recursive chunking
│   ├── vector_store.py        # Session-isolated Qdrant collections
│   ├── retrieval_pipeline.py  # HyDE + RRF + Cohere reranking
│   ├── rag_graph.py           # LangGraph 10-node agentic pipeline
│   ├── guardrails.py          # Financial input/output guardrails
│   └── models.py              # Pydantic schemas & LangGraph state
├── evaluation/
│   ├── golden_qa.json         # 20 golden QA pairs
│   ├── evaluate_ragas.py      # RAGAS evaluation pipeline
│   └── evaluate_deepeval.py   # DeepEval evaluation pipeline
├── documents/                  # Place Tradebulls PDFs here
├── app.py                     # Streamlit UI
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # App + Qdrant services
├── requirements.txt
└── README.md
```

---

## 👤 Author

**Rajneel Chavan** — BTech, Symbiosis Institute of Technology, Pune

- GitHub: [github.com/Rajneel-Chavan](https://github.com/Rajneel-Chavan)
- LinkedIn: [linkedin.com/in/rajneel-chavan](https://linkedin.com/in/rajneel-chavan)
- Email: rajneelchavan16@gmail.com

---

## 📜 License

MIT License
