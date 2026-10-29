# 🏆 SportsMate AI — Presidency University Sports Assistant

SportsMate AI is a local & cloud-ready Retrieval-Augmented Generation (RAG) assistant designed for Presidency University, Bengaluru. It provides students, athletes, and staff with instant, verified answers regarding campus sports facilities, regulations, the Sports Council, and athletic activities.

---

## 🌟 Key Features

- **Strict Document Grounding**: Powered by **ChromaDB** with **233 pre-indexed chunks** from official university policy documents and student handbooks.
- **Accurate RAG Pipeline**:
  - **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  - **Generation Model**: `google/flan-t5-small`
  - **Strict Hallucination Guards**: If documents do not support an answer, SportsMate transparently reports it.
- **Modern Glassmorphic UI**: Built with Streamlit featuring a dark-navy palette, interactive starter questions, and source citation badges.
- **Verified Source Tracking**: Every response links to the specific document name and page number.

---

## 🚀 Running Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gt1432/sportsmate.git
   cd sportsmate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment on Render

This repository includes a `render.yaml` blueprint for one-click deployment:

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- **Environment**: Python 3.11+
