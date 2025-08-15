# import os
# import json
# import time
# import uuid
# import re
# from datetime import datetime, timezone
# from typing import List, Dict, Any, Optional

# import streamlit as st
# from dotenv import load_dotenv
# import requests
# from bs4 import BeautifulSoup
# import trafilatura
# from streamlit_extras.app_refresh import st_autorefresh

# # Embeddings + LLM (Gemini)
# import google.generativeai as genai

# # Vector store helpers
# import numpy as np

# # Transformers for fake news classification
# from transformers import pipeline

# # Sentence-Transformers fallback
# from sentence_transformers import SentenceTransformer

# load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")
# CHROMADB_TENANT = os.getenv("CHROMADB_TENANT", "435c9546-30ce-47da-9431-c759362c04b0")
# CHROMADB_DATABASE = os.getenv("CHROMADB_DATABASE", "Dummy DB")
# CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "news_rag")
# USE_CHROMA_LOCAL = os.getenv("USE_CHROMA_LOCAL", "false").lower() == "true"

# APP_TITLE = "Real-time News RAG - Gemini + Chroma + Serper"
# MAX_CHUNK = 900
# CHUNK_OVERLAP = 150

# if GEMINI_API_KEY:
#     genai.configure(api_key=GEMINI_API_KEY)

# # Initialize metrics and monitoring state
# if "metrics" not in st.session_state:
#     st.session_state.metrics = {
#         "retrieval_feedback": [],  # list of bools
#         "latency": {               # per-operation latencies (seconds)
#             "rag": [],
#             "fact_check": [],
#             "ingest": []
#         },
#         "content_quality": {
#             "freshness_scores": [],     # 0..100
#             "bias_scores": [],          # 0..1 placeholder
#             "verification_status": []   # 0..1
#         },
#         "monitoring_enabled": False,
#         "last_monitor_ts": None
#     }

# def utcnow_iso() -> str:
#     return datetime.now(timezone.utc).isoformat()

# def normalize_text(s: str) -> str:
#     return re.sub(r"\s+", " ", s).strip()

# def chunk_text(text: str, max_len: int = MAX_CHUNK, overlap: int = CHUNK_OVERLAP):
#     text = text.strip()
#     if not text:
#         return []
#     chunks = []
#     i = 0
#     while i < len(text):
#         end = min(len(text), i + max_len)
#         chunks.append(text[i:end])
#         i = end - overlap
#         if i < 0:
#             i = 0
#         if end == len(text):
#             break
#     return chunks

# def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
#     try:
#         headers = {"User-Agent": "Mozilla/5.0"}
#         resp = requests.get(url, headers=headers, timeout=timeout)
#         if resp.status_code == 200:
#             return resp.text
#     except Exception:
#         return None
#     return None

# def extract_main_text(url: str):
#     downloaded = trafilatura.fetch_url(url)
#     if downloaded:
#         txt = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
#         if txt and len(txt.split()) > 50:
#             return {"title": None, "text": normalize_text(txt)}
#     html = fetch_url(url)
#     if not html:
#         return {"title": None, "text": ""}
#     soup = BeautifulSoup(html, "lxml")
#     title = soup.title.text.strip() if soup.title else None
#     ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
#     body = normalize_text(" ".join(ps))
#     return {"title": title, "text": body}

# _minilm_model = None
# def get_minilm():
#     global _minilm_model
#     if _minilm_model is None:
#         _minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     return _minilm_model

# def embed_texts(texts: List[str]):
#     texts = [t if t is not None else "" for t in texts]
#     if GEMINI_API_KEY:
#         try:
#             model = genai.GenerativeModel("text-embedding-004")
#             embs = []
#             for t in texts:
#                 r = model.embed_content(content=t)
#                 embs.append(r["embedding"])
#             return embs
#         except Exception:
#             pass
#     m = get_minilm()
#     return m.encode(texts, normalize_embeddings=True).tolist()

# class VectorStore:
#     def __init__(self):
#         self.kind = None
#         self.client = None
#         self.collection = None
#         self.dim = None
#         self._init_store()

#     def _init_store(self):
#         try:
#             import chromadb
#             if not USE_CHROMA_LOCAL and CHROMADB_API_KEY:
#                 self.client = chromadb.CloudClient(
#                     api_key=CHROMADB_API_KEY,
#                     tenant=CHROMADB_TENANT,
#                     database=CHROMADB_DATABASE,
#                 )
#                 self.collection = self.client.get_or_create_collection(
#                     CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
#                 )
#                 self.kind = "chroma-cloud"
#             else:
#                 self.client = chromadb.PersistentClient(path="./data/chroma")
#                 self.collection = self.client.get_or_create_collection(
#                     CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
#                 )
#                 self.kind = "chroma-local"
#         except Exception:
#             import faiss
#             self.kind = "faiss"
#             self.dim = 768
#             self.index = None
#             self.meta = {}

#     def _ensure_faiss(self, dim: int):
#         import faiss
#         if getattr(self, "index", None) is None:
#             self.dim = dim
#             self.index = faiss.IndexFlatIP(dim)

#     def add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
#         vectors = embed_texts(texts)
#         if self.kind and self.kind.startswith("chroma"):
#             self.collection.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=vectors)
#         else:
#             import faiss
#             arr = np.array(vectors).astype("float32")
#             faiss.normalize_L2(arr)
#             self._ensure_faiss(arr.shape[1])
#             self.index.add(arr)
#             base = len(self.meta)
#             for i, m in enumerate(metadatas):
#                 self.meta[base + i] = {**m, "text": texts[i], "id": ids[i]}

#     def query(self, query_text: str, k: int = 8):
#         qv = embed_texts([query_text])[0]
#         if self.kind and self.kind.startswith("chroma"):
#             res = self.collection.query(query_embeddings=[qv], n_results=k)
#             out = []
#             for i in range(len(res["ids"][0])):
#                 out.append({
#                     "id": res["ids"][0][i],
#                     "text": res["documents"][0][i],
#                     "metadata": res["metadatas"][0][i],
#                 })
#             return out
#         else:
#             import faiss
#             arr = np.array([qv]).astype("float32")
#             faiss.normalize_L2(arr)
#             D, I = self.index.search(arr, k)
#             out = []
#             for idx in I[0]:
#                 if idx in self.meta:
#                     out.append(self.meta[idx])
#             return out

# VS = VectorStore()

# def serper_search(query: str, num: int = 8):
#     if not SERPER_API_KEY:
#         return []
#     url = "https://google.serper.dev/search"
#     headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
#     payload = {"q": query, "num": num}
#     try:
#         r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
#         r.raise_for_status()
#         data = r.json()
#         items = []
#         for it in data.get("organic", [])[:num]:
#             items.append({
#                 "title": it.get("title"),
#                 "snippet": it.get("snippet"),
#                 "url": it.get("link"),
#             })
#         return items
#     except Exception:
#         return []

# @st.cache_resource(show_spinner=False)
# def get_fake_news_pipeline():
#     return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection", truncation=True)

# def extract_claims_gemini(text: str, max_claims: int = 5):
#     if not GEMINI_API_KEY:
#         sents = re.split(r"(?<=[.!?])\s+", text)
#         return [normalize_text(s) for s in sents if len(s.split()) > 8][:max_claims]
#     sys = "Extract short, checkable claims with entities, numbers, and dates. Return a bullet list."
#     user = f"Text:\n{text}\n\nExtract up to {max_claims} atomic claims."
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     out = model.generate_content([{"role": "user", "parts": [sys + "\n\n" + user]}])
#     lines = [normalize_text(l) for l in out.text.splitlines() if l.strip()]
#     claims = []
#     for l in lines:
#         l = re.sub(r"^[\\-\\*\\d\\.\\)\\s]+", "", l)
#         if len(l) > 8:
#             claims.append(l)
#     return claims[:max_claims]

# def gemini_answer(question: str, context_chunks: List[str]) -> str:
#     context = "\n\n".join([f"[CTX {i+1}] {c}" for i, c in enumerate(context_chunks)])
#     prompt = f"Use ONLY the provided context and say if uncertain.\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"
#     if not GEMINI_API_KEY:
#         return f"(Gemini key missing) Context length: {len(context)} chars.\nQ: {question}\nA: {context_chunks[0][:300] if context_chunks else 'No context'}"
#     model = genai.GenerativeModel("gemini-1.5-pro")
#     out = model.generate_content(prompt)
#     return out.text

# def score_credibility(evidence: List[Dict[str, str]], verdict_text: str) -> int:
#     v = verdict_text.lower()
#     v_sig = 0.55
#     if "supported" in v:
#         v_sig = 0.9
#     elif "refuted" in v:
#         v_sig = 0.2
#     domains = {re.sub(r"^www\\.", "", (e.get("url") or "").split("/")[2]) for e in evidence if e.get("url")}
#     consensus = min(len(domains) / 4.0, 1.0)
#     return int(round(100 * (0.35 * 0.7 + 0.35 * v_sig + 0.3 * consensus)))

# def fact_check_claim(claim: str):
#     hits = serper_search(claim, num=8)
#     ev_text = "\n".join([f"- {h['title']} | {h['url']} :: {h.get('snippet','')}" for h in hits])
#     if GEMINI_API_KEY:
#         model = genai.GenerativeModel("gemini-1.5-pro")
#         prompt = f"Claim: \"{claim}\"\n\nEvidence:\n{ev_text}\n\nDecide: Supported, Refuted, Needs more evidence."
#         out = model.generate_content(prompt)
#         verdict = out.text
#     else:
#         verdict = "Verdict: Needs more evidence"
#     score = score_credibility(hits, verdict)
#     return {"verdict": verdict, "score": score, "evidence": hits}

# def ingest_url(url: str, label: Optional[str] = None):
#     article = extract_main_text(url)
#     title = article.get("title") or label or url
#     text = article.get("text", "")
#     if len(text) < 200:
#         return {"ok": False, "msg": "Failed to extract sufficient text."}
#     chunks = chunk_text(text)
#     ids = [str(uuid.uuid4()) for _ in chunks]
#     metas = [{"url": url, "title": title, "chunk": i, "ingested_at": utcnow_iso()} for i, _ in enumerate(chunks)]
#     VS.add(ids=ids, texts=chunks, metadatas=metas)
#     return {"ok": True, "title": title, "chunks": len(chunks)}

# # Simple helper to compute a freshness score (0..100) from ingested_at
# def freshness_from_ingested(ingested_at_iso: Optional[str]) -> Optional[float]:
#     if not ingested_at_iso:
#         return None
#     try:
#         age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(ingested_at_iso)).days
#     except Exception:
#         return None
#     return max(0.0, 100.0 - float(age_days))

# # Continuous monitoring routine (runs on rerun tick when enabled)
# def run_monitoring_tick():
#     # For now, compute rolling averages; could be extended to re-query sources
#     st.session_state.metrics["last_monitor_ts"] = utcnow_iso()

# st.set_page_config(page_title=APP_TITLE, layout="wide")
# st.title(APP_TITLE)

# with st.sidebar:
#     st.markdown("Configuration")
#     st.write(f"Vector store: {VS.kind}")
#     st.write(f"Chroma collection: {CHROMA_COLLECTION}")
#     st.write("Gemini: " + ("Enabled" if GEMINI_API_KEY else "Disabled"))
#     st.write("Serper.dev: " + ("Enabled" if SERPER_API_KEY else "Disabled"))

# TAB1, TAB2, TAB3, TAB4 = st.tabs(["Ingest Article", "Ask (RAG)", "Fact-Check", "Metrics"])

# with TAB1:
#     st.subheader("Ingest a news article by URL")
#     url = st.text_input("Article URL", value="")
#     label = st.text_input("Optional label", value="")
#     if st.button("Ingest URL"):
#         t0 = time.time()
#         with st.spinner("Scraping and embedding..."):
#             res = ingest_url(url, label or None)
#         ingest_latency = time.time() - t0
#         st.session_state.metrics["latency"]["ingest"].append(ingest_latency)
#         if res.get("ok"):
#             st.success(f"Ingested '{res['title']}' with {res['chunks']} chunks.")
#             # Freshness baseline for this item (score for display stats)
#             st.session_state.metrics["content_quality"]["freshness_scores"].append(100.0)
#         else:
#             st.error(res.get("msg", "Unknown error"))

# with TAB2:
#     st.subheader("Ask a question (retrieval-augmented)")
#     q = st.text_input("Your question", value="What happened to Tesla stock today?")
#     k = st.slider("Top-k chunks", 2, 12, 6)
#     if st.button("Search and Answer"):
#         t0 = time.time()
#         hits = VS.query(q, k=k)
#         chunks = [h.get("text") or h.get("metadata", {}).get("text", "") for h in hits]
#         answer = gemini_answer(q, chunks)
#         rag_latency = time.time() - t0
#         st.session_state.metrics["latency"]["rag"].append(rag_latency)

#         st.markdown("Answer")
#         st.write(answer)

#         st.markdown("Context Chunks")
#         for i, h in enumerate(hits, 1):
#             meta = h.get("metadata", h)
#             st.markdown(f"[CTX {i}] {meta.get('title','')} - {meta.get('url','')}")
#             # Freshness from ingested_at if present
#             f = freshness_from_ingested(meta.get("ingested_at"))
#             if f is not None:
#                 st.session_state.metrics["content_quality"]["freshness_scores"].append(f)

#         feedback = st.radio("Was this retrieval accurate?", ["Yes", "No"], horizontal=True, key=f"feedback_{uuid.uuid4()}")
#         if feedback:
#             st.session_state.metrics["retrieval_feedback"].append(feedback == "Yes")

#         # Optional placeholder bias score per interaction (0.5 neutral)
#         st.session_state.metrics["content_quality"]["bias_scores"].append(0.5)

# with TAB3:
#     st.subheader("Fact-check a claim")
#     claim = st.text_area("Enter a claim to verify", value="Company X reported a 25% YoY revenue increase in Q2 2025.")
#     if st.button("Check Claim"):
#         t0 = time.time()
#         with st.spinner("Searching evidence..."):
#             result = fact_check_claim(claim)
#         fc_latency = time.time() - t0
#         st.session_state.metrics["latency"]["fact_check"].append(fc_latency)

#         st.markdown("Verdict")
#         st.write(result["verdict"])
#         st.progress(result["score"] / 100.0, text=f"Credibility score: {result['score']}")

#         # Map verdict score 0..100 to 0..1 and store
#         st.session_state.metrics["content_quality"]["verification_status"].append(result["score"] / 100.0)

#         st.markdown("Evidence")
#         for h in result["evidence"]:
#             st.markdown(f"- {h.get('title','(no title)')} - {h.get('url','')} - {h.get('snippet','')}")

# with TAB4:
#     st.subheader("Analytics and Evaluation Metrics")

#     # Continuous Monitoring toggle
#     st.session_state.metrics["monitoring_enabled"] = st.checkbox(
#         "Enable Continuous Monitoring",
#         value=st.session_state.metrics["monitoring_enabled"]
#     )

#     # If monitoring is enabled, run a lightweight tick and auto-refresh
#     if st.session_state.metrics["monitoring_enabled"]:
#         run_monitoring_tick()
#         st.info(f"Monitoring active. Last check: {st.session_state.metrics['last_monitor_ts']}")
#         # Auto refresh every 60 seconds while the tab is open
#         st_autorefresh(interval=60 * 1000, key="monitor_refresh")

#     else:
#         st.info("Monitoring is off.")

#     # Retrieval Accuracy
#     if st.session_state.metrics["retrieval_feedback"]:
#         acc = sum(st.session_state.metrics["retrieval_feedback"]) / len(st.session_state.metrics["retrieval_feedback"])
#         st.metric("Retrieval Accuracy", f"{acc*100:.1f}%")

#     # Response Latency by operation
#     if st.session_state.metrics["latency"]["rag"]:
#         avg_rag = sum(st.session_state.metrics["latency"]["rag"]) / len(st.session_state.metrics["latency"]["rag"])
#         st.metric("Average RAG Latency (s)", f"{avg_rag:.2f}")
#     if st.session_state.metrics["latency"]["fact_check"]:
#         avg_fc = sum(st.session_state.metrics["latency"]["fact_check"]) / len(st.session_state.metrics["latency"]["fact_check"])
#         st.metric("Average Fact-Check Latency (s)", f"{avg_fc:.2f}")
#     if st.session_state.metrics["latency"]["ingest"]:
#         avg_ing = sum(st.session_state.metrics["latency"]["ingest"]) / len(st.session_state.metrics["latency"]["ingest"])
#         st.metric("Average Ingest Latency (s)", f"{avg_ing:.2f}")

#     # Content Quality Metrics
#     if st.session_state.metrics["content_quality"]["freshness_scores"]:
#         avg_freshness = sum(st.session_state.metrics["content_quality"]["freshness_scores"]) / len(st.session_state.metrics["content_quality"]["freshness_scores"])
#         st.metric("Average Content Freshness", f"{avg_freshness:.1f}")
#     if st.session_state.metrics["content_quality"]["verification_status"]:
#         avg_verif = sum(st.session_state.metrics["content_quality"]["verification_status"]) / len(st.session_state.metrics["content_quality"]["verification_status"])
#         st.metric("Average Verification Score", f"{avg_verif*100:.1f}%")
#     if st.session_state.metrics["content_quality"]["bias_scores"]:
#         avg_bias = sum(st.session_state.metrics["content_quality"]["bias_scores"]) / len(st.session_state.metrics["content_quality"]["bias_scores"])
#         st.metric("Average Bias Score (0..1)", f"{avg_bias:.2f}")

#     st.markdown("System Improvement Recommendations")
#     st.markdown(
#         "- Enable continuous monitoring for real-time content updates\n"
#         "- Implement user feedback collection for retrieval accuracy\n"
#         "- Add more bias detection models for comprehensive analysis\n"
#         "- Implement temporal fact verification with scheduled re-checks\n"
#         "- Add source reliability scoring based on historical accuracy"
#     )

# st.caption(f"{datetime.now().year} - RAG demo.")








import os
import json
import time
import uuid
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import trafilatura

# Embeddings + LLM (Gemini)
import google.generativeai as genai

# Vector store helpers
import numpy as np

# Transformers for fake news classification
from transformers import pipeline

# Sentence-Transformers fallback
from sentence_transformers import SentenceTransformer

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")
CHROMADB_TENANT = os.getenv("CHROMADB_TENANT", "435c9546-30ce-47da-9431-c759362c04b0")
CHROMADB_DATABASE = os.getenv("CHROMADB_DATABASE", "Dummy DB")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "news_rag")
USE_CHROMA_LOCAL = os.getenv("USE_CHROMA_LOCAL", "false").lower() == "true"

APP_TITLE = "Real-time News RAG - Gemini + Chroma + Serper"
MAX_CHUNK = 900
CHUNK_OVERLAP = 150

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize metrics and monitoring state
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "retrieval_feedback": [],  # list of bools
        "latency": {               # per-operation latencies (seconds)
            "rag": [],
            "fact_check": [],
            "ingest": []
        },
        "content_quality": {
            "freshness_scores": [],     # 0..100
            "bias_scores": [],          # 0..1 placeholder
            "verification_status": []   # 0..1
        },
        "monitoring_enabled": False,
        "last_monitor_ts": None
    }

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, max_len: int = MAX_CHUNK, overlap: int = CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + max_len)
        chunks.append(text[i:end])
        i = end - overlap
        if i < 0:
            i = 0
        if end == len(text):
            break
    return chunks

def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        return None
    return None

def extract_main_text(url: str):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        txt = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if txt and len(txt.split()) > 50:
            return {"title": None, "text": normalize_text(txt)}
    html = fetch_url(url)
    if not html:
        return {"title": None, "text": ""}
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.text.strip() if soup.title else None
    ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body = normalize_text(" ".join(ps))
    return {"title": title, "text": body}

_minilm_model = None
def get_minilm():
    global _minilm_model
    if _minilm_model is None:
        _minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _minilm_model

def embed_texts(texts: List[str]):
    texts = [t if t is not None else "" for t in texts]
    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("text-embedding-004")
            embs = []
            for t in texts:
                r = model.embed_content(content=t)
                embs.append(r["embedding"])
            return embs
        except Exception:
            pass
    m = get_minilm()
    return m.encode(texts, normalize_embeddings=True).tolist()

class VectorStore:
    def __init__(self):
        self.kind = None
        self.client = None
        self.collection = None
        self.dim = None
        self._init_store()

    def _init_store(self):
        try:
            import chromadb
            if not USE_CHROMA_LOCAL and CHROMADB_API_KEY:
                self.client = chromadb.CloudClient(
                    api_key=CHROMADB_API_KEY,
                    tenant=CHROMADB_TENANT,
                    database=CHROMADB_DATABASE,
                )
                self.collection = self.client.get_or_create_collection(
                    CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
                )
                self.kind = "chroma-cloud"
            else:
                self.client = chromadb.PersistentClient(path="./data/chroma")
                self.collection = self.client.get_or_create_collection(
                    CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
                )
                self.kind = "chroma-local"
        except Exception:
            import faiss
            self.kind = "faiss"
            self.dim = 768
            self.index = None
            self.meta = {}

    def _ensure_faiss(self, dim: int):
        import faiss
        if getattr(self, "index", None) is None:
            self.dim = dim
            self.index = faiss.IndexFlatIP(dim)

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        vectors = embed_texts(texts)
        if self.kind and self.kind.startswith("chroma"):
            self.collection.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=vectors)
        else:
            import faiss
            arr = np.array(vectors).astype("float32")
            faiss.normalize_L2(arr)
            self._ensure_faiss(arr.shape[1])
            self.index.add(arr)
            base = len(self.meta)
            for i, m in enumerate(metadatas):
                self.meta[base + i] = {**m, "text": texts[i], "id": ids[i]}

    def query(self, query_text: str, k: int = 8):
        qv = embed_texts([query_text])[0]
        if self.kind and self.kind.startswith("chroma"):
            res = self.collection.query(query_embeddings=[qv], n_results=k)
            out = []
            for i in range(len(res["ids"][0])):
                out.append({
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "metadata": res["metadatas"][0][i],
                })
            return out
        else:
            import faiss
            arr = np.array([qv]).astype("float32")
            faiss.normalize_L2(arr)
            D, I = self.index.search(arr, k)
            out = []
            for idx in I[0]:
                if idx in self.meta:
                    out.append(self.meta[idx])
            return out

VS = VectorStore()

def serper_search(query: str, num: int = 8):
    if not SERPER_API_KEY:
        return []
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num}
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=15)
        r.raise_for_status()
        data = r.json()
        items = []
        for it in data.get("organic", [])[:num]:
            items.append({
                "title": it.get("title"),
                "snippet": it.get("snippet"),
                "url": it.get("link"),
            })
        return items
    except Exception:
        return []

@st.cache_resource(show_spinner=False)
def get_fake_news_pipeline():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection", truncation=True)

def extract_claims_gemini(text: str, max_claims: int = 5):
    if not GEMINI_API_KEY:
        sents = re.split(r"(?<=[.!?])\s+", text)
        return [normalize_text(s) for s in sents if len(s.split()) > 8][:max_claims]
    sys = "Extract short, checkable claims with entities, numbers, and dates. Return a bullet list."
    user = f"Text:\n{text}\n\nExtract up to {max_claims} atomic claims."
    model = genai.GenerativeModel("gemini-1.5-flash")
    out = model.generate_content([{"role": "user", "parts": [sys + "\n\n" + user]}])
    lines = [normalize_text(l) for l in out.text.splitlines() if l.strip()]
    claims = []
    for l in lines:
        l = re.sub(r"^[\-\*\d\.\)\s]+", "", l)
        if len(l) > 8:
            claims.append(l)
    return claims[:max_claims]

def gemini_answer(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join([f"[CTX {i+1}] {c}" for i, c in enumerate(context_chunks)])
    prompt = f"Use ONLY the provided context and say if uncertain.\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"
    if not GEMINI_API_KEY:
        return f"(Gemini key missing) Context length: {len(context)} chars.\nQ: {question}\nA: {context_chunks[0][:300] if context_chunks else 'No context'}"
    model = genai.GenerativeModel("gemini-1.5-pro")
    out = model.generate_content(prompt)
    return out.text

def score_credibility(evidence: List[Dict[str, str]], verdict_text: str) -> int:
    v = verdict_text.lower()
    v_sig = 0.55
    if "supported" in v:
        v_sig = 0.9
    elif "refuted" in v:
        v_sig = 0.2
    domains = {re.sub(r"^www\.", "", (e.get("url") or "").split("/")[2]) for e in evidence if e.get("url")}
    consensus = min(len(domains) / 4.0, 1.0)
    return int(round(100 * (0.35 * 0.7 + 0.35 * v_sig + 0.3 * consensus)))

def fact_check_claim(claim: str):
    hits = serper_search(claim, num=8)
    ev_text = "\n".join([f"- {h['title']} | {h['url']} :: {h.get('snippet','')}" for h in hits])
    if GEMINI_API_KEY:
        model = genai.GenerativeModel("gemini-1.5-pro")
        prompt = f"Claim: \"{claim}\"\n\nEvidence:\n{ev_text}\n\nDecide: Supported, Refuted, Needs more evidence."
        out = model.generate_content(prompt)
        verdict = out.text
    else:
        verdict = "Verdict: Needs more evidence"
    score = score_credibility(hits, verdict)
    return {"verdict": verdict, "score": score, "evidence": hits}

def ingest_url(url: str, label: Optional[str] = None):
    article = extract_main_text(url)
    title = article.get("title") or label or url
    text = article.get("text", "")
    if len(text) < 200:
        return {"ok": False, "msg": "Failed to extract sufficient text."}
    chunks = chunk_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metas = [{"url": url, "title": title, "chunk": i, "ingested_at": utcnow_iso()} for i, _ in enumerate(chunks)]
    VS.add(ids=ids, texts=chunks, metadatas=metas)
    return {"ok": True, "title": title, "chunks": len(chunks)}

def freshness_from_ingested(ingested_at_iso: Optional[str]) -> Optional[float]:
    if not ingested_at_iso:
        return None
    try:
        age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(ingested_at_iso)).days
    except Exception:
        return None
    return max(0.0, 100.0 - float(age_days))

def run_monitoring_tick():
    st.session_state.metrics["last_monitor_ts"] = utcnow_iso()

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.markdown("Configuration")
    st.write(f"Vector store: {VS.kind}")
    st.write(f"Chroma collection: {CHROMA_COLLECTION}")
    st.write("Gemini: " + ("Enabled" if GEMINI_API_KEY else "Disabled"))
    st.write("Serper.dev: " + ("Enabled" if SERPER_API_KEY else "Disabled"))

TAB1, TAB2, TAB3, TAB4 = st.tabs(["Ingest Article", "Ask (RAG)", "Fact-Check", "Metrics"])

with TAB1:
    st.subheader("Ingest a news article by URL")
    url = st.text_input("Article URL", value="")
    label = st.text_input("Optional label", value="")
    if st.button("Ingest URL"):
        t0 = time.time()
        with st.spinner("Scraping and embedding..."):
            res = ingest_url(url, label or None)
        ingest_latency = time.time() - t0
        st.session_state.metrics["latency"]["ingest"].append(ingest_latency)
        if res.get("ok"):
            st.success(f"Ingested '{res['title']}' with {res['chunks']} chunks.")
            st.session_state.metrics["content_quality"]["freshness_scores"].append(100.0)
        else:
            st.error(res.get("msg", "Unknown error"))

with TAB2:
    st.subheader("Ask a question (retrieval-augmented)")
    q = st.text_input("Your question", value="What happened to Tesla stock today?")
    k = st.slider("Top-k chunks", 2, 12, 6)
    if st.button("Search and Answer"):
        t0 = time.time()
        hits = VS.query(q, k=k)
        chunks = [h.get("text") or h.get("metadata", {}).get("text", "") for h in hits]
        answer = gemini_answer(q, chunks)
        rag_latency = time.time() - t0
        st.session_state.metrics["latency"]["rag"].append(rag_latency)

        st.markdown("Answer")
        st.write(answer)

        st.markdown("Context Chunks")
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata", h)
            st.markdown(f"[CTX {i}] {meta.get('title','')} - {meta.get('url','')}")
            f = freshness_from_ingested(meta.get("ingested_at"))
            if f is not None:
                st.session_state.metrics["content_quality"]["freshness_scores"].append(f)

        feedback = st.radio("Was this retrieval accurate?", ["Yes", "No"], horizontal=True, key=f"feedback_{uuid.uuid4()}")
        if feedback:
            st.session_state.metrics["retrieval_feedback"].append(feedback == "Yes")

        st.session_state.metrics["content_quality"]["bias_scores"].append(0.5)

with TAB3:
    st.subheader("Fact-check a claim")
    claim = st.text_area("Enter a claim to verify", value="Company X reported a 25% YoY revenue increase in Q2 2025.")
    if st.button("Check Claim"):
        t0 = time.time()
        with st.spinner("Searching evidence..."):
            result = fact_check_claim(claim)
        fc_latency = time.time() - t0
        st.session_state.metrics["latency"]["fact_check"].append(fc_latency)

        st.markdown("Verdict")
        st.write(result["verdict"])
        st.progress(result["score"] / 100.0, text=f"Credibility score: {result['score']}")

        st.session_state.metrics["content_quality"]["verification_status"].append(result["score"] / 100.0)

        st.markdown("Evidence")
        for h in result["evidence"]:
            st.markdown(f"- {h.get('title','(no title)')} - {h.get('url','')} - {h.get('snippet','')}")

with TAB4:
    st.subheader("Analytics and Evaluation Metrics")
    st.session_state.metrics["monitoring_enabled"] = st.checkbox(
        "Enable Continuous Monitoring",
        value=st.session_state.metrics["monitoring_enabled"]
    )
    if st.session_state.metrics["monitoring_enabled"]:
        run_monitoring_tick()
        st.info(f"Monitoring active. Last check: {st.session_state.metrics['last_monitor_ts']}")
        # Built-in st_autorefresh replacement
        st.rerun()
    else:
        st.info("Monitoring is off.")

    if st.session_state.metrics["retrieval_feedback"]:
        acc = sum(st.session_state.metrics["retrieval_feedback"]) / len(st.session_state.metrics["retrieval_feedback"])
        st.metric("Retrieval Accuracy", f"{acc*100:.1f}%")

    if st.session_state.metrics["latency"]["rag"]:
        avg_rag = sum(st.session_state.metrics["latency"]["rag"]) / len(st.session_state.metrics["latency"]["rag"])
        st.metric("Average RAG Latency (s)", f"{avg_rag:.2f}")
    if st.session_state.metrics["latency"]["fact_check"]:
        avg_fc = sum(st.session_state.metrics["latency"]["fact_check"]) / len(st.session_state.metrics["latency"]["fact_check"])
        st.metric("Average Fact-Check Latency (s)", f"{avg_fc:.2f}")
