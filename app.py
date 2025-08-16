# app.py
"""
Real-time News RAG with Fact-Checking — Enhanced Single-file Streamlit App
Stack: Gemini (LLM + Embeddings), ChromaDB Cloud, Serper.dev (Google results), web scraping, FAISS (local fallback),
       HF transformer for fake-news classification, bias detection, temporal verification
"""

import os
import json
import time
import math
import uuid
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
import threading

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup
import trafilatura

# Embeddings + LLM (Gemini)
import google.generativeai as genai

# Vector math
import numpy as np

# Transformers for fake news classification
from transformers import pipeline

# Sentence-Transformers fallback for embeddings (if Gemini missing)
from sentence_transformers import SentenceTransformer

# ---------- Config & Setup ----------
load_dotenv()

# Prefer secrets, fallback to env
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", os.getenv("SERPER_API_KEY"))

CHROMADB_API_KEY = st.secrets.get("CHROMADB_API_KEY", os.getenv("CHROMADB_API_KEY"))
CHROMADB_TENANT = st.secrets.get("CHROMADB_TENANT", os.getenv("CHROMADB_TENANT", ""))
CHROMADB_DATABASE = st.secrets.get("CHROMADB_DATABASE", os.getenv("CHROMADB_DATABASE", ""))

USE_CHROMA_LOCAL = os.getenv("USE_CHROMA_LOCAL", "false").lower() == "true"
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "news_rag")

APP_TITLE = "Real-time News RAG with Fact-Checking — Enhanced (Gemini Flash)"
MAX_CHUNK = 900
CHUNK_OVERLAP = 150

# Monitoring
MONITORING_INTERVAL = 30  # minutes

# Configure Gemini (if key present)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------- Utilities ----------
def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, max_len: int = MAX_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    i = 0
    while i < len(text):
        end = min(len(text), i + max_len)
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = max(0, end - overlap)
    return chunks

def calculate_content_freshness(ingested_at: str) -> float:
    """Exponential decay: 24h ~ 0.37, 48h ~ 0.14; clamp to [0,1]."""
    try:
        ing = datetime.fromisoformat(ingested_at.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        hours = (now - ing).total_seconds() / 3600
        return float(max(0.0, min(1.0, math.exp(-hours / 24))))
    except Exception:
        return 0.5

# ---------- Scraping ----------
def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None

def extract_main_text(url: str) -> Dict[str, Any]:
    """Try trafilatura; fallback to simple BeautifulSoup paragraphs."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            txt = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if txt and len(txt.split()) > 50:
                return {"title": None, "text": normalize_text(txt)}
    except Exception:
        pass

    html = fetch_url(url)
    if not html:
        return {"title": None, "text": ""}
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.text.strip() if soup.title else None
    ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return {"title": title, "text": normalize_text(" ".join(ps))}

# ---------- Embeddings ----------
_minilm_model = None
def get_minilm():
    global _minilm_model
    if _minilm_model is None:
        _minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _minilm_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Prefer Gemini embeddings; fallback to MiniLM."""
    texts = [t or "" for t in texts]
    if GEMINI_API_KEY:
        try:
            # text-embedding-004: 768-dim
            out = []
            for t in texts:
                r = genai.embed_content(model="text-embedding-004", content=t)
                out.append(r["embedding"])
            return out
        except Exception:
            pass
    # Fallback: MiniLM (384-dim)
    m = get_minilm()
    return m.encode(texts, normalize_embeddings=True).tolist()

# ---------- Vector Store ----------
class VectorStore:
    """
    Priority:
    1) Chroma Cloud (if CHROMADB_API_KEY and not USE_CHROMA_LOCAL)
    2) Chroma Local (PersistentClient)
    3) FAISS in-memory fallback
    """
    def __init__(self):
        self.kind = None
        self.client = None
        self.collection = None
        # Always define FAISS-related members to avoid AttributeErrors downstream
        self.index = None
        self.dim = None
        self.meta: Dict[int, Dict[str, Any]] = {}
        self._init_store()

    def _init_store(self):
        try:
            import chromadb
            if CHROMADB_API_KEY and not USE_CHROMA_LOCAL:
                self.client = chromadb.CloudClient(
                    api_key=CHROMADB_API_KEY,
                    tenant=CHROMADB_TENANT or None,
                    database=CHROMADB_DATABASE or None,
                )
                self.collection = self.client.get_or_create_collection(
                    CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
                )
                self.kind = "chroma-cloud"
                return
            # else local
            self.client = chromadb.PersistentClient(path="./data/chroma")
            self.collection = self.client.get_or_create_collection(
                CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
            )
            self.kind = "chroma-local"
        except Exception:
            # Final fallback: FAISS
            self.kind = "faiss"
            self.index = None
            self.dim = None
            self.meta = {}

    def _ensure_faiss(self, dim: int):
        import faiss  # type: ignore
        if self.index is None:
            self.dim = dim
            self.index = faiss.IndexFlatIP(dim)

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        vectors = embed_texts(texts)
        if self.kind and self.kind.startswith("chroma"):
            try:
                self.collection.upsert(
                    ids=ids, documents=texts, metadatas=metadatas, embeddings=vectors
                )
                return
            except Exception:
                # If Chroma fails at runtime, silently fall back to FAISS for robustness
                self.kind = "faiss"
                self.index = None
                self.dim = None
                self.meta = {}

        # FAISS path
        import faiss  # type: ignore
        arr = np.array(vectors).astype("float32")
        if arr.ndim != 2:
            return
        faiss.normalize_L2(arr)  # cosine
        self._ensure_faiss(arr.shape[1])
        base = getattr(self.index, "ntotal", 0)
        self.index.add(arr)
        # Map FAISS ids to metadata
        for i, m in enumerate(metadatas):
            self.meta[base + i] = {**m, "text": texts[i], "id": ids[i]}

    def query(self, query_text: str, k: int = 8) -> List[Dict[str, Any]]:
        qv = embed_texts([query_text])[0]

        if self.kind and self.kind.startswith("chroma"):
            try:
                res = self.collection.query(query_embeddings=[qv], n_results=k)
                out = []
                # If collection empty, Chroma returns empty lists
                if not res.get("ids"):
                    return out
                for i in range(len(res["ids"][0])):
                    out.append(
                        {
                            "id": res["ids"][0][i],
                            "text": res["documents"][0][i],
                            "metadata": res["metadatas"][0][i],
                        }
                    )
                return out
            except Exception:
                # Soft-fallback to FAISS within same run
                self.kind = "faiss"
                # Continue below into FAISS path

        # FAISS path (safe empty behavior)
        import faiss  # type: ignore
        if self.index is None or getattr(self.index, "ntotal", 0) == 0:
            return []
        arr = np.array([qv]).astype("float32")
        faiss.normalize_L2(arr)
        D, I = self.index.search(arr, min(k, getattr(self.index, "ntotal", 0)))
        out = []
        for idx in I[0]:
            if int(idx) in self.meta:
                out.append(self.meta[int(idx)])
        return out

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        if self.kind and self.kind.startswith("chroma"):
            try:
                res = self.collection.get()
                return res.get("metadatas", []) or []
            except Exception:
                return []
        return list(self.meta.values())

# ---------- Serper.dev ----------
def serper_search(query: str, num: int = 8) -> List[Dict[str, Any]]:
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
            items.append(
                {
                    "title": it.get("title"),
                    "snippet": it.get("snippet"),
                    "url": it.get("link"),
                }
            )
        return items
    except Exception:
        return []

# ---------- Fake News & Bias ----------
@st.cache_resource(show_spinner=False)
def get_fake_news_pipeline():
    return pipeline(
        "text-classification",
        model="mrm8488/bert-tiny-finetuned-fake-news-detection",
        truncation=True,
    )

def detect_bias(text: str) -> Dict[str, Any]:
    bias_keywords = {
        "political": [
            "democrat",
            "republican",
            "liberal",
            "conservative",
            "left-wing",
            "right-wing",
            "socialist",
            "capitalist",
        ],
        "economic": [
            "market",
            "economy",
            "inflation",
            "recession",
            "unemployment",
            "gdp",
            "trade deficit",
            "wealth gap",
        ],
        "social": [
            "race",
            "gender",
            "immigration",
            "religion",
            "culture",
            "identity",
            "privilege",
            "discrimination",
        ],
    }

    text_lower = (text or "").lower()
    bias_scores = {}
    for bias_type, keywords in bias_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        bias_scores[bias_type] = min(score / max(1, len(keywords)), 1.0)

    overall_bias = max(bias_scores.values()) if bias_scores else 0.0

    if GEMINI_API_KEY and overall_bias > 0.3:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""Analyze this text for potential bias. Focus on political, economic, and social bias.
Text: {text[:500]}

Return a JSON with:
- bias_type
- confidence
- explanation
- severity
"""
            resp = model.generate_content(prompt)
            try:
                bias_scores["llm_analysis"] = json.loads(resp.text)
            except Exception:
                bias_scores["llm_analysis"] = {"error": "Failed to parse LLM response"}
        except Exception:
            pass

    return {
        "overall_bias": overall_bias,
        "bias_breakdown": bias_scores,
        "detected_bias_types": [
            k for k, v in bias_scores.items() if k != "llm_analysis" and v > 0.3
        ],
    }

# ---------- RAG & Fact-Checking ----------
def extract_claims_gemini(text: str, max_claims: int = 5) -> List[str]:
    if not GEMINI_API_KEY:
        sents = re.split(r"(?<=[.!?])\s+", text or "")
        return [normalize_text(s) for s in sents if len(s.split()) > 8][:max_claims]
    model = genai.GenerativeModel("gemini-1.5-flash")
    sys = "You are a precise information extraction system. Extract short, checkable claims with entities, numbers, and dates."
    user = f"Text:\n{text}\n\nExtract up to {max_claims} atomic claims (<= 25 words each). Return a bullet list."
    out = model.generate_content(sys + "\n\n" + user)
    lines = [normalize_text(l) for l in (out.text or "").splitlines() if l.strip()]
    claims = []
    for l in lines:
        l = re.sub(r"^[•\-\*\d\.\)\s]+", "", l)
        if len(l) > 8:
            claims.append(l)
    return claims[:max_claims]

def gemini_answer(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join([f"[CTX {i+1}] {c}" for i, c in enumerate(context_chunks)])
    prompt = f"""You are a careful financial/news assistant.
Use ONLY the provided context and say if information is uncertain.
Cite with [CTX i] markers.

Question: {question}

Context:
{context}

Answer:"""
    if not GEMINI_API_KEY:
        return f"(Gemini key missing) Context chunks: {len(context_chunks)}"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        out = model.generate_content(prompt)
        return out.text
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return f"Rate limit. Based on {len(context_chunks)} chunks:\n" + "\n".join(
                [f"[CTX {i+1}] {c[:200]}..." for i, c in enumerate(context_chunks[:3])]
            )
        return f"API error: {str(e)[:120]}"

def temporal_fact_verification(claim: str, evidence: List[Dict[str, str]]) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": "LLM unavailable"}
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Analyze this claim for temporal relevance.

Claim: "{claim}"
Evidence sources: {[e.get('url', '') for e in evidence[:3]]}

Return JSON with keys: temporal_relevance ("current"|"outdated"|"unknown"), confidence (0-1), reason, recommended_actions (list)
"""
        resp = model.generate_content(prompt)
        try:
            return json.loads(resp.text)
        except Exception:
            return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": "Parse error"}
    except Exception:
        return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": "LLM error"}

def enhanced_score_credibility(evidence: List[Dict[str, str]], verdict_text: str, temporal_factors: Dict[str, Any] = None) -> Dict[str, Any]:
    v = (verdict_text or "").lower()
    v_sig = 0.55
    if "supported" in v:
        v_sig = 0.9
    elif "refuted" in v:
        v_sig = 0.2

    domains = {
        re.sub(r"^www\.", "", (e.get("url") or "").split("/")[2]) 
        for e in evidence if e.get("url") and "://" in e.get("url")
    }
    consensus = min(len(domains) / 4.0, 1.0)

    base_score = int(round(100 * (0.35 * 0.7 + 0.35 * v_sig + 0.3 * consensus)))
    enhanced_score = base_score

    is_recent = temporal_factors.get("temporal_relevance") == "current" if temporal_factors else False
    if is_recent:
        enhanced_score = min(100, enhanced_score + 10)

    reliable_domains = {"reuters.com", "ap.org", "bbc.com", "npr.org", "wsj.com", "nytimes.com"}
    reliable_sources = sum(1 for d in domains if d in reliable_domains)
    enhanced_score = min(100, enhanced_score + reliable_sources * 5)

    return {
        "score": enhanced_score,
        "base_score": base_score,
        "temporal_bonus": enhanced_score - base_score,
        "source_reliability_bonus": reliable_sources * 5,
        "consensus_score": consensus * 100,
        "verdict_significance": v_sig * 100,
    }

def fact_check_claim(claim: str) -> Dict[str, Any]:
    hits = serper_search(claim, num=8)
    ev_text = "\n".join([f"- {h.get('title','')} | {h.get('url','')} :: {h.get('snippet','')}" for h in hits])

    if GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""Claim: "{claim}"

Evidence (titles | urls | snippets):
{ev_text}

Decide one of: Supported, Refuted, Needs more evidence.
Give a brief reason (<= 3 sentences) and list 2-3 citations (urls).
Format strictly:
Verdict: <one>
Why: <short>
Citations:
- <url>
- <url>
"""
            out = model.generate_content(prompt)
            verdict = out.text
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                verdict = "Verdict: Needs more evidence\nWhy: Rate limited.\nCitations:"
            else:
                verdict = f"Verdict: Needs more evidence\nWhy: API error: {str(e)[:80]}\nCitations:"
    else:
        verdict = "Verdict: Needs more evidence\nWhy: Gemini key missing.\nCitations:"

    temporal = temporal_fact_verification(claim, hits)
    cred = enhanced_score_credibility(hits, verdict, temporal)
    bias = detect_bias(claim)

    return {
        "verdict": verdict,
        "credibility": cred,
        "evidence": hits,
        "temporal_analysis": temporal,
        "bias_analysis": bias,
    }

# ---------- Ingestion ----------
def ingest_url(url: str, label: Optional[str] = None) -> Dict[str, Any]:
    article = extract_main_text(url)
    title = article.get("title") or label or url
    text = article.get("text", "")
    if len(text) < 200:
        return {"ok": False, "msg": "Failed to extract sufficient text."}

    chunks = chunk_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    ingested_at = utcnow_iso()

    metas = [
        {
            "url": url,
            "title": title,
            "chunk": i,
            "ingested_at": ingested_at,
            "content_length": len(chunk),
            "freshness_score": 1.0,
            "last_verified": ingested_at,
            "verification_count": 0,
        }
        for i, chunk in enumerate(chunks)
    ]

    VS.add(ids=ids, texts=chunks, metadatas=metas)
    bias_analysis = detect_bias(text)
    return {
        "ok": True,
        "title": title,
        "chunks": len(chunks),
        "bias_analysis": bias_analysis,
        "ingested_at": ingested_at,
    }

# ---------- Monitoring ----------
class ContentMonitor:
    def __init__(self):
        self.running = False
        self.monitor_thread = None

    def start_monitoring(self):
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            st.success("Continuous monitoring started")

    def stop_monitoring(self):
        self.running = False
        st.info("Continuous monitoring stopped")

    def _monitor_loop(self):
        while self.running:
            try:
                self._update_content_freshness()
                self._check_for_updates()
                time.sleep(MONITORING_INTERVAL * 60)
            except Exception as e:
                # Don't spam Streamlit from background thread; just backoff
                time.sleep(60)

    def _update_content_freshness(self):
        all_metadata = VS.get_all_metadata()
        for meta in all_metadata:
            if "ingested_at" in meta:
                meta["freshness_score"] = calculate_content_freshness(meta["ingested_at"])

    def _check_for_updates(self):
        all_metadata = VS.get_all_metadata()
        now = datetime.now(timezone.utc)
        for meta in all_metadata:
            lv = meta.get("last_verified")
            if not lv:
                continue
            try:
                last_v = datetime.fromisoformat(lv.replace('Z', '+00:00'))
                hours = (now - last_v).total_seconds() / 3600
                if hours > 24:
                    # In a real system you might enqueue a recheck job here
                    pass
            except Exception:
                pass

# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Create store once per session
if "VS" not in st.session_state:
    st.session_state["VS"] = VectorStore()
VS = st.session_state["VS"]

if "content_monitor" not in st.session_state:
    st.session_state["content_monitor"] = ContentMonitor()
content_monitor: ContentMonitor = st.session_state["content_monitor"]

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    st.write(f"Vector store: **{VS.kind or 'unknown'}**")
    st.write(f"Chroma collection: `{CHROMA_COLLECTION}`")
    st.write("Gemini: " + ("✅" if GEMINI_API_KEY else "❌"))
    st.write("Serper.dev: " + ("✅" if SERPER_API_KEY else "❌"))
    if GEMINI_API_KEY:
        st.info("Using Gemini 1.5 Flash")
    st.markdown("### Monitoring Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start Monitoring"):
            content_monitor.start_monitoring()
    with c2:
        if st.button("Stop Monitoring"):
            content_monitor.stop_monitoring()
    st.markdown("### System Status")
    st.write("Monitoring: " + ("🟢 Active" if content_monitor.running else "⏹ Inactive"))
    st.caption("Tip: set keys in Secrets or .env, then restart.")

TAB1, TAB2, TAB3, TAB4, TAB5 = st.tabs(
    ["Ingest Article", "Ask (RAG)", "Fact-Check a Claim", "Content Monitor", "Analytics"]
)

with TAB1:
    st.subheader("Ingest a news article by URL")
    url = st.text_input("Article URL", value="")
    label = st.text_input("Optional label / title override", value="")
    if st.button("Ingest URL", type="primary"):
        with st.spinner("Scraping, analyzing, and embedding..."):
            res = ingest_url(url, label or None)
        if res.get("ok"):
            st.success(f"Ingested '{res['title']}' into {VS.kind} with {res['chunks']} chunks.")
            if "bias_analysis" in res:
                st.markdown("### 🔍 Bias Analysis")
                bias = res["bias_analysis"]
                st.write(f"Overall bias score: {bias['overall_bias']:.2f}")
                if bias["detected_bias_types"]:
                    st.warning(f"Detected bias types: {', '.join(bias['detected_bias_types'])}")
                else:
                    st.success("No significant bias detected")
            st.write(f"Ingested at: {res['ingested_at']}")
        else:
            st.error(res.get("msg", "Unknown error"))

with TAB2:
    st.subheader("Ask a question (retrieval-augmented)")
    q = st.text_input("Your question", value="What happened to Tesla stock today?")
    k = st.slider("Top-k chunks", 2, 12, 6)

    if st.button("Search & Answer"):
        with st.spinner("Retrieving context and generating answer..."):
            hits = VS.query(q, k=k)
            chunks = [h.get("text") or h.get("metadata", {}).get("text", "") for h in hits]
            answer = gemini_answer(q, chunks)

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Context Chunks with Freshness Scores")
        if not hits:
            st.info("No local context yet. Ingest some articles first.")
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata", h)
            freshness = meta.get("freshness_score", 0.5)
            st.markdown(f"**[CTX {i}]** {meta.get('title','(untitled)')} — {meta.get('url','')}")
            st.write(f"Chunk {meta.get('chunk','?')} | Freshness: {freshness:.2f}")
            body = (h.get("text") or meta.get("text", "")) or ""
            st.caption(body[:500] + ("..." if len(body) > 500 else ""))

with TAB3:
    st.subheader("Enhanced Fact-check a claim")
    claim = st.text_area(
        "Enter a concise claim to verify",
        value="Company X reported a 25% YoY revenue increase in Q2 2025.",
        height=100,
    )

    if st.button("Extract Claims from Text Above"):
        cands = extract_claims_gemini(claim)
        if cands:
            st.info("Select a mined claim below, or edit manually and click 'Check Claim'.")
            st.text_input("Mined claim (editable)", value=cands[0], key="mined_claim")

    if st.button("Check Claim", type="primary"):
        with st.spinner("Searching evidence, analyzing bias, and verifying temporally..."):
            result = fact_check_claim(claim)

        st.markdown("### Verdict")
        st.write(result["verdict"])

        cred = result["credibility"]
        st.progress(cred["score"] / 100.0, text=f"Enhanced Credibility Score: {cred['score']}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Base Score", f"{cred['base_score']}")
            st.metric("Temporal Bonus", f"+{cred['temporal_bonus']}")
        with col2:
            st.metric("Source Reliability", f"+{cred['source_reliability_bonus']}")
            st.metric("Consensus", f"{cred['consensus_score']:.1f}")

        st.markdown("### Temporal Analysis")
        t = result["temporal_analysis"]
        st.write(f"**Relevance:** {t.get('temporal_relevance','unknown')}")
        st.write(f"**Confidence:** {t.get('confidence',0.0):.2f}")
        st.write(f"**Reason:** {t.get('reason','')}")

        st.markdown("### Bias Analysis")
        bias = result["bias_analysis"]
        st.write(f"**Overall Bias:** {bias['overall_bias']:.2f}")
        if bias["detected_bias_types"]:
            st.warning(f"**Detected Bias Types:** {', '.join(bias['detected_bias_types'])}")

        st.markdown("### Evidence (Serper)")
        if not result["evidence"]:
            st.info("No Serper results (check SERPER_API_KEY).")
        for h in result["evidence"]:
            st.markdown(f"- [{h.get('title','(no title)')}]({h.get('url','')}) — {h.get('snippet','')}")

        st.divider()
        st.markdown("### Local Context Matches")
        local_hits = VS.query(claim, k=5)
        if not local_hits:
            st.info("No local matches yet.")
        for i, h in enumerate(local_hits, 1):
            meta = h.get("metadata", h)
            freshness = meta.get("freshness_score", 0.5)
            st.markdown(f"**[{i}]** {meta.get('title','(untitled)')} — Freshness: {freshness:.2f}")

with TAB4:
    st.subheader("Content Monitoring Dashboard")
    if st.button("Refresh Monitoring Data"):
        st.rerun()

    try:
        all_metadata = VS.get_all_metadata()
        if all_metadata:
            st.markdown("### Content Overview")
            total_chunks = len(all_metadata)
            total_articles = len(set(m.get("url", "") for m in all_metadata))
            freshness_scores = [m.get("freshness_score", 0.5) for m in all_metadata if "freshness_score" in m]
            avg_freshness = (sum(freshness_scores) / len(freshness_scores)) if freshness_scores else 0.0

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Chunks", total_chunks)
            with col2: st.metric("Total Articles", total_articles)
            with col3: st.metric("Avg Freshness", f"{avg_freshness:.2f}")

            st.markdown("### Content Age Distribution")
            ages = []
            for meta in all_metadata:
                if "ingested_at" in meta:
                    try:
                        ing = datetime.fromisoformat(meta["ingested_at"].replace('Z', '+00:00'))
                        ages.append((datetime.now(timezone.utc) - ing).total_seconds() / 3600)
                    except Exception:
                        pass
            if ages:
                counts, bins = np.histogram(ages, bins=10)
                ranges = [f"{bins[i]:.0f}-{bins[i+1]:.0f}h" for i in range(len(counts))]
                hist_df = pd.DataFrame({"Age range": ranges, "Count": counts})
                st.bar_chart(hist_df.set_index("Age range"))
                old_content = [a for a in ages if a > 48]
                if old_content:
                    st.warning(f"{len(old_content)} chunks are older than 48h and may need re-verification")

            st.markdown("### Recent Activity")
            recent = sorted(all_metadata, key=lambda x: x.get("ingested_at", ""), reverse=True)[:10]
            for meta in recent:
                if "title" in meta and "ingested_at" in meta:
                    st.write(f"• **{meta['title']}** - {meta['ingested_at'][:19]}")
        else:
            st.info("No content found yet.")
    except Exception as e:
        st.error(f"Error loading monitoring data: {e}")

with TAB5:
    st.subheader("Analytics & Evaluation Metrics")
    try:
        all_metadata = VS.get_all_metadata()
        st.markdown("### System Performance Metrics")

        st.markdown("#### Retrieval Accuracy")
        if all_metadata:
            total_chunks = len(all_metadata)
            total_articles = len(set(m.get("url", "") for m in all_metadata))
            chunk_counts = {}
            for meta in all_metadata:
                url = meta.get("url", "unknown")
                chunk_counts[url] = chunk_counts.get(url, 0) + 1
            avg_chunks_per_article = total_chunks / total_articles if total_articles > 0 else 0

            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Chunks", total_chunks)
            with col2: st.metric("Total Articles", total_articles)
            with col3: st.metric("Avg Chunks/Article", f"{avg_chunks_per_article:.1f}")

            if chunk_counts:
                chunk_df = pd.DataFrame(list(chunk_counts.items()), columns=["URL", "Chunks"]).sort_values("Chunks", ascending=False).head(10)
                st.bar_chart(chunk_df.set_index("URL"))
        else:
            st.info("No content available for metrics.")

        st.markdown("#### Response Latency")
        st.info("Latency tracking to be implemented.")

        st.markdown("#### Content Quality Metrics")
        if all_metadata:
            freshness_scores = [m.get("freshness_score", 0.5) for m in all_metadata if "freshness_score" in m]
            if freshness_scores:
                avg_f = sum(freshness_scores) / len(freshness_scores)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Avg Freshness", f"{avg_f:.2f}")
                with col2: st.metric("Min Freshness", f"{min(freshness_scores):.2f}")
                with col3: st.metric("Max Freshness", f"{max(freshness_scores):.2f}")
                st.bar_chart(pd.DataFrame({"Freshness": freshness_scores}))

            # Age analysis
            ages = []
            for meta in all_metadata:
                if "ingested_at" in meta:
                    try:
                        ing = datetime.fromisoformat(meta["ingested_at"].replace('Z', '+00:00'))
                        ages.append((datetime.now(timezone.utc) - ing).total_seconds() / 3600)
                    except Exception:
                        pass
            if ages:
                avg_age = sum(ages) / len(ages)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric("Avg Age (h)", f"{avg_age:.1f}")
                with col2: st.metric("Oldest (h)", f"{max(ages):.1f}")
                with col3: st.metric("Newest (h)", f"{min(ages):.1f}")
                st.bar_chart(pd.DataFrame({"Age (hours)": ages}))
        # System health
        st.markdown("#### System Health")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Vector Store", "🟢 Healthy" if VS.kind else "🔴 Error")
        with c2: st.metric("Gemini API", "🟢 Available" if GEMINI_API_KEY else "🔴 Missing")
        with c3: st.metric("Serper API", "🟢 Available" if SERPER_API_KEY else "🔴 Missing")

        st.markdown("### Recommendations")
        if all_metadata:
            total_chunks = len(all_metadata)
            total_articles = len(set(m.get("url", "") for m in all_metadata))
            recs = []
            fr = [m.get("freshness_score", 0.5) for m in all_metadata if "freshness_score" in m]
            if fr:
                avg_f = sum(fr) / len(fr)
                if avg_f < 0.5:
                    recs.append("Content freshness is low — ingest newer articles.")
                elif avg_f < 0.7:
                    recs.append("Content freshness is moderate — consider refreshing.")
                else:
                    recs.append("Content freshness is good.")
            if total_articles < 5:
                recs.append("Low content diversity — ingest more sources.")
            elif total_articles < 20:
                recs.append("Moderate content diversity — consider adding more.")
            else:
                recs.append("Good content diversity.")
            avg_chunks_per_article = (total_chunks / total_articles) if total_articles else 0
            if avg_chunks_per_article < 3:
                recs.append("Low chunk count — try more overlap for context recall.")
            elif avg_chunks_per_article > 10:
                recs.append("High chunk count — consider larger chunk sizes for efficiency.")
            for r in recs:
                st.write("• " + r)
        else:
            st.info("No content yet — ingest articles to see recommendations.")
    except Exception as e:
        st.error(f"Analytics error: {e}")

st.divider()
st.markdown("#### Quick fake-news classifier (headline/lead text)")
news_txt = st.text_input("Paste headline or short paragraph", value="Fed announces unexpected rate cut this Friday")
if st.button("Classify"):
    with st.spinner("Running transformer..."):
        clf = get_fake_news_pipeline()
        out = clf(news_txt)[0]
    st.write(out)
    bias_result = detect_bias(news_txt)
    st.markdown("**Bias Analysis:**")
    st.write(f"Overall bias: {bias_result['overall_bias']:.2f}")
    if bias_result["detected_bias_types"]:
        st.warning(f"Detected bias types: {', '.join(bias_result['detected_bias_types'])}")

st.caption(f"{datetime.now().year} -- Demo RAG")
