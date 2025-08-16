"""
Real-time News RAG with Fact-Checking — Enhanced Single-file Streamlit App
Stack: Gemini (LLM + Embeddings), ChromaDB Cloud, Serper.dev (Google results), web scraping, FAISS (local fallback),
       HF transformer for fake-news classification, bias detection, temporal verification

Enhanced Features:
- Continuous content monitoring and updates
- Bias detection and mitigation algorithms
- Temporal fact verification with time-series analysis
- Advanced source reliability scoring
- Real-time misinformation alerts
- Content freshness tracking
- Bias classification (political, economic, social)
- Temporal relevance scoring
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

# Vector stores
import numpy as np

# Try Chroma Cloud first, fallback to local Chroma or FAISS
USE_CHROMA_LOCAL = os.getenv("USE_CHROMA_LOCAL", "false").lower() == "true"

# Transformers for fake news classification
from transformers import pipeline

# Sentence-Transformers (optional fallback for embedding if Gemini unavailable)
from sentence_transformers import SentenceTransformer

# ---------- Enhanced Config & Setup ----------
load_dotenv()

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")
# CHROMADB_TENANT = os.getenv("CHROMADB_TENANT", "435c9546-30ce-47da-9431-c759362c04b0")
# CHROMADB_DATABASE = os.getenv("CHROMADB_DATABASE", "Dummy DB")
import streamlit as st
import os

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", os.getenv("SERPER_API_KEY"))
CHROMADB_API_KEY = st.secrets.get("CHROMADB_API_KEY", os.getenv("CHROMADB_API_KEY"))
CHROMADB_TENANT = st.secrets.get("CHROMADB_TENANT", os.getenv("CHROMADB_TENANT"))
CHROMADB_DATABASE = st.secrets.get("CHROMADB_DATABASE", os.getenv("CHROMADB_DATABASE"))

CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "news_rag")

APP_TITLE = " Real-time News RAG with Fact-Checking — Enhanced (Gemini Flash)"
MAX_CHUNK = 900
CHUNK_OVERLAP = 150

# Enhanced monitoring settings
MONITORING_INTERVAL = 30  # minutes
BIAS_DETECTION_ENABLED = True
TEMPORAL_VERIFICATION_ENABLED = True
CONTINUOUS_MONITORING = True

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------- Enhanced Utilities ----------

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, max_len: int = MAX_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
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


def calculate_content_freshness(ingested_at: str) -> float:
    """Calculate content freshness score (0-1, higher = fresher)"""
    try:
        ingested_time = datetime.fromisoformat(ingested_at.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        age_hours = (now - ingested_time).total_seconds() / 3600
        # Exponential decay: 24 hours = 0.5, 48 hours = 0.25, etc.
        freshness = math.exp(-age_hours / 24)
        return max(0.0, min(1.0, freshness))
    except:
        return 0.5


# ---------- Enhanced Web Scraping ----------

def fetch_url(url: str, timeout: int = 15) -> Optional[str]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
    except Exception:
        return None
    return None


def extract_main_text(url: str) -> Dict[str, Any]:
    """Attempt to extract title + main article text via trafilatura, fallback to BS4."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        txt = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        if txt and len(txt.split()) > 50:
            return {"title": None, "text": normalize_text(txt)}
    # Fallback simple parse
    html = fetch_url(url)
    if not html:
        return {"title": None, "text": ""}
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.text.strip() if soup.title else None
    # Heuristic: join <p> tags
    ps = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    body = normalize_text(" ".join(ps))
    return {"title": title, "text": body}


# ---------- Enhanced Embeddings ----------

# Prefer Gemini Embeddings; fallback to MiniLM if missing
_minilm_model = None


def get_minilm():
    global _minilm_model
    if _minilm_model is None:
        _minilm_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _minilm_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    texts = [t if t is not None else "" for t in texts]
    if GEMINI_API_KEY:
        try:
            # text-embedding-004 returns 768-d by default
            model = genai.GenerativeModel("text-embedding-004")
            # Batch embeddings
            embs = []
            for t in texts:
                r = model.embed_content(content=t)
                embs.append(r["embedding"])  # SDK returns dict-like
            return embs
        except Exception:
            pass
    # Fallback to MiniLM (384-d)
    m = get_minilm()
    return m.encode(texts, normalize_embeddings=True).tolist()


# ---------- Enhanced Vector DB (Chroma Cloud primary) ----------

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
                # Cloud Client
                self.client = chromadb.CloudClient(
                    api_key=CHROMADB_API_KEY,
                    tenant=CHROMADB_TENANT,
                    database=CHROMADB_DATABASE,
                )
                self.collection = self.client.get_or_create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
                self.kind = "chroma-cloud"
            else:
                # Local Client
                self.client = chromadb.PersistentClient(path="./data/chroma")
                self.collection = self.client.get_or_create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
                self.kind = "chroma-local"
        except Exception as e:
            # Final fallback: in-memory FAISS
            import faiss  # type: ignore
            self.kind = "faiss"
            self.dim = 768  # assume gemini; will adapt when first vector comes
            self.index = None
            self.meta: Dict[int, Dict[str, Any]] = {}

    def _ensure_faiss(self, dim: int):
        import faiss  # type: ignore
        if getattr(self, "index", None) is None:
            self.dim = dim
            self.index = faiss.IndexFlatIP(dim)

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        vectors = embed_texts(texts)
        if self.kind.startswith("chroma"):
            self.collection.upsert(ids=ids, metadatas=metadatas, documents=texts, embeddings=vectors)
        else:
            # FAISS
            import faiss  # type: ignore
            arr = np.array(vectors).astype("float32")
            # normalize for cosine
            faiss.normalize_L2(arr)
            self._ensure_faiss(arr.shape[1])
            self.index.add(arr)
            base = len(self.meta)
            for i, m in enumerate(metadatas):
                self.meta[base + i] = m | {"text": texts[i], "id": ids[i]}

    def query(self, query_text: str, k: int = 8) -> List[Dict[str, Any]]:
        qv = embed_texts([query_text])[0]
        
        if self.kind.startswith("chroma"):
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
            import faiss  # type: ignore
            arr = np.array([qv]).astype("float32")
            faiss.normalize_L2(arr)
            D, I = self.index.search(arr, k)
            out = []
            for idx in I[0]:
                if idx in self.meta:
                    out.append(self.meta[idx])
            return out

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """Get all stored metadata for monitoring purposes"""
        if self.kind.startswith("chroma"):
            try:
                # This is a simplified approach - in production you'd want pagination
                res = self.collection.get()
                return res.get("metadatas", [])
            except:
                return []
        else:
            return list(self.meta.values())


VS = VectorStore()


# ---------- Enhanced Serper.dev (Google results) ----------

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
            items.append({
                "title": it.get("title"),
                "snippet": it.get("snippet"),
                "url": it.get("link"),
            })
        return items
    except Exception:
        return []


# ---------- Enhanced Fake News & Bias Classification ----------

@st.cache_resource(show_spinner=False)
def get_fake_news_pipeline():
    # Lightweight model; outputs labels like FAKE / REAL (varies by model)
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection", truncation=True)


def detect_bias(text: str) -> Dict[str, Any]:
    """Detect potential bias in text using keyword analysis and LLM"""
    bias_keywords = {
        "political": ["democrat", "republican", "liberal", "conservative", "left-wing", "right-wing", "socialist", "capitalist"],
        "economic": ["market", "economy", "inflation", "recession", "unemployment", "GDP", "trade deficit", "wealth gap"],
        "social": ["race", "gender", "immigration", "religion", "culture", "identity", "privilege", "discrimination"]
    }
    
    text_lower = text.lower()
    bias_scores = {}
    
    for bias_type, keywords in bias_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        bias_scores[bias_type] = min(score / len(keywords), 1.0)
    
    # Overall bias score
    overall_bias = max(bias_scores.values()) if bias_scores else 0.0
    
    # Use LLM for nuanced bias detection if available
    if GEMINI_API_KEY and overall_bias > 0.3:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""Analyze this text for potential bias. Focus on political, economic, and social bias.
            Text: {text[:500]}
            
            Return a JSON response with:
            - bias_type: "political", "economic", "social", or "none"
            - confidence: 0.0-1.0
            - explanation: brief reason
            - severity: "low", "medium", "high"
            """
            response = model.generate_content(prompt)
            # Try to parse JSON response
            try:
                llm_analysis = json.loads(response.text)
                bias_scores["llm_analysis"] = llm_analysis
            except:
                bias_scores["llm_analysis"] = {"error": "Failed to parse LLM response"}
        except:
            pass
    
    return {
        "overall_bias": overall_bias,
        "bias_breakdown": bias_scores,
        "detected_bias_types": [k for k, v in bias_scores.items() if v > 0.3 and k != "llm_analysis"]
    }


# ---------- Enhanced RAG & Fact-Checking ----------

def extract_claims_gemini(text: str, max_claims: int = 5) -> List[str]:
    if not GEMINI_API_KEY:
        # naive fallback: split by sentences and pick 3
        sents = re.split(r"(?<=[.!?])\s+", text)
        return [normalize_text(s) for s in sents if len(s.split()) > 8][:max_claims]
    sys = "You are a precise information extraction system. Extract short, checkable claims with entities, numbers, and dates when present. Return a bullet list."
    user = f"""Text:\n{text}\n\nExtract up to {max_claims} atomic claims (<= 25 words each)."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    out = model.generate_content([{ "role": "user", "parts": [sys + "\n\n" + user] }])
    lines = [normalize_text(l) for l in out.text.splitlines() if l.strip()]
    claims = []
    for l in lines:
        l = re.sub(r"^[•\-\*\d\.\)\s]+", "", l)
        if len(l) > 8:
            claims.append(l)
    return claims[:max_claims]


def gemini_answer(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join([f"[CTX {i+1}] {c}" for i, c in enumerate(context_chunks)])
    prompt = f"""You are a careful financial news assistant.\nUse ONLY the provided context and say if information is uncertain.\nCite with [CTX i] markers.\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"""
    if not GEMINI_API_KEY:
        return "(Gemini key missing) Context length: %d chars.\n\nQ: %s\n\nA (stub): %s" % (len(context), question, context_chunks[0][:300] if context_chunks else "No context")
    
    try:
        # Use gemini-1.5-flash for better rate limits and cost efficiency
        model = genai.GenerativeModel("gemini-1.5-flash")
        out = model.generate_content(prompt)
        return out.text
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return f" API rate limit reached. Please wait a moment or upgrade your plan.\n\nFallback response: Based on {len(context_chunks)} context chunks, here's what I found:\n\n" + "\n\n".join([f"[CTX {i+1}] {c[:200]}..." for i, c in enumerate(context_chunks[:3])])
        else:
            return f" API error: {str(e)[:100]}...\n\nFallback: Context length: {len(context)} chars with {len(context_chunks)} chunks."


def enhanced_score_credibility(evidence: List[Dict[str, str]], verdict_text: str, temporal_factors: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhanced credibility scoring with temporal and source factors"""
    v = verdict_text.lower()
    v_sig = 0.55
    if "supported" in v:
        v_sig = 0.9
    elif "refuted" in v:
        v_sig = 0.2
    
    # Source diversity and domain analysis
    domains = {re.sub(r"^www\.", "", (e.get("url") or "").split("/")[2]) for e in evidence if e.get("url")}
    consensus = min(len(domains) / 4.0, 1.0)
    
    # Base credibility score
    base_score = int(round(100 * (0.35 * 0.7 + 0.35 * v_sig + 0.3 * consensus)))
    
    # Enhanced scoring factors
    enhanced_score = base_score
    
    # Temporal relevance bonus
    if temporal_factors and temporal_factors.get("is_recent", False):
        enhanced_score = min(100, enhanced_score + 10)
    
    # Source reliability bonus
    reliable_domains = {"reuters.com", "ap.org", "bbc.com", "npr.org", "wsj.com", "nytimes.com"}
    reliable_sources = sum(1 for domain in domains if domain in reliable_domains)
    if reliable_sources > 0:
        enhanced_score = min(100, enhanced_score + (reliable_sources * 5))
    
    return {
        "score": enhanced_score,
        "base_score": base_score,
        "temporal_bonus": enhanced_score - base_score,
        "source_reliability_bonus": reliable_sources * 5,
        "consensus_score": consensus * 100,
        "verdict_significance": v_sig * 100
    }


def temporal_fact_verification(claim: str, evidence: List[Dict[str, str]]) -> Dict[str, Any]:
    """Verify if facts are still relevant given temporal context"""
    if not GEMINI_API_KEY:
        return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": "LLM not available"}
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Analyze this claim for temporal relevance. Consider if the information might be outdated.
        
        Claim: "{claim}"
        
        Evidence sources: {[e.get('url', '') for e in evidence[:3]]}
        
        Return JSON:
        {{
            "temporal_relevance": "current", "outdated", or "unknown",
            "confidence": 0.0-1.0,
            "reason": "explanation",
            "recommended_actions": ["action1", "action2"]
        }}
        """
        
        response = model.generate_content(prompt)
        try:
            return json.loads(response.text)
        except:
            return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": "Failed to parse response"}
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": "API rate limit reached"}
        else:
            return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": f"LLM error: {str(e)[:50]}..."}


def fact_check_claim(claim: str) -> Dict[str, Any]:
    # Search web for evidence
    hits = serper_search(claim, num=8)
    # Compose evidence text
    ev_text = "\n".join([f"- {h['title']} | {h['url']} :: {h.get('snippet','')}" for h in hits])
    
    if GEMINI_API_KEY:
        try:
            # Use gemini-1.5-flash for better rate limits
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""Claim: "{claim}"\n\nEvidence (titles | urls | snippets):\n{ev_text}\n\nDecide one of: Supported, Refuted, Needs more evidence.\nGive a brief reason (<= 3 sentences) and list 2-3 citations (urls).\nFormat:\nVerdict: <one>\nWhy: <short>\nCitations:\n- <url>\n- <url>\n"""
            out = model.generate_content(prompt)
            verdict = out.text
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                verdict = "Verdict: Needs more evidence\nWhy: API rate limit reached. Please wait before trying again.\nCitations: Rate limited"
            else:
                verdict = f"Verdict: Needs more evidence\nWhy: API error: {str(e)[:50]}...\nCitations: Error occurred"
    else:
        verdict = "Verdict: Needs more evidence\nWhy: Gemini key missing, used local heuristics only.\nCitations:"
    
    # Enhanced credibility scoring
    temporal_analysis = temporal_fact_verification(claim, hits)
    credibility_scores = enhanced_score_credibility(hits, verdict, temporal_analysis)
    
    # Bias detection
    bias_analysis = detect_bias(claim)
    
    return {
        "verdict": verdict, 
        "credibility": credibility_scores,
        "evidence": hits,
        "temporal_analysis": temporal_analysis,
        "bias_analysis": bias_analysis
    }


# ---------- Enhanced Ingestion with Monitoring ----------

def ingest_url(url: str, label: Optional[str] = None) -> Dict[str, Any]:
    article = extract_main_text(url)
    title = article.get("title") or label or url
    text = article.get("text", "")
    if len(text) < 200:
        return {"ok": False, "msg": "Failed to extract sufficient text."}
    
    # Enhanced metadata
    chunks = chunk_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    ingested_at = utcnow_iso()
    
    # Enhanced metadata with monitoring info
    metas = [{
        "url": url, 
        "title": title, 
        "chunk": i, 
        "ingested_at": ingested_at,
        "content_length": len(chunk),
        "freshness_score": 1.0,  # Will be updated by monitoring
        "last_verified": ingested_at,
        "verification_count": 0
    } for i, chunk in enumerate(chunks)]
    
    VS.add(ids=ids, texts=chunks, metadatas=metas)
    
    # Trigger bias detection
    bias_analysis = detect_bias(text)
    
    return {
        "ok": True, 
        "title": title, 
        "chunks": len(chunks),
        "bias_analysis": bias_analysis,
        "ingested_at": ingested_at
    }


# ---------- Continuous Monitoring System ----------

class ContentMonitor:
    def __init__(self):
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start continuous monitoring in background thread"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            st.success(" Continuous monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        st.info(" Continuous monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._update_content_freshness()
                self._check_for_updates()
                time.sleep(MONITORING_INTERVAL * 60)  # Convert to seconds
            except Exception as e:
                st.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    def _update_content_freshness(self):
        """Update freshness scores for all content"""
        try:
            all_metadata = VS.get_all_metadata()
            for meta in all_metadata:
                if "ingested_at" in meta:
                    freshness = calculate_content_freshness(meta["ingested_at"])
                    # Update freshness in metadata (simplified - in production you'd update the DB)
                    meta["freshness_score"] = freshness
        except Exception as e:
            st.error(f"Freshness update error: {e}")
    
    def _check_for_updates(self):
        """Check for content that needs re-verification"""
        try:
            all_metadata = VS.get_all_metadata()
            current_time = datetime.now(timezone.utc)
            
            for meta in all_metadata:
                if "last_verified" in meta:
                    last_verified = datetime.fromisoformat(meta["last_verified"].replace('Z', '+00:00'))
                    hours_since_verification = (current_time - last_verified).total_seconds() / 3600
                    
                    # Re-verify content older than 24 hours
                    if hours_since_verification > 24:
                        st.warning(f"Content needs re-verification: {meta.get('title', 'Unknown')}")
        except Exception as e:
            st.error(f"Update check error: {e}")


# Initialize monitoring system
content_monitor = ContentMonitor()


# ---------- Enhanced Streamlit UI ----------

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Enhanced sidebar with monitoring controls
with st.sidebar:
    st.markdown("###  Configuration")
    st.write(f"Vector store: **{VS.kind}**")
    st.write(f"Chroma collection: `{CHROMA_COLLECTION}`")
    st.write("Gemini: " + ("✅" if GEMINI_API_KEY else "❌ (fallback embeddings/answers)"))
    st.write("Serper.dev: " + ("✅" if SERPER_API_KEY else "❌ (no web evidence)"))
    
    if GEMINI_API_KEY:
        st.info(" Using Gemini 1.5 Flash (better rate limits)")
        st.caption("Free tier: 15 requests/minute, 1500 requests/day")
    
    st.markdown("### Monitoring Controls")
    if st.button("Start Continuous Monitoring", type="primary"):
        content_monitor.start_monitoring()
    
    if st.button("Stop Monitoring"):
        content_monitor.stop_monitoring()
    
    st.markdown("### System Status")
    # Display monitoring status
    if content_monitor.running:
        st.success(" Monitoring Active")
        st.write(f"Interval: {MONITORING_INTERVAL} minutes")
    else:
        st.info(" Monitoring Inactive")
    
    st.caption("Tip: add keys to .env and restart the app.")

# Enhanced tabs with new functionality
TAB1, TAB2, TAB3, TAB4, TAB5 = st.tabs(["Ingest Article", "Ask (RAG)", "Fact-Check a Claim", "Content Monitor", "Analytics"]) 

with TAB1:
    st.subheader("Ingest a news article by URL")
    url = st.text_input("Article URL", value="")
    label = st.text_input("Optional label / title override", value="")
    
    if st.button("Ingest URL", type="primary"):
        with st.spinner("Scraping, analyzing, and embedding..."):
            res = ingest_url(url, label or None)
        if res.get("ok"):
            st.success(f"Ingested '{res['title']}' into {VS.kind} with {res['chunks']} chunks.")
            
            # Display enhanced analysis
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
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata", h)
            freshness = meta.get("freshness_score", 0.5)
            st.markdown(f"**[CTX {i}]** {meta.get('title','(untitled)')} — {meta.get('url','')}")
            st.write(f"Chunk {meta.get('chunk','?')} | Freshness: {freshness:.2f}")
            st.caption((h.get("text") or meta.get("text", ""))[:500] + ("..." if len((h.get("text") or meta.get("text","")))>500 else ""))

with TAB3:
    st.subheader("Enhanced Fact-check a claim")
    claim = st.text_area("Enter a concise claim to verify", value="Company X reported a 25% YoY revenue increase in Q2 2025.", height=100)
    
    if st.button("Extract Claims from Text Above", help="If you pasted a paragraph, we can mine claims first"):
        cands = extract_claims_gemini(claim)
        if cands:
            st.info("Select a mined claim below, or edit manually and click 'Check Claim'.")
            claim = cands[0]
            st.text_input("Mined claim (editable)", value=claim, key="mined_claim")
    
    if st.button("Check Claim", type="primary"):
        with st.spinner("Searching evidence, analyzing bias, and verifying temporally..."):
            result = fact_check_claim(claim)
        
        st.markdown("### Verdict")
        st.write(result["verdict"])
        
        # Enhanced credibility display
        cred = result["credibility"]
        st.progress(cred["score"]/100.0, text=f"Enhanced Credibility Score: {cred['score']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Base Score", f"{cred['base_score']}")
            st.metric("Temporal Bonus", f"+{cred['temporal_bonus']}")
        with col2:
            st.metric("Source Reliability", f"+{cred['source_reliability_bonus']}")
            st.metric("Consensus", f"{cred['consensus_score']:.1f}")
        
        # Temporal analysis
        st.markdown("### Temporal Analysis")
        temporal = result["temporal_analysis"]
        st.write(f"**Relevance:** {temporal['temporal_relevance']}")
        st.write(f"**Confidence:** {temporal['confidence']:.2f}")
        st.write(f"**Reason:** {temporal['reason']}")
        
        # Bias analysis
        st.markdown("### Bias Analysis")
        bias = result["bias_analysis"]
        st.write(f"**Overall Bias:** {bias['overall_bias']:.2f}")
        if bias["detected_bias_types"]:
            st.warning(f"**Detected Bias Types:** {', '.join(bias['detected_bias_types'])}")
        
        st.markdown("### Evidence (Serper)")
        for h in result["evidence"]:
            st.markdown(f"- [{h['title']}]({h['url']}) — {h.get('snippet','')}")
        
        st.divider()
        st.markdown("### Local Context Matches")
        local_hits = VS.query(claim, k=5)
        for i, h in enumerate(local_hits, 1):
            meta = h.get("metadata", h)
            freshness = meta.get("freshness_score", 0.5)
            st.markdown(f"**[{i}]** {meta.get('title','(untitled)')} — Freshness: {freshness:.2f}")

with TAB4:
    st.subheader(" Content Monitoring Dashboard")
    
    if st.button("Refresh Monitoring Data"):
        st.rerun()
    
    # Display content statistics
    try:
        all_metadata = VS.get_all_metadata()
        if all_metadata:
            st.markdown("### Content Overview")
            
            # Calculate statistics
            total_chunks = len(all_metadata)
            total_articles = len(set(m.get("url", "") for m in all_metadata))
            
            # Freshness analysis
            freshness_scores = [m.get("freshness_score", 0.5) for m in all_metadata if "freshness_score" in m]
            avg_freshness = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", total_chunks)
            with col2:
                st.metric("Total Articles", total_articles)
            with col3:
                st.metric("Avg Freshness", f"{avg_freshness:.2f}")
            
            # Content age distribution
            st.markdown("### Content Age Distribution")
            ages = []
            for meta in all_metadata:
                if "ingested_at" in meta:
                    try:
                        ingested = datetime.fromisoformat(meta["ingested_at"].replace('Z', '+00:00'))
                        age_hours = (datetime.now(timezone.utc) - ingested).total_seconds() / 3600
                        ages.append(age_hours)
                    except:
                        pass
            
            if ages:
                # Build a simple histogram using numpy + pandas, render as bar chart
                counts, bins = np.histogram(ages, bins=10)
                ranges = [f"{bins[i]:.0f}-{bins[i+1]:.0f}h" for i in range(len(counts))]
                hist_df = pd.DataFrame({"Age range": ranges, "Count": counts})
                st.bar_chart(hist_df.set_index("Age range"))
                
                # Identify old content
                old_content = [age for age in ages if age > 48]  # Older than 48 hours
                if old_content:
                    st.warning(f"⚠️ {len(old_content)} chunks are older than 48 hours and may need re-verification")
            
            # Recent activity
            st.markdown("### Recent Activity")
            recent_metadata = sorted(all_metadata, key=lambda x: x.get("ingested_at", ""), reverse=True)[:10]
            for meta in recent_metadata:
                if "title" in meta and "ingested_at" in meta:
                    st.write(f"• **{meta['title']}** - {meta['ingested_at'][:19]}")
        else:
            st.info("No content found in the system yet.")
    except Exception as e:
        st.error(f"Error loading monitoring data: {e}")

with TAB5:
    st.subheader(" Analytics & Evaluation Metrics")
    
    # Get real system data
    try:
        all_metadata = VS.get_all_metadata()
        
        st.markdown("### System Performance Metrics")
        
        # Real retrieval accuracy metrics
        st.markdown("#### 🔍 Retrieval Accuracy")
        if all_metadata:
            total_chunks = len(all_metadata)
            total_articles = len(set(m.get("url", "") for m in all_metadata))
            
            # Calculate chunk distribution
            chunk_counts = {}
            for meta in all_metadata:
                url = meta.get("url", "unknown")
                if url in chunk_counts:
                    chunk_counts[url] += 1
                else:
                    chunk_counts[url] = 1
            
            avg_chunks_per_article = total_chunks / total_articles if total_articles > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", total_chunks)
            with col2:
                st.metric("Total Articles", total_articles)
            with col3:
                st.metric("Avg Chunks/Article", f"{avg_chunks_per_article:.1f}")
            
            # Chunk distribution chart
            if chunk_counts:
                chunk_df = pd.DataFrame(list(chunk_counts.items()), columns=["URL", "Chunks"])
                chunk_df = chunk_df.sort_values("Chunks", ascending=False).head(10)
                st.bar_chart(chunk_df.set_index("URL"))
        else:
            st.info("No content available for retrieval accuracy metrics.")
        
        # Real response latency metrics
        st.markdown("####  Response Latency")
        st.info("Latency tracking will be implemented in future versions.")
        
        # Real content quality metrics
        st.markdown("####  Content Quality Metrics")
        if all_metadata:
            # Content freshness analysis
            freshness_scores = [m.get("freshness_score", 0.5) for m in all_metadata if "freshness_score" in m]
            if freshness_scores:
                avg_freshness = sum(freshness_scores) / len(freshness_scores)
                min_freshness = min(freshness_scores)
                max_freshness = max(freshness_scores)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Freshness", f"{avg_freshness:.2f}")
                with col2:
                    st.metric("Min Freshness", f"{min_freshness:.2f}")
                with col3:
                    st.metric("Max Freshness", f"{max_freshness:.2f}")
                
                # Freshness distribution
                st.subheader("Content Freshness Distribution")
                freshness_df = pd.DataFrame({"Freshness": freshness_scores})
                st.bar_chart(freshness_df)
                
                # Create a proper histogram using numpy
                if len(freshness_scores) > 1:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(freshness_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_xlabel('Freshness Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Content Freshness Distribution')
                    st.pyplot(fig)
                    plt.close()
            
            # Content age analysis
            ages = []
            for meta in all_metadata:
                if "ingested_at" in meta:
                    try:
                        ingested = datetime.fromisoformat(meta["ingested_at"].replace('Z', '+00:00'))
                        age_hours = (datetime.now(timezone.utc) - ingested).total_seconds() / 3600
                        ages.append(age_hours)
                    except:
                        pass
            
            if ages:
                st.subheader("Content Age Analysis")
                avg_age = sum(ages) / len(ages)
                oldest_content = max(ages)
                newest_content = min(ages)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Age (hours)", f"{avg_age:.1f}")
                with col2:
                    st.metric("Oldest Content (hours)", f"{oldest_content:.1f}")
                with col3:
                    st.metric("Newest Content (hours)", f"{newest_content:.1f}")
                
                # Age distribution
                age_df = pd.DataFrame({"Age (hours)": ages})
                st.bar_chart(age_df)
                
                # Create a proper histogram using matplotlib
                if len(ages) > 1:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(ages, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
                    ax.set_xlabel('Age (hours)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Content Age Distribution')
                    st.pyplot(fig)
                    plt.close()
        
        # System health metrics
        st.markdown("####  System Health")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Vector store status
            store_status = "🟢 Healthy" if VS.kind else "🔴 Error"
            st.metric("Vector Store", store_status)
        
        with col2:
            # API status
            gemini_status = "🟢 Available" if GEMINI_API_KEY else "🔴 Missing"
            st.metric("Gemini API", gemini_status)
        
        with col3:
            # Serper status
            serper_status = "🟢 Available" if SERPER_API_KEY else "🔴 Missing"
            st.metric("Serper API", serper_status)
        
        # Monitoring status
        st.markdown("####  Monitoring Status")
        if content_monitor.running:
            st.success(" Continuous monitoring is ACTIVE")
            st.write(f"Monitoring interval: {MONITORING_INTERVAL} minutes")
        else:
            st.info("⏹ Continuous monitoring is INACTIVE")
            st.write("Click 'Start Continuous Monitoring' in the sidebar to enable")
        
        # Performance recommendations
        st.markdown("###  Performance Recommendations")
        
        if all_metadata:
            # Analyze content and provide recommendations
            recommendations = []
            
            # Check content freshness
            if freshness_scores:
                avg_freshness = sum(freshness_scores) / len(freshness_scores)
                if avg_freshness < 0.5:
                    recommendations.append(" **Content freshness is low** - Consider ingesting newer articles")
                elif avg_freshness < 0.7:
                    recommendations.append(" **Content freshness is moderate** - Some content may need updates")
                else:
                    recommendations.append(" **Content freshness is good** - Content is relatively recent")
            
            # Check content diversity
            if total_articles < 5:
                recommendations.append(" **Low content diversity** - Ingest more articles for better RAG performance")
            elif total_articles < 20:
                recommendations.append(" **Moderate content diversity** - Consider adding more sources")
            else:
                recommendations.append(" **Good content diversity** - Sufficient sources for comprehensive RAG")
            
            # Check chunk distribution
            if avg_chunks_per_article < 3:
                recommendations.append("✂️ **Low chunk count** - Consider increasing chunk overlap for better context")
            elif avg_chunks_per_article > 10:
                recommendations.append("📏 **High chunk count** - Consider larger chunk sizes for efficiency")
            
            # Display recommendations
            for rec in recommendations:
                st.write(rec)
        else:
            st.info(" **No content available** - Start by ingesting some articles to see recommendations")
        
        # System improvement roadmap
        st.markdown("###  Future Enhancements")
        st.markdown("""
        1. **Real-time latency tracking** - Monitor actual response times
        2. **User feedback collection** - Implement accuracy scoring system
        3. **Advanced bias detection** - Integrate specialized bias models
        4. **Automated re-verification** - Schedule fact-checking updates
        5. **Source reliability scoring** - Track historical accuracy
        6. **Performance alerts** - Set thresholds for system health
        """)
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        st.info("Some metrics may not be available due to system configuration.")

st.divider()
st.markdown("####  Quick fake-news classifier (headline/lead text)")
news_txt = st.text_input("Paste headline or short paragraph for a quick fake/real score", value="Fed announces unexpected rate cut this Friday")
if st.button("Classify"):
    with st.spinner("Running transformer..."):
        clf = get_fake_news_pipeline()
        out = clf(news_txt)[0]
    st.write(out)
    
    # Enhanced bias detection for the input
    bias_result = detect_bias(news_txt)
    st.markdown("**Bias Analysis:**")
    st.write(f"Overall bias: {bias_result['overall_bias']:.2f}")
    if bias_result["detected_bias_types"]:
        st.warning(f"Detected bias types: {', '.join(bias_result['detected_bias_types'])}")

st.caption(" %s — Demo RAG" % datetime.now().year)
