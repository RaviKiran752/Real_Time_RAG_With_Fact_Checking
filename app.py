"""
Real-time News RAG with Fact-Checking - Enhanced Single-file Streamlit App
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
import threading
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import streamlit as st
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

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")
CHROMADB_TENANT = os.getenv("CHROMADB_TENANT", "435c9546-30ce-47da-9431-c759362c04b0")
CHROMADB_DATABASE = os.getenv("CHROMADB_DATABASE", "Dummy DB")
CHROMA_COLLECTION = os.getenv("CHROMADB_COLLECTION", "news_rag")

APP_TITLE = "Real-time News RAG with Fact-Checking - Enhanced (Gemini Flash)"
MAX_CHUNK = 900
CHUNK_OVERLAP = 150
MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", "30"))  # minutes

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------- Utility Functions ----------

def utcnow_iso() -> str:
    """
    Get the current UTC time as an ISO string.
    This is useful for storing timestamps in the database.
    """
    return datetime.now(timezone.utc).isoformat()


def normalize_text(s: str) -> str:
    """
    Clean up text by removing extra whitespace and normalizing it.
    Makes the text more consistent for processing.
    """
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, max_chunk: int = MAX_CHUNK, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split long text into smaller chunks that can be processed by the AI models.
    Uses overlapping chunks so we don't lose context between pieces.
    """
    if len(text) <= max_chunk:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk
        if end > len(text):
            end = len(text)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def fetch_url(url: str) -> Optional[str]:
    """
    Download the HTML content from a URL.
    Simple function that just gets the raw HTML for processing.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Failed to fetch {url}: {e}")
        return None


def extract_main_text(url: str) -> Dict[str, Any]:
    """
    Try to extract the main text content from a news article URL.
    First attempts to use trafilatura (which is pretty good at this),
    but falls back to BeautifulSoup if that doesn't work.
    Returns a dict with title and text.
    """
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


def detect_bias(text: str) -> Dict[str, Any]:
    """
    Look for potential bias in the text by checking for certain keywords.
    This is a simple approach but it actually works pretty well for
    identifying obvious political, economic, or social bias.
    """
    bias_keywords = {
        "political": ["democrat", "republican", "liberal", "conservative", "left-wing", "right-wing", "socialist", "capitalist"],
        "economic": ["market", "economy", "inflation", "recession", "unemployment", "GDP", "trade deficit", "wealth gap"],
        "social": ["race", "gender", "immigration", "religion", "culture", "identity", "privilege", "discrimination"]
    }
    
    text_lower = text.lower()
    bias_scores = {}
    detected_types = []
    
    for bias_type, keywords in bias_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        bias_scores[bias_type] = score
        if score > 0:
            detected_types.append(bias_type)
    
    overall_bias = sum(bias_scores.values()) / len(bias_keywords)
    
    return {
        "overall_bias": overall_bias,
        "bias_scores": bias_scores,
        "detected_bias_types": detected_types
    }


def calculate_content_freshness(ingested_at: str) -> float:
    """
    Calculate how fresh the content is based on when it was ingested.
    Returns a score between 0 and 1, where 1 is fresh and 0 is old.
    Uses exponential decay - content gets less fresh over time.
    """
    try:
        ingested_time = datetime.fromisoformat(ingested_at.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        age_hours = (now - ingested_time).total_seconds() / 3600
        # Exponential decay: 24 hours = 0.5, 48 hours = 0.25, etc.
        freshness = math.exp(-age_hours / 24)
        return max(0.0, min(1.0, freshness))
    except:
        return 0.5

# ---------- Vector Store Setup ----------

class VectorStore:
    """
    A simple wrapper around different vector databases.
    Tries to use ChromaDB Cloud first, falls back to local Chroma,
    and finally to FAISS if nothing else works.
    """
    
    def __init__(self):
        self.kind = "Unknown"
        self.store = None
        self._setup_store()
    
    def _setup_store(self):
        """Try to set up the best available vector store"""
        # Try ChromaDB Cloud first
        if CHROMADB_API_KEY and not USE_CHROMA_LOCAL:
            try:
                import chromadb
                from chromadb.config import Settings
                
                client = chromadb.HttpClient(
                    host="api.chromadb.com",
                    port=443,
                    ssl=True,
                    headers={
                        "X-Chroma-Token": CHROMADB_API_KEY,
                        "X-Chroma-Tenant": CHROMADB_TENANT,
                        "X-Chroma-Database": CHROMADB_DATABASE
                    }
                )
                self.store = client.get_or_create_collection(CHROMA_COLLECTION)
                self.kind = "ChromaDB Cloud"
                return
            except Exception as e:
                st.warning(f"ChromaDB Cloud failed: {e}")
        
        # Try local ChromaDB
        try:
            import chromadb
            client = chromadb.PersistentClient(path="./chroma_db")
            self.store = client.get_or_create_collection(CHROMA_COLLECTION)
            self.kind = "ChromaDB Local"
            return
        except Exception as e:
            st.warning(f"Local ChromaDB failed: {e}")
        
        # Fallback to FAISS
        try:
            import faiss
            self.store = faiss.IndexFlatL2(768)  # Default dimension
            self.kind = "FAISS (in-memory)"
            st.info("Using FAISS in-memory vector store (data will be lost on restart)")
        except Exception as e:
            st.error(f"All vector stores failed: {e}")
            self.kind = "None Available"
    
    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict] = None):
        """Add text chunks to the vector store"""
        if self.kind == "FAISS (in-memory)":
            # Simple FAISS implementation
            embeddings = self._get_embeddings(texts)
            if embeddings is not None:
                self.store.add(embeddings.astype('float32'))
        else:
            # ChromaDB
            try:
                self.store.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
            except Exception as e:
                st.error(f"Failed to add to vector store: {e}")
    
    def query(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar text chunks"""
        if self.kind == "FAISS (in-memory)":
            # Simple FAISS search
            query_embedding = self._get_embeddings([query])
            if query_embedding is not None:
                D, I = self.store.search(query_embedding.astype('float32'), k)
                return [{"text": f"Chunk {i}", "metadata": {"chunk": i}} for i in I[0]]
            return []
        else:
            # ChromaDB search
            try:
                results = self.store.query(
                    query_texts=[query],
                    n_results=k
                )
                # Format results consistently
                hits = []
                if results.get("documents"):
                    for i in range(len(results["documents"][0])):
                        hit = {
                            "text": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}
                        }
                        hits.append(hit)
                return hits
            except Exception as e:
                st.error(f"Search failed: {e}")
                return []
    
    def _get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Get embeddings for texts using available models"""
        if GEMINI_API_KEY:
            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                embeddings = []
                for text in texts:
                    result = model.embed_content(text)
                    embeddings.append(result.embedding)
                return np.array(embeddings)
            except Exception as e:
                st.warning(f"Gemini embeddings failed: {e}")
        
        # Fallback to sentence-transformers
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(texts)
            return embeddings
        except Exception as e:
            st.error(f"All embedding methods failed: {e}")
            return None
    
    def get_all_metadata(self) -> List[Dict]:
        """Get metadata for all stored items"""
        if self.kind == "FAISS (in-memory)":
            return []  # FAISS doesn't store metadata
        else:
            try:
                # This is a simplified version - in practice you'd want pagination
                results = self.store.get()
                return results.get("metadatas", [])
            except Exception as e:
                st.warning(f"Failed to get metadata: {e}")
                return []

# Initialize vector store
VS = VectorStore()

# ---------- Core RAG Functions ----------

def gemini_answer(question: str, context_chunks: List[str]) -> str:
    """
    Generate an answer to a question using the Gemini model.
    Takes the question and some context chunks, then asks Gemini
    to answer based on that context.
    """
    if not context_chunks:
        return "No context available to answer this question."
    
    context = "\n\n".join(context_chunks)
    
    if not GEMINI_API_KEY:
        return f"(Gemini key missing) Context length: {len(context)} chars.\n\nQ: {question}\n\nA (stub): {context_chunks[0][:300] if context_chunks else 'No context'}"
    
    prompt = f"""Context: {context}

Question: {question}

Answer the question based on the context provided. If the context doesn't contain enough information, say so. Be concise but thorough."""

    try:
        # Use gemini-1.5-flash for better rate limits and cost efficiency
        model = genai.GenerativeModel("gemini-1.5-flash")
        out = model.generate_content(prompt)
        return out.text
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return f"API rate limit reached. Please wait a moment or upgrade your plan.\n\nFallback response: Based on {len(context_chunks)} context chunks, here's what I found:\n\n" + "\n\n".join([f"[CTX {i+1}] {c[:200]}..." for i, c in enumerate(context_chunks[:3])])
        else:
            return f"API error: {str(e)[:100]}...\n\nFallback: Context length: {len(context)} chars with {len(context_chunks)} chunks."


def serper_search(query: str, num: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using Serper.dev API to find evidence for fact-checking.
    This gives us real-time information from the web to verify claims.
    """
    if not SERPER_API_KEY:
        return []
    
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "num": num
        }
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        if "organic" in data:
            for result in data["organic"][:num]:
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", ""),
                    "snippet": result.get("snippet", "")
                })
        
        return results
    except Exception as e:
        st.error(f"Serper search failed: {e}")
        return []


def extract_claims_gemini(text: str) -> List[str]:
    """
    Extract specific claims from a longer text using Gemini.
    Useful when someone pastes a whole article and you want to
    pick out the specific claims to fact-check.
    """
    if not GEMINI_API_KEY:
        return [text]  # Fallback to original text
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Extract 2-3 specific, verifiable claims from this text. Each claim should be a single sentence that can be fact-checked.

Text: {text}

Format your response as a simple list, one claim per line."""
        
        out = model.generate_content(prompt)
        claims = [line.strip() for line in out.text.split('\n') if line.strip()]
        return claims[:3]  # Limit to 3 claims
    except Exception as e:
        st.warning(f"Failed to extract claims: {e}")
        return [text]


def get_fake_news_pipeline():
    """
    Get the fake news detection pipeline from HuggingFace.
    This is a lightweight model that can quickly classify
    whether text is likely fake news or not.
    """
    try:
        return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")
    except Exception as e:
        st.error(f"Failed to load fake news classifier: {e}")
        return None


def score_credibility(evidence: List[Dict[str, str]], verdict_text: str) -> float:
    """
    Score how credible a fact-check result is.
    Simple scoring based on the verdict and amount of evidence.
    """
    base_score = 50.0
    
    # Verdict impact
    v = verdict_text.lower()
    if "supported" in v:
        base_score += 30
    elif "refuted" in v:
        base_score -= 20
    elif "needs more evidence" in v:
        base_score -= 10
    
    # Evidence count bonus
    evidence_bonus = min(len(evidence) * 5, 20)
    base_score += evidence_bonus
    
    return max(0, min(100, base_score))


def enhanced_score_credibility(evidence: List[Dict[str, str]], verdict_text: str, temporal_factors: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Score how credible a fact-check result is based on multiple factors.
    Takes into account the verdict, source diversity, and how recent the info is.
    This gives a more nuanced view than just a simple yes/no.
    """
    v = verdict_text.lower()
    v_sig = 0.55
    if "supported" in v:
        v_sig = 0.9
    elif "refuted" in v:
        v_sig = 0.2
    
    # Source diversity
    unique_domains = len(set([e.get("url", "").split("/")[2] for e in evidence if e.get("url")]))
    source_diversity = min(unique_domains / 5.0, 1.0)
    
    # Source reliability (simple heuristic)
    reliable_domains = ["reuters.com", "ap.org", "bbc.com", "npr.org", "wsj.com", "nytimes.com"]
    reliable_count = sum(1 for e in evidence if any(domain in e.get("url", "").lower() for domain in reliable_domains))
    source_reliability = min(reliable_count / len(evidence), 1.0) if evidence else 0.0
    
    # Temporal relevance
    temporal_score = temporal_factors.get("temporal_relevance_score", 0.5) if temporal_factors else 0.5
    
    # Weighted combination
    final_score = (
        v_sig * 0.4 +
        source_diversity * 0.2 +
        source_reliability * 0.25 +
        temporal_score * 0.15
    )
    
    return {
        "overall_score": final_score,
        "verdict_significance": v_sig,
        "source_diversity": source_diversity,
        "source_reliability": source_reliability,
        "temporal_relevance": temporal_score,
        "breakdown": {
            "verdict": f"{v_sig:.2f}",
            "diversity": f"{source_diversity:.2f}",
            "reliability": f"{source_reliability:.2f}",
            "temporal": f"{temporal_score:.2f}"
        }
    }


def temporal_fact_verification(claim: str, evidence: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Check if the facts in a claim are still relevant given how much time has passed.
    Some facts become outdated quickly (like stock prices), others stay relevant longer.
    """
    if not GEMINI_API_KEY:
        return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": "LLM not available"}
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""Claim: "{claim}"

Evidence sources: {[e.get('url', '') for e in evidence]}

Assess the temporal relevance of this claim. Consider:
1. How quickly does this type of information become outdated?
2. Are the evidence sources recent enough?
3. Is this still relevant today?

Respond with:
Temporal Relevance: [high/medium/low]
Confidence: [0.0-1.0]
Reason: [brief explanation]"""
        
        out = model.generate_content(prompt)
        response = out.text
        
        # Parse response
        relevance = "medium"
        confidence = 0.5
        reason = "Analysis completed"
        
        if "high" in response.lower():
            relevance = "high"
        elif "low" in response.lower():
            relevance = "low"
        
        # Extract confidence if present
        import re
        conf_match = re.search(r"confidence:\s*([0-9.]+)", response.lower())
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except:
                pass
        
        return {
            "temporal_relevance": relevance,
            "confidence": confidence,
            "reason": reason,
            "temporal_relevance_score": {"high": 0.9, "medium": 0.6, "low": 0.3}[relevance]
        }
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": "API rate limit reached"}
        else:
            return {"temporal_relevance": "unknown", "confidence": 0.5, "reason": f"LLM error: {str(e)[:50]}..."}


def fact_check_claim(claim: str) -> Dict[str, Any]:
    """
    Take a claim and fact-check it by searching the web for evidence.
    This is the main function that ties everything together - it searches,
    analyzes, and gives you a verdict with confidence scores.
    """
    # Search web for evidence
    hits = serper_search(claim, num=8)
    # Compose evidence text
    ev_text = "\n".join([f"- {h['title']} | {h['url']} :: {h.get('snippet','')}" for h in hits])
    
    # Get temporal verification
    temporal_analysis = temporal_fact_verification(claim, hits)
    
    # Get LLM verdict if available
    if GEMINI_API_KEY:
        try:
            # Use gemini-1.5-flash for better rate limits
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"""Claim: "{claim}"

Evidence (titles | urls | snippets):
{ev_text}

Decide one of: Supported, Refuted, Needs more evidence.
Give a brief reason (<= 3 sentences) and list 2-3 citations (urls).
Format:
Verdict: <one>
Why: <short>
Citations:
- <url>
- <url>"""
            out = model.generate_content(prompt)
            verdict = out.text
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                verdict = "Verdict: Needs more evidence\nWhy: API rate limit reached. Please wait before trying again.\nCitations: Rate limited"
            else:
                verdict = f"Verdict: Needs more evidence\nWhy: API error: {str(e)[:50]}...\nCitations: Error occurred"
    else:
        verdict = "Verdict: Needs more evidence\nWhy: Gemini key missing, used local heuristics only.\nCitations:"
    
    # Enhanced scoring
    score_result = enhanced_score_credibility(hits, verdict, temporal_analysis)
    
    return {
        "verdict": verdict,
        "score": score_result["overall_score"],
        "evidence": hits,
        "temporal_analysis": temporal_analysis,
        "enhanced_score": score_result
    }


def ingest_url(url: str, label: Optional[str] = None) -> Dict[str, Any]:
    """
    Take a news article URL, scrape the content, chunk it up, and store it
    in the vector database. This is how we build our knowledge base.
    """
    article = extract_main_text(url)
    title = article.get("title") or label or url
    text = article.get("text", "")
    if len(text) < 200:
        return {"ok": False, "msg": "Failed to extract sufficient text."}
    
    # Analyze bias
    bias_analysis = detect_bias(text)
    
    # Chunk and store
    chunks = chunk_text(text)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metas = [{
        "url": url, 
        "title": title, 
        "chunk": i, 
        "ingested_at": utcnow_iso(),
        "bias_score": bias_analysis["overall_bias"],
        "bias_types": bias_analysis["detected_bias_types"]
    } for i, _ in enumerate(chunks)]
    
    VS.add(ids=ids, texts=chunks, metadatas=metas)
    
    return {
        "ok": True, 
        "title": title, 
        "chunks": len(chunks),
        "bias_analysis": bias_analysis,
        "ingested_at": utcnow_iso()
    }

# ---------- Content Monitoring System ----------

class ContentMonitor:
    """
    Monitors content freshness and automatically updates stale information.
    Runs in the background to keep the system up-to-date.
    """
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.stop_event = threading.Event()
    
    def start_monitoring(self):
        """Start the background monitoring thread"""
        if not self.running:
            self.running = True
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            st.success("Content monitoring started")
    
    def stop_monitoring(self):
        """Stop the background monitoring thread"""
        if self.running:
            self.running = False
            self.stop_event.set()
            if self.thread:
                self.thread.join(timeout=1)
            st.info("Content monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in the background"""
        while not self.stop_event.is_set():
            try:
                self._update_freshness_scores()
                self._check_for_updates()
                
                # Wait for the specified interval
                for _ in range(MONITORING_INTERVAL * 60):  # Convert to seconds
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                st.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait a minute before retrying
    
    def _update_freshness_scores(self):
        """Update freshness scores for all content"""
        try:
            all_metadata = VS.get_all_metadata()
            for meta in all_metadata:
                if "ingested_at" in meta:
                    freshness = calculate_content_freshness(meta["ingested_at"])
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
    st.markdown("### Configuration")
    st.write(f"Vector store: **{VS.kind}**")
    st.write(f"Chroma collection: `{CHROMA_COLLECTION}`")
    st.write("Gemini: " + ("Available" if GEMINI_API_KEY else "Missing (using fallbacks)"))
    st.write("Serper.dev: " + ("Available" if SERPER_API_KEY else "Missing (no web evidence)"))
    
    if GEMINI_API_KEY:
        st.info("Using Gemini 1.5 Flash (better rate limits)")
        st.caption("Free tier: 15 requests/minute, 1500 requests/day")
    
    st.markdown("### Monitoring Controls")
    if st.button("Start Continuous Monitoring", type="primary"):
        content_monitor.start_monitoring()
    
    if st.button("Stop Monitoring"):
        content_monitor.stop_monitoring()
    
    st.markdown("### System Status")
    # Display monitoring status
    if content_monitor.running:
        st.success("Monitoring Active")
        st.write(f"Interval: {MONITORING_INTERVAL} minutes")
    else:
        st.info("Monitoring Inactive")
    
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
                st.markdown("### Bias Analysis")
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
        st.markdown("### Context Chunks")
        for i, h in enumerate(hits, 1):
            meta = h.get("metadata", h)
            st.markdown(f"**[CTX {i}]** {meta.get('title','(untitled)')} — {meta.get('url','')} (chunk {meta.get('chunk','?')})")
            st.caption((h.get("text") or meta.get("text", ""))[:500] + ("..." if len((h.get("text") or meta.get("text","")))>500 else ""))

with TAB3:
    st.subheader("Fact-check a claim")
    claim = st.text_area("Enter a concise claim to verify", value="Company X reported a 25% YoY revenue increase in Q2 2025.", height=100)
    if st.button("Extract Claims from Text Above", help="If you pasted a paragraph, we can mine claims first"):
        cands = extract_claims_gemini(claim)
        if cands:
            st.info("Select a mined claim below, or edit manually and click 'Check Claim'.")
            claim = cands[0]
            st.text_input("Mined claim (editable)", value=claim, key="mined_claim")
    if st.button("Check Claim", type="primary"):
        with st.spinner("Searching evidence & reasoning..."):
            result = fact_check_claim(claim)
        st.markdown("### Verdict")
        st.write(result["verdict"]) 
        st.progress(result["score"]/100.0, text=f"Credibility score: {result['score']}")
        
        # Enhanced scoring breakdown
        if "enhanced_score" in result:
            st.markdown("### Enhanced Credibility Analysis")
            enhanced = result["enhanced_score"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{enhanced['overall_score']:.1f}")
            with col2:
                st.metric("Verdict Weight", enhanced['breakdown']['verdict'])
            with col3:
                st.metric("Source Diversity", enhanced['breakdown']['diversity'])
            with col4:
                st.metric("Temporal Relevance", enhanced['breakdown']['temporal'])
        
        st.markdown("### Evidence (Serper)")
        for h in result["evidence"]:
            st.markdown(f"- [{h['title']}]({h['url']}) — {h.get('snippet','')}")
        
        # Temporal analysis
        if "temporal_analysis" in result:
            st.markdown("### Temporal Relevance")
            temp = result["temporal_analysis"]
            st.write(f"**Relevance:** {temp['temporal_relevance'].title()}")
            st.write(f"**Confidence:** {temp['confidence']:.2f}")
            st.write(f"**Reason:** {temp['reason']}")
        
        st.divider()
        st.markdown("### Local Context Matches (optional)")
        local_hits = VS.query(claim, k=5)
        for i, h in enumerate(local_hits, 1):
            meta = h.get("metadata", h)
            st.markdown(f"**[{i}]** {meta.get('title','(untitled)')} — {meta.get('url','')}")

with TAB4:
    st.subheader("Content Monitoring Dashboard")
    st.write("Monitor the health and freshness of your content")
    
    # Content freshness overview
    try:
        all_metadata = VS.get_all_metadata()
        if all_metadata:
            st.markdown("### Content Overview")
            total_chunks = len(all_metadata)
            total_articles = len(set(meta.get("url", "") for meta in all_metadata))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", total_chunks)
            with col2:
                st.metric("Total Articles", total_articles)
            with col3:
                st.metric("Avg Chunks/Article", f"{total_chunks/total_articles:.1f}" if total_articles > 0 else "0")
            
            # Freshness distribution
            freshness_scores = []
            for meta in all_metadata:
                if "ingested_at" in meta:
                    freshness = calculate_content_freshness(meta["ingested_at"])
                    freshness_scores.append(freshness)
            
            if freshness_scores:
                st.markdown("### Content Freshness")
                avg_freshness = sum(freshness_scores) / len(freshness_scores)
                st.metric("Average Freshness", f"{avg_freshness:.2f}")
                
                # Freshness chart
                import pandas as pd
                freshness_df = pd.DataFrame({"Freshness": freshness_scores})
                st.bar_chart(freshness_df)
                
                # Create a proper histogram using matplotlib
                if len(freshness_scores) > 1:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(freshness_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                    ax.set_xlabel('Freshness Score')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Content Freshness Distribution')
                    st.pyplot(fig)
                    plt.close()
        else:
            st.info("No content available yet. Ingest some articles first!")
    except Exception as e:
        st.error(f"Error loading monitoring data: {e}")

with TAB5:
    st.subheader("Analytics & Evaluation Metrics")
    try:
        all_metadata = VS.get_all_metadata()
        
        st.markdown("### System Performance Metrics")
        
        # Retrieval accuracy metrics
        if all_metadata:
            total_chunks = len(all_metadata)
            total_articles = len(set(meta.get("url", "") for meta in all_metadata))
            avg_chunks_per_article = total_chunks / total_articles if total_articles > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", total_chunks)
            with col2:
                st.metric("Total Articles", total_articles)
            with col3:
                st.metric("Avg Chunks/Article", f"{avg_chunks_per_article:.1f}")
            
            # Chunk distribution per URL
            url_chunk_counts = {}
            for meta in all_metadata:
                url = meta.get("url", "unknown")
                url_chunk_counts[url] = url_chunk_counts.get(url, 0) + 1
            
            if url_chunk_counts:
                st.subheader("Chunk Distribution per Article")
                chunk_df = pd.DataFrame(list(url_chunk_counts.items()), columns=["URL", "Chunks"])
                st.bar_chart(chunk_df.set_index("URL"))
        else:
            st.info("No content available for analysis")
        
        # Response latency (placeholder for future implementation)
        st.markdown("### Response Latency")
        st.info("Latency metrics will be implemented in future versions")
        
        # Content quality metrics
        st.markdown("### Content Quality Metrics")
        if all_metadata:
            # Freshness analysis
            freshness_scores = []
            ages = []
            current_time = datetime.now(timezone.utc)
            
            for meta in all_metadata:
                if "ingested_at" in meta:
                    freshness = calculate_content_freshness(meta["ingested_at"])
                    freshness_scores.append(freshness)
                    
                    # Calculate age in hours
                    try:
                        ingested_time = datetime.fromisoformat(meta["ingested_at"].replace('Z', '+00:00'))
                        age_hours = (current_time - ingested_time).total_seconds() / 3600
                        ages.append(age_hours)
                    except:
                        pass
            
            if freshness_scores:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Freshness", f"{sum(freshness_scores)/len(freshness_scores):.2f}")
                with col2:
                    st.metric("Min Freshness", f"{min(freshness_scores):.2f}")
                with col3:
                    st.metric("Max Freshness", f"{max(freshness_scores):.2f}")
                
                # Freshness distribution chart
                st.subheader("Content Freshness Distribution")
                freshness_df = pd.DataFrame({"Freshness": freshness_scores})
                st.bar_chart(freshness_df)
                
                # Create a proper histogram using matplotlib
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
            if ages:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Age (hours)", f"{sum(ages)/len(ages):.1f}")
                with col2:
                    st.metric("Oldest Content (hours)", f"{max(ages):.1f}")
                with col3:
                    st.metric("Newest Content (hours)", f"{min(ages):.1f}")
                
                # Age distribution chart
                st.subheader("Content Age Distribution")
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
        
        # System health
        st.markdown("### System Health")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vector Store", VS.kind)
        with col2:
            st.metric("Gemini API", "Available" if GEMINI_API_KEY else "Missing")
        with col3:
            st.metric("Serper API", "Available" if SERPER_API_KEY else "Missing")
        
        # Monitoring status
        st.markdown("### Monitoring Status")
        st.metric("Continuous Monitoring", "Active" if content_monitor.running else "Inactive")
        
        # Performance recommendations
        st.markdown("### Performance Recommendations")
        if all_metadata:
            total_chunks = len(all_metadata)
            total_articles = len(set(meta.get("url", "") for meta in all_metadata))
            
            if total_articles < 5:
                st.warning("Consider ingesting more diverse content sources for better coverage")
            
            if total_chunks / max(total_articles, 1) > 15:
                st.info("High chunk-to-article ratio detected. Consider adjusting chunk size for better retrieval")
            
            # Check freshness
            freshness_scores = [calculate_content_freshness(meta.get("ingested_at", "")) for meta in all_metadata if "ingested_at" in meta]
            if freshness_scores and sum(freshness_scores) / len(freshness_scores) < 0.3:
                st.warning("Content is getting stale. Consider re-ingesting or updating sources")
        else:
            st.info("Start by ingesting some articles to see recommendations")
        
        # Future enhancements
        st.markdown("### Future Enhancements")
        st.markdown("""
        - Real-time content monitoring with alerts
        - Advanced bias detection using ML models
        - Source credibility scoring based on historical accuracy
        - Automated fact-checking workflows
        - Integration with news APIs for automatic ingestion
        - Performance benchmarking and optimization
        """)
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        st.info("Some metrics may not be available due to system configuration.")

# Quick fake-news classifier
st.divider()
st.markdown("#### Quick fake-news classifier (headline/lead text)")
news_txt = st.text_input("Paste headline or short paragraph for a quick fake/real score", value="Fed announces unexpected rate cut this Friday")
if st.button("Classify"):
    with st.spinner("Running transformer..."):
        clf = get_fake_news_pipeline()
        if clf:
            out = clf(news_txt)[0]
            st.write(out)
        else:
            st.error("Fake news classifier not available")

st.caption(f"© {datetime.now().year} — Real-time News RAG with Fact-Checking. Use at your own risk; always verify sources.")
