# 📰 Real-time News RAG with Fact-Checking

A comprehensive Retrieval-Augmented Generation (RAG) system that continuously ingests news articles, identifies potential misinformation, and provides fact-checked responses with source verification and credibility scoring.

## 🎯 Problem Statement

This system addresses the critical challenge of information verification in the digital age by:
- **Real-time news article ingestion and processing**
- **Misinformation detection algorithms** using transformer-based classification
- **Source credibility assessment and scoring** with multi-source verification
- **Fact-checking with evidence retrieval** from web search and local knowledge base
- **Continuous content updates and monitoring** for temporal fact verification

## 🏗️ Architecture & Technical Stack

### Core Components
- **LLM & Embeddings**: Google Gemini (1.5 Pro + text-embedding-004)
- **Vector Database**: ChromaDB Cloud (primary) with local ChromaDB and FAISS fallbacks
- **Web Evidence**: Serper.dev for Google search results
- **Content Processing**: Trafilatura + BeautifulSoup for article extraction
- **Misinformation Detection**: HuggingFace transformers (BERT-based fake news classifier)
- **UI Framework**: Streamlit for interactive web interface

### Technical Features
- **Multi-tier fallback system** ensuring reliability even with API failures
- **Intelligent text chunking** with overlap for context preservation
- **Source diversity scoring** for credibility assessment
- **Real-time web scraping** with robust error handling
- **Scalable vector storage** supporting both cloud and local deployments

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API keys for Gemini, Serper.dev, and ChromaDB Cloud

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd real_time_news_rag_with_fact_checking
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   Create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   SERPER_API_KEY=your_serper_api_key_here
   CHROMADB_API_KEY=your_chromadb_api_key_here
   CHROMADB_TENANT=435c9546-30ce-47da-9431-c759362c04b0
   CHROMADB_DATABASE="Dummy DB"
   CHROMA_COLLECTION=news_rag
   
   # Optional: Use local ChromaDB instead of cloud
   USE_CHROMA_LOCAL=true
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 🔧 Configuration

### API Keys Required
- **Gemini API Key**: For LLM responses and embeddings
- **Serper.dev API Key**: For web search evidence retrieval
- **ChromaDB Cloud API Key**: For vector storage (with local fallback)

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `SERPER_API_KEY` | Serper.dev API key | Optional |
| `CHROMADB_API_KEY` | ChromaDB Cloud API key | Optional |
| `USE_CHROMA_LOCAL` | Use local ChromaDB | `false` |

## 📱 User Interface

The application features three main tabs:

### 1. **Ingest Article**
- Input news article URLs for processing
- Automatic text extraction and chunking
- Vector embedding and storage
- Real-time processing feedback

### 2. **Ask (RAG)**
- Query the knowledge base with natural language
- Retrieval-augmented generation using stored context
- Configurable retrieval parameters (top-k chunks)
- Source attribution and context display

### 3. **Fact-Check a Claim**
- Input specific claims for verification
- Automatic claim extraction from longer text
- Web evidence search and analysis
- Credibility scoring and verdict generation
- Source citation and local context matching

### Additional Features
- **Quick Fake News Classifier**: Instant classification of headlines
- **Real-time Configuration Status**: API key validation and system status
- **Responsive Design**: Optimized for desktop and mobile use

## 🔍 How It Works

### 1. **Content Ingestion Pipeline**
```
URL → Web Scraping → Text Extraction → Chunking → Embedding → Vector Storage
```

### 2. **RAG Query Flow**
```
User Question → Query Embedding → Vector Search → Context Retrieval → LLM Generation → Answer
```

### 3. **Fact-Checking Process**
```
Claim → Web Evidence Search → Multi-Source Analysis → LLM Reasoning → Verdict + Score
```

### 4. **Misinformation Detection**
```
Text Input → BERT Classification → Fake/Real Probability → Confidence Score
```

## 📊 Key Features

### Real-time Processing
- **Continuous ingestion** of news articles
- **Live fact-checking** with web evidence
- **Instant classification** of content credibility

### Intelligent Retrieval
- **Semantic search** using state-of-the-art embeddings
- **Context-aware chunking** preserving article coherence
- **Multi-source verification** for comprehensive fact-checking

### Scalable Architecture
- **Cloud-native design** with local fallbacks
- **Efficient vector storage** supporting large-scale deployments
- **Modular components** for easy extension and customization

### Robust Error Handling
- **Graceful degradation** when APIs are unavailable
- **Multiple fallback strategies** ensuring system reliability
- **Comprehensive logging** for debugging and monitoring

## 🧪 Evaluation Metrics

The system provides several evaluation dimensions:

- **Retrieval Accuracy**: Context relevance scoring
- **Response Latency**: End-to-end processing time
- **Credibility Scoring**: Multi-factor source assessment
- **Misinformation Detection**: Classification accuracy
- **Source Diversity**: Evidence breadth and quality

## 🔮 Future Enhancements

- **Multi-modal support** for images and videos
- **Advanced bias detection** algorithms
- **Temporal fact verification** with time-series analysis
- **Collaborative fact-checking** with user contributions
- **API endpoints** for integration with other systems

## 🛠️ Development

### Project Structure
```
real_time_news_rag_with_fact_checking/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── .env                  # Environment configuration (create this)
└── data/                 # Local data storage (auto-created)
    └── chroma/          # Local ChromaDB storage
```

### Code Organization
- **Single-file architecture** for rapid deployment
- **Modular functions** with clear separation of concerns
- **Comprehensive error handling** and fallback mechanisms
- **Type hints** for better code maintainability

### Testing
- **Streamlit interface testing** with user interactions
- **API integration testing** with various configurations
- **Fallback mechanism validation** for reliability testing

## 📚 API Documentation

### Vector Store Operations
- `add(ids, texts, metadatas)`: Store new content
- `query(query_text, k)`: Retrieve relevant context
- `_init_store()`: Initialize storage backend

### Fact-Checking Functions
- `fact_check_claim(claim)`: Verify claim with evidence
- `extract_claims_gemini(text)`: Extract checkable claims
- `score_credibility(evidence, verdict)`: Calculate credibility score

### Content Processing
- `ingest_url(url, label)`: Process and store article
- `extract_main_text(url)`: Extract article content
- `chunk_text(text, max_len, overlap)`: Create searchable chunks

## 🚨 Limitations & Considerations

### Current Limitations
- **API rate limits** may affect high-volume usage
- **Web scraping** depends on site accessibility
- **Model accuracy** varies with content complexity
- **Storage costs** for cloud vector databases

### Best Practices
- **Verify sources** independently for critical decisions
- **Monitor API usage** to avoid rate limiting
- **Regular model updates** for improved accuracy
- **Backup strategies** for local deployments

## 🤝 Contributing

This project is designed for educational purposes and coursework submission. Contributions are welcome for:

- **Bug fixes** and performance improvements
- **New feature implementations**
- **Documentation enhancements**
- **Testing and validation**

## 📄 License

This project is created for educational coursework purposes. Please ensure compliance with:

- **API terms of service** for external services
- **Content usage rights** for scraped materials
- **Academic integrity** requirements

## 📞 Support

For technical support or questions:

- **Check the configuration** in the sidebar
- **Verify API keys** are properly set
- **Review error messages** for specific issues
- **Test with sample URLs** to validate functionality

---

**⚠️ Disclaimer**: This system is designed for educational and research purposes. Always verify information from multiple sources and exercise critical thinking when evaluating claims. The system provides assistance but should not be the sole basis for important decisions.
