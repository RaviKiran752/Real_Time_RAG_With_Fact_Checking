# 🚀 Deployment Guide

This guide covers deploying your Real-time News RAG with Fact-Checking system to various platforms.

## 🌐 Streamlit Cloud (Recommended)

### 1. **Prepare Your Repository**
- Ensure all files are committed to GitHub
- Verify `.gitignore` excludes sensitive files (`.env`, `data/`, etc.)
- Check that `requirements.txt` is up to date

### 2. **Deploy to Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository and branch
5. Set the main file path to `app.py`
6. Click "Deploy!"

### 3. **Configure Environment Variables**
In Streamlit Cloud dashboard:
1. Go to your app's settings
2. Navigate to "Secrets"
3. Add your environment variables:
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key"
   SERPER_API_KEY = "your_serper_api_key"
   CHROMADB_API_KEY = "your_chromadb_api_key"
   CHROMADB_TENANT = "435c9546-30ce-47da-9431-c759362c04b0"
   CHROMADB_DATABASE = "Dummy DB"
   CHROMA_COLLECTION = "news_rag"
   ```

### 4. **Update app.py for Streamlit Cloud**
The app automatically detects Streamlit Cloud and uses the secrets configuration.

## 🐳 Docker Deployment

### 1. **Create Dockerfile**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. **Build and Run**
```bash
docker build -t news-rag .
docker run -p 8501:8501 --env-file .env news-rag
```

## ☁️ Hugging Face Spaces

### 1. **Create Space**
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Streamlit" as SDK
4. Upload your files

### 2. **Configure Secrets**
In your Space settings, add the same environment variables as Streamlit Cloud.

## 🐍 Local Development

### 1. **Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. **Run Locally**
```bash
streamlit run app.py
```

### 3. **Environment Setup**
Create `.env` file with your API keys (see README.md).

## 🔧 Configuration Options

### **Vector Store Selection**
- **ChromaDB Cloud**: Set `CHROMADB_API_KEY` (recommended for production)
- **Local ChromaDB**: Set `USE_CHROMA_LOCAL=true`
- **FAISS**: Automatic fallback if ChromaDB unavailable

### **API Fallbacks**
- **Gemini**: Primary LLM and embedding provider
- **Sentence Transformers**: Local embedding fallback
- **Serper.dev**: Web evidence retrieval (optional)

## 📊 Performance Optimization

### **For Production**
1. **Use ChromaDB Cloud** for better scalability
2. **Enable caching** for frequently accessed content
3. **Monitor API usage** to avoid rate limits
4. **Use GPU** for transformer models if available

### **For Development**
1. **Use local ChromaDB** to avoid cloud costs
2. **Limit chunk sizes** for faster processing
3. **Cache models** locally to avoid re-downloads

## 🚨 Troubleshooting

### **Common Issues**

#### **API Key Errors**
- Verify all API keys are correctly set
- Check API key permissions and quotas
- Ensure no extra spaces or characters

#### **Import Errors**
- Verify all dependencies are installed
- Check Python version compatibility (3.11+)
- Clear cache: `pip cache purge`

#### **ChromaDB Connection Issues**
- Verify cloud credentials
- Check network connectivity
- Try local fallback: `USE_CHROMA_LOCAL=true`

#### **Memory Issues**
- Reduce chunk sizes in `app.py`
- Use smaller embedding models
- Enable garbage collection

### **Debug Mode**
Add to your environment:
```bash
export STREAMLIT_LOG_LEVEL=debug
streamlit run app.py --logger.level=debug
```

## 📈 Scaling Considerations

### **Horizontal Scaling**
- Deploy multiple Streamlit instances behind a load balancer
- Use Redis for session sharing
- Implement database connection pooling

### **Vertical Scaling**
- Increase memory allocation for larger models
- Use GPU acceleration for transformer operations
- Optimize chunk sizes based on available resources

## 🔒 Security Best Practices

### **API Key Management**
- Never commit API keys to version control
- Use environment variables or secrets management
- Rotate keys regularly
- Monitor API usage for anomalies

### **Data Privacy**
- Implement user authentication if needed
- Log access patterns for audit trails
- Sanitize user inputs
- Respect robots.txt for web scraping

## 📝 Monitoring & Logging

### **Health Checks**
- Monitor API response times
- Track error rates
- Check vector store performance
- Monitor memory usage

### **Analytics**
- Track user interactions
- Monitor fact-checking accuracy
- Analyze retrieval performance
- Measure system latency

## 🎯 Next Steps

After successful deployment:

1. **Test all functionality** with sample URLs and claims
2. **Monitor performance** and optimize as needed
3. **Gather user feedback** for improvements
4. **Plan scaling strategy** based on usage patterns
5. **Implement monitoring** and alerting systems

---

**Need Help?** Check the main README.md for detailed usage instructions and troubleshooting tips.
