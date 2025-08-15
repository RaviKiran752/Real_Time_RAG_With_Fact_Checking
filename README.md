# Real-time News RAG with Fact-Checking

Hey there! This is my project for building a real-time news analysis system that can actually fact-check claims and detect misinformation. I got tired of reading news and wondering if it's true, so I built this thing that uses AI to help figure out what's real and what's not.

## What This Does

Basically, it's a smart news reader that:
- Scrapes news articles from URLs you give it
- Breaks them down into chunks and stores them in a vector database
- Lets you ask questions and get answers based on the articles you've loaded
- Fact-checks claims by searching the web for evidence
- Detects bias in news content
- Monitors how fresh your content is over time
- Gives you credibility scores for fact-check results

## How It Works

The system uses a few key technologies:
- **Gemini 1.5 Flash** for generating answers and analyzing text (chose this over Pro for better rate limits)
- **ChromaDB** (Cloud or local) for storing and searching through article chunks
- **Serper.dev** for searching the web to find evidence for fact-checking
- **HuggingFace transformers** for detecting fake news
- **Streamlit** for the web interface

## Quick Start

1. **Clone this repo** and navigate to the directory
2. **Set up Python environment** (3.11+ recommended)
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Create a .env file** with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_key_here
   SERPER_API_KEY=your_serper_key_here
   CHROMADB_API_KEY=your_chromadb_key_here
   CHROMADB_TENANT=your_tenant_id
   CHROMADB_DATABASE=your_database_name
   ```
5. **Run the app**: `streamlit run app.py`

## What You Can Do

### 1. Ingest Articles
Paste a news article URL and the system will:
- Scrape the content
- Break it into manageable chunks
- Store it in the vector database
- Analyze it for bias
- Track when it was added

### 2. Ask Questions (RAG)
Ask questions about the articles you've loaded:
- "What happened to Tesla stock today?"
- "What are the main points about climate change?"
- The system finds relevant chunks and generates answers

### 3. Fact-Check Claims
Paste a claim and get it fact-checked:
- Searches the web for evidence
- Analyzes the evidence
- Gives you a verdict (Supported/Refuted/Needs more evidence)
- Provides credibility scores
- Shows temporal relevance (how fresh the info is)
- Source attribution and context display

## Key Features

- **Real-time monitoring**: Background processes keep your content fresh
- **Bias detection**: Identifies political, economic, and social bias
- **Temporal verification**: Checks if facts are still relevant
- **Enhanced credibility scoring**: Multi-factor analysis of fact-check results
- **Fallback systems**: Works even if some APIs are down
- **Clean UI**: Easy to use interface with tabs for different functions

## API Keys You'll Need

- **Gemini**: Get this from Google AI Studio (free tier: 15 req/min, 1500 req/day)
- **Serper.dev**: For web search during fact-checking
- **ChromaDB**: For cloud vector storage (optional, falls back to local)

## Architecture

The system is built around a few core components:
- **Vector Store**: Handles storage and retrieval of article chunks
- **Content Monitor**: Background thread that keeps content fresh
- **Fact-Checker**: Combines web search with AI analysis
- **Bias Detector**: Keyword-based bias identification
- **UI Layer**: Streamlit interface for easy interaction

## Why I Built It This Way

I wanted something that could actually work in real-world scenarios, not just a demo. So I focused on:
- **Reliability**: Multiple fallback options if APIs fail
- **Performance**: Efficient chunking and retrieval
- **Usability**: Clean interface that doesn't require technical knowledge
- **Scalability**: Can handle growing amounts of content
- **Monitoring**: Built-in tools to see how the system is performing

## Limitations

- **Rate limits**: Free API tiers have limits
- **Content freshness**: Web content changes, so facts can become outdated
- **Bias detection**: Current approach is keyword-based (could be improved with ML)
- **Source verification**: Relies on web search results (not perfect)

## Future Improvements

I'm thinking about adding:
- Better bias detection using ML models
- Source credibility scoring based on historical accuracy
- Automated fact-checking workflows
- Integration with news APIs for automatic ingestion
- Performance benchmarking tools
- Real-time alerts for misinformation detection
## Troubleshooting

- **"Module not found" errors**: Make sure you're in the right virtual environment
- **API rate limits**: Wait a bit or upgrade your API plan
- **ChromaDB issues**: Check your API keys and tenant settings
- **Content not loading**: Verify your .env file has the right keys

## Contributing

Feel free to fork this and improve it! Some areas that could use work:
- Better error handling
- More sophisticated bias detection
- Performance optimization
- Additional data sources
- Better UI/UX

## License

This is just a demo project I built for learning. Use it however you want, but remember it's not production-ready and you should always verify sources yourself.

---

Built with Python, Streamlit, and a lot of coffee. Hope you find it useful!
