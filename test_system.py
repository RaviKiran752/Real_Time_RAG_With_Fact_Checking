#!/usr/bin/env python3
"""
Test script for Real-time News RAG with Fact-Checking system
Run this to verify all components are working correctly
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test that all required packages can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import google.generativeai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Google Generative AI import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB imported successfully")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        import transformers
        print("✅ Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import faiss
        print("✅ FAISS imported successfully")
    except ImportError as e:
        print(f"❌ FAISS import failed: {e}")
        return False
    
    try:
        import trafilatura
        print("✅ Trafilatura imported successfully")
    except ImportError as e:
        print(f"❌ Trafilatura import failed: {e}")
        return False
    
    try:
        import requests
        import bs4
        print("✅ Web scraping libraries imported successfully")
    except ImportError as e:
        print(f"❌ Web scraping libraries import failed: {e}")
        return False
    
    return True

def test_environment():
    """Test environment variables and configuration"""
    print("\n🔧 Testing environment configuration...")
    
    load_dotenv()
    
    # Check required API keys
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print("✅ Gemini API key found")
    else:
        print("⚠️  Gemini API key not found (will use fallbacks)")
    
    serper_key = os.getenv("SERPER_API_KEY")
    if serper_key:
        print("✅ Serper.dev API key found")
    else:
        print("⚠️  Serper.dev API key not found (web evidence disabled)")
    
    chroma_key = os.getenv("CHROMADB_API_KEY")
    if chroma_key:
        print("✅ ChromaDB API key found")
    else:
        print("⚠️  ChromaDB API key not found (will use local fallback)")
    
    return True

def test_app_import():
    """Test that the main app can be imported"""
    print("\n📱 Testing app import...")
    
    try:
        # Import the main app module
        sys.path.append('.')
        import app
        print("✅ Main app imported successfully")
        print(f"   Vector store type: {app.VS.kind}")
        print(f"   App title: {app.APP_TITLE}")
        return True
    except Exception as e:
        print(f"❌ App import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without API calls"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        import app
        
        # Test text processing
        test_text = "This is a test article about artificial intelligence and machine learning."
        chunks = app.chunk_text(test_text, max_len=50, overlap=10)
        print(f"✅ Text chunking: {len(chunks)} chunks created")
        
        # Test text normalization
        normalized = app.normalize_text("  Multiple    spaces   and\nnewlines  ")
        print(f"✅ Text normalization: '{normalized}'")
        
        # Test claim extraction (fallback mode)
        claims = app.extract_claims_gemini("Test paragraph with multiple sentences. Second sentence here. Third sentence.")
        print(f"✅ Claim extraction: {len(claims)} claims extracted")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Real-time News RAG System Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_environment,
        test_app_import,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is ready to run.")
        print("\nTo start the application:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("   1. Install missing packages: pip install -r requirements.txt")
        print("   2. Check your .env file configuration")
        print("   3. Verify Python version (3.11+)")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
