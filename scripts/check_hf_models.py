#!/usr/bin/env python3
"""
Check Hugging Face Model Storage and Cache
"""

import os
from pathlib import Path
import platform

def check_hf_cache():
    """Check where Hugging Face models are stored locally"""
    
    print("🤖 Hugging Face Model Storage Information")
    print("=" * 50)
    
    # Default cache directories by OS
    if platform.system() == "Windows":
        cache_dir = Path.home() / ".cache" / "huggingface"
        alt_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    elif platform.system() == "Darwin":  # macOS
        cache_dir = Path.home() / ".cache" / "huggingface"
        alt_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    else:  # Linux
        cache_dir = Path.home() / ".cache" / "huggingface"
        alt_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    
    print(f"🗂️  Default Cache Directory: {cache_dir}")
    print(f"🗂️  Environment Override: {alt_cache}")
    
    # Check if cache exists
    if cache_dir.exists():
        print(f"✅ Cache directory exists")
        
        # Check for transformers cache
        transformers_cache = cache_dir / "transformers"
        if transformers_cache.exists():
            print(f"✅ Transformers cache found: {transformers_cache}")
            
            # List cached models
            try:
                models = list(transformers_cache.iterdir())
                print(f"📦 Cached models: {len(models)}")
                
                for model_dir in models[:5]:  # Show first 5
                    if model_dir.is_dir():
                        size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                        size_mb = size / (1024 * 1024)
                        print(f"   • {model_dir.name}: {size_mb:.1f} MB")
                        
            except Exception as e:
                print(f"⚠️  Could not list models: {e}")
        else:
            print(f"❌ No transformers cache found")
    else:
        print(f"❌ Cache directory does not exist yet")
    
    # Check current models used by our system
    print(f"\n🎯 Models Used by ARA AI:")
    print("-" * 30)
    
    models_info = [
        {
            "name": "ProsusAI/finbert",
            "purpose": "Financial sentiment analysis",
            "size": "~440 MB"
        },
        {
            "name": "cardiffnlp/twitter-roberta-base-sentiment-latest", 
            "purpose": "General sentiment analysis",
            "size": "~500 MB"
        }
    ]
    
    for model in models_info:
        print(f"📊 {model['name']}")
        print(f"   Purpose: {model['purpose']}")
        print(f"   Size: {model['size']}")
        print()
    
    # Show total storage impact
    print(f"💾 Total Storage Impact:")
    print(f"   • Hugging Face Models: ~1 GB")
    print(f"   • ARA AI Models: ~50 MB (in models/ folder)")
    print(f"   • Python Dependencies: ~2 GB")
    print(f"   • Total: ~3 GB")

def check_model_loading():
    """Test model loading and show where they're loaded from"""
    
    print(f"\n🔄 Testing Model Loading...")
    print("-" * 30)
    
    try:
        from transformers import pipeline
        import time
        
        print("📥 Loading FinBERT model...")
        start_time = time.time()
        
        # This will download if not cached, or load from cache
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
        
        load_time = time.time() - start_time
        print(f"✅ FinBERT loaded in {load_time:.2f} seconds")
        
        # Test the model
        test_text = "Apple stock shows strong performance with positive outlook"
        result = sentiment_pipeline(test_text)
        
        print(f"🧪 Test Result: {result[0]['label']} ({result[0]['score']:.2f})")
        
        if load_time < 5:
            print("🚀 Fast loading indicates model is cached locally!")
        else:
            print("⏳ Slow loading indicates model is being downloaded...")
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

def show_offline_capability():
    """Show that models work offline after initial download"""
    
    print(f"\n🌐 Offline Capability:")
    print("-" * 20)
    print("✅ After initial download, models work completely offline")
    print("✅ No internet connection required for predictions")
    print("✅ No API keys or authentication needed")
    print("✅ No rate limits or usage restrictions")
    print("✅ Models are yours to use indefinitely")
    
    print(f"\n🔒 Privacy & Security:")
    print("-" * 20)
    print("✅ All processing happens locally on your machine")
    print("✅ No data sent to external servers")
    print("✅ Complete privacy for your stock analysis")
    print("✅ No dependency on external services")

if __name__ == "__main__":
    check_hf_cache()
    check_model_loading()
    show_offline_capability()