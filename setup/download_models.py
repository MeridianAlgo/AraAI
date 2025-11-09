#!/usr/bin/env python3
"""
Download and cache lightweight AI models for Ara AI
Pre-downloads models to avoid delays during first use
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def download_models():
    """Download all required AI models"""
    print(" Downloading Lightweight AI Models for Ara AI")
    print("=" * 60)
    
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        import torch
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        device = 0 if gpu_available else -1
        
        print(f" GPU Available: {gpu_available}")
        print(f" Device: {'GPU' if gpu_available else 'CPU'}")
        print()
        
        # Model configurations (lightweight and accurate)
        models = [
            {
                'name': 'TinyLlama Conversational AI (Main)',
                'model_id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                'task': 'text-generation',
                'size': '2.2GB',
                'accuracy': '85%'
            },
            {
                'name': 'FinBERT (Financial Sentiment)',
                'model_id': 'ProsusAI/finbert',
                'task': 'sentiment-analysis',
                'size': '440MB',
                'accuracy': '91%'
            },
            {
                'name': 'BART Zero-Shot Classifier',
                'model_id': 'facebook/bart-large-mnli',
                'task': 'zero-shot-classification',
                'size': '1.6GB',
                'accuracy': '89%'
            },
            {
                'name': 'DialoGPT Fallback AI',
                'model_id': 'microsoft/DialoGPT-medium',
                'task': 'text-generation',
                'size': '350MB',
                'accuracy': '82%'
            }
        ]
        
        downloaded = 0
        failed = 0
        
        for model_info in models:
            try:
                print(f" Downloading {model_info['name']}...")
                print(f"   Model: {model_info['model_id']}")
                print(f"   Size: {model_info['size']} | Accuracy: {model_info['accuracy']}")
                
                # Download model
                if model_info['task'] == 'sentiment-analysis':
                    pipeline(
                        "sentiment-analysis",
                        model=model_info['model_id'],
                        device=device,
                        return_all_scores=True
                    )
                elif model_info['task'] == 'zero-shot-classification':
                    pipeline(
                        "zero-shot-classification",
                        model=model_info['model_id'],
                        device=device
                    )
                elif model_info['task'] == 'text-generation':
                    if 'TinyLlama' in model_info['model_id']:
                        # Special handling for TinyLlama
                        pipeline(
                            "text-generation",
                            model=model_info['model_id'],
                            device=device,
                            model_kwargs={
                                "torch_dtype": torch.float16 if gpu_available else torch.float32,
                                "device_map": "auto" if gpu_available else None
                            },
                            max_new_tokens=512
                        )
                    else:
                        # Standard text generation
                        pipeline(
                            "text-generation",
                            model=model_info['model_id'],
                            device=device,
                            max_length=300
                        )
                
                print(f"    Downloaded successfully")
                downloaded += 1
                
            except Exception as e:
                print(f"    Failed: {e}")
                failed += 1
            
            print()
        
        print("=" * 60)
        print(" DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f" Successfully downloaded: {downloaded}")
        print(f" Failed downloads: {failed}")
        print(f" Total models: {len(models)}")
        
        if downloaded > 0:
            print("\n Models ready for use!")
            print("Ara AI will now load faster on first use.")
        
        if failed > 0:
            print(f"\n  {failed} models failed to download.")
            print("Ara AI will attempt to download them during first use.")
        
        print("\n Tips:")
        print("• Models are cached locally for faster loading")
        print("• GPU acceleration will be used automatically if available")
        print("• You can run this script again to update models")
        
        return downloaded > 0
        
    except ImportError:
        print(" Transformers library not installed")
        print("Run: pip install transformers torch")
        return False
    except Exception as e:
        print(f" Download failed: {e}")
        return False

def check_model_cache():
    """Check which models are already cached"""
    try:
        from transformers import AutoTokenizer
        
        models_to_check = [
            'ProsusAI/finbert',
            'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'facebook/bart-large-mnli',
            'microsoft/DialoGPT-medium'
        ]
        
        print(" Checking cached models...")
        
        cached = 0
        for model_id in models_to_check:
            try:
                # Try to load tokenizer (lightweight check)
                AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                print(f" {model_id} - Cached")
                cached += 1
            except:
                print(f" {model_id} - Not cached")
        
        print(f"\n {cached}/{len(models_to_check)} models cached")
        return cached == len(models_to_check)
        
    except ImportError:
        print(" Transformers not available")
        return False

def main():
    """Main function"""
    print(" Ara AI Model Downloader")
    print("Downloads lightweight, accurate AI models for enhanced analysis")
    print()
    
    # Check current cache status
    all_cached = check_model_cache()
    
    if all_cached:
        print("\n All models already cached!")
        print("No download needed.")
        return True
    
    print("\n Starting model download...")
    success = download_models()
    
    if success:
        print("\n Next steps:")
        print("• Run 'python ara.py AAPL' to test predictions")
        print("• Run 'python ara.py --ai-analysis MSFT' for AI analysis")
        print("• Models will load faster now!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)