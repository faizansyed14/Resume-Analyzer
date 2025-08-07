#!/usr/bin/env python3
"""
Setup script to download all required models and dependencies
Run this after installing requirements.txt
"""

import subprocess
import sys
import nltk

def install_spacy_model():
    """Download spaCy transformer model"""
    try:
        print("Downloading spaCy transformer model...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_trf"])
        print("✓ spaCy model downloaded successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download spaCy model: {e}")

def download_nltk_data():
    """Download required NLTK data"""
    nltk_downloads = [
        'punkt',
        'stopwords', 
        'wordnet',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'omw-1.4'
    ]
    
    print("Downloading NLTK data...")
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
            print(f"✓ Downloaded {item}")
        except Exception as e:
            print(f"✗ Failed to download {item}: {e}")

def setup_tesseract():
    """Setup instructions for Tesseract OCR"""
    print("\n" + "="*50)
    print("TESSERACT OCR SETUP REQUIRED")
    print("="*50)
    print("Please install Tesseract OCR for your system:")
    print()
    print("Ubuntu/Debian:")
    print("  sudo apt-get install tesseract-ocr")
    print()
    print("macOS:")
    print("  brew install tesseract")
    print()
    print("Windows:")
    print("  Download from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("  Add to PATH: C:\\Program Files\\Tesseract-OCR")
    print()

def download_huggingface_models():
    """Pre-download HuggingFace models to cache"""
    try:
        print("Pre-downloading HuggingFace models...")
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from sentence_transformers import SentenceTransformer
        
        # Download NER model
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        print("✓ HuggingFace NER model cached")
        
        # Download sentence transformer models
        models_to_cache = [
            'all-MiniLM-L6-v2',
            'all-mpnet-base-v2'
        ]
        
        for model_name in models_to_cache:
            model = SentenceTransformer(model_name)
            print(f"✓ Cached sentence transformer: {model_name}")
            
    except Exception as e:
        print(f"✗ Failed to cache HuggingFace models: {e}")

def setup_language_tool():
    """Setup Language Tool"""
    try:
        print("Setting up Language Tool...")
        import language_tool_python
        tool = language_tool_python.LanguageTool('en-US')
        tool.close()
        print("✓ Language Tool setup successful")
    except Exception as e:
        print(f"✗ Language Tool setup failed: {e}")

def verify_installation():
    """Verify all components are working"""
    print("\n" + "="*50)
    print("VERIFYING INSTALLATION")
    print("="*50)
    
    # Test spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_trf")
        doc = nlp("Test sentence for verification.")
        print("✓ spaCy transformer model working")
    except Exception as e:
        print(f"✗ spaCy verification failed: {e}")
    
    # Test NLTK
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        tokens = word_tokenize("Test sentence for verification.")
        stops = set(stopwords.words('english'))
        print("✓ NLTK working")
    except Exception as e:
        print(f"✗ NLTK verification failed: {e}")
    
    # Test sentence transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding = model.encode("Test sentence")
        print("✓ Sentence transformers working")
    except Exception as e:
        print(f"✗ Sentence transformers verification failed: {e}")
    
    # Test transformers
    try:
        from transformers import pipeline
        print("✓ HuggingFace transformers available")
    except Exception as e:
        print(f"✗ HuggingFace transformers verification failed: {e}")
    
    # Test other libraries
    libraries_to_test = [
        'phonenumbers',
        'email_validator', 
        'nameparser',
        'textstat',
        'sklearn',
        'numpy',
        'pandas'
    ]
    
    for lib in libraries_to_test:
        try:
            __import__(lib)
            print(f"✓ {lib} working")
        except ImportError as e:
            print(f"✗ {lib} not available: {e}")

def main():
    """Main setup function"""
    print("="*60)
    print("RESUME ANALYZER - ADVANCED SETUP")
    print("="*60)
    print()
    
    print("Step 1: Installing spaCy models...")
    install_spacy_model()
    print()
    
    print("Step 2: Downloading NLTK data...")
    download_nltk_data()
    print()
    
    print("Step 3: Caching HuggingFace models...")
    download_huggingface_models()
    print()
    
    print("Step 4: Setting up Language Tool...")
    setup_language_tool()
    print()
    
    setup_tesseract()
    
    print("Step 5: Verifying installation...")
    verify_installation()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("Your enhanced resume analyzer is ready to use.")
    print("Make sure Tesseract OCR is installed for full functionality.")
    print()

if __name__ == "__main__":
    main()