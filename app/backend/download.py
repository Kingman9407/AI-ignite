"""
download.py
Downloads and caches offline models for clinical documentation system.
Run this once before using the system.
"""

import os
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration

def download_models():
    """Download BioClinicalBERT and FLAN-T5-Small models to local cache."""
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 60)
    print("DOWNLOADING CLINICAL NLP MODELS")
    print("=" * 60)
    
    # Download BioClinicalBERT for embeddings
    print("\n[1/2] Downloading BioClinicalBERT...")
    print("      Purpose: Clinical text embeddings")
    
    bio_bert_path = os.path.join(models_dir, "bio_clinical_bert")
    
    try:
        tokenizer_bio = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        model_bio = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        
        tokenizer_bio.save_pretrained(bio_bert_path)
        model_bio.save_pretrained(bio_bert_path)
        
        print("      ✓ BioClinicalBERT downloaded successfully")
        print(f"      ✓ Saved to: {bio_bert_path}")
    except Exception as e:
        print(f"      ✗ Error downloading BioClinicalBERT: {e}")
        return False
    
    # Download FLAN-T5-Small for text generation
    print("\n[2/2] Downloading FLAN-T5-Small...")
    print("      Purpose: Deterministic nursing note generation")
    
    flan_t5_path = os.path.join(models_dir, "flan_t5_small")
    
    try:
        tokenizer_t5 = T5Tokenizer.from_pretrained("google/flan-t5-small")
        model_t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        
        tokenizer_t5.save_pretrained(flan_t5_path)
        model_t5.save_pretrained(flan_t5_path)
        
        print("      ✓ FLAN-T5-Small downloaded successfully")
        print(f"      ✓ Saved to: {flan_t5_path}")
    except Exception as e:
        print(f"      ✗ Error downloading FLAN-T5-Small: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL MODELS DOWNLOADED SUCCESSFULLY")
    print("=" * 60)
    print("\nYou can now run main.py offline.")
    print("No internet connection will be required during runtime.")
    
    return True

if __name__ == "__main__":
    success = download_models()
    if not success:
        print("\n✗ Model download failed. Please check your internet connection.")
        exit(1)