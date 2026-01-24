import os
import json
import numpy as np
import torch
import gc
import spacy
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ================= CONFIGURATION =================
# 1. OPTIMIZE MEMORY ALLOCATION
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATA_FOLDER = "extracted_texts_new" # Check if this matches your folder name
CACHE_DIR = Path("cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# !!! CONTROL SWITCHES !!!
TEST_MODE = False           
SAVE_RAW_EMBEDDINGS = True 

KEYWORDS = [
    "groundwater withdrawal", "ground-water monitoring", "underground water abstraction", 
    "groundwater permits", "groundwater rights", "well and borehole drilling licenses", 
    "conjunctive management", "groundwater protection zones", "aquifer recharge", 
    "transboundary aquifers"
]

EMBEDDING_MODELS = [
    {
        "name": "Qwen/Qwen3-Embedding-8B",
        "short_name": "Qwen3-8B",
        "batch_size": 1, 
        "max_chunks": None,
        "prompt": None,
        "trust_remote": True,
        "fp16": True
    },
    {
        "name": "BAAI/bge-m3",
        "short_name": "BGE-M3",
        "batch_size": 8, # Increased for 3090
        "max_chunks": None,
        "prompt": None,
        "trust_remote": True,
        "fp16": True
    },
    {
        "name": "jinaai/jina-embeddings-v3",
        "short_name": "Jina-v3",
        "batch_size": 8, # Increased for 3090
        "max_chunks": None,
        "prompt": None,
        "trust_remote": True,
        "fp16": True,
        "is_jina": True
    }
]

def clean_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def load_and_chunk_documents(folder):
    print("\n" + "="*60)
    print(f"STEP 1: Loading Documents from {folder}")
    print("="*60)
    
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 3500000 
    
    path_obj = Path(folder)
    if not path_obj.exists():
        print(f"ERROR: Folder {folder} does not exist!")
        return {}

    files = list(path_obj.glob("*.txt"))
    
    if TEST_MODE:
        print(f"⚠ TEST MODE ACTIVE: Processing only first 5 documents")
        files = files[:5]
    else:
        print(f"Found {len(files)} text files")
    
    doc_chunks = {}
    
    for f in tqdm(files, desc="Chunking"):
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')
            if len(text) < 100: continue
            
            doc = nlp(text[:3500000])
            chunks = []
            for sent in doc.sents:
                s = sent.text.strip()
                if (len(s.split()) >= 10 and len(s) >= 50 and 
                    "Page" not in s[:15] and "---" not in s):
                    chunks.append(s)
            
            if chunks:
                doc_chunks[f.stem] = chunks
                
        except Exception as e:
            print(f"  Error reading {f.name}: {e}")
    
    return doc_chunks

def load_model(config, device):
    clean_memory() # Clear before loading
    print(f"  Loading {config['name']}...")
    
    kwargs = {"trust_remote_code": config.get("trust_remote", False)}
    
    # 1. CONFIGURATION FOR QWEN (Needs Flash Attention for Memory)
    if "Qwen" in config["name"] and device == "cuda":
        kwargs["model_kwargs"] = {
            "torch_dtype": torch.float16,
            "attn_implementation": "flash_attention_2" 
        }

    # 2. CONFIGURATION FOR JINA (CRITICAL FIX)
    # Jina V3 crashes if we force flash_attention_2. 
    # We also force use_flash_attn=False to prevent its internal assertion error.
    elif "jina" in config["name"]:
        kwargs["model_kwargs"] = {
            "torch_dtype": torch.float16,
            "use_flash_attn": False 
        }

    # 3. CONFIGURATION FOR BGE / OTHERS
    elif config.get("fp16") and device == "cuda":
        kwargs["model_kwargs"] = {
            "torch_dtype": torch.float16
        }
    
    # Attempt to load
    try:
        return SentenceTransformer(config["name"], device=device, **kwargs)
    except Exception as e:
        print(f"  ⚠ Standard load failed: {e}")
        
        # FALLBACK: If Qwen fails with Flash Attn, try Native (might OOM, but better than crash)
        if "model_kwargs" in kwargs and "attn_implementation" in kwargs["model_kwargs"]:
            print("  ↻ Retrying without Flash Attention...")
            del kwargs["model_kwargs"]["attn_implementation"]
            return SentenceTransformer(config["name"], device=device, **kwargs)
        raise e

def process_model(model, config, chunks_map):
    print(f"\n  Processing with {config['short_name']}...")
    
    print("  Encoding keywords...")
    kw_args = {"show_progress_bar": False, "normalize_embeddings": True}
    if config.get("is_jina"): kw_args["task"] = "retrieval.query"
    
    prompts = [config["prompt"] + k for k in KEYWORDS] if config.get("prompt") else KEYWORDS
    kw_embeds = model.encode(prompts, **kw_args)
    
    results = {}
    max_chunks = config.get("max_chunks", None)
    
    raw_embed_dir = CACHE_DIR / "raw_embeddings" / config["short_name"]
    if SAVE_RAW_EMBEDDINGS:
        raw_embed_dir.mkdir(parents=True, exist_ok=True)
    
    for country, chunks in tqdm(chunks_map.items(), desc=f"  {config['short_name']}", leave=False):
        chunks_subset = chunks[:max_chunks]
        if not chunks_subset: continue
        
        try:
            doc_args = {"batch_size": config["batch_size"], "show_progress_bar": False, "normalize_embeddings": True}
            if config.get("is_jina"): doc_args["task"] = "retrieval.passage"
            
            chunk_embeds = model.encode(chunks_subset, **doc_args)
            
            sim_matrix = cosine_similarity(kw_embeds, chunk_embeds)
            
            # Global Mean Metric
            keyword_means = np.mean(sim_matrix, axis=1)
            global_score = np.mean(keyword_means)

            if SAVE_RAW_EMBEDDINGS:
                np.savez_compressed(
                    raw_embed_dir / f"{country}.npz", 
                    embeddings=chunk_embeds,
                    text_chunks=chunks_subset,
                    chunk_scores=np.max(sim_matrix, axis=0)
                )
            
            results[country] = {
                "weighted_score": float(global_score), 
                "keyword_scores": {k: float(s) for k, s in zip(KEYWORDS, keyword_means)},
                "chunk_count": len(chunks_subset)
            }
            
        except Exception as e:
            print(f"    Error {country}: {e}")
            clean_memory() # Panic clear if an error occurs inside loop
            
    return results

def main():
    print("\n" + "="*70)
    print(" WATER POLICY EMBEDDING ANALYSIS (FULL DOCS) ".center(70))
    print("="*70)
    
    clean_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"VRAM Free: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")
    
    chunks = load_and_chunk_documents(DATA_FOLDER)
    if not chunks: return
    
    for config in EMBEDDING_MODELS:
        cache_path = CACHE_DIR / f"country_scores_{config['short_name']}.json"
        
        if cache_path.exists() and not TEST_MODE:
             print(f"\n  ℹ Note: {cache_path} exists. Skipping...")
             continue 
            
        model = load_model(config, device)
        if model:
            scores = process_model(model, config, chunks)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(scores, f, indent=2)
            
            # AGGRESSIVE CLEANUP AFTER EACH MODEL
            del model
            clean_memory()

    print("\n" + "="*60)
    print("EMBEDDING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()