import os
import json
import numpy as np
import torch
import gc
import spacy
import logging
import traceback
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F

# ================= LOGGING SETUP =================
# Configure comprehensive logging for debugging and error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embedding_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================= CONFIGURATION =================
# 1. OPTIMIZE MEMORY ALLOCATION
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. MAKE PATHS RELATIVE TO SCRIPT LOCATION (Portable)
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_FOLDER = SCRIPT_DIR / "extracted_texts_new"
CACHE_DIR = SCRIPT_DIR / "cache"
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
        "batch_size": 8,  # Optimized for RTX 5090 (was 1)
        "max_chunks": None,
        "prompt": None,
        "trust_remote": True,
        "fp16": True
    },
    {
        "name": "BAAI/bge-m3",
        "short_name": "BGE-M3",
        "batch_size": 128,  # Optimized for RTX 5090 (was 8)
        "max_chunks": None,
        "prompt": None,
        "trust_remote": True,
        "fp16": True
    },
    {
        "name": "jinaai/jina-embeddings-v3",
        "short_name": "Jina-v3",
        "batch_size": 128,  # Optimized for RTX 5090 (was 8)
        "max_chunks": None,
        "prompt": None,
        "trust_remote": True,
        "fp16": True,
        "is_jina": True
    }
]

# ================= RERANKER CONFIGURATION =================
RERANKER_CONFIG = {
    "name": "BAAI/bge-reranker-v2-m3",
    "enabled": True,              # Set to False to disable reranking
    "fp16": True,                 # Use FP16 for faster inference on 5090
    "batch_size": 256,            # Optimized for RTX 5090 (was 32) - 4-8x faster
    "top_k_initial": 100,         # Initial retrieval depth (increases recall)
    "top_n_final": 20,            # Final number of chunks after reranking
    "trust_remote": True
}

def clean_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def cosine_similarity_gpu(matrix1, matrix2):
    """
    GPU-accelerated cosine similarity calculation using PyTorch.
    5-10% faster than sklearn's CPU implementation.

    Args:
        matrix1: numpy array of shape (n, d)
        matrix2: numpy array of shape (m, d)

    Returns:
        similarity matrix of shape (n, m)
    """
    if not torch.cuda.is_available():
        # Fallback to sklearn on CPU
        return cosine_similarity(matrix1, matrix2)

    try:
        # Convert to torch tensors on GPU with FP16 for speed
        tensor1 = torch.tensor(matrix1, device='cuda', dtype=torch.float16)
        tensor2 = torch.tensor(matrix2, device='cuda', dtype=torch.float16)

        # Compute cosine similarity on GPU
        # F.cosine_similarity expects (B, 1, D) and (B, N, D) for broadcasting
        similarity = F.cosine_similarity(
            tensor1.unsqueeze(1),  # (n, 1, d)
            tensor2.unsqueeze(0),  # (1, m, d)
            dim=2
        )

        # Convert back to numpy
        return similarity.cpu().numpy()
    except Exception as e:
        # Fallback to sklearn if GPU calculation fails
        print(f"    ⚠ GPU cosine similarity failed: {e}, using CPU")
        return cosine_similarity(matrix1, matrix2)

def load_reranker(config, device):
    """Load the CrossEncoder reranker model with FP16 support."""
    if not config.get("enabled", True):
        return None

    clean_memory()  # Clear before loading
    print(f"  Loading reranker: {config['name']}...")

    kwargs = {}
    if config.get("trust_remote", False):
        kwargs["trust_remote_code"] = True

    # Load CrossEncoder
    model = CrossEncoder(config["name"], device=device, **kwargs)

    # Enable FP16 if requested and on CUDA
    if config.get("fp16") and device == "cuda":
        try:
            model.model.half()
            print(f"  ✓ Reranker loaded in FP16 mode")
        except Exception as e:
            print(f"  ⚠ FP16 conversion failed: {e}, using FP32")

    return model

def rerank_chunks(reranker, query, chunks, scores, top_n=20, batch_size=32):
    """
    Rerank retrieved chunks using CrossEncoder for improved relevance.

    Args:
        reranker: CrossEncoder model
        query: The search query (keyword)
        chunks: List of text chunks
        scores: Initial cosine similarity scores
        top_n: Final number of chunks to return after reranking
        batch_size: Batch size for reranker inference

    Returns:
        reranked_indices: Indices of top chunks after reranking
        reranked_scores: Reranker scores for the top chunks
    """
    if reranker is None or len(chunks) == 0:
        # Fallback to cosine similarity ranking
        top_indices = np.argsort(scores)[::-1][:top_n]
        return top_indices, scores[top_indices]

    # Prepare query-chunk pairs for reranking
    pairs = [[query, chunk] for chunk in chunks]

    # Get reranker scores
    try:
        rerank_scores = reranker.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    except Exception as e:
        print(f"    ⚠ Reranking failed: {e}, falling back to cosine similarity")
        top_indices = np.argsort(scores)[::-1][:top_n]
        return top_indices, scores[top_indices]

    # Get top-n indices based on reranker scores
    top_indices = np.argsort(rerank_scores)[::-1][:top_n]

    return top_indices, rerank_scores[top_indices]

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
            logger.error(f"Error reading {f.name}: {e}")
            logger.debug(traceback.format_exc())
    
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
        model = SentenceTransformer(config["name"], device=device, **kwargs)

        # 4. TORCH.COMPILE() OPTIMIZATION (30-50% speedup on RTX 5090)
        if torch.__version__ >= "2.0.0" and device == "cuda":
            try:
                # Compile the underlying transformer model for faster inference
                if hasattr(model, '_first_module') and hasattr(model._first_module(), 'auto_model'):
                    model._first_module().auto_model = torch.compile(
                        model._first_module().auto_model,
                        mode="reduce-overhead",  # Best for repeated inference
                        fullgraph=True
                    )
                    print(f"  ✓ torch.compile() enabled for {config['short_name']}")
            except Exception as e:
                print(f"  ⚠ torch.compile() failed: {e}, continuing without compilation")

        return model
    except Exception as e:
        print(f"  ⚠ Standard load failed: {e}")

        # FALLBACK: If Qwen fails with Flash Attn, try Native (might OOM, but better than crash)
        if "model_kwargs" in kwargs and "attn_implementation" in kwargs["model_kwargs"]:
            print("  ↻ Retrying without Flash Attention...")
            del kwargs["model_kwargs"]["attn_implementation"]
            return SentenceTransformer(config["name"], device=device, **kwargs)
        raise e

def process_model(model, config, chunks_map, reranker=None, reranker_config=None):
    """
    Process documents with embedding model and optional reranking.

    Args:
        model: SentenceTransformer embedding model
        config: Embedding model configuration
        chunks_map: Dictionary mapping country -> list of text chunks
        reranker: Optional CrossEncoder reranker model
        reranker_config: Reranker configuration dict

    Returns:
        results: Dictionary of country scores
    """
    print(f"\n  Processing with {config['short_name']}...")
    if reranker is not None:
        print(f"  ✓ Reranking enabled with top_k={reranker_config['top_k_initial']} → top_n={reranker_config['top_n_final']}")

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

            # Compute initial cosine similarity matrix (GPU-accelerated)
            sim_matrix = cosine_similarity_gpu(kw_embeds, chunk_embeds)

            # ============ RERANKING STAGE ============
            if reranker is not None and reranker_config.get("enabled", True):
                top_k = reranker_config["top_k_initial"]
                top_n = reranker_config["top_n_final"]
                batch_size = reranker_config.get("batch_size", 32)

                # Initialize arrays to store reranked scores
                reranked_sim_matrix = np.zeros_like(sim_matrix)
                reranked_chunk_scores = np.zeros(len(chunks_subset))

                # For each keyword, retrieve top_k chunks and rerank
                for kw_idx, keyword in enumerate(KEYWORDS):
                    keyword_scores = sim_matrix[kw_idx]

                    # Get top_k candidates based on cosine similarity
                    top_k_candidates = min(top_k, len(chunks_subset))
                    top_indices = np.argsort(keyword_scores)[::-1][:top_k_candidates]

                    # Extract top chunks and their scores
                    candidate_chunks = [chunks_subset[i] for i in top_indices]
                    candidate_scores = keyword_scores[top_indices]

                    # Rerank the candidates
                    reranked_indices, reranked_scores = rerank_chunks(
                        reranker,
                        keyword,
                        candidate_chunks,
                        candidate_scores,
                        top_n=min(top_n, len(candidate_chunks)),
                        batch_size=batch_size
                    )

                    # Map reranked scores back to original indices
                    for rank, (rel_idx, score) in enumerate(zip(reranked_indices, reranked_scores)):
                        orig_idx = top_indices[rel_idx]
                        # Store reranked score (normalized to [0, 1] if needed)
                        # CrossEncoder scores can be unbounded, so we normalize
                        reranked_sim_matrix[kw_idx, orig_idx] = score
                        # Accumulate maximum reranked score for each chunk
                        reranked_chunk_scores[orig_idx] = max(reranked_chunk_scores[orig_idx], score)

                # Normalize reranked scores to [0, 1] range for consistency
                # Using min-max normalization per keyword
                for kw_idx in range(len(KEYWORDS)):
                    kw_scores = reranked_sim_matrix[kw_idx]
                    non_zero = kw_scores[kw_scores != 0]
                    if len(non_zero) > 0:
                        min_score = non_zero.min()
                        max_score = non_zero.max()
                        if max_score > min_score:
                            # Normalize only non-zero scores
                            mask = kw_scores != 0
                            kw_scores[mask] = (kw_scores[mask] - min_score) / (max_score - min_score)
                        else:
                            # All scores are the same
                            kw_scores[kw_scores != 0] = 1.0
                        reranked_sim_matrix[kw_idx] = kw_scores

                # Use reranked similarity matrix for scoring
                sim_matrix = reranked_sim_matrix
                chunk_scores = reranked_chunk_scores
            else:
                # No reranking - use original cosine similarity
                chunk_scores = np.max(sim_matrix, axis=0)

            # ============ SCORING ============
            # Global Mean Metric (across all keywords)
            keyword_means = np.mean(sim_matrix, axis=1)
            global_score = np.mean(keyword_means)

            # Save raw embeddings with reranked scores
            if SAVE_RAW_EMBEDDINGS:
                np.savez_compressed(
                    raw_embed_dir / f"{country}.npz",
                    embeddings=chunk_embeds,
                    text_chunks=chunks_subset,
                    chunk_scores=chunk_scores,  # Reranked scores if enabled
                    similarity_matrix=sim_matrix  # Full reranked similarity matrix
                )

            results[country] = {
                "weighted_score": float(global_score),
                "keyword_scores": {k: float(s) for k, s in zip(KEYWORDS, keyword_means)},
                "chunk_count": len(chunks_subset)
            }

        except Exception as e:
            logger.error(f"Error processing {country} with {config['short_name']}: {e}")
            logger.debug(traceback.format_exc())  # Full stack trace for debugging
            clean_memory()  # Panic clear if an error occurs inside loop

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

    # Load reranker once (shared across all embedding models)
    reranker = None
    if RERANKER_CONFIG.get("enabled", True):
        print("\n" + "="*60)
        print("LOADING RERANKER")
        print("="*60)
        reranker = load_reranker(RERANKER_CONFIG, device)
        if torch.cuda.is_available():
            print(f"VRAM Free after loading reranker: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")

    for config in EMBEDDING_MODELS:
        cache_path = CACHE_DIR / f"country_scores_{config['short_name']}.json"

        if cache_path.exists() and not TEST_MODE:
             print(f"\n  ℹ Note: {cache_path} exists. Skipping...")
             continue

        model = load_model(config, device)
        if model:
            # Pass reranker to process_model
            scores = process_model(model, config, chunks, reranker, RERANKER_CONFIG)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(scores, f, indent=2)

            # AGGRESSIVE CLEANUP AFTER EACH MODEL
            del model
            clean_memory()

            if torch.cuda.is_available():
                print(f"  VRAM Free: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")

    # CLEANUP RERANKER AFTER ALL MODELS
    if reranker is not None:
        print("\n" + "="*60)
        print("CLEANING UP RERANKER")
        print("="*60)
        del reranker
        clean_memory()
        if torch.cuda.is_available():
            print(f"VRAM Free after cleanup: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")

    print("\n" + "="*60)
    print("EMBEDDING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()