import os
import json
import numpy as np
import torch
import gc
import logging
import traceback
import threading
import queue
import re
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import expit
import torch.nn.functional as F
from rank_bm25 import BM25Okapi

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

# ================= SLIDING WINDOW CONFIGURATION =================
WINDOW_SIZE_WORDS = 400      # ~512 tokens (approx. 400 words)
OVERLAP_WORDS = 100          # ~128 tokens (approx. 100 words)
MIN_CHUNK_WORDS = 10         # Minimum words for a valid chunk

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
        "batch_size": 64,  # RTX 5090 optimized (was 8)
        "max_chunks": None,
        "prompt": None,
        "trust_remote": True,
        "fp16": True
    },
    {
        "name": "BAAI/bge-m3",
        "short_name": "BGE-M3",
        "batch_size": 512,  # RTX 5090 optimized (was 128)
        "max_chunks": None,
        "prompt": None,
        "trust_remote": True,
        "fp16": True
    },
    {
        "name": "jinaai/jina-embeddings-v3",
        "short_name": "Jina-v3",
        "batch_size": 512,  # RTX 5090 optimized (was 128)
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
    "batch_size": 512,            # RTX 5090 optimized (was 256)
    "top_k_initial": 100,         # Initial retrieval depth (increases recall)
    "top_n_final": 20,            # Final number of chunks after reranking
    "trust_remote": True
}

# ================= HYBRID SEARCH CONFIGURATION =================
HYBRID_SEARCH_CONFIG = {
    "enabled": True,
    "vector_weight": 0.7,         # Weight for cosine similarity score
    "bm25_weight": 0.3,           # Weight for BM25 score
    "bm25_normalization": "minmax"  # "minmax" or "sigmoid"
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
        print(f"    Warning: GPU cosine similarity failed: {e}, using CPU")
        return cosine_similarity(matrix1, matrix2)

# ================= SLIDING WINDOW CHUNKING =================
def sliding_window_chunk(text, window_size=WINDOW_SIZE_WORDS, overlap=OVERLAP_WORDS):
    """
    Split text into overlapping chunks using a sliding window approach.

    Args:
        text: Input text string
        window_size: Number of words per chunk (~400 words = ~512 tokens)
        overlap: Number of overlapping words between chunks (~100 words = ~128 tokens)

    Returns:
        List of text chunks
    """
    # Clean text: normalize whitespace and remove page markers
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'---+', ' ', text)  # Remove horizontal rules

    # Split into words while preserving boundaries
    words = text.split()

    if len(words) < MIN_CHUNK_WORDS:
        return []

    chunks = []
    step = window_size - overlap

    if step <= 0:
        step = window_size // 2  # Fallback to 50% overlap

    i = 0
    while i < len(words):
        # Get window of words
        chunk_words = words[i:i + window_size]

        # Skip if chunk is too small (except for last chunk)
        if len(chunk_words) >= MIN_CHUNK_WORDS:
            chunk_text = ' '.join(chunk_words)

            # Filter out page number patterns at the start
            if not re.match(r'^Page\s+\d+', chunk_text[:15]):
                chunks.append(chunk_text)

        # Move to next window
        i += step

        # If we're at the end and have a small remaining piece, we've already
        # captured it in the last full window due to overlap
        if i >= len(words):
            break

    return chunks


def tokenize_for_bm25(text):
    """
    Simple tokenizer for BM25 - lowercase and split on non-alphanumeric.

    Args:
        text: Input text string

    Returns:
        List of tokens
    """
    return re.findall(r'\b\w+\b', text.lower())


def compute_bm25_scores(chunks, keywords, normalization="minmax"):
    """
    Compute BM25 scores for keywords against chunks.

    Args:
        chunks: List of text chunks
        keywords: List of keyword strings
        normalization: "minmax" or "sigmoid" for score normalization

    Returns:
        numpy array of shape (num_keywords, num_chunks) with normalized scores
    """
    if not chunks:
        return np.zeros((len(keywords), 0))

    # Tokenize chunks for BM25
    tokenized_chunks = [tokenize_for_bm25(chunk) for chunk in chunks]

    # Initialize BM25 index
    bm25 = BM25Okapi(tokenized_chunks)

    # Compute scores for each keyword
    scores_matrix = np.zeros((len(keywords), len(chunks)))

    for kw_idx, keyword in enumerate(keywords):
        tokenized_query = tokenize_for_bm25(keyword)
        raw_scores = bm25.get_scores(tokenized_query)
        scores_matrix[kw_idx] = raw_scores

    # Normalize scores to 0-1 range
    if normalization == "sigmoid":
        # Sigmoid normalization (handles any range, centers around 0.5)
        scores_matrix = expit(scores_matrix)
    else:
        # MinMax normalization per keyword (default)
        for kw_idx in range(len(keywords)):
            row = scores_matrix[kw_idx]
            min_val, max_val = row.min(), row.max()
            if max_val > min_val:
                scores_matrix[kw_idx] = (row - min_val) / (max_val - min_val)
            else:
                scores_matrix[kw_idx] = 0.0  # All same score -> set to 0

    return scores_matrix


def compute_hybrid_scores(vector_scores, bm25_scores, config=HYBRID_SEARCH_CONFIG):
    """
    Combine vector similarity and BM25 scores using weighted fusion.

    Args:
        vector_scores: numpy array of cosine similarity scores (keywords x chunks)
        bm25_scores: numpy array of normalized BM25 scores (keywords x chunks)
        config: Hybrid search configuration dict

    Returns:
        numpy array of hybrid scores (keywords x chunks)
    """
    if not config.get("enabled", True):
        return vector_scores

    vector_weight = config.get("vector_weight", 0.7)
    bm25_weight = config.get("bm25_weight", 0.3)

    # Weighted fusion: combined = (0.7 * vector) + (0.3 * bm25)
    hybrid_scores = (vector_weight * vector_scores) + (bm25_weight * bm25_scores)

    return hybrid_scores


# ================= PRODUCER-CONSUMER PATTERN =================
class DocumentProducer(threading.Thread):
    """
    Background thread that reads files, applies sliding window chunking,
    and pushes (country_name, chunks) tuples into a queue.
    """

    def __init__(self, folder, output_queue, test_mode=False):
        super().__init__(daemon=True)
        self.folder = Path(folder)
        self.output_queue = output_queue
        self.test_mode = test_mode
        self.total_files = 0
        self.processed_files = 0
        self.error_count = 0

    def run(self):
        """Producer thread main loop."""
        try:
            if not self.folder.exists():
                logger.error(f"Folder {self.folder} does not exist!")
                self.output_queue.put(None)  # Signal completion
                return

            files = list(self.folder.glob("*.txt"))

            if self.test_mode:
                logger.info(f"TEST MODE: Processing only first 5 documents")
                files = files[:5]

            self.total_files = len(files)
            logger.info(f"Producer: Found {self.total_files} text files to process")

            for f in files:
                try:
                    text = f.read_text(encoding='utf-8', errors='ignore')

                    # Skip very short files
                    if len(text) < 100:
                        continue

                    # Apply sliding window chunking
                    chunks = sliding_window_chunk(text)

                    if chunks:
                        # Push to queue for consumer
                        self.output_queue.put((f.stem, chunks))
                        self.processed_files += 1

                except Exception as e:
                    logger.error(f"Producer error reading {f.name}: {e}")
                    logger.debug(traceback.format_exc())
                    self.error_count += 1

            logger.info(f"Producer: Completed. {self.processed_files} documents chunked, {self.error_count} errors")

        finally:
            # Signal that producer is done
            self.output_queue.put(None)


def load_and_chunk_documents_threaded(folder, test_mode=TEST_MODE):
    """
    Load and chunk documents using producer-consumer pattern.

    Producer: Background thread reads files and chunks them
    Consumer: Main thread collects results into dictionary

    Args:
        folder: Path to folder containing .txt files
        test_mode: If True, only process first 5 files

    Returns:
        Dictionary mapping country_name -> list of chunks
    """
    print("\n" + "="*60)
    print(f"STEP 1: Loading Documents from {folder}")
    print("="*60)
    print("  Using Sliding Window chunking (400 words, 100 overlap)")
    print("  Producer-Consumer threading enabled")

    # Create queue for producer-consumer communication
    doc_queue = queue.Queue(maxsize=50)  # Buffer up to 50 documents

    # Start producer thread
    producer = DocumentProducer(folder, doc_queue, test_mode)
    producer.start()

    # Consumer: collect results from queue
    doc_chunks = {}
    pbar = None

    while True:
        item = doc_queue.get()

        if item is None:
            # Producer signaled completion
            break

        country_name, chunks = item
        doc_chunks[country_name] = chunks

        # Initialize progress bar after first item (we now know producer started)
        if pbar is None:
            pbar = tqdm(total=producer.total_files, desc="Chunking", unit="docs")

        pbar.update(1)

    # Wait for producer to fully complete
    producer.join(timeout=5.0)

    if pbar is not None:
        pbar.close()

    print(f"  Loaded {len(doc_chunks)} documents with chunks")

    return doc_chunks


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
            print(f"  Reranker loaded in FP16 mode")
        except Exception as e:
            print(f"  Warning: FP16 conversion failed: {e}, using FP32")

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
        print(f"    Warning: Reranking failed: {e}, falling back to cosine similarity")
        top_indices = np.argsort(scores)[::-1][:top_n]
        return top_indices, scores[top_indices]

    # Get top-n indices based on reranker scores
    top_indices = np.argsort(rerank_scores)[::-1][:top_n]

    return top_indices, rerank_scores[top_indices]

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
                    print(f"  torch.compile() enabled for {config['short_name']}")
            except Exception as e:
                print(f"  Warning: torch.compile() failed: {e}, continuing without compilation")

        return model
    except Exception as e:
        print(f"  Warning: Standard load failed: {e}")

        # FALLBACK: If Qwen fails with Flash Attn, try Native (might OOM, but better than crash)
        if "model_kwargs" in kwargs and "attn_implementation" in kwargs["model_kwargs"]:
            print("  Retrying without Flash Attention...")
            del kwargs["model_kwargs"]["attn_implementation"]
            return SentenceTransformer(config["name"], device=device, **kwargs)
        raise e

def process_model(model, config, chunks_map, reranker=None, reranker_config=None):
    """
    Process documents with embedding model, hybrid search, and optional reranking.

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

    # Log enabled features
    if HYBRID_SEARCH_CONFIG.get("enabled", True):
        print(f"  Hybrid search enabled: {HYBRID_SEARCH_CONFIG['vector_weight']:.0%} vector + {HYBRID_SEARCH_CONFIG['bm25_weight']:.0%} BM25")
    if reranker is not None:
        print(f"  Reranking enabled with top_k={reranker_config['top_k_initial']} -> top_n={reranker_config['top_n_final']}")

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
            vector_sim_matrix = cosine_similarity_gpu(kw_embeds, chunk_embeds)

            # ============ HYBRID SEARCH (BM25 + Vector) ============
            if HYBRID_SEARCH_CONFIG.get("enabled", True):
                # Compute BM25 scores for this document's chunks
                bm25_scores = compute_bm25_scores(
                    chunks_subset,
                    KEYWORDS,
                    normalization=HYBRID_SEARCH_CONFIG.get("bm25_normalization", "minmax")
                )

                # Combine vector and BM25 scores
                sim_matrix = compute_hybrid_scores(vector_sim_matrix, bm25_scores, HYBRID_SEARCH_CONFIG)
            else:
                sim_matrix = vector_sim_matrix

            # ============ RERANKING STAGE ============
            if reranker is not None and reranker_config.get("enabled", True):
                top_k = reranker_config["top_k_initial"]
                top_n = reranker_config["top_n_final"]
                batch_size = reranker_config.get("batch_size", 32)

                # Initialize arrays to store reranked scores
                reranked_sim_matrix = np.zeros_like(sim_matrix)
                reranked_chunk_scores = np.zeros(len(chunks_subset))

                # For each keyword, retrieve top_k chunks (using hybrid scores) and rerank
                for kw_idx, keyword in enumerate(KEYWORDS):
                    keyword_scores = sim_matrix[kw_idx]

                    # Get top_k candidates based on HYBRID score (BM25 + Vector)
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
                        # Store reranked score (CrossEncoder logits)
                        reranked_sim_matrix[kw_idx, orig_idx] = score
                        # Accumulate maximum reranked score for each chunk
                        reranked_chunk_scores[orig_idx] = max(reranked_chunk_scores[orig_idx], score)

                # CRITICAL: Convert CrossEncoder logits to probabilities using sigmoid
                # This preserves cross-country comparisons (unlike min-max normalization)
                reranked_sim_matrix = expit(reranked_sim_matrix)
                reranked_chunk_scores = expit(reranked_chunk_scores)

                # Use reranked similarity matrix for scoring
                sim_matrix = reranked_sim_matrix
                chunk_scores = reranked_chunk_scores
            else:
                # No reranking - use hybrid similarity scores
                chunk_scores = np.max(sim_matrix, axis=0)

            # ============ SCORING ============
            # Global Mean Metric (across all keywords)
            keyword_means = np.mean(sim_matrix, axis=1)
            global_score = np.mean(keyword_means)

            # Save raw embeddings with hybrid/reranked scores
            if SAVE_RAW_EMBEDDINGS:
                np.savez_compressed(
                    raw_embed_dir / f"{country}.npz",
                    embeddings=chunk_embeds,
                    text_chunks=chunks_subset,
                    chunk_scores=chunk_scores,  # Hybrid/reranked scores
                    similarity_matrix=sim_matrix  # Hybrid similarity matrix
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
    print("  Refactored: Sliding Window + Producer-Consumer + Hybrid BM25")

    clean_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"VRAM Free: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")

    # Use threaded producer-consumer for document loading
    chunks = load_and_chunk_documents_threaded(DATA_FOLDER)
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
             print(f"\n  Note: {cache_path} exists. Skipping...")
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
