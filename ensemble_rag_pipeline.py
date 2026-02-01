#!/usr/bin/env python3
"""
===============================================================================
MULTI-MODEL ENSEMBLE RAG + RERANKING PIPELINE
===============================================================================
Analyzes National Water Laws from 160+ countries to detect 6 groundwater
governance dimensions using:

1. Ensemble Retrieval (Wide Net): BGE-M3 + Jina-v3 for maximum recall
2. Cross-Encoder Reranker (Judge): BGE-Reranker-v2-M3 for precision scoring
3. Semantic Description Strategy: Rich queries for legal concept matching

Hardware Target: NVIDIA RTX 5090 (24GB VRAM)
Optimizations: torch.float16, torch.compile, large batch sizes (256-512)

Author: Claude Code
Date: 2025
===============================================================================
"""

import os
import gc
import json
import logging
import re
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from tqdm import tqdm

# Lazy imports for optional dependencies
spacy = None
SentenceTransformer = None
CrossEncoder = None


# ===============================================================================
# LOGGING SETUP
# ===============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble_rag_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===============================================================================
# GPU MEMORY CONFIGURATION
# ===============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ===============================================================================
# PATH CONFIGURATION
# ===============================================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_FOLDER = SCRIPT_DIR / "extracted_texts_new"
OUTPUT_DIR = SCRIPT_DIR / "ensemble_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ===============================================================================
# CONTROL SWITCHES
# ===============================================================================
TEST_MODE = True  # Set to False for full processing
TEST_FILES = [
    "AD.txt",                          # Andorra
    "RU_water code_rus_ENGLISH.txt",   # Russia Water Code
    "CR.txt"                           # Costa Rica
]


# ===============================================================================
# SEMANTIC DESCRIPTION SEARCH QUERIES (The "Semantic Description" Strategy)
# ===============================================================================
SEARCH_QUERIES = {
    "Scope_and_Inclusion": (
        "Explicit inclusion of groundwater, underground water, or aquifers within the scope of the water law; "
        "provisions subjecting groundwater to regulatory requirements."
    ),
    "Access_and_Abstraction": (
        "Permit or licence requirements for abstraction; volumetric limits or caps; priority rules; "
        "authorization, registration, or technical standards for wells; well closure or abandonment obligations."
    ),
    "Monitoring_and_Reporting": (
        "Statutory obligations to monitor or report groundwater levels or quality; metering requirements; "
        "establishment of groundwater registers or databases."
    ),
    "Protection_and_Pollution": (
        "Groundwater or aquifer protection zones established in water law; restrictions on contaminating activities; "
        "liability or remediation obligations embedded in water legislation."
    ),
    "Authority_and_Enforcement": (
        "Designation of a competent authority for groundwater; statutory coordination duties with other agencies; "
        "inspection powers; administrative or criminal sanctions for non-compliance."
    ),
    "Conj_Management_and_Transboundary": (
        "Aquifer or basin-based regulatory units; legal linkage between surface water and groundwater regulation; "
        "managed aquifer recharge provisions; transboundary aquifer cooperation mechanisms."
    )
}


# ===============================================================================
# ENSEMBLE RETRIEVAL CONFIGURATION (The "Wide Net")
# ===============================================================================
@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""
    name: str
    short_name: str
    batch_size: int = 512  # RTX 5090 optimized
    trust_remote: bool = True
    fp16: bool = True
    is_jina: bool = False
    top_k: int = 50  # Top-K candidates per query


EMBEDDING_MODELS = [
    EmbeddingModelConfig(
        name="BAAI/bge-m3",
        short_name="BGE-M3",
        batch_size=512,
        top_k=50
    ),
    EmbeddingModelConfig(
        name="jinaai/jina-embeddings-v3",
        short_name="Jina-v3",
        batch_size=512,
        is_jina=True,
        top_k=50
    )
]


# ===============================================================================
# RERANKER CONFIGURATION (The "Judge")
# ===============================================================================
@dataclass
class RerankerConfig:
    """Configuration for the cross-encoder reranker."""
    name: str = "BAAI/bge-reranker-v2-m3"
    batch_size: int = 512  # RTX 5090 optimized
    top_n_final: int = 10  # Final number of chunks per dimension
    fp16: bool = True
    trust_remote: bool = True


RERANKER_CONFIG = RerankerConfig()


# ===============================================================================
# SPACY CHUNKING CONFIGURATION
# ===============================================================================
@dataclass
class ChunkingConfig:
    """Configuration for SpaCy-based text chunking."""
    spacy_model: str = "en_core_web_sm"
    max_sentences_per_chunk: int = 5  # ~400-500 words per chunk
    min_chunk_words: int = 20
    max_chunk_words: int = 600
    overlap_sentences: int = 1  # Overlap for context preservation


CHUNKING_CONFIG = ChunkingConfig()


# ===============================================================================
# MEMORY MANAGEMENT UTILITIES
# ===============================================================================
def clean_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def get_vram_info() -> str:
    """Get current VRAM usage information."""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return f"VRAM: {used/1024**3:.2f}GB used / {total/1024**3:.2f}GB total ({free/1024**3:.2f}GB free)"
    return "CUDA not available"


def log_memory_state(prefix: str = ""):
    """Log current memory state."""
    if prefix:
        logger.info(f"{prefix} - {get_vram_info()}")
    else:
        logger.info(get_vram_info())


# ===============================================================================
# MODEL MANAGER CONTEXT MANAGER
# ===============================================================================
@contextmanager
def model_manager(model_name: str, model_loader, *args, **kwargs):
    """
    Context manager for safe model loading and unloading.

    Ensures proper GPU memory cleanup after model use to prevent OOM errors
    when switching between large models.

    Args:
        model_name: Human-readable name for logging
        model_loader: Callable that returns the loaded model
        *args, **kwargs: Arguments passed to model_loader

    Yields:
        The loaded model

    Example:
        with model_manager("BGE-M3", load_embedding_model, config, device) as model:
            embeddings = model.encode(texts)
    """
    model = None
    try:
        logger.info(f"Loading model: {model_name}")
        log_memory_state("Before loading")

        model = model_loader(*args, **kwargs)

        log_memory_state("After loading")
        yield model

    except Exception as e:
        logger.error(f"Error with model {model_name}: {e}")
        logger.debug(traceback.format_exc())
        raise

    finally:
        # Aggressive cleanup
        if model is not None:
            logger.info(f"Unloading model: {model_name}")
            del model

        clean_memory()
        log_memory_state("After cleanup")


# ===============================================================================
# SPACY CHUNKING
# ===============================================================================
def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load SpaCy model with automatic download if needed."""
    global spacy
    if spacy is None:
        import spacy as sp
        spacy = sp

    try:
        return spacy.load(model_name)
    except OSError:
        logger.info(f"SpaCy model '{model_name}' not found. Downloading...")
        from spacy.cli import download
        download(model_name)
        return spacy.load(model_name)


def clean_text(text: str) -> str:
    """
    Clean and normalize text for chunking.

    Args:
        text: Raw input text

    Returns:
        Cleaned text with normalized whitespace and removed artifacts
    """
    # Remove page markers
    text = re.sub(r'---+\s*PAGE\s*\d+\s*---+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'---+', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    return text.strip()


def is_noise_chunk(text: str) -> bool:
    """
    Detect if a chunk is likely noise/boilerplate.

    Args:
        text: Chunk text

    Returns:
        True if chunk appears to be noise
    """
    text_lower = text.lower()

    # Too short to be meaningful legal content
    word_count = len(text.split())
    if word_count < CHUNKING_CONFIG.min_chunk_words:
        return True

    # Table of contents patterns
    if re.match(r'^(table\s+of\s+contents|contents|index)\s*$', text_lower):
        return True

    # Pure numbering or reference patterns
    if re.match(r'^[\d\.\s]+$', text):
        return True

    # Page number patterns
    if re.match(r'^page\s+\d+', text_lower):
        return True

    # Check for high ratio of numbers/symbols to text
    alpha_chars = sum(1 for c in text if c.isalpha())
    if len(text) > 10 and alpha_chars / len(text) < 0.5:
        return True

    return False


def chunk_document_spacy(text: str, nlp) -> List[str]:
    """
    Chunk document using SpaCy sentence segmentation.

    Uses sentence-based chunking with overlap for context preservation.

    Args:
        text: Document text
        nlp: Loaded SpaCy model

    Returns:
        List of text chunks
    """
    # Clean the text
    text = clean_text(text)

    if len(text) < 100:
        return []

    # Parse with SpaCy for sentence segmentation
    # Use a large enough character limit for legal documents
    doc = nlp(text[:1_000_000])  # SpaCy limit

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    if not sentences:
        return []

    chunks = []
    config = CHUNKING_CONFIG

    i = 0
    while i < len(sentences):
        # Collect sentences for this chunk
        chunk_sentences = []
        chunk_word_count = 0

        for j in range(i, min(i + config.max_sentences_per_chunk, len(sentences))):
            sent = sentences[j]
            sent_words = len(sent.split())

            # Check if adding this sentence exceeds max words
            if chunk_word_count + sent_words > config.max_chunk_words and chunk_sentences:
                break

            chunk_sentences.append(sent)
            chunk_word_count += sent_words

        if chunk_sentences:
            chunk_text = ' '.join(chunk_sentences)

            # Filter noise chunks
            if not is_noise_chunk(chunk_text):
                chunks.append(chunk_text)

        # Move forward with overlap
        step = len(chunk_sentences) - config.overlap_sentences
        if step <= 0:
            step = 1
        i += step

        # Prevent infinite loop
        if i <= 0:
            i = 1

    return chunks


def load_and_chunk_documents(folder: Path, test_mode: bool = False) -> Dict[str, List[str]]:
    """
    Load documents and chunk them using SpaCy.

    Args:
        folder: Path to folder containing .txt files
        test_mode: If True, only process TEST_FILES

    Returns:
        Dictionary mapping country_name -> list of chunks
    """
    print("\n" + "=" * 70)
    print(" STEP 1: LOADING & CHUNKING DOCUMENTS (SpaCy) ".center(70))
    print("=" * 70)

    if not folder.exists():
        logger.error(f"Data folder not found: {folder}")
        return {}

    # Load SpaCy model
    logger.info("Loading SpaCy model for sentence segmentation...")
    nlp = load_spacy_model(CHUNKING_CONFIG.spacy_model)

    # Disable unnecessary pipeline components for speed
    nlp.disable_pipes(["ner", "lemmatizer", "attribute_ruler"])

    # Get files to process
    if test_mode:
        files = []
        for test_file in TEST_FILES:
            # Try exact match first
            exact_path = folder / test_file
            if exact_path.exists():
                files.append(exact_path)
                logger.info(f"  Found test file: {test_file}")
            else:
                logger.warning(f"Test file not found: {test_file}")
        logger.info(f"TEST MODE: Processing {len(files)} files: {[f.name for f in files]}")
    else:
        files = list(folder.glob("*.txt"))
        logger.info(f"FULL MODE: Processing {len(files)} files")

    # Process files
    doc_chunks = {}

    for f in tqdm(files, desc="Chunking documents"):
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')

            # Skip very short files
            if len(text) < 200:
                logger.warning(f"Skipping short file: {f.name} ({len(text)} chars)")
                continue

            chunks = chunk_document_spacy(text, nlp)

            if chunks:
                # Use filename stem as country name
                country_name = f.stem
                doc_chunks[country_name] = chunks
                logger.debug(f"  {country_name}: {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error processing {f.name}: {e}")

    print(f"\n  Loaded {len(doc_chunks)} documents")
    print(f"  Total chunks: {sum(len(c) for c in doc_chunks.values())}")

    return doc_chunks


# ===============================================================================
# EMBEDDING MODEL LOADING
# ===============================================================================
def load_embedding_model(config: EmbeddingModelConfig, device: str):
    """
    Load embedding model with optimizations.

    Args:
        config: Model configuration
        device: Target device ('cuda' or 'cpu')

    Returns:
        Loaded SentenceTransformer model
    """
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer as ST
        SentenceTransformer = ST

    clean_memory()

    kwargs = {"trust_remote_code": config.trust_remote}

    # Configure model-specific settings
    if config.is_jina:
        # Jina V3: CRITICAL - disable flash attention to prevent assertion errors
        kwargs["model_kwargs"] = {
            "torch_dtype": torch.float16 if config.fp16 else torch.float32,
            "use_flash_attn": False
        }
    elif config.fp16 and device == "cuda":
        kwargs["model_kwargs"] = {
            "torch_dtype": torch.float16
        }

    # Load model
    model = SentenceTransformer(config.name, device=device, **kwargs)

    # Apply torch.compile for 30-50% speedup on RTX 5090
    if torch.__version__ >= "2.0.0" and device == "cuda":
        try:
            if hasattr(model, '_first_module') and hasattr(model._first_module(), 'auto_model'):
                model._first_module().auto_model = torch.compile(
                    model._first_module().auto_model,
                    mode="reduce-overhead",
                    fullgraph=True
                )
                logger.info(f"torch.compile() enabled for {config.short_name}")
        except Exception as e:
            logger.warning(f"torch.compile() failed for {config.short_name}: {e}")

    return model


# ===============================================================================
# RERANKER MODEL LOADING
# ===============================================================================
def load_reranker_model(config: RerankerConfig, device: str):
    """
    Load cross-encoder reranker model.

    Args:
        config: Reranker configuration
        device: Target device

    Returns:
        Loaded CrossEncoder model
    """
    global CrossEncoder
    if CrossEncoder is None:
        from sentence_transformers import CrossEncoder as CE
        CrossEncoder = CE

    clean_memory()

    kwargs = {}
    if config.trust_remote:
        kwargs["trust_remote_code"] = True

    model = CrossEncoder(config.name, device=device, **kwargs)

    # Enable FP16 for faster inference
    if config.fp16 and device == "cuda":
        try:
            model.model.half()
            logger.info("Reranker loaded in FP16 mode")
        except Exception as e:
            logger.warning(f"FP16 conversion failed: {e}")

    return model


# ===============================================================================
# COSINE SIMILARITY (GPU-ACCELERATED)
# ===============================================================================
def cosine_similarity_gpu(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    GPU-accelerated cosine similarity calculation.

    Args:
        matrix1: Query embeddings (n, d)
        matrix2: Document embeddings (m, d)

    Returns:
        Similarity matrix (n, m)
    """
    if not torch.cuda.is_available():
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(matrix1, matrix2)

    try:
        tensor1 = torch.tensor(matrix1, device='cuda', dtype=torch.float16)
        tensor2 = torch.tensor(matrix2, device='cuda', dtype=torch.float16)

        similarity = F.cosine_similarity(
            tensor1.unsqueeze(1),
            tensor2.unsqueeze(0),
            dim=2
        )

        return similarity.cpu().numpy()
    except Exception as e:
        logger.warning(f"GPU cosine similarity failed: {e}, using CPU")
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(matrix1, matrix2)


# ===============================================================================
# ENSEMBLE RETRIEVAL
# ===============================================================================
@dataclass
class RetrievedChunk:
    """A retrieved chunk with metadata."""
    text: str
    chunk_idx: int
    score: float
    source_model: str


def retrieve_all_queries_with_model(
    model,
    config: EmbeddingModelConfig,
    queries: Dict[str, str],
    chunks: List[str],
    top_k: int = 50
) -> Dict[str, List[RetrievedChunk]]:
    """
    Retrieve top-K chunks for ALL queries using a single model load.

    OPTIMIZATION: Encodes chunks ONCE, then searches with all queries.
    This avoids redundant chunk encoding per query.

    Args:
        model: Loaded SentenceTransformer model
        config: Model configuration
        queries: Dictionary of dimension_name -> query_text
        chunks: List of document chunks
        top_k: Number of top candidates to retrieve per query

    Returns:
        Dictionary of dimension_name -> List[RetrievedChunk]
    """
    if not chunks:
        return {dim: [] for dim in queries.keys()}

    # Encode chunks ONCE (this is the expensive operation)
    chunk_args = {
        "batch_size": config.batch_size,
        "normalize_embeddings": True,
        "show_progress_bar": False
    }
    if config.is_jina:
        chunk_args["task"] = "retrieval.passage"

    chunk_embeds = model.encode(chunks, **chunk_args)

    # Encode ALL queries at once
    query_args = {"normalize_embeddings": True}
    if config.is_jina:
        query_args["task"] = "retrieval.query"

    query_list = list(queries.values())
    dim_names = list(queries.keys())
    query_embeds = model.encode(query_list, **query_args)

    # Compute similarities for all queries at once (batch operation)
    # Shape: (num_queries, num_chunks)
    all_similarities = cosine_similarity_gpu(query_embeds, chunk_embeds)

    # Extract top-K for each query
    results: Dict[str, List[RetrievedChunk]] = {}

    for query_idx, dim_name in enumerate(dim_names):
        similarities = all_similarities[query_idx]
        top_k_actual = min(top_k, len(chunks))
        top_indices = np.argsort(similarities)[::-1][:top_k_actual]

        dim_results = []
        for idx in top_indices:
            dim_results.append(RetrievedChunk(
                text=chunks[idx],
                chunk_idx=int(idx),
                score=float(similarities[idx]),
                source_model=config.short_name
            ))
        results[dim_name] = dim_results

    return results


def process_all_countries_with_model(
    model,
    config: EmbeddingModelConfig,
    doc_chunks: Dict[str, List[str]],
    queries: Dict[str, str],
    top_k: int = 50
) -> Dict[str, Dict[str, List[RetrievedChunk]]]:
    """
    Process ALL countries with a single model load.

    OPTIMIZATION: Load model once, process all countries, then unload.

    Args:
        model: Loaded SentenceTransformer model
        config: Model configuration
        doc_chunks: Dictionary of country_name -> List[chunks]
        queries: Dictionary of dimension_name -> query_text
        top_k: Number of top candidates per query

    Returns:
        Nested dict: country_name -> dimension_name -> List[RetrievedChunk]
    """
    all_results: Dict[str, Dict[str, List[RetrievedChunk]]] = {}

    for country_name, chunks in tqdm(
        doc_chunks.items(),
        desc=f"  {config.short_name}",
        leave=False
    ):
        country_results = retrieve_all_queries_with_model(
            model, config, queries, chunks, top_k
        )
        all_results[country_name] = country_results

    return all_results


def merge_retrieval_results(
    results_by_model: List[Dict[str, Dict[str, List[RetrievedChunk]]]]
) -> Dict[str, Dict[str, List[RetrievedChunk]]]:
    """
    Merge retrieval results from multiple models with deduplication.

    For duplicate chunks (same chunk_idx), keeps the one with highest score.

    Args:
        results_by_model: List of results from each model
            Each element: country_name -> dimension_name -> List[RetrievedChunk]

    Returns:
        Merged results: country_name -> dimension_name -> List[RetrievedChunk]
    """
    merged: Dict[str, Dict[str, List[RetrievedChunk]]] = {}

    # Get all country names
    all_countries: Set[str] = set()
    for model_results in results_by_model:
        all_countries.update(model_results.keys())

    for country in all_countries:
        merged[country] = {}

        # Get all dimensions
        all_dims: Set[str] = set()
        for model_results in results_by_model:
            if country in model_results:
                all_dims.update(model_results[country].keys())

        for dim in all_dims:
            # Merge chunks from all models, deduplicate by chunk_idx
            chunk_map: Dict[int, RetrievedChunk] = {}

            for model_results in results_by_model:
                if country in model_results and dim in model_results[country]:
                    for chunk in model_results[country][dim]:
                        existing = chunk_map.get(chunk.chunk_idx)
                        if existing is None or chunk.score > existing.score:
                            chunk_map[chunk.chunk_idx] = chunk

            # Sort by score descending
            merged_chunks = sorted(
                chunk_map.values(),
                key=lambda x: x.score,
                reverse=True
            )
            merged[country][dim] = merged_chunks

    return merged


# ===============================================================================
# RERANKING
# ===============================================================================
@dataclass
class RankedChunk:
    """A reranked chunk with final score."""
    text: str
    chunk_idx: int
    reranker_score: float  # Sigmoid of logits (probability)
    retrieval_score: float  # Original retrieval score
    source_models: List[str]


def rerank_chunks(
    reranker,
    query: str,
    candidates: List[RetrievedChunk],
    config: RerankerConfig
) -> List[RankedChunk]:
    """
    Rerank retrieved chunks using cross-encoder.

    Args:
        reranker: Loaded CrossEncoder model
        query: Search query
        candidates: Retrieved chunk candidates
        config: Reranker configuration

    Returns:
        List of RankedChunk objects with final scores
    """
    if not candidates:
        return []

    # Prepare query-chunk pairs
    pairs = [[query, chunk.text] for chunk in candidates]

    # Get reranker scores
    try:
        logits = reranker.predict(
            pairs,
            batch_size=config.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Fallback: return top-N by retrieval score
        return [
            RankedChunk(
                text=c.text,
                chunk_idx=c.chunk_idx,
                reranker_score=c.score,
                retrieval_score=c.score,
                source_models=[c.source_model]
            )
            for c in candidates[:config.top_n_final]
        ]

    # Convert logits to probabilities using sigmoid
    probabilities = expit(logits)

    # Create ranked chunks
    ranked = []
    for i, (chunk, prob) in enumerate(zip(candidates, probabilities)):
        ranked.append(RankedChunk(
            text=chunk.text,
            chunk_idx=chunk.chunk_idx,
            reranker_score=float(prob),
            retrieval_score=chunk.score,
            source_models=[chunk.source_model]
        ))

    # Sort by reranker score
    ranked.sort(key=lambda x: x.reranker_score, reverse=True)

    # Return top-N
    return ranked[:config.top_n_final]


# ===============================================================================
# MAIN PIPELINE
# ===============================================================================
@dataclass
class DimensionResult:
    """Result for a single regulatory dimension."""
    dimension: str
    query: str
    top_chunks: List[Dict[str, Any]]
    max_score: float
    mean_score: float


@dataclass
class CountryResult:
    """Complete results for a country."""
    country: str
    dimensions: Dict[str, DimensionResult]
    overall_score: float  # Mean of all dimension max scores


def rerank_all_countries(
    reranker,
    merged_results: Dict[str, Dict[str, List[RetrievedChunk]]],
    queries: Dict[str, str],
    config: RerankerConfig
) -> Dict[str, Dict[str, List[RankedChunk]]]:
    """
    Rerank ALL countries and dimensions with a single reranker load.

    OPTIMIZATION: Load reranker once, process everything, then unload.

    Args:
        reranker: Loaded CrossEncoder model
        merged_results: country_name -> dimension_name -> List[RetrievedChunk]
        queries: dimension_name -> query_text
        config: Reranker configuration

    Returns:
        country_name -> dimension_name -> List[RankedChunk]
    """
    all_ranked: Dict[str, Dict[str, List[RankedChunk]]] = {}

    for country_name, dim_candidates in tqdm(
        merged_results.items(),
        desc="  Reranking",
        leave=False
    ):
        all_ranked[country_name] = {}

        for dim_name, candidates in dim_candidates.items():
            query = queries[dim_name]
            ranked = rerank_chunks(reranker, query, candidates, config)
            all_ranked[country_name][dim_name] = ranked

    return all_ranked


def build_country_results(
    ranked_results: Dict[str, Dict[str, List[RankedChunk]]],
    queries: Dict[str, str]
) -> Dict[str, CountryResult]:
    """
    Build final CountryResult objects from ranked results.

    Args:
        ranked_results: country_name -> dimension_name -> List[RankedChunk]
        queries: dimension_name -> query_text

    Returns:
        Dictionary of country_name -> CountryResult
    """
    results: Dict[str, CountryResult] = {}

    for country_name, dim_ranked in ranked_results.items():
        dimension_results: Dict[str, DimensionResult] = {}

        for dim_name, ranked_chunks in dim_ranked.items():
            query = queries[dim_name]

            # Convert to serializable format
            top_chunks = []
            for rank, chunk in enumerate(ranked_chunks):
                top_chunks.append({
                    "rank": rank + 1,
                    "text": chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                    "full_text": chunk.text,
                    "reranker_score": round(chunk.reranker_score, 4),
                    "retrieval_score": round(chunk.retrieval_score, 4),
                    "chunk_idx": chunk.chunk_idx
                })

            scores = [c.reranker_score for c in ranked_chunks]

            dimension_results[dim_name] = DimensionResult(
                dimension=dim_name,
                query=query,
                top_chunks=top_chunks,
                max_score=max(scores) if scores else 0.0,
                mean_score=sum(scores) / len(scores) if scores else 0.0
            )

        # Ensure all dimensions are present (even if empty)
        for dim_name, query in queries.items():
            if dim_name not in dimension_results:
                dimension_results[dim_name] = DimensionResult(
                    dimension=dim_name,
                    query=query,
                    top_chunks=[],
                    max_score=0.0,
                    mean_score=0.0
                )

        # Calculate overall score
        dim_max_scores = [r.max_score for r in dimension_results.values()]
        overall_score = sum(dim_max_scores) / len(dim_max_scores) if dim_max_scores else 0.0

        results[country_name] = CountryResult(
            country=country_name,
            dimensions=dimension_results,
            overall_score=overall_score
        )

    return results


def serialize_results(results: Dict[str, CountryResult]) -> Dict[str, Any]:
    """
    Serialize results to JSON-compatible format.

    Structure: Country -> Dimension -> Score/Evidence

    Args:
        results: Dictionary of country results

    Returns:
        JSON-serializable dictionary
    """
    output = {
        "metadata": {
            "pipeline": "Multi-Model Ensemble RAG + Reranking",
            "models": {
                "ensemble_retrievers": [c.name for c in EMBEDDING_MODELS],
                "reranker": RERANKER_CONFIG.name
            },
            "dimensions": list(SEARCH_QUERIES.keys()),
            "configuration": {
                "top_k_per_model": EMBEDDING_MODELS[0].top_k,
                "top_n_final": RERANKER_CONFIG.top_n_final,
                "chunking": {
                    "method": "spacy_sentence",
                    "max_sentences": CHUNKING_CONFIG.max_sentences_per_chunk,
                    "overlap_sentences": CHUNKING_CONFIG.overlap_sentences
                }
            }
        },
        "countries": {}
    }

    for country_name, result in results.items():
        country_data = {
            "overall_score": round(result.overall_score, 4),
            "dimensions": {}
        }

        for dim_name, dim_result in result.dimensions.items():
            country_data["dimensions"][dim_name] = {
                "query": dim_result.query,
                "max_score": round(dim_result.max_score, 4),
                "mean_score": round(dim_result.mean_score, 4),
                "evidence": dim_result.top_chunks
            }

        output["countries"][country_name] = country_data

    return output


def main():
    """
    Main pipeline execution with OPTIMIZED model loading.

    OPTIMIZATION: Each model is loaded ONCE, processes ALL countries, then unloaded.
    This reduces model load/unload cycles from O(countries × dimensions × models)
    to O(models), saving hours of overhead.

    Pipeline:
    1. Load & chunk all documents
    2. For each embedding model:
       - Load model ONCE
       - Process ALL countries with ALL queries
       - Unload model
    3. Merge results from all models
    4. Load reranker ONCE
    5. Rerank ALL countries
    6. Unload reranker
    7. Build final results & save
    """
    print("\n" + "=" * 70)
    print(" MULTI-MODEL ENSEMBLE RAG + RERANKING PIPELINE ".center(70))
    print("=" * 70)
    print(f"\nEnsemble Models: {[c.short_name for c in EMBEDDING_MODELS]}")
    print(f"Reranker: {RERANKER_CONFIG.name}")
    print(f"Dimensions: {len(SEARCH_QUERIES)}")
    print(f"Test Mode: {TEST_MODE}")
    print("\n[OPTIMIZED] Models loaded once per stage, not per query")

    # Initialize
    clean_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    log_memory_state("Initial")

    # =========================================================================
    # STEP 1: Load and chunk documents
    # =========================================================================
    doc_chunks = load_and_chunk_documents(DATA_FOLDER, test_mode=TEST_MODE)

    if not doc_chunks:
        logger.error("No documents loaded. Exiting.")
        return

    # =========================================================================
    # STEP 2: ENSEMBLE RETRIEVAL (Load each model ONCE, process ALL countries)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 2: ENSEMBLE RETRIEVAL ".center(70))
    print("=" * 70)
    print(f"  Processing {len(doc_chunks)} countries × {len(SEARCH_QUERIES)} dimensions")

    results_by_model: List[Dict[str, Dict[str, List[RetrievedChunk]]]] = []

    for config in EMBEDDING_MODELS:
        print(f"\n  Loading {config.short_name}...")

        with model_manager(config.short_name, load_embedding_model, config, device) as model:
            # Process ALL countries with this model
            model_results = process_all_countries_with_model(
                model, config, doc_chunks, SEARCH_QUERIES, config.top_k
            )
            results_by_model.append(model_results)

            logger.info(f"  {config.short_name}: Processed {len(model_results)} countries")

    # =========================================================================
    # STEP 3: MERGE RESULTS FROM ALL MODELS
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 3: MERGING ENSEMBLE RESULTS ".center(70))
    print("=" * 70)

    merged_results = merge_retrieval_results(results_by_model)
    logger.info(f"  Merged results for {len(merged_results)} countries")

    # Free memory from individual model results
    del results_by_model
    clean_memory()

    # =========================================================================
    # STEP 4: RERANKING (Load reranker ONCE, process ALL countries)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 4: RERANKING ".center(70))
    print("=" * 70)

    with model_manager("BGE-Reranker-v2-M3", load_reranker_model, RERANKER_CONFIG, device) as reranker:
        ranked_results = rerank_all_countries(
            reranker, merged_results, SEARCH_QUERIES, RERANKER_CONFIG
        )

    # Free memory from merged results
    del merged_results
    clean_memory()

    # =========================================================================
    # STEP 5: BUILD FINAL RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 5: BUILDING FINAL RESULTS ".center(70))
    print("=" * 70)

    results = build_country_results(ranked_results, SEARCH_QUERIES)

    # Log summary for each country
    for country_name, result in results.items():
        logger.info(
            f"  {country_name}: Overall Score = {result.overall_score:.4f}, "
            f"Dimensions = {len(result.dimensions)}"
        )

    # =========================================================================
    # STEP 6: SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STEP 6: SAVING RESULTS ".center(70))
    print("=" * 70)

    output_data = serialize_results(results)

    # Generate output filename
    mode_suffix = "TEST" if TEST_MODE else "FULL"
    output_file = OUTPUT_DIR / f"ensemble_rag_results_{mode_suffix}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print(" SUMMARY ".center(70))
    print("=" * 70)

    # Sort countries by overall score
    sorted_countries = sorted(
        results.items(),
        key=lambda x: x[1].overall_score,
        reverse=True
    )

    print(f"\nCountries processed: {len(results)}")
    print(f"Model loads: {len(EMBEDDING_MODELS)} retrievers + 1 reranker = {len(EMBEDDING_MODELS) + 1} total")
    print("\nTop Countries by Overall Score:")
    for i, (country, result) in enumerate(sorted_countries[:10], 1):
        print(f"  {i:2}. {country}: {result.overall_score:.4f}")

    print("\nDimension Breakdown (Top Country):")
    if sorted_countries:
        top_country, top_result = sorted_countries[0]
        for dim_name, dim_result in top_result.dimensions.items():
            print(f"  {dim_name}: max={dim_result.max_score:.4f}, mean={dim_result.mean_score:.4f}")

    # Final cleanup
    clean_memory()
    log_memory_state("Final")

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE ".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
