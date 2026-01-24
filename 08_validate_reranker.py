import numpy as np
import pandas as pd
import re
from pathlib import Path

# ================= CONFIGURATION =================

CACHE_DIR = Path("cache/raw_embeddings")
OUTPUT_DIR = Path("results/validation_reranker")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. TARGET COUNTRIES (Subset for focused validation)
TARGET_COUNTRIES = ["Tajikistan", "Vietnam", "China", "Uzbekistan"]

# 2. ISO MAPPING (for robust file matching)
ISO_MAP = {
    "Tajikistan": "TJ",
    "Vietnam": "VN",
    "China": "CN",
    "Uzbekistan": "UZ"
}

# 3. KEYWORDS - MUST MATCH 01_embeddings_ALL.py for index alignment
# This is the CANONICAL list from the embedding script
KEYWORDS = [
    "groundwater withdrawal", "ground-water monitoring", "underground water abstraction",
    "groundwater permits", "groundwater rights", "well and borehole drilling licenses",
    "conjunctive management", "groundwater protection zones", "aquifer recharge",
    "transboundary aquifers"
]

# 4. HARD KEYWORDS (subset for validation)
HARD_KEYWORDS = ["groundwater permits", "transboundary aquifers", "aquifer recharge"]

# 5. MODELS TO VALIDATE
MODEL_NAMES = ["Qwen3-8B", "BGE-M3", "Jina-v3"]

# ================= UTILS =================

def normalize_name(name):
    """Normalize for matching: lower case, no spaces/underscores."""
    name = str(name).lower().replace("_", "").replace(" ", "")
    name = re.sub(r"\(.*?\)", "", name).strip()
    return name

def map_files_to_targets(model_dir, target_countries):
    """
    Robust file mapper for target countries.
    Returns: dict {country_name: Path}
    """
    mapping = {}
    all_files = list(model_dir.glob("*.npz"))
    file_lookup_norm = {normalize_name(f.stem): f for f in all_files}

    for target in target_countries:
        found = None
        target_norm = normalize_name(target)
        target_iso = ISO_MAP.get(target, "").upper()

        # STRATEGY 1: ISO PREFIX MATCH
        if not found and target_iso:
            for f in all_files:
                stem_upper = f.stem.upper()
                if stem_upper == target_iso or \
                   stem_upper.startswith(target_iso + "_") or \
                   stem_upper.startswith(target_iso + " "):
                    found = f
                    break

        # STRATEGY 2: Full Normalized Name
        if not found and target_norm in file_lookup_norm:
            found = file_lookup_norm[target_norm]

        # STRATEGY 3: Partial Name Match
        if not found:
            for stem_norm, fpath in file_lookup_norm.items():
                if target_norm in stem_norm:
                    found = fpath
                    break

        # STRATEGY 4: Special Case for Vietnam
        if not found and target == "Vietnam":
            for f in all_files:
                if "VIET" in f.stem.upper() and "NAM" in f.stem.upper():
                    found = f
                    break

        if found:
            mapping[target] = found

    return mapping

def get_keyword_index(keyword):
    """Get the index of a keyword in the KEYWORDS list."""
    try:
        return KEYWORDS.index(keyword)
    except ValueError:
        print(f"❌ ERROR: Keyword '{keyword}' not found in KEYWORDS list!")
        return None

def extract_top_chunk_from_reranker(npz_path, keyword):
    """
    Extract the top-1 chunk for a given keyword using the reranked similarity_matrix.

    Args:
        npz_path: Path to .npz file
        keyword: Keyword string

    Returns:
        dict: {"score": float, "text": str, "chunk_idx": int} or None if error
    """
    try:
        # Load the .npz file
        data = np.load(npz_path, allow_pickle=True)

        # Check if similarity_matrix exists (reranked scores)
        if 'similarity_matrix' not in data:
            print(f"    ⚠ No similarity_matrix in {npz_path.name}, skipping")
            return None

        similarity_matrix = data['similarity_matrix']
        text_chunks = data['text_chunks']

        # Get keyword index
        kw_idx = get_keyword_index(keyword)
        if kw_idx is None:
            return None

        # Ensure keyword index is valid
        if kw_idx >= similarity_matrix.shape[0]:
            print(f"    ⚠ Keyword index {kw_idx} out of bounds for {npz_path.name}")
            return None

        # Get scores for this keyword (row in similarity matrix)
        keyword_scores = similarity_matrix[kw_idx]

        # Find the chunk with the highest score
        if len(keyword_scores) == 0:
            return None

        max_idx = np.argmax(keyword_scores)
        max_score = keyword_scores[max_idx]

        # Extract the corresponding text chunk
        if max_idx >= len(text_chunks):
            print(f"    ⚠ Chunk index {max_idx} out of bounds for {npz_path.name}")
            return None

        top_chunk_text = str(text_chunks[max_idx]).strip()

        return {
            "score": float(max_score),
            "text": top_chunk_text,
            "chunk_idx": int(max_idx)
        }

    except Exception as e:
        print(f"    ❌ Error processing {npz_path.name}: {e}")
        return None

# ================= MAIN LOGIC =================

def main():
    print("\n" + "="*70)
    print(" RERANKER VALIDATION - TOP HITS EXTRACTION ".center(70))
    print("="*70)

    all_results = []

    # Process each model
    for model_name in MODEL_NAMES:
        model_dir = CACHE_DIR / model_name

        if not model_dir.exists():
            print(f"\n❌ Cache directory missing for {model_name}: {model_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing Model: {model_name}")
        print(f"{'='*60}")

        # Map target countries to files
        file_map = map_files_to_targets(model_dir, TARGET_COUNTRIES)

        if not file_map:
            print(f"  ⚠ No target countries found in {model_name} cache")
            continue

        print(f"  Found {len(file_map)}/{len(TARGET_COUNTRIES)} countries")

        # Process each country
        for country, npz_path in file_map.items():
            print(f"\n  Country: {country}")

            # Process each hard keyword
            for keyword in HARD_KEYWORDS:
                result = extract_top_chunk_from_reranker(npz_path, keyword)

                if result is None:
                    continue

                # Store the result
                all_results.append({
                    "Model": model_name,
                    "Country": country,
                    "Keyword": keyword,
                    "Reranked_Score": result["score"],
                    "Chunk_Index": result["chunk_idx"],
                    "Top_Chunk_Text": result["text"]
                })

                # Print summary
                text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
                print(f"    ✓ {keyword:25} | Score: {result['score']:.4f} | Idx: {result['chunk_idx']:4d}")
                print(f"      └─ {text_preview}")

    # ================= SAVE RESULTS =================

    if not all_results:
        print("\n❌ No results to save. Check if .npz files contain 'similarity_matrix'.")
        return

    df = pd.DataFrame(all_results)

    # Sort by model, country, keyword for readability
    df = df.sort_values(["Model", "Country", "Keyword"]).reset_index(drop=True)

    # Save to CSV
    csv_path = OUTPUT_DIR / "reranker_top_hits.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "="*70)
    print(f"✓ COMPLETE: Saved {len(df)} results to {csv_path}")
    print("="*70)

    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"  Total entries: {len(df)}")
    print(f"  Models: {df['Model'].nunique()}")
    print(f"  Countries: {df['Country'].nunique()}")
    print(f"  Keywords: {df['Keyword'].nunique()}")
    print(f"\n  Average Reranked Score: {df['Reranked_Score'].mean():.4f}")
    print(f"  Min Score: {df['Reranked_Score'].min():.4f}")
    print(f"  Max Score: {df['Reranked_Score'].max():.4f}")

    # Show per-keyword statistics
    print("\n📈 Per-Keyword Average Scores:")
    for kw in HARD_KEYWORDS:
        avg_score = df[df['Keyword'] == kw]['Reranked_Score'].mean()
        print(f"  {kw:30} -> {avg_score:.4f}")

if __name__ == "__main__":
    main()
