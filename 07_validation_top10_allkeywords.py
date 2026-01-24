import numpy as np
import pandas as pd
import torch
import gc
import os
import re
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ================= CONFIGURATION =================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CACHE_DIR = Path("cache/raw_embeddings")
OUTPUT_DIR = Path("results/validation_targets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. THE SPECIFIC COUNTRIES YOU WANT
TARGET_COUNTRIES = [
    "Tajikistan", "Turkmenistan", "Vietnam", "China", "Belarus", 
    "Georgia", "Iran", "Ukraine", "Lithuania", "Somalia"
]

# 2. ISO MAPPING
ISO_MAP = {
    "Tajikistan": "TJ",
    "Turkmenistan": "TM",
    "Vietnam": "VN",
    "China": "CN",
    "Belarus": "BY",
    "Georgia": "GE",
    "Iran": "IR",
    "Ukraine": "UA",
    "Lithuania": "LT",
    "Somalia": "SO"
}

# 3. KEYWORDS
KEYWORDS = [
    "groundwater withdrawal", "ground-water monitoring", "underground water abstraction", 
    "groundwater permits", "groundwater rights", "well and borehole drilling licenses", 
    "conjunctive management", "groundwater protection zones", "aquifer recharge", 
    "transboundary aquifers"
]

# 4. MODELS
MODEL_CONFIGS = [
    {"name": "Qwen/Qwen3-Embedding-8B", "short": "Qwen3-8B"},
    {"name": "jinaai/jina-embeddings-v3", "short": "Jina-v3", "trust": True}
]

# 5. DEPTH
SEARCH_DEPTH = 150 

# ================= UTILS =================

def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def normalize_name(name):
    """Normalize for matching: lower case, no spaces/underscores."""
    name = str(name).lower().replace("_", "").replace(" ", "")
    name = re.sub(r"\(.*?\)", "", name).strip()
    return name

def map_files_to_targets(model_dir):
    """
    Robust Mapper:
    1. Exact Name match
    2. ISO Prefix match (CRITICAL FIX: Matches 'TJ_Water Code...')
    3. Partial Name match
    """
    mapping = {} 
    
    # Get all .npz files
    all_files = list(model_dir.glob("*.npz"))
    
    # Pre-calculate lookups
    # 1. Full normalized stems
    file_lookup_norm = {normalize_name(f.stem): f for f in all_files}
    
    print(f"  Mapping files for {len(TARGET_COUNTRIES)} targets...")

    for target in TARGET_COUNTRIES:
        found = None
        target_norm = normalize_name(target)
        target_iso = ISO_MAP.get(target, "").upper() # e.g., "TJ"

        # STRATEGY 1: ISO PREFIX MATCH (The Fix)
        # Looks for "TJ_" or "TJ " at start of filename
        if not found and target_iso:
            for f in all_files:
                stem_upper = f.stem.upper()
                # Check strict equality OR prefix with separator
                if stem_upper == target_iso or \
                   stem_upper.startswith(target_iso + "_") or \
                   stem_upper.startswith(target_iso + " "):
                    found = f
                    break
        
        # STRATEGY 2: Full Normalized Name (e.g. "china.npz")
        if not found and target_norm in file_lookup_norm:
            found = file_lookup_norm[target_norm]

        # STRATEGY 3: Partial Name Match (e.g. "republic_of_tajikistan")
        if not found:
            for stem_norm, fpath in file_lookup_norm.items():
                if target_norm in stem_norm:
                    found = fpath
                    break
        
        # STRATEGY 4: Special Case for Vietnam (Vietnam vs Viet Nam)
        if not found and target == "Vietnam":
            for f in all_files:
                if "VIET" in f.stem.upper() and "NAM" in f.stem.upper():
                    found = f
                    break

        if found:
            mapping[target] = found
            # Print truncated filename for cleaner output
            fname = found.name if len(found.name) < 40 else found.name[:37] + "..."
            print(f"    ✓ {target.ljust(15)} -> {fname}")
        else:
            print(f"    ❌ {target.ljust(15)} -> NOT FOUND in {model_dir.name}")
            
    return mapping

def load_model(model_name):
    print(f"  Loading {model_name}...")
    clean_memory()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_kwargs = {
        "torch_dtype": torch.float16,
        "use_safetensors": True 
    }
    
    if "jina" in model_name.lower():
        model_kwargs["use_flash_attn"] = False
        
    try:
        return SentenceTransformer(
            model_name, 
            device=device, 
            trust_remote_code=True,
            model_kwargs=model_kwargs
        )
    except Exception as e:
        print(f"  ❌ Error loading {model_name}: {e}")
        return None

# ================= MAIN LOGIC =================

def main():
    print("\n=== TARGETED CONSENSUS EXTRACTION (ISO FIX) ===")
    
    # Structure: data_store[keyword][country][model_name] = {text: score}
    data_store = {kw: {c: {} for c in TARGET_COUNTRIES} for kw in KEYWORDS}

    # --- STEP 1: COLLECT DATA MODEL BY MODEL ---
    for config in MODEL_CONFIGS:
        short_name = config["short"]
        model_dir = CACHE_DIR / short_name
        
        if not model_dir.exists():
            print(f"Cache missing for {short_name}")
            continue
            
        # Map target countries to files
        file_map = map_files_to_targets(model_dir)
        if not file_map:
            print(f"  No targets found in {short_name} cache.")
            continue
            
        # Load Model
        model = load_model(config["name"])
        if not model: continue
        
        # Process Keywords
        for kw in KEYWORDS:
            # print(f"  Scanning '{kw}'...")
            kw_embed = model.encode(kw, normalize_embeddings=True)
            
            for country, filepath in file_map.items():
                try:
                    data = np.load(filepath)
                    chunks = data['text_chunks']
                    embeds = data['embeddings']
                    
                    if len(embeds) == 0: continue
                    
                    # Score
                    scores = embeds @ kw_embed.T
                    
                    # Get indices of Top N
                    # We negate scores to use argsort for descending order
                    top_indices = np.argsort(-scores)[:SEARCH_DEPTH]
                    
                    # Store results
                    results = {}
                    for idx in top_indices:
                        score = float(scores[idx])
                        if score < 0.25: continue # Basic noise filter
                        text = str(chunks[idx]).strip()
                        results[text] = score
                        
                    data_store[kw][country][short_name] = results
                    
                except Exception as e:
                    print(f"    Error reading {country}: {e}")
                    
        # Cleanup
        del model
        clean_memory()

    # --- STEP 2: FIND INTERSECTION & SAVE ---
    print("\n=== CALCULATING CONSENSUS ===")
    final_rows = []

    for kw in KEYWORDS:
        for country in TARGET_COUNTRIES:
            
            # Get dicts for available models
            models_data = [data_store[kw][country].get(m['short'], {}) for m in MODEL_CONFIGS]
            
            # If any model is missing data (empty dict), we can't intersect
            if any(len(d) == 0 for d in models_data):
                continue
                
            # Find Intersection of Texts
            common_texts = set(models_data[0].keys())
            for d in models_data[1:]:
                common_texts = common_texts.intersection(d.keys())
            
            # Convert to list of dicts with scores
            candidates = []
            for text in common_texts:
                scores = [d[text] for d in models_data]
                avg_score = sum(scores) / len(scores)
                candidates.append({
                    "Keyword": kw,
                    "Country": country,
                    "Consensus_Text": text,
                    "Avg_Score": avg_score,
                    "Qwen_Score": models_data[0].get(text, 0),
                    "Jina_Score": models_data[1].get(text, 0)
                })
            
            # Sort by Avg Score
            candidates.sort(key=lambda x: x["Avg_Score"], reverse=True)
            
            # Keep Top 20 (to ensure we cover the "at least 10" request)
            top_candidates = candidates[:20]
            
            final_rows.extend(top_candidates)
            
        # --- STEP 3: SAVE TO EXCEL ---
        if not final_rows:
            print("❌ No overlapping sentences found. Try increasing SEARCH_DEPTH.")
            return
    
        excel_path = OUTPUT_DIR / "Target_Countries_Consensus.xlsx"
        print(f"\nSaving {len(final_rows)} rows to {excel_path}...")
        
        try:
            # FIXED: Added engine='xlsxwriter'
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                for kw in KEYWORDS:
                    subset = pd.DataFrame([r for r in final_rows if r["Keyword"] == kw])
                    if subset.empty: continue
                    
                    # Cleanup cols
                    subset = subset.drop(columns=["Keyword"])
                    cols = ["Country", "Avg_Score", "Consensus_Text", "Qwen_Score", "Jina_Score"]
                    subset = subset[cols]
                    
                    sheet_name = kw[:30].replace(" ", "_")
                    subset.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Format
                    workbook = writer.book
                    worksheet = writer.sheets[sheet_name]
                    
                    # Now add_format will work because we are using XlsxWriter
                    text_fmt = workbook.add_format({'text_wrap': True, 'valign': 'top'})
                    
                    worksheet.set_column('A:A', 15) # Country
                    worksheet.set_column('B:B', 10) # Score
                    worksheet.set_column('C:C', 80, text_fmt) # Text
                    
            print("✓ Done! Excel file created.")
            
        except Exception as e:
            print(f"⚠ Excel save failed ({e}). Saving as CSV.")
            csv_path = OUTPUT_DIR / "Target_Countries_Consensus.csv"
            pd.DataFrame(final_rows).to_csv(csv_path, index=False)
            print("✓ Done! CSV created.")

if __name__ == "__main__":
    main()