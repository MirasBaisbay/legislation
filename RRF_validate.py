import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import os

# ================= CONFIGURATION =================
RESULTS_DIR = Path("results/final_analysis")
FIGURES_DIR = Path("results/figures/robustness")
os.makedirs(FIGURES_DIR, exist_ok=True)

INPUT_FILE = RESULTS_DIR / "final_rankings.csv" # The file from Step 02 (without correction is fine for now)

def analyze_length_bias():
    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Ensure we have the necessary columns
    # If Step 02 didn't save Avg_Length, we need to recalculate or ensure it's there
    if 'Avg_Length' not in df.columns or 'RRF_Score' not in df.columns:
        print("❌ Missing columns. Ensure 'final_rankings.csv' has 'Avg_Length' and 'RRF_Score'")
        return

    # Filter out empty docs
    df = df[df['Avg_Length'] > 0]

    # Calculate Correlations
    pearson_corr, p_p = pearsonr(df['Avg_Length'], df['RRF_Score'])
    spearman_corr, p_s = spearmanr(df['Avg_Length'], df['RRF_Score'])
    
    print("\n" + "="*40)
    print("ROBUSTNESS DIAGNOSTIC: LENGTH BIAS")
    print("="*40)
    print(f"Pearson r (Linear):   {pearson_corr:.4f} (p={p_p:.4f})")
    print(f"Spearman ρ (Rank):    {spearman_corr:.4f} (p={p_s:.4f})")
    print("-" * 40)
    
    # Interpretation
    if abs(pearson_corr) < 0.3:
        print("✅ LOW BIAS: Length does not significantly drive ranking.")
        print("RECOMMENDATION: Do NOT apply correction.")
    elif abs(pearson_corr) < 0.6:
        print("⚠️ MODERATE CORRELATION: Likely a 'Capacity Signal'.")
        print("RECOMMENDATION: Report this as a feature of comprehensive law, do not correct.")
    else:
        print("❌ HIGH BIAS: Length is dominating the score.")
        print("RECOMMENDATION: You SHOULD apply the log-correction.")

    # --- VISUALIZATION FOR SUPPLEMENTARY MATERIALS ---
    plt.figure(figsize=(8, 6))
    
    # Log scale usually visualizes this relationship better
    sns.scatterplot(x=df['Avg_Length'], y=df['RRF_Score'], alpha=0.6, color='#2c3e50')
    
    # Add trendline
    sns.regplot(x=df['Avg_Length'], y=df['RRF_Score'], scatter=False, color='#e74c3c')
    
    plt.xscale('log') # Log scale for length is standard in NLP analysis
    plt.title(f"Robustness Check: Policy Score vs. Document Length\n(Pearson r={pearson_corr:.2f})", fontsize=14)
    plt.xlabel("Document Length (Chunks) - Log Scale", fontsize=12)
    plt.ylabel("Policy Intensity (RRF Score)", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    save_path = FIGURES_DIR / "S1_length_robustness_check.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Saved diagnostic plot: {save_path}")

if __name__ == "__main__":
    analyze_length_bias()