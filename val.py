import pandas as pd
from pathlib import Path

# Paths
qwen_path = Path("results/viz_data/heatmap_data_Qwen3-8B.csv")
openai_path = Path("results/viz_data/heatmap_data_OpenAI.csv")

def check_alignment():
    print("Running Sanity Check...")
    
    df_qwen = pd.read_csv(qwen_path)
    df_openai = pd.read_csv(openai_path)
    
    # 1. Check Row Counts
    print(f"Qwen Rows:   {len(df_qwen)}")
    print(f"OpenAI Rows: {len(df_openai)}")
    
    if len(df_qwen) != len(df_openai):
        print("❌ FAIL: Row counts do not match!")
        diff = set(df_qwen['Country']) ^ set(df_openai['Country'])
        print(f"Mismatched Countries: {diff}")
        return

    # 2. Check Column Names
    if list(df_qwen.columns) != list(df_openai.columns):
        print("❌ FAIL: Column names do not match!")
        return

    # 3. Check Country Sorting
    # We enforce sorting to ensure alignment
    df_qwen = df_qwen.sort_values("Country").reset_index(drop=True)
    df_openai = df_openai.sort_values("Country").reset_index(drop=True)
    
    if not df_qwen['Country'].equals(df_openai['Country']):
         print("❌ FAIL: Countries are not identical or sorted differently!")
         for i in range(len(df_qwen)):
             c1 = df_qwen.iloc[i]['Country']
             c2 = df_openai.iloc[i]['Country']
             if c1 != c2:
                 print(f"  Mismatch at row {i}: Qwen='{c1}' vs OpenAI='{c2}'")
         return

    print("✅ SUCCESS: Both files are perfectly aligned (167 countries, same columns).")

if __name__ == "__main__":
    check_alignment()