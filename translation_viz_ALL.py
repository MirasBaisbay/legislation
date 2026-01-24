import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Colors strictly consistent across all charts
COLORS = {
    'Google': '#5B9BD5',      # Blue
    'Gemini': '#A64D79',      # Purple
    'GPT': '#70AD47'          # Green
}

# Mapping internal column names to Display Names
DISPLAY_NAMES = {
    'Google': 'Google Translate',
    'Gemini': 'Gemini-2.5-Pro',
    'GPT': 'GPT-4.1-mini'
}

# File list for Language Family mapping (Spider Chart)
file_list_text = """
AD_Water Law_catalan.txt
AF_ Water Affairs Management Law (2020)_dari.txt
AO_Water Law_portuguese .txt
AZE_Water code_ Azerbaijani.txt
BF_Water Law_french.txt
BG_Water act_bulg.txt
BIH_Water Law_Bos, Hr, Sr.txt
BI_Water Code_french.txt
BJ_Water LAw_2010_french.txt
BOL_Water Law_spanish.txt
BY_Water Code_rus.txt
CF_Water Code_french.txt
CG_Water Code_french.txt
CI_Water Code_french.txt
CL_Codigo de Auguas_1981_spanish.txt
CM_Water Law_1998_french.txt
COD_Water Code_french.txt
CO_Código Nacional de Recursos Naturales_1974_spanish.txt
CR_Ley de Aguas_1942_spanish.txt
CU_Water Law_spanish.txt
CV_Water Code_portuguese .txt
CY_The Protection and Management of Water Law of 2004 (No. 13(I)_2004)_greek.txt
CZ_ Waters and Amendments to Some Acts (The Water Act)_Czech.txt
DE_Federal Water Act_deu.txt
DJ_Water Code_french.txt
DK_Water Supply Act_No.125 of 2017_Da.txt
DO_Law5852_spanish.txt
DZ_Water Law_french.txt
EC_Water Law_spanish.txt
EE_Water Code_Estonian.txt
EG_Water Law_arabic.txt
ES_Water Law_Es.txt
ET_Water Resources Management_amharic.txt
FI_Water Act_fin.txt
FR_Water and Aquatic Environments Law_french.txt
GA_Env code_2014_french.txt
GE_Water Law_georgian.txt
GN_Water Code_1994_french.txt
GQ_Law3_spanish.txt
GR_ Law 3199_2003 on Water Protection and Management_greek.txt
GW_Water Code_portuguese.txt
HN_Water Law_spanish.txt
HR_Water Act_croatian.txt
HU_ 1995. évi LVII. törvény a vízgazdálkodásról_hungarian.txt
ID_ Law No. 17 of 2019 on Water Resources_indonesian.txt
IL_Water Law,5719-1959_hebrew.txt
IQ_Water Law_arabic.txt
IR_Water Law_farsi.txt
IS_Water Act_ icelandic.txt
IT_Environmental Code_ita .txt
JO_Water Authority Law_arabic.txt
KM_Water Code_french.txt
KR_Water Act_korean.txt
KW_ Environmental Protection Law_arabic.txt
LA_Water Law_lao.txt
LB_Water Law_arabic.txt
LI_Water Act_german.txt
LT_Law on Water of the Republic of Lithuania_lithuanian.txt
LU_ Law of 19 December 2008 on Water_french.txt
LV_Water management Law_latvian.txt
LY_Water Law_arabic.txt
MA_Loi_36-15_2016_french.txt
MD_Water Law_ romanian.txt
ME_Water Law_Montenegrin.txt
MG_Water Act_french.txt
ML_Water Code_french.txt
MR_Water Code_french.txt
MV_Water Act_ Dhivehi.txt
MX_National Water Law_spanish.txt
MZ_Water Law_ portuguese .txt
NE_Water Code_french.txt
NI_Ley General de Aguas Nacionales_2007_spanish.txt
NL_Water Act_2009_dutch.txt
OM_Water Law_arabic.txt
PE_Water Law_spanish.txt
PL_Water Law Act_polish.txt
PT_ Law No. 58_2005, of 29 December (Water Law)_portuguese.txt
PY_Water Law_spanish.txt
RO_ Water Law No. 107_1996_romanian.txt
RS_Water Law_serbian.txt
RU_water code_rus.txt
SA_Saudi water law_arabic.txt
SD_Water resources act_arabic.txt
SI_ Zakon o vodah (ZV-1)_slovenian.txt
SK_ Act No. 364_2004 Coll. on Waters (Water Act)_slovak.txt
SM_Regulation No.1_italian.txt
SN_Code-de-l-eau_1981_french.txt
SO_Water Act_somali.txt
ST_Water Law_portuguese.txt
SV_General water law_2021_spanish.txt
SY_Water Law_arabic.txt
TD_Water Code_french.txt
TG_Water Code_french.txt
TJ_Water Code_rus.txt
TM_Water Code_rus.txt
TN_Water Code_arabic.txt
TR_Water Law_turkish.txt
UA_Water Code_rus.txt
UY_Water Code_1978_spanish.txt
UZ_Water Code_uzb.txt
VE_Ley de Aguas_2007_spanish.txt
VN_Law on Water_2023_vietnamese.txt
YE_Water Law_arabic.txt
"""

# Simplified mapping for Spider Chart
# Ensure keys are lowercase for matching
family_map = {
    'catalan': 'Romance', 'spanish': 'Romance', 'portuguese': 'Romance', 
    'french': 'Romance', 'romanian': 'Romance', 'italian': 'Romance', 'es': 'Romance', 'ita': 'Romance',
    'german': 'Germanic', 'dutch': 'Germanic', 'icelandic': 'Germanic', 'da': 'Germanic', 'deu': 'Germanic', 'english': 'Germanic',
    'rus': 'Slavic', 'bulg': 'Slavic', 'bos': 'Slavic', 'czech': 'Slavic', 'croatian': 'Slavic', 'serbian': 'Slavic', 'slovenian': 'Slavic', 'slovak': 'Slavic', 'polish': 'Slavic', 'montenegrin': 'Slavic', 'ukrainian': 'Slavic',
    'arabic': 'Semitic', 'hebrew': 'Semitic', 'amharic': 'Semitic',
    'turkish': 'Turkic', 'azerbaijani': 'Turkic', 'uzb': 'Turkic', 'kazakh': 'Turkic', 'kyrgyz': 'Turkic', 'turkmen': 'Turkic',
    'farsi': 'Iranian', 'dari': 'Iranian', 'tajik': 'Iranian', 'pashto': 'Iranian',
    'greek': 'Hellenic',
    'estonian': 'Uralic', 'finnish': 'Uralic', 'hungarian': 'Uralic', 'fin': 'Uralic',
    'lithuanian': 'Baltic', 'latvian': 'Baltic',
    'indonesian': 'Austronesian', 'malay': 'Austronesian',
    'georgian': 'Kartvelian',
    'korean': 'Koreanic',
    'vietnamese': 'Austroasiatic', 'khmer': 'Austroasiatic',
    'lao': 'Kra-Dai', 'thai': 'Kra-Dai',
    'dhivehi': 'Indo-Aryan', 'hindi': 'Indo-Aryan', 'bengali': 'Indo-Aryan', 'nepali': 'Indo-Aryan', 'urdu': 'Indo-Aryan', 'sinhala': 'Indo-Aryan',
    'somali': 'Cushitic'
}

def get_family_from_filename(filename):
    """
    Robust extraction of language family from filename.
    """
    # 1. Clean the filename first
    clean_name = filename.strip()
    
    # 2. Try to extract the language part at the end (e.g., "_french.txt")
    # Using a regex that looks for the last segment after an underscore, ignoring .txt
    match = re.search(r'_([a-zA-Z,\s]+)\.txt$', clean_name)
    
    if match:
        raw_lang = match.group(1).lower().strip()
        
        # Special case for comma-separated languages like "Bos, Hr, Sr"
        if 'bos' in raw_lang: return 'Slavic' 
        
        # Check against our map
        for key, fam in family_map.items():
            # Check if the key is IN the raw_lang string (e.g. "portuguese " contains "portuguese")
            if key in raw_lang:
                return fam
                
    return 'Other'

def map_codes_to_families(raw_df):
    """
    Maps each file in the dataframe to a language family and prints the mapping for verification.
    """
    print("\n--- DEBUG: Language Family Mapping ---")
    
    # Create the mapping dictionary
    file_to_family = {}
    
    # We iterate through the DataFrame 'File' column (which contains country codes usually)
    # BUT we need the original filenames to detect language.
    # Assuming 'scores_raw.csv' has a 'File' column that matches the PREFIX of the text files (e.g., 'AD', 'AF')
    # OR if 'File' contains the full filename.
    
    # Strategy: We will use the 'file_list_text' variable you provided to build a map: CODE -> FAMILY
    # Then we map the DataFrame's 'File' column (which usually contains codes like 'AD', 'AF') using this map.
    
    # 1. Parse the file list provided in the script
    # (Note: I'm not using the huge string variable here to save space in the chat, 
    #  but in your real script, you MUST include the 'file_list_text' string defined in your previous message)
    
    # FOR THIS DEBUG TO WORK, ensure 'file_list_text' is defined in the script!
    # I will assume it is available globally as per your original code structure.
    
    code_to_family = {}
    
    # Split the long string into lines
    try:
        lines = file_list_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Extract Country Code (Prefix)
            code_match = re.match(r'^([A-Z]{2,3})_', line)
            if code_match:
                code = code_match.group(1)
                family = get_family_from_filename(line)
                code_to_family[code] = family
                
                # Print any 'Other' classifications for debugging
                if family == 'Other':
                    print(f"[WARNING] Classified as OTHER: {line}")
                # else:
                #     print(f"OK: {code} -> {family}") # Uncomment to see all
                    
    except NameError:
        print("Error: 'file_list_text' variable not found. Please ensure it is defined.")
        return raw_df

    # 2. Map the DataFrame
    # Assuming the 'File' column in raw_df contains the Country Codes (e.g. 'AD', 'AF')
    raw_df['Family'] = raw_df['File'].map(code_to_family).fillna('Other')
    
    # 3. Print Summary
    print("\n--- Family Distribution ---")
    print(raw_df['Family'].value_counts())
    print("--------------------------------------\n")
    
    return raw_df

def create_spider_chart(df_mapped):
    """
    Create spider chart by language family using nanmean and fixed scale.
    """
    # Use nanmean to ignore missing values
    family_scores = df_mapped.groupby('Family')[['Google', 'Gemini', 'GPT']].agg(np.nanmean)
    
    categories = list(family_scores.index)
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] 

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='black', size=11)
    
    # FIXED SCALE 0.0 to 1.0
    plt.ylim(0, 1.0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    
    for model in ['Google', 'Gemini', 'GPT']:
        values = family_scores[model].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=DISPLAY_NAMES[model], color=COLORS[model])
        ax.fill(angles, values, color=COLORS[model], alpha=0.05)

    # UPDATED TITLE with X -> ENG
    plt.title('Translation Performance by Language Family\n(X → ENG)', size=16, y=1.08, fontweight='bold')
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.savefig('spider_chart_families.png', dpi=300, bbox_inches='tight')
    print("Generated: spider_chart_families.png")
    plt.close()

def create_all_models_chart_with_stats(df):
    """
    Create chart with all three models and statistics box
    """
    subset = df[['File', 'Google', 'Gemini', 'GPT']].dropna().copy()
    subset = subset.sort_values(by='File')
    n_countries = len(subset)
    
    # Setup Figure
    fig_width = max(20, n_countries * 0.3)
    plt.figure(figsize=(fig_width, 8))
    
    # Bar Positions
    x = np.arange(n_countries)
    width = 0.25
    
    # Create Grouped Bars (With Updated Labels)
    plt.bar(x - width, subset['Google'], width, label='Google Translate', 
            color=COLORS['Google'], alpha=0.9, edgecolor='white', linewidth=0.5)
    plt.bar(x, subset['Gemini'], width, label='Gemini-2.5-Pro', 
            color=COLORS['Gemini'], alpha=0.9, edgecolor='white', linewidth=0.5)
    plt.bar(x + width, subset['GPT'], width, label='GPT-4.1-mini', 
            color=COLORS['GPT'], alpha=0.9, edgecolor='white', linewidth=0.5)
    
    # Calculate and plot mean lines
    for model_name, color in COLORS.items():
        mean_val = subset[model_name].mean()
        plt.axhline(y=mean_val, color=color, linestyle='--', linewidth=2, 
                   alpha=0.7, zorder=5)
    
    # Create statistics text box (With Updated Names)
    stats_text = "Model Statistics:\n" + "="*35 + "\n"
    for model_name in ['Google', 'Gemini', 'GPT']:
        disp_name = DISPLAY_NAMES[model_name]
        mean_val = subset[model_name].mean()
        median_val = subset[model_name].median()
        std_val = subset[model_name].std()
        stats_text += f"{disp_name:16s}: μ={mean_val:.3f}  Med={median_val:.3f}  σ={std_val:.3f}\n"
    
    # Add text box in top left
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
             fontfamily='monospace')
    
    # Labels and formatting
    plt.xlabel('Country Code (ISO)', fontsize=14, fontweight='bold')
    plt.ylabel('COMET-Kiwi Score', fontsize=14, fontweight='bold')
    plt.xticks(x, subset['File'], rotation=90, fontsize=8, ha='center')
    
    # UPDATED TITLE with X -> ENG
    plt.title(f'Translation Quality Comparison (X → ENG) - All Models (n={n_countries})', 
              fontsize=18, pad=20, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12, framealpha=0.95)
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    plt.xlim(-1, n_countries)
    
    max_score = subset[['Google', 'Gemini', 'GPT']].max().max()
    plt.ylim(0, max(0.8, max_score * 1.1))
    plt.yticks(np.arange(0, 0.9, 0.1))
    
    plt.tight_layout()
    plt.savefig('all_models_with_stats.png', dpi=300, bbox_inches='tight')
    print("Generated: all_models_with_stats.png")
    plt.close()

def create_pairwise_chart_with_stats(df, model_x, model_y):
    """
    Create pairwise comparison chart with statistics box
    """
    subset = df[['File', model_x, model_y]].dropna().copy()
    subset = subset.sort_values(by='File')
    n_countries = len(subset)
    
    # Get display names
    name_x = DISPLAY_NAMES[model_x]
    name_y = DISPLAY_NAMES[model_y]
    
    # Setup Figure
    fig_width = max(20, n_countries * 0.3)
    plt.figure(figsize=(fig_width, 8))
    
    # Bar Positions
    x = np.arange(n_countries)
    width = 0.35
    
    # Create Bars
    plt.bar(x - width/2, subset[model_x], width, label=name_x, 
            color=COLORS[model_x], alpha=0.9, edgecolor='white', linewidth=0.5)
    plt.bar(x + width/2, subset[model_y], width, label=name_y, 
            color=COLORS[model_y], alpha=0.9, edgecolor='white', linewidth=0.5)
    
    # Calculate and plot mean lines
    mean_x_val = subset[model_x].mean()
    mean_y_val = subset[model_y].mean()
    
    plt.axhline(y=mean_x_val, color=COLORS[model_x], linestyle='--', linewidth=2, 
               alpha=0.7, zorder=5)
    plt.axhline(y=mean_y_val, color=COLORS[model_y], linestyle='--', linewidth=2, 
               alpha=0.7, zorder=5)
    
    # Create statistics text box
    stats_text = "Model Statistics:\n" + "="*35 + "\n"
    for model_name in [model_x, model_y]:
        disp_name = DISPLAY_NAMES[model_name]
        mean_val = subset[model_name].mean()
        median_val = subset[model_name].median()
        std_val = subset[model_name].std()
        stats_text += f"{disp_name:16s}: μ={mean_val:.3f}  Med={median_val:.3f}  σ={std_val:.3f}\n"
    
    # Calculate difference
    diff = abs(mean_x_val - mean_y_val)
    better_key = model_x if mean_x_val > mean_y_val else model_y
    better_name = DISPLAY_NAMES[better_key]
    
    stats_text += "="*35 + "\n"
    stats_text += f"Difference: {diff:.3f}\n"
    stats_text += f"Better: {better_name}"
    
    # Add text box in top left
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9),
             fontfamily='monospace')
    
    # Labels and formatting
    plt.xlabel('Country Code (ISO)', fontsize=14, fontweight='bold')
    plt.ylabel('COMET-Kiwi Score', fontsize=14, fontweight='bold')
    plt.xticks(x, subset['File'], rotation=90, fontsize=8, ha='center')
    
    # UPDATED TITLE with X -> ENG
    plt.title(f'Translation Quality Comparison (X → ENG): {name_x} vs {name_y} (n={n_countries})', 
              fontsize=18, pad=20, fontweight='bold')
    plt.legend(loc='upper right', fontsize=12, framealpha=0.95)
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    plt.xlim(-1, n_countries)
    
    max_score = subset[[model_x, model_y]].max().max()
    plt.ylim(0, max(0.8, max_score * 1.1))
    plt.yticks(np.arange(0, 0.9, 0.1))
    
    filename = f'pairwise_{model_x}_vs_{model_y}_with_stats.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Generated: {filename}")
    plt.close()

def generate_all_visualizations():
    """
    Main function to generate all 5 charts
    """
    try:
        raw_df = pd.read_csv('scores_raw.csv')
    except FileNotFoundError:
        print("Error: 'scores_raw.csv' not found.")
        return
    
    # Rename GT to Google for consistency
    if 'GT' in raw_df.columns:
        raw_df = raw_df.rename(columns={'GT': 'Google'})
    
    # Map families for spider chart (with Debugging)
    df_mapped = map_codes_to_families(raw_df)
    
    print("\n" + "="*60)
    print("GENERATING TRANSLATION QUALITY VISUALIZATIONS")
    print("="*60 + "\n")
    
    # 1. Spider Chart
    print("1/5 Creating spider chart by language family...")
    create_spider_chart(df_mapped)
    
    # 2. All Models Chart with Stats
    print("2/5 Creating all models comparison with statistics...")
    create_all_models_chart_with_stats(df_mapped)
    
    # 3-5. Pairwise Comparisons with Stats
    pairwise_combinations = [
        ('Google', 'Gemini'),
        ('Google', 'GPT'),
        ('Gemini', 'GPT')
    ]
    
    for i, (model_x, model_y) in enumerate(pairwise_combinations, start=3):
        # Using internal keys for filename generation to keep them short/safe
        print(f"{i}/5 Creating pairwise comparison: {DISPLAY_NAMES[model_x]} vs {DISPLAY_NAMES[model_y]}...")
        create_pairwise_chart_with_stats(df_mapped, model_x, model_y)
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. spider_chart_families.png")
    print("  2. all_models_with_stats.png")
    print("  3. pairwise_Google_vs_Gemini_with_stats.png")
    print("  4. pairwise_Google_vs_GPT_with_stats.png")
    print("  5. pairwise_Gemini_vs_GPT_with_stats.png")
    print("="*60 + "\n")

if __name__ == "__main__":
    generate_all_visualizations()