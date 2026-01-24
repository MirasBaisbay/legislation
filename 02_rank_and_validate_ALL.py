import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kendalltau
import re

# ================= CONFIGURATION =================
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results/final_analysis")
OPENAI_FILE = Path("keyword_similarities_OPENAI.parquet") 
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_FILES = {
    "Qwen3-8B": "country_scores_Qwen3-8B.json",
    "BGE-M3": "country_scores_BGE-M3.json",
    "Jina-v3": "country_scores_Jina-v3.json"
}

OPENAI_MAPPING = {
    "sim_groundwater_withdrawal": "groundwater withdrawal",
    "sim_groundwater_monitoring": "ground-water monitoring",
    "sim_underground_water_abstraction": "underground water abstraction",
    "sim_groundwater_permits": "groundwater permits",
    "sim_groundwater_rights": "groundwater rights",
    "sim_well_and_borehole_drilling_licenses": "well and borehole drilling licenses",
    "sim_conjunctive_management": "conjunctive management",
    "sim_groundwater_protection_zones": "groundwater protection zones",
    "sim_aquifer_recharge": "aquifer recharge",
    "sim_transboundary_aquifers": "transboundary aquifers"
}

# --- NAME CLEANING DICTIONARIES (Full List) ---
ISO_TO_COUNTRY = {
    "AD": "Andorra", "AF": "Afghanistan", "AG": "Antigua and Barbuda", "AL": "Albania", "AO": "Angola",
    "AR": "Argentina", "AT": "Austria", "AU": "Australia", "AZ": "Azerbaijan", "AZE": "Azerbaijan",
    "BA": "Bosnia and Herzegovina", "BB": "Barbados", "BD": "Bangladesh", "BE": "Belgium", "BF": "Burkina Faso",
    "BG": "Bulgaria", "BH": "Bahrain", "BI": "Burundi", "BJ": "Benin", "BN": "Brunei", "BO": "Bolivia", "BOL": "Bolivia",
    "BR": "Brazil", "BS": "Bahamas", "BT": "Bhutan", "BW": "Botswana", "BY": "Belarus", "BZ": "Belize",
    "CA": "Canada", "CD": "DR Congo", "COD": "DR Congo", "CF": "Central African Republic", "CG": "Congo",
    "CH": "Switzerland", "CI": "Ivory Coast", "CL": "Chile", "CM": "Cameroon", "CN": "China", "CO": "Colombia",
    "CR": "Costa Rica", "CU": "Cuba", "CV": "Cape Verde", "CY": "Cyprus", "CZ": "Czech Republic",
    "DE": "Germany", "DJ": "Djibouti", "DK": "Denmark", "DM": "Dominica", "DO": "Dominican Republic", "DZ": "Algeria",
    "EC": "Ecuador", "EE": "Estonia", "EG": "Egypt", "ER": "Eritrea", "ES": "Spain", "ET": "Ethiopia",
    "FI": "Finland", "FJ": "Fiji", "FM": "Micronesia", "FR": "France",
    "GA": "Gabon", "GB": "United Kingdom", "GD": "Grenada", "GE": "Georgia", "GH": "Ghana", "GM": "Gambia",
    "GN": "Guinea", "GQ": "Equatorial Guinea", "GR": "Greece", "GT": "Guatemala", "GW": "Guinea-Bissau", "GY": "Guyana",
    "HN": "Honduras", "HR": "Croatia", "HT": "Haiti", "HU": "Hungary",
    "ID": "Indonesia", "IE": "Ireland", "IL": "Israel", "IN": "India", "IQ": "Iraq", "IR": "Iran", "IS": "Iceland", "IT": "Italy",
    "JM": "Jamaica", "JO": "Jordan", "JP": "Japan",
    "KE": "Kenya", "KG": "Kyrgyzstan", "KH": "Cambodia", "KI": "Kiribati", "KM": "Comoros", "KN": "Saint Christopher and Nevis",
    "KP": "North Korea", "KR": "South Korea", "KW": "Kuwait", "KZ": "Kazakhstan",
    "LA": "Laos", "LB": "Lebanon", "LC": "Saint Lucia", "LI": "Liechtenstein", "LK": "Sri Lanka",
    "LR": "Liberia", "LS": "Lesotho", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "LY": "Libya",
    "MA": "Morocco", "MC": "Monaco", "MD": "Moldova", "ME": "Montenegro", "MG": "Madagascar", "MH": "Marshall Islands",
    "MK": "North Macedonia", "ML": "Mali", "MM": "Myanmar", "MN": "Mongolia", "MR": "Mauritania", "MT": "Malta",
    "MU": "Mauritius", "MV": "Maldives", "MW": "Malawi", "MX": "Mexico", "MY": "Malaysia", "MZ": "Mozambique",
    "NA": "Namibia", "NE": "Niger", "NG": "Nigeria", "NI": "Nicaragua", "NL": "Netherlands", "NO": "Norway",
    "NP": "Nepal", "NR": "Nauru", "NZ": "New Zealand",
    "OM": "Oman",
    "PA": "Panama", "PE": "Peru", "PG": "Papua New Guinea", "PH": "Philippines", "PK": "Pakistan", "PL": "Poland",
    "PT": "Portugal", "PW": "Palau", "PY": "Paraguay",
    "QA": "Qatar",
    "RO": "Romania", "RS": "Serbia", "RU": "Russia", "RW": "Rwanda",
    "SA": "Saudi Arabia", "SB": "Solomon Islands", "SC": "Seychelles", "SD": "Sudan", "SE": "Sweden", "SG": "Singapore",
    "SI": "Slovenia", "SK": "Slovakia", "SL": "Sierra Leone", "SM": "San Marino", "SN": "Senegal", "SO": "Somalia",
    "SR": "Suriname", "SS": "South Sudan", "ST": "Sao Tome and Principe", "SV": "El Salvador", "SY": "Syria", "SZ": "Eswatini",
    "TD": "Chad", "TG": "Togo", "TH": "Thailand", "TJ": "Tajikistan", "TM": "Turkmenistan", "TN": "Tunisia",
    "TO": "Tonga", "TR": "Turkey", "TT": "Trinidad and Tobago", "TV": "Tuvalu", "TW": "Taiwan", "TZ": "Tanzania",
    "UA": "Ukraine", "UG": "Uganda", "US": "United States", "UY": "Uruguay", "UZ": "Uzbekistan",
    "VC": "Saint Vincent and the Grenadines", "VE": "Venezuela", "VN": "Vietnam", "VU": "Vanuatu",
    "WS": "Samoa",
    "YE": "Yemen",
    "ZA": "South Africa", "ZM": "Zambia", "ZW": "Zimbabwe",
    "BIH": "Bosnia and Herzegovina",
}

OPENAI_NAME_FIXES = {
    "Democratic Republic of the Congo": "DR Congo",
    "Congo, Democratic Republic of the": "DR Congo",
    "Republic of Korea": "South Korea",
    "Korea, Republic of": "South Korea",
    "Democratic People's Republic of Korea": "North Korea",
    "Korea, Democratic People's Republic of": "North Korea",
    "Cabo Verde": "Cape Verde",
    "Viet Nam": "Vietnam",
    "Syrian Arab Republic": "Syria",
    "Lao People's Democratic Republic": "Laos",
    "United Republic of Tanzania": "Tanzania",
    "Tanzania, United Republic of": "Tanzania",
    "Russian Federation": "Russia",
    "Micronesia (Federated States of)": "Micronesia",
    "Brunei Darussalam": "Brunei",
    "Cote d'Ivoire": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "Eswatini (Swaziland)": "Eswatini",
    "Czechia": "Czech Republic",
    "Bolivia, Plurinational State of": "Bolivia",
    "Iran, Islamic Republic of": "Iran",
    "Moldova, Republic of": "Moldova",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Macedonia, The Former Yugoslav Republic of": "North Macedonia"
}

def normalize_key(filename):
    clean = str(filename).replace(".txt", "").strip()
    match = re.match(r"^([A-Z]{2,3})_", clean)
    if match:
        code = match.group(1)
        if code in ISO_TO_COUNTRY: return ISO_TO_COUNTRY[code]
    if clean in ISO_TO_COUNTRY: return ISO_TO_COUNTRY[clean]
    if clean in OPENAI_NAME_FIXES: return OPENAI_NAME_FIXES[clean]
    return clean

def load_data():
    data = {}
    print("Loading Model Scores...")
    
    # 1. Load Local Models
    for name, fname in MODEL_FILES.items():
        path = CACHE_DIR / fname
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            data[name] = {}
            for k, v in raw.items():
                clean = normalize_key(k)
                data[name][clean] = v
            print(f"  ✓ {name}: {len(data[name])} countries")

    # 2. Load OpenAI
    if OPENAI_FILE.exists():
        try:
            df = pd.read_parquet(OPENAI_FILE).rename(columns=OPENAI_MAPPING)
            
            # OpenAI typically has one row per chunk. 
            # We count chunks by counting rows per country.
            
            # Normalize names first in the dataframe to ensure grouping is correct
            if 'country_name' not in df.columns:
                 # Fallback if only country_code exists
                 df['country_name'] = df['country_code'].map(ISO_TO_COUNTRY)
            
            df['clean_country'] = df['country_name'].apply(normalize_key)
            
            grouped = df.groupby('clean_country')
            openai_scores = {}
            
            for country, group in grouped:
                openai_scores[country] = {
                    "weighted_score": float(group[list(OPENAI_MAPPING.values())].mean().mean()),
                    "chunk_count": len(group) # Count rows = Count chunks
                }
            data["OpenAI"] = openai_scores
            print(f"  ✓ OpenAI: {len(openai_scores)} countries")
        except Exception as e:
            print(f"  ✗ OpenAI Error: {e}")

    return data

def generate_rankings(data):
    print("\nGenerating Rankings with Length Stats...")
    
    # Master list of countries
    countries = set()
    for m in data: countries.update(data[m].keys())
    
    results = []
    
    for c in countries:
        # 1. Calculate Average Length
        # We average the 'chunk_count' across all models that have data for this country
        lengths = []
        for m in data:
            if c in data[m] and 'chunk_count' in data[m][c]:
                lengths.append(data[m][c]['chunk_count'])
        
        avg_len = np.mean(lengths) if lengths else 0
        
        # 2. Calculate RRF Score
        rrf_score = 0
        k = 60
        
        # We also want to track individual scores for sanity checking
        model_scores = {} 
        
        for m in data:
            # Rank based on weighted_score
            all_scores = [data[m][cnt]['weighted_score'] for cnt in data[m]]
            all_scores.sort(reverse=True)
            
            if c in data[m]:
                s = data[m][c]['weighted_score']
                rank = all_scores.index(s) + 1
                rrf_score += 1 / (k + rank)
                model_scores[f"{m}_score"] = s
            else:
                model_scores[f"{m}_score"] = 0
        
        row = {
            "Country_Name": c,
            "country_id": c, # Redundant but useful for different script expectations
            "Avg_Length": avg_len,
            "RRF_Score": rrf_score
        }
        row.update(model_scores)
        results.append(row)

    df = pd.DataFrame(results)
    
    # Sort by RRF
    df = df.sort_values("RRF_Score", ascending=False).reset_index(drop=True)
    df["Final_Rank"] = df.index + 1
    
    # Save
    out_path = RESULTS_DIR / "final_rankings.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df)} rows to {out_path}")
    print(f"  ✓ Columns: {list(df.columns)}")
    return df

def main():
    data = load_data()
    if data:
        generate_rankings(data)

if __name__ == "__main__":
    main()