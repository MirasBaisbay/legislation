import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import re

# ================= CONFIGURATION =================
# Make paths relative to script location (Portable)
SCRIPT_DIR = Path(__file__).parent.resolve()
CACHE_DIR = SCRIPT_DIR / "cache"
OUTPUT_DIR = SCRIPT_DIR / "results" / "viz_data"
OPENAI_FILE = SCRIPT_DIR / "keyword_similarities_OPENAI.parquet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_FILES = {
    "Qwen3-8B": "country_scores_Qwen3-8B.json",
    "BGE-M3": "country_scores_BGE-M3.json",
    "Jina-v3": "country_scores_Jina-v3.json"
}

KEYWORDS = [
    "groundwater withdrawal", "ground-water monitoring", "underground water abstraction", 
    "groundwater permits", "groundwater rights", "well and borehole drilling licenses", 
    "conjunctive management", "groundwater protection zones", "aquifer recharge", 
    "transboundary aquifers"
]

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

# --- NAME CLEANING DICTIONARIES ---
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

def clean_parentheses(name):
    if "(" in name: return name.split("(")[0].strip()
    return name

def extract_country_from_messy_name(raw_name):
    match = re.match(r"^([A-Z]{2,3})_", raw_name)
    if match:
        code = match.group(1)
        if code in ISO_TO_COUNTRY:
            return ISO_TO_COUNTRY[code]
    return None

def normalize_key(filename):
    """Robust normalization."""
    raw_name = str(filename).replace(".txt", "").strip()
    
    if raw_name in OPENAI_NAME_FIXES:
        return OPENAI_NAME_FIXES[raw_name]
    
    extracted = extract_country_from_messy_name(raw_name)
    if extracted:
        return extracted
        
    if raw_name in ISO_TO_COUNTRY:
        return ISO_TO_COUNTRY[raw_name]
        
    cleaned = clean_parentheses(raw_name)
    if cleaned != raw_name:
        if cleaned in OPENAI_NAME_FIXES:
            return OPENAI_NAME_FIXES[cleaned]
        return cleaned
        
    return raw_name

def get_gold_standard_from_qwen():
    """
    Instead of trusting the folder, we trust the Qwen JSON file.
    If Qwen has 167 countries, that is our Master List.
    """
    qwen_file = CACHE_DIR / "country_scores_Qwen3-8B.json"
    if not qwen_file.exists():
        print(f"⚠ Warning: {qwen_file} not found. Cannot build Master List.")
        return set()
        
    with open(qwen_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    gold_set = set()
    for k in data.keys():
        gold_set.add(normalize_key(k))
        
    print(f"✓ Built Gold Standard List from Qwen Cache: {len(gold_set)} countries")
    return gold_set

def convert_local_models(valid_countries):
    print(f"Converting Local Models...")
    for name, fname in MODEL_FILES.items():
        path = CACHE_DIR / fname
        if not path.exists(): continue
        
        with open(path, 'r') as f:
            data = json.load(f)
            
        rows = []
        for country, stats in data.items():
            clean_name = normalize_key(country)
            
            # STRICT FILTER: Only keep if in Gold Standard
            if valid_countries and clean_name not in valid_countries:
                continue

            row = {"Country": clean_name}
            if 'keyword_scores' in stats:
                row.update(stats['keyword_scores'])
            rows.append(row)
            
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.groupby("Country").mean().reset_index()
        
        df.to_csv(OUTPUT_DIR / f"heatmap_data_{name}.csv", index=False)
        print(f"  ✓ Saved heatmap_data_{name}.csv ({len(df)} rows)")

def convert_openai(valid_countries):
    print(f"Converting OpenAI Data...")
    if not OPENAI_FILE.exists():
        print("  ✗ OpenAI file missing")
        return

    df = pd.read_parquet(OPENAI_FILE)
    df = df.rename(columns=OPENAI_MAPPING)
    
    if 'country_name' in df.columns:
        id_col = 'country_name'
    else:
        id_col = 'doc_id' 
    
    cols = list(OPENAI_MAPPING.values())
    grouped = df.groupby(id_col)[cols].mean().reset_index()
    
    # Normalize Names
    grouped['Country'] = grouped[id_col].apply(normalize_key)
    
    # STRICT FILTER: Filter OpenAI down to the 167 Qwen countries
    if valid_countries:
        grouped = grouped[grouped['Country'].isin(valid_countries)]
    
    final_df = grouped.groupby('Country')[cols].mean().reset_index()
    
    final_df.to_csv(OUTPUT_DIR / "heatmap_data_OpenAI.csv", index=False)
    print(f"  ✓ Saved heatmap_data_OpenAI.csv ({len(final_df)} rows)")

def main():
    print("\n" + "="*60)
    print("GENERATING VISUALIZATION DATA (CSV)")
    print("="*60)
    
    # 1. Build Master List from Qwen Cache (Trusting the 167 count)
    valid_countries = get_gold_standard_from_qwen()
    
    if len(valid_countries) == 0:
        print("⚠ ERROR: Could not build country list.")
        return

    # 2. Convert and Filter
    convert_local_models(valid_countries)
    convert_openai(valid_countries)

if __name__ == "__main__":
    main()