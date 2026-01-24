import pandas as pd
from pathlib import Path

# ================= CONFIGURATION =================
OPENAI_FILE = Path("keyword_similarities_OPENAI.parquet")

# Standardized list of countries we expect (from your previous 02 script context)
# This mimics the "Local" dataset you are trying to match against.
EXPECTED_LOCAL_NAMES = {
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", 
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", 
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", 
    "Cambodia", "Cameroon", "Canada", "Cape Verde", "Central African Republic", "Chad", "Chile", "China", "Colombia", 
    "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "DR Congo", "Denmark", "Djibouti", 
    "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", 
    "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", 
    "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", 
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", 
    "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", 
    "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", 
    "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", 
    "Mozambique", "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", 
    "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", 
    "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", "Saint Christopher and Nevis", 
    "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", 
    "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", 
    "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", 
    "Syria", "Tajikistan", "Tanzania", "Thailand", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", 
    "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", 
    "Uzbekistan", "Vanuatu", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
}

# The fixes currently applied in Script 02
CURRENT_FIXES = {
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
    "Russian Federation": "Russia",
    "Micronesia (Federated States of)": "Micronesia",
    "Brunei Darussalam": "Brunei",
    "Cote d'Ivoire": "Ivory Coast",
    "Eswatini (Swaziland)": "Eswatini",
    "Czechia": "Czech Republic",
}

def analyze_parquet():
    print("="*60)
    print("OPENAI PARQUET ANALYSIS")
    print("="*60)

    if not OPENAI_FILE.exists():
        print(f"✗ File not found: {OPENAI_FILE}")
        return

    try:
        df = pd.read_parquet(OPENAI_FILE)
        print(f"✓ Successfully loaded {len(df):,} rows.")
        print(f"  Columns found: {list(df.columns)}")
        
        # Identify Country Name Column
        col_name = None
        if 'country_name' in df.columns:
            col_name = 'country_name'
        elif 'country' in df.columns:
            col_name = 'country'
        
        if not col_name:
            print("⚠ Could not find 'country_name' column. Showing sample data:")
            print(df.head())
            return

        # Get Unique Countries
        unique_countries = sorted(df[col_name].dropna().unique())
        print(f"\n✓ Found {len(unique_countries)} unique countries in OpenAI file.")
        
        # --- ANALYSIS ---
        
        matches = []
        mismatches = []
        fixed_by_dict = []
        
        print("\n--- NAME MATCHING REPORT ---")
        
        for name in unique_countries:
            # 1. Direct Match
            if name in EXPECTED_LOCAL_NAMES:
                matches.append(name)
                continue
                
            # 2. Match via Parenthesis Cleaning
            clean_name = name.split("(")[0].strip()
            if clean_name in EXPECTED_LOCAL_NAMES:
                matches.append(f"{name} -> {clean_name} (Auto-Cleaned)")
                continue
                
            # 3. Match via Fix Dictionary
            if name in CURRENT_FIXES:
                fixed_by_dict.append(f"{name} -> {CURRENT_FIXES[name]}")
                continue
                
            # 4. No Match Found
            mismatches.append(name)

        print(f"Matched directly or via cleaning: {len(matches)}")
        print(f"Fixed by dictionary: {len(fixed_by_dict)}")
        print(f"STILL MISMATCHED: {len(mismatches)}")
        
        if fixed_by_dict:
            print("\n--- FIXED BY DICTIONARY ---")
            for f in fixed_by_dict:
                print(f"  ✓ {f}")
                
        if mismatches:
            print("\n--- ⚠ UNMATCHED COUNTRIES (Add these to Script 02) ---")
            for m in mismatches:
                print(f"  ✗ {m}")
        else:
            print("\n✓ ALL COUNTRIES ACCOUNTED FOR!")

        # Show full list if user wants
        # print("\nFull OpenAI Country List:")
        # print(unique_countries)

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    analyze_parquet()