import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler

# --- 1. GLOBAL FONT SETTINGS (ARIAL) ---
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ================= CONFIGURATION =================
# Make paths relative to script location (Portable)
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results" / "final_analysis"
FIGURES_DIR = SCRIPT_DIR / "results" / "figures" / "final_submission"
os.makedirs(FIGURES_DIR, exist_ok=True)

INPUT_FILE = RESULTS_DIR / "final_rankings.csv"

# Color Threshold for Dendrogram
DENDRO_THRESHOLD = 8 

# --- 2. NEW COLOR PALETTE ---
REGION_COLORS = {
    'Africa':        '#3B6B36',  # Dark Green
    'Europe':        '#4A9037',  # Medium Green
    'South America': '#60A73D',  # Light Green
    'North America': '#093A58',  # Dark Blue
    'Asia':          '#00598B',  # Medium Blue
    'Oceania':       '#0086CB',  # Bright Blue
    'Other':         '#20B2DD'   # Cyan
}

def get_region(country_name):
    regions = {
        'Europe': ['Albania','Andorra','Austria','Belarus','Belgium','Bosnia and Herzegovina','Bulgaria','Croatia','Cyprus','Czech Republic','Denmark','Estonia','Finland','France','Germany','Greece','Hungary','Iceland','Ireland','Italy','Latvia','Liechtenstein','Lithuania','Luxembourg','Malta','Moldova','Monaco','Montenegro','Netherlands','North Macedonia','Norway','Poland','Portugal','Romania','Russia','San Marino','Serbia','Slovakia','Slovenia','Spain','Sweden','Switzerland','Ukraine','United Kingdom'],
        'Asia': ['Afghanistan','Armenia','Azerbaijan','Bahrain','Bangladesh','Bhutan','Brunei','Cambodia','China','Georgia','India','Indonesia','Iran','Iraq','Israel','Japan','Jordan','Kazakhstan','Kuwait','Kyrgyzstan','Laos','Lebanon','Malaysia','Maldives','Mongolia','Myanmar','Nepal','North Korea','Oman','Pakistan','Philippines','Qatar','Saudi Arabia','Singapore','South Korea','Sri Lanka','Syria','Tajikistan','Thailand','Timor-Leste','Turkey','Turkmenistan','United Arab Emirates','Uzbekistan','Vietnam','Yemen'],
        'Africa': ['Algeria','Angola','Benin','Botswana','Burkina Faso','Burundi','Cameroon','Cape Verde','Central African Republic','Chad','Comoros','Congo','DR Congo','Djibouti','Egypt','Equatorial Guinea','Eritrea','Eswatini','Ethiopia','Gabon','Gambia','Ghana','Guinea','Guinea-Bissau','Ivory Coast','Kenya','Lesotho','Liberia','Libya','Madagascar','Malawi','Mali','Mauritania','Mauritius','Morocco','Mozambique','Namibia','Niger','Nigeria','Rwanda','Sao Tome and Principe','Senegal','Seychelles','Sierra Leone','Somalia','South Africa','South Sudan','Sudan','Tanzania','Togo','Tunisia','Uganda','Zambia','Zimbabwe'],
        'North America': ['Antigua and Barbuda','Bahamas','Barbados','Belize','Canada','Costa Rica','Cuba','Dominica','Dominican Republic','El Salvador','Grenada','Guatemala','Haiti','Honduras','Jamaica','Mexico','Nicaragua','Panama','Saint Christopher and Nevis','Saint Lucia','Saint Vincent and the Grenadines','Trinidad and Tobago','United States'],
        'South America': ['Argentina','Bolivia','Brazil','Chile','Colombia','Ecuador','Guyana','Paraguay','Peru','Suriname','Uruguay','Venezuela'],
        'Oceania': ['Australia','Fiji','Kiribati','Marshall Islands','Micronesia','Nauru','New Zealand','Palau','Papua New Guinea','Samoa','Solomon Islands','Tonga','Tuvalu','Vanuatu']
    }
    for region, countries in regions.items():
        if country_name in countries: return region
    return "Other"

def load_data():
    if not INPUT_FILE.exists():
        print(f"❌ File not found: {INPUT_FILE}")
        return None
    df = pd.read_csv(INPUT_FILE)

    # FIX NAMES
    df['Country_Name'] = df['Country_Name'].replace(
        "Congo, The Democratic Republic of the", "DR Congo"
    )

    # MERGE DUPLICATES
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    df = df.groupby('Country_Name', as_index=False)[numeric_cols].mean()

    # ASSIGN REGION
    df['Region'] = df['Country_Name'].apply(get_region)
    
    # Recalculate Rank
    df = df.sort_values("RRF_Score", ascending=False).reset_index(drop=True)
    df["Final_Rank"] = df.index + 1

    return df

def plot_dendrogram_viz(df):
    feature_cols = ['RRF_Score', 'Qwen3-8B_score', 'BGE-M3_score', 'Jina-v3_score', 'OpenAI-text-embedding-3-large_score']
    if 'OpenAI-text-embedding-3-large_score' not in df.columns and 'OpenAI_score' in df.columns:
        feature_cols[-1] = 'OpenAI_score'
        
    cols_to_use = [c for c in feature_cols if c in df.columns]
    features = df[cols_to_use].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    plt.figure(figsize=(16, 8))
    linked = linkage(X, 'ward')
    
    dendrogram(
        linked, 
        orientation='top', 
        labels=df['Country_Name'].values, 
        distance_sort='descending', 
        show_leaf_counts=True,
        leaf_font_size=8,
        leaf_rotation=90,
        color_threshold=DENDRO_THRESHOLD 
    )
    
    plt.axhline(y=DENDRO_THRESHOLD, c='grey', lw=1, linestyle='dashed')
    plt.title(f'Hierarchical Clustering (Ward Linkage) - Cut at d={DENDRO_THRESHOLD}', fontsize=16, fontweight='bold', fontname='Arial')
    plt.xlabel("Countries", fontsize=12, fontname='Arial')
    plt.ylabel("Euclidean Distance", fontsize=12, fontname='Arial')
    plt.xticks(fontname='Arial')
    plt.yticks(fontname='Arial')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_S2_Dendrogram.png", dpi=300)
    plt.close()
    print(f"✓ Generated Figure_S2_Dendrogram.png (Threshold={DENDRO_THRESHOLD})")

def plot_world_map(df):
    if not PLOTLY_AVAILABLE: return

    df['Hover_Text'] = df.apply(lambda x: f"Rank: #{int(x['Final_Rank'])}<br>Score: {x['RRF_Score']:.4f}", axis=1)

    fig = px.choropleth(
        df,
        locations="Country_Name",
        locationmode='country names',
        color="RRF_Score",
        hover_name="Country_Name",
        hover_data={"RRF_Score": False, "Country_Name": False, "Hover_Text": True},
        color_continuous_scale="Viridis",
        title="Global Groundwater Policy Intensity Index"
    )
    
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
        margin=dict(l=0, r=0, t=50, b=0),
        font=dict(family="Arial", size=12) 
    )

    fig.write_html(FIGURES_DIR / "Figure_1_World_Map.html")
    print("✓ Generated Figure_1_World_Map.html")

def plot_rrf_ranking_bar(df):
    """
    Generates Horizontal Bar Chart with Scores displayed at the end of each bar.
    """
    print("Generating RRF Ranking Bar Chart...")

    df_sorted = df.sort_values('RRF_Score', ascending=False)
    counts = df_sorted['Region'].value_counts()
    
    n_countries = len(df_sorted)
    # Ensure enough vertical space for text
    fig_height = max(10, n_countries * 0.25)

    plt.figure(figsize=(12, fig_height)) 

    ax = sns.barplot(
        data=df_sorted,
        y='Country_Name',
        x='RRF_Score',
        hue='Region',
        palette=REGION_COLORS,
        dodge=False 
    )

    # --- ADD SCORES TEXT ---
    # Iterate through the sorted dataframe. 
    # Since seaborn plots in the order of the dataframe for y-axis:
    for i, (index, row) in enumerate(df_sorted.iterrows()):
        score = row['RRF_Score']
        # Place text slightly to the right of the bar end
        ax.text(
            score, 
            i, 
            f" {score:.4f}", 
            va='center', 
            ha='left', 
            fontsize=10, 
            fontname='Arial',
            color='black'
        )

    # Extend X-axis slightly so numbers don't get cut off
    max_score = df_sorted['RRF_Score'].max()
    plt.xlim(0, max_score * 1.1)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [f"{label} ({counts.get(label, 0)})" for label in labels]
    
    ax.legend(handles, new_labels, title='UN Regions (Count)', 
              title_fontsize='12', fontsize='10', 
              bbox_to_anchor=(1.01, 1.0), loc='upper left')

    plt.title("Countries by Groundwater Coverage in Water Legislation", fontsize=16, fontweight='bold', pad=20, fontname='Arial')
    plt.xlabel("RRF Score", fontsize=14, fontname='Arial')
    plt.ylabel("", fontsize=14, fontname='Arial')
    plt.xticks(fontsize=12, fontname='Arial')
    plt.yticks(fontsize=10, fontname='Arial')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Invert Y axis to show Rank #1 at top
    plt.ylim(len(df_sorted) - 0.5, -0.5) 

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_RRF_Ranking_Bar.png", dpi=300)
    plt.close()
    print("✓ Generated Figure_RRF_Ranking_Bar.png")

def plot_regional_summary(df):
    """
    Generates Vertical Bar Chart with new colors.
    """
    print("Generating Regional Summary Graph...")

    summary = df.groupby('Region').agg(
        Count=('Country_Name', 'count'),
        Avg_Score=('RRF_Score', 'mean')
    ).reset_index()

    summary = summary.sort_values('Count', ascending=False)

    plt.figure(figsize=(12, 8))
    
    bars = plt.bar(
        summary['Region'], 
        summary['Count'], 
        color=[REGION_COLORS.get(r, '#999') for r in summary['Region']],
        edgecolor='white',
        linewidth=0.5
    )

    for bar, avg_score in zip(bars, summary['Avg_Score']):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height + 0.2, 
            f"Avg Score: {avg_score:.3f}", 
            ha='center', 
            va='bottom',
            fontsize=10, 
            fontweight='bold',
            fontname='Arial',
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8E71C", ec="black", alpha=0.9)
        )

    plt.title("Regional Differences in Water Legislation by Groundwater Coverage", fontsize=16, fontweight='bold', pad=20, fontname='Arial')
    plt.xlabel("", fontsize=12)
    plt.ylabel("Number of Countries in Study", fontsize=12, fontname='Arial')
    plt.xticks(rotation=45, ha='right', fontsize=11, fontname='Arial')
    plt.yticks(fontname='Arial')
    
    plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
    plt.ylim(0, summary['Count'].max() * 1.15)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "Figure_Regional_Summary.png", dpi=300)
    plt.close()
    print("✓ Generated Figure_Regional_Summary.png")

def main():
    print("GENERATING FINAL FIGURES (Arial Font + New Colors + Scores)...")
    df = load_data()
    if df is not None:
        plot_dendrogram_viz(df)
        plot_world_map(df)
        plot_rrf_ranking_bar(df) 
        plot_regional_summary(df) 
    print("\nAll figures saved to:", FIGURES_DIR)

if __name__ == "__main__":
    main()