import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from difflib import SequenceMatcher

st.set_page_config(page_title="Analyse Concurrentielle", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .block-container {padding: 1.2rem;}
    h1, h2, h3 {color: #2c3e50;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Analyse Concurrentielle")

DATA_PATH = "companies.xlsx"

@st.cache_data(ttl=300)
def load_excel(path):
    return pd.read_excel(path)

df = None
if os.path.exists(DATA_PATH):
    try:
        df = load_excel(DATA_PATH)
    except Exception as e:
        st.error(f"Erreur lecture {DATA_PATH}: {e}")
        df = None

if df is None:
    uploaded_file = st.file_uploader("📂 Téléversez le fichier Excel (si 'companies.xlsx' absent)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
    else:
        st.info("📁 Aucune donnée chargée. Téléversez un fichier ou placez 'companies.xlsx' dans le dossier de l'app.")
        st.stop()

cols = list(df.columns)
cols_lower = [c.lower() for c in cols]

def find_col(keyword_tokens, year=None):
    for i, col in enumerate(cols):
        cl = col.lower()
        if all(tok.lower() in cl for tok in keyword_tokens):
            if year is None or str(year) in cl:
                return col
    return None

def get_column_for(metric_key, year):
    col = find_col(metric_key, year)
    if not col and year is not None:
        col = find_col(metric_key, None)
    return col

YEARS = [2020, 2021, 2022, 2023]
SECTOR_COL = None
COMPANY_COL = None
COMPANY_COL2 = None

for c in cols:
    lc = c.lower()
    if "secteur" in lc:
        SECTOR_COL = c
    if ("raison" in lc and "kerix" in lc) or ("raison" in lc and "sociale" in lc):
        COMPANY_COL = c
    if ("raison" in lc and "nouvelle" in lc):
        COMPANY_COL2 = c
if COMPANY_COL is None:
    string_cols = [c for c in cols if df[c].dtype == object]
    COMPANY_COL = string_cols[0] if string_cols else cols[0]

if SECTOR_COL is None:
    for c in cols:
        if "secteur" in c.lower():
            SECTOR_COL = c
if SECTOR_COL is None:
    df['Secteur'] = 'Tous'
    SECTOR_COL = 'Secteur'

st.sidebar.header("🔍 Filtres")
sector_list = list(df[SECTOR_COL].dropna().unique())
sector_choice = st.sidebar.selectbox("🏣 Sélectionner un secteur", ["Tous"] + sorted(sector_list))
metric_display_year = st.sidebar.selectbox("Année pour affichages (top / parts de marché)", YEARS, index=3)
top_n = st.sidebar.number_input("Nombre d'entreprises à afficher", min_value=3, max_value=200, value=10, step=1)

def safe_to_numeric(series):
    return pd.to_numeric(series, errors='coerce')

METRIC_TOKENS = {
    "CA": ["chiffre", "affaires"],
    "RE": ["resultat", "exploitation"],
    "CP": ["charges", "personnel"],
    "EBIT_CA": ["marge", "ebit", "ca"],
    "EBIT_CP": ["marge", "ebit", "cp"],
    "CP_CA": ["marge", "cp", "ca"],
}

def build_metric_matrix(df_input, metric_key):
    cols_found = {}
    for y in YEARS:
        col = get_column_for(METRIC_TOKENS[metric_key], y)
        cols_found[y] = col
    matrix = pd.DataFrame({COMPANY_COL: df_input[COMPANY_COL]})
    for y in YEARS:
        c = cols_found[y]
        if c is not None:
            matrix[str(y)] = safe_to_numeric(df_input[c])
        else:
            matrix[str(y)] = np.nan
    return matrix, cols_found

st.markdown("---")

sector_df = df if sector_choice == "Tous" else df[df[SECTOR_COL] == sector_choice]
n_companies_sector = len(sector_df)

st.subheader(f"Secteur : {sector_choice} - {n_companies_sector} entreprises")
agg = {}
for key in ["CA", "RE", "CP"]:
    mat, cols_map = build_metric_matrix(sector_df, key)
    totals = {}
    means = {}
    for y in YEARS:
        totals[y] = mat[str(y)].sum(skipna=True)
        means[y] = mat[str(y)].mean(skipna=True)
    agg[key] = {"totals": totals, "means": means, "cols_map": cols_map}

def safe_div(a, b):
    try:
        if b == 0 or pd.isna(b):
            return np.nan
        return a / b
    except Exception:
        return np.nan

latest = metric_display_year
agg_CA_latest = agg["CA"]["totals"].get(latest, np.nan)
agg_RE_latest = agg["RE"]["totals"].get(latest, np.nan)
agg_CP_latest = agg["CP"]["totals"].get(latest, np.nan)

ebit_ca_ratio = safe_div(agg_RE_latest, agg_CA_latest)
ebit_cp_ratio = safe_div(agg_RE_latest, agg_CP_latest)
cp_ca_ratio = safe_div(agg_CP_latest, agg_CA_latest)

col1, col2, col3 = st.columns(3)
col1.metric(f"CA total ({latest})", f"{int(agg_CA_latest):,}".replace(",", ".") if not pd.isna(agg_CA_latest) else "N/A")
col2.metric(f"Résultat d'exploitation total ({latest})", f"{int(agg_RE_latest):,}".replace(",", ".") if not pd.isna(agg_RE_latest) else "N/A")
col3.metric(f"Charges personnel total ({latest})", f"{int(agg_CP_latest):,}".replace(",", ".") if not pd.isna(agg_CP_latest) else "N/A")

st.write(f"Ratios agrégés ({latest}): EBIT/CA = {ebit_ca_ratio:.2%} - EBIT/CP = {ebit_cp_ratio:.2%} - CP/CA = {cp_ca_ratio:.2%}")

def compute_company_cagrs(df_input, metric_key):
    mat, cols_map = build_metric_matrix(df_input, metric_key)
    cagr_list = []
    for idx, row in mat.iterrows():
        v0 = row[str(YEARS[0])]
        vN = row[str(YEARS[-1])]
        if pd.notna(v0) and pd.notna(vN) and v0 > 0 and vN > 0:
            try:
                cagr = (vN / v0) ** (1 / (len(YEARS) - 1)) - 1
                if isinstance(cagr, complex):
                    continue
                cagr_list.append(float(cagr))
            except Exception:
                continue
    return cagr_list

st.subheader(f"CAGR moyens du secteur {sector_choice}")
cagr_results = {}
cagr_display = []
for key in ["CA", "RE", "CP"]:
    cagr_list = compute_company_cagrs(sector_df, key)
    mean_cagr = np.mean(cagr_list) if len(cagr_list) > 0 else np.nan
    if isinstance(mean_cagr, complex):
        mean_cagr = np.nan
    cagr_results[key] = mean_cagr
    cagr_display.append((key, mean_cagr, len(cagr_list)))

for key, mean_cagr, count in cagr_display:
    if not pd.isna(mean_cagr) and np.isfinite(mean_cagr):
        st.write(f"- {key} : CAGR moyen = {mean_cagr:.2%} (calculé sur {count} entreprises)")
    else:
        st.write(f"- {key} : CAGR moyen = N/A")

st.markdown("### 📈 Évolution moyenne secteur")
years = YEARS
ca_means = [agg["CA"]["means"].get(y, np.nan) for y in years]
re_means = [agg["RE"]["means"].get(y, np.nan) for y in years]

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=years,
        y=ca_means,
        name="CA moyen ",
        marker_color="#2c7fb8",
        width=0.35,
        offsetgroup=1
    )
)

fig.add_trace(
    go.Bar(
        x=years,
        y=re_means,
        name="RE moyen ",
        marker_color="#de2d26",
        width=0.35,
        offsetgroup=2,
        yaxis="y2"
    )
)

fig.update_layout(
    title=f"CA moyen vs RE moyen par entreprise - Secteur: {sector_choice}",
    xaxis=dict(title="Année"),
    yaxis=dict(title="CA (Dhs)"),
    yaxis2=dict(
        title="RE (Dhs)",
        overlaying="y",
        side="right"
    ),
    barmode="group",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader(f"Parts de marché ({latest})")
for key, label in [("CA", "Chiffre d'affaires"), ("RE", "Résultat d'exploitation"), ("CP", "Charges personnel")]:
    col = get_column_for(METRIC_TOKENS[key], latest)
    if col:
        temp = sector_df[[COMPANY_COL, COMPANY_COL2, col]].copy()
        temp[col] = safe_to_numeric(temp[col]).fillna(0)
        temp['Entreprise'] = temp[COMPANY_COL].combine_first(temp[COMPANY_COL2])
        total = temp[col].sum()
        if total == 0:
            st.write(f"Aucune valeur non-nulle pour {label} ({latest}) - impossible de calculer parts.")
            continue
        temp['Part'] = temp[col] / total
        temp = temp.sort_values(by=col, ascending=False).head(top_n)
        temp_display = temp[['Entreprise', col, 'Part']].copy()
        temp_display[col] = temp_display[col].apply(lambda x: f"{int(x):,}".replace(",", "."))
        temp_display['Part'] = (temp_display['Part'] * 100).round(2).astype(str) + '%'
        st.write(f"Parts de marché par {label} ({latest}) - secteur {sector_choice}")
        st.dataframe(
            temp_display.rename(columns={col: label, 'Part': 'Part (%)'}),
            use_container_width=True
        )
    else:
        st.write(f"Aucune colonne trouvée pour {label} ({latest}) - impossible de calculer les parts de marché.")

st.markdown("---")
st.subheader("🧾 Comparaison multi-entreprises")
selected_companies = st.multiselect("Sélectionnez (multiselect)", sorted(df[COMPANY_COL].dropna().unique()), default=[])
compare_metrics = {
    "Chiffre d'affaires": "CA",
    "Résultat d'exploitation": "RE",
    "Charges personnel": "CP",
    "EBIT/CA": "EBIT_CA",
    "EBIT/CP": "EBIT_CP",
    "CP/CA": "CP_CA"
}
for label, key in compare_metrics.items():
    fig = go.Figure()
    plotted_any = False
    for company in selected_companies:
        comp_df = df[df[COMPANY_COL].str.lower().str.strip() == company.lower().strip()]
        if comp_df.empty:
            st.warning(f"Entreprise '{company}' non trouvée dans les données - ignorée pour {label}.")
            continue
        values = []
        for y in YEARS:
            if key in ["CA", "RE", "CP"]:
                col = get_column_for(METRIC_TOKENS[key], y)
            else:
                col = get_column_for(METRIC_TOKENS[key], y)
            if col and col in comp_df.columns:
                val = safe_to_numeric(comp_df.iloc[0][col])
            else:
                val = np.nan
            values.append(val)
        if all(pd.isna(values)):
            st.info(f"Aucune donnée 2020-2023 trouvée pour '{company}' - métrique {label}.")
            continue
        plotted_any = True
        fig.add_trace(go.Scatter(
            x=YEARS,
            y=[0 if pd.isna(v) else v for v in values],
            mode='lines+markers',
            name=company,
            hovertemplate=f"<b>{company}</b><br>Année: %{{x}}<br>{label}: %{{y:,.0f}}<extra></extra>".replace(",", ".")
        ))
    sector_cagr = cagr_results.get(key if key in cagr_results else key, np.nan)
    if key not in cagr_results:
        try:
            cgrs = compute_company_cagrs(sector_df, key)
            sector_cagr = np.mean(cgrs) if len(cgrs) > 0 else np.nan
        except Exception:
            sector_cagr = np.nan
    sector_matrix, _ = build_metric_matrix(sector_df, key)
    sector_2020_mean = sector_matrix['2020'].mean(skipna=True)
    if not pd.isna(sector_2020_mean) and not pd.isna(sector_cagr) and np.isfinite(sector_cagr):
        sector_proj = [sector_2020_mean * ((1 + sector_cagr) ** (y - 2020)) for y in YEARS]
        fig.add_trace(go.Scatter(
            x=YEARS,
            y=sector_proj,
            mode='lines+markers',
            name=f"Référence secteur (CAGR moyen {sector_cagr:.2%})",
            line=dict(dash='dash', color='black'),
            hovertemplate=f"Référence secteur (%{{x}}): %{{y:,.0f}}".replace(",", ".") + "<extra></extra>"
        ))
    fig.update_layout(
        title=f"Comparaison sur 2020-2023 - {label}",
        xaxis=dict(tickmode='array', tickvals=YEARS),
        yaxis_title=label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420
    )
    if plotted_any:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("🔎 Détails entreprise")
company_single = st.selectbox("Sélectionner une entreprise pour détails", ["Aucune"] + sorted(list(df[COMPANY_COL].dropna().unique())))
if company_single != "Aucune":
    comp_df = df[df[COMPANY_COL] == company_single]
    if comp_df.empty:
        st.write("Entreprise non trouvée.")
    else:
        metrics_for_display = ["CA", "RE", "CP", "EBIT_CA"]
        for key in metrics_for_display:
            mat, _ = build_metric_matrix(comp_df, key)
            values = [mat[str(y)].iloc[0] if str(y) in mat.columns else np.nan for y in YEARS]
            if all(pd.isna(values)):
                continue
            title = {
                "CA": "Chiffre d'affaires",
                "RE": "Résultat d'exploitation",
                "CP": "Charges personnel",
                "EBIT_CA": "Marge EBIT/CA"
            }.get(key, key)
            fig = px.line(x=YEARS, y=[0 if pd.isna(v) else v for v in values], markers=True, labels={'x': 'Année', 'y': title}, title=f"{title} - {company_single}")
            st.plotly_chart(fig, use_container_width=True)

with st.expander("🔎 Afficher les données brutes"):
    st.dataframe(df, use_container_width=True)