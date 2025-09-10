import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from difflib import get_close_matches
import json
import colorsys
METRIC_TOKENS = {
    "CA": ["chiffre", "affaires"],
    "RE": ["resultat", "exploitation"],
    "CP": ["charges", "personnel"],
    "EBIT_CA": ["marge", "ebit", "ca"],
    "EBIT_CP": ["marge", "ebit", "cp"],
    "CP_CA": ["marge", "cp", "ca"],
}
st.set_page_config(page_title="Analyse Des Entreprises", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .block-container {padding: 1.2rem;}
    h1, h2, h3 {color: #2c3e50;}
    .group-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .company-card {
        background-color: white;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #dee2e6;
        margin: 0.2rem 0;
    }
    .stMultiSelect > label {font-weight: bold;}
    .group-creation {background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 1rem 0;}
    label[for="radio-Mode d'affichage"] > div[data-testid="stMarkdownContainer"] p {
        font-size: 20px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)
st.title("🏢 Analyse Des Entreprises")
DATA_PATH = "companies.xlsx"
@st.cache_data(ttl=300)
def load_excel(path):
    return pd.read_excel(path)
def format_number(v, percent=False):
    if pd.isna(v):
        return ""
    if percent:
        return f"{v*100:.2f}%"
    if abs(v) >= 1e9:
        return f"{v/1e9:.2f} B"
    elif abs(v) >= 1e6:
        return f"{v/1e6:.2f} M"
    else:
        return f"{v:,.0f}"
COMPANY_COL = None
def safe_find_company(df, company_name, company_col):
    if not isinstance(company_name, str) or not company_name.strip():
        return pd.DataFrame()
    company_name = company_name.strip()
    if company_col is None:
        return pd.DataFrame()
    company_col = company_col.strip()
    comp_df = df[df[company_col].str.strip() == company_name]
    if not comp_df.empty:
        return comp_df
    comp_df = df[df[company_col].str.lower().str.strip() == company_name.lower()]
    if not comp_df.empty:
        return comp_df
    try:
        all_company_names = df[company_col].dropna().astype(str).str.strip().str.lower().unique().tolist()
        if company_name.lower() in all_company_names:
            return df[df[company_col].str.lower().str.strip() == company_name.lower()]
        if len(all_company_names) > 0 and isinstance(company_name, str):
            matches = get_close_matches(
                company_name.lower(),
                all_company_names,
                n=1,
                cutoff=0.6
            )
            if matches:
                matched_name = matches[0]
                return df[df[company_col].str.lower().str.strip() == matched_name]
    except Exception as e:
        st.warning(f"Error in fuzzy matching for '{company_name}': {e}")
    return pd.DataFrame()
def aggregate_group_data(group_entities, df, metric_key, year):
    if df is None or df.empty:
        return np.nan
    if not isinstance(group_entities, (list, tuple)) and not isinstance(group_entities, dict):
        st.warning(f"Invalid group_entities type: {type(group_entities)}")
        return np.nan
    if metric_key not in METRIC_TOKENS:
        st.warning(f"Metric key '{metric_key}' not found in METRIC_TOKENS. Available: {list(METRIC_TOKENS.keys())}")
        return np.nan
    total_values = []
    if isinstance(group_entities, (list, tuple)):
        for entity in group_entities:
            if isinstance(entity, dict):
                if 'companies' in entity:
                    group_values = aggregate_group_data(entity['companies'], df, metric_key, year)
                    if pd.notna(group_values):
                        total_values.append(group_values)
            else:
                try:
                    company_name = str(entity).strip()
                    if not company_name:
                        continue
                    comp_df = safe_find_company(df, company_name, COMPANY_COL)
                    if not comp_df.empty:
                        col = get_column_for(metric_key, year)
                        if col and col in comp_df.columns:
                            value = safe_to_numeric(comp_df.iloc[0][col])
                            if pd.notna(value):
                                total_values.append(value)
                        else:
                            total_values.append(np.nan)
                    else:
                        total_values.append(np.nan)
                except Exception as e:
                    st.warning(f"Error processing entity '{entity}': {e}")
                    total_values.append(np.nan)
    else:
        try:
            company_name = str(group_entities).strip()
            if not company_name:
                return np.nan
            comp_df = safe_find_company(df, company_name, COMPANY_COL)
            if not comp_df.empty:
                col = get_column_for(metric_key, year)
                if col and col in comp_df.columns:
                    value = safe_to_numeric(comp_df.iloc[0][col])
                    if pd.notna(value):
                        return value
            return np.nan
        except Exception as e:
            st.warning(f"Error processing entity '{group_entities}': {e}")
            return np.nan
    valid_values = [v for v in total_values if pd.notna(v) and not isinstance(v, (str, bool))]
    if metric_key in ["EBIT_CA", "EBIT_CP", "CP_CA"]:
        if valid_values:
            ca_vals = [aggregate_group_data(group_entities, df, "CA", year) for _ in range(len(valid_values))]
            re_vals = [aggregate_group_data(group_entities, df, "RE", year) for _ in range(len(valid_values))]
            cp_vals = [aggregate_group_data(group_entities, df, "CP", year) for _ in range(len(valid_values))]
            if metric_key == "EBIT_CA" and ca_vals[0] != 0:
                return np.mean([re/ca if ca != 0 else np.nan for re, ca in zip(re_vals, ca_vals)])
            elif metric_key == "EBIT_CP" and cp_vals[0] != 0:
                return np.mean([re/cp if cp != 0 else np.nan for re, cp in zip(re_vals, cp_vals)])
            elif metric_key == "CP_CA" and ca_vals[0] != 0:
                return np.mean([cp/ca if ca != 0 else np.nan for cp, ca in zip(cp_vals, ca_vals)])
        return np.nan
    return sum(valid_values) if valid_values else np.nan
def create_group_representation(group_entities, company_name_map=None):
    if company_name_map is None:
        company_name_map = {}
    html_content = ""
    for i, entity in enumerate(group_entities):
        if isinstance(entity, dict):
            subgroup_html = f"""
            <div class="group-card">
                <strong>🗂️ {entity['name']}</strong>
                <div style="margin-left: 1rem;">
                    {create_group_representation(entity['companies'], company_name_map)}
                </div>
            </div>
            """
            html_content += subgroup_html
        else:
            display_name = company_name_map.get(str(entity), str(entity))
            html_content += f"""
            <div class="company-card">
                🏢 {display_name}
            </div>
            """
    return html_content
def flatten_group_entities(group_entities):
    companies = []
    for entity in group_entities:
        if isinstance(entity, dict) and 'companies' in entity:
            companies.extend(flatten_group_entities(entity['companies']))
        elif isinstance(entity, str):
            companies.append(entity)
    return companies
class GroupManager:
    def __init__(self):
        self.groups = {}
        self.company_to_group = {}
    def create_group(self, name, companies=None):
        if companies is None:
            companies = []
        clean_companies = []
        for company in companies:
            if isinstance(company, str) and company.strip():
                clean_companies.append(company.strip())
        self.groups[name] = {'name': name, 'companies': clean_companies}
        for company in clean_companies:
            self.company_to_group[company] = name
    def get_group_data(self, group_name, df, metric_key, year):
        if df is None or df.empty:
            return np.nan
        if metric_key not in METRIC_TOKENS:
            st.warning(f"Metric key '{metric_key}' not found in METRIC_TOKENS. Available: {list(METRIC_TOKENS.keys())}")
            return np.nan
        group = self.groups.get(group_name)
        if not isinstance(group, dict) or 'companies' not in group:
            st.warning(f"Group '{group_name}' malformed, resetting to empty.")
            self.groups[group_name] = {'name': group_name, 'companies': []}
            group = self.groups[group_name]
        clean_companies = []
        for company in group['companies']:
            if isinstance(company, str) and company.strip():
                clean_companies.append(company.strip())
        group['companies'] = clean_companies
        return aggregate_group_data(clean_companies, df, metric_key, year)
    def get_all_groups(self):
        return list(self.groups.keys())
    def save_session(self):
        st.session_state.groups = {
            'groups': self.groups,
            'company_to_group': self.company_to_group
        }
    def load_session(self):
        if 'groups' in st.session_state:
            self.groups = st.session_state.groups.get('groups', {})
            self.company_to_group = st.session_state.groups.get('company_to_group', {})
            for group_name, group_data in list(self.groups.items()):
                if not isinstance(group_data, dict) or 'companies' not in group_data:
                    self.groups[group_name] = {'name': group_name, 'companies': []}
                else:
                    clean_companies = []
                    for company in group_data.get('companies', []):
                        if isinstance(company, str) and company.strip():
                            clean_companies.append(company.strip())
                    group_data['companies'] = clean_companies
if 'group_manager' not in st.session_state:
    st.session_state.group_manager = GroupManager()
group_manager = st.session_state.group_manager
group_manager.load_session()
def plot_multi_metric(
    df,
    years,
    var_name,
    cols,
    yaxis_title,
    chart_title=None,
    multi_sectors=None,
    color_map=None,
    width=0.8,
    template=None,
    font_size=18,
):
    if multi_sectors is None:
        multi_sectors = df['Secteur'].unique().tolist()
    if color_map is None:
        palette = px.colors.qualitative.Plotly
        color_map = {s: palette[i % len(palette)] for i, s in enumerate(multi_sectors)}
    bar_count = len(multi_sectors)
    bar_width = 0.25
    fig = go.Figure()
    for i, sector in enumerate(multi_sectors):
        sector_df = df[df["Secteur"] == sector]
        if sector_df.empty:
            continue
        vals = [float(sector_df.get(col, pd.Series(dtype=float)).mean(skipna=True)) for col in cols]
        vals = np.array(vals, dtype=float)
        texts = [format_number(v, percent=("marge" in var_name.lower())) for v in vals]
        offsetgroup = f"{sector}_{var_name}"
        scatter_offset = (i - bar_count/2 + 0.5) * bar_width + 0.13
        fig.add_trace(go.Bar(
            x=years,
            y=vals,
            name=f"{sector} – {var_name}",
            marker_color=color_map[sector],
            width=bar_width,
            offset=0,
            text=texts,
            textposition="auto",
            textfont=dict(size=font_size-4),
            offsetgroup=offsetgroup
        ))
        pct_change = np.array([np.nan] + list(vals[1:] / vals[:-1] - 1)) if len(vals)>1 else np.array([np.nan]*len(vals))
        fig.add_trace(go.Scatter(
            x=[y + scatter_offset for y in years],
            y=vals,
            mode="lines+text",
            name=f"{sector} – Var {var_name}",
            line=dict(shape='spline', dash="dot", color=color_map[sector]),
            text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct_change],
            textposition="top center",
            textfont=dict(size=font_size-6),
            hovertemplate=f"%{{x}}<br><b>{sector}</b>: %{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=chart_title or var_name, font=dict(size=font_size+8),),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=600,
        template=template,
        font=dict(size=font_size),
        xaxis=dict(tickmode="array", tickvals=years, title=dict(text="Année", font=dict(size=font_size)), tickfont=dict(size=font_size), showgrid=False),
        yaxis=dict(title=yaxis_title, tickformat="~s", showgrid=False, gridcolor="rgba(200,200,200,0.3)", tickfont=dict(size=font_size), visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=font_size-2)),
        hovermode="x unified",
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
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
    if not isinstance(keyword_tokens, (list, tuple)):
        keyword_tokens = [str(keyword_tokens)]
    for i, col in enumerate(cols):
        if pd.isna(col):
            continue
        cl = str(col).lower()
        if all(str(tok).lower() in cl for tok in keyword_tokens if tok):
            if year is None or str(year) in cl:
                return col
    return None
def get_column_for(metric_key, year):
    if metric_key not in METRIC_TOKENS:
        st.warning(f"Invalid metric_key '{metric_key}'. Available metrics: {list(METRIC_TOKENS.keys())}")
        return None
    tokens = METRIC_TOKENS[metric_key]
    col = find_col(tokens, year)
    if not col and year is not None:
        col = find_col(tokens, None)
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
def build_metric_matrix(df_input, metric_key):
    if df_input is None or df_input.empty:
        return pd.DataFrame(), {}
    if metric_key not in METRIC_TOKENS:
        st.warning(f"Invalid metric_key '{metric_key}' in build_metric_matrix. Available: {list(METRIC_TOKENS.keys())}")
        return pd.DataFrame({COMPANY_COL: df_input[COMPANY_COL]}), {}
    cols_found = {}
    tokens = METRIC_TOKENS[metric_key]
    for y in YEARS:
        col = get_column_for(metric_key, y)
        cols_found[y] = col
    matrix = pd.DataFrame({COMPANY_COL: df_input[COMPANY_COL]})
    for y in YEARS:
        c = cols_found[y]
        if c is not None and c in df_input.columns:
            matrix[str(y)] = safe_to_numeric(df_input[c])
        else:
            matrix[str(y)] = np.nan
    return matrix, cols_found
if sector_choice == "Tous":
    sector_df = df
else:
    sector_df = df[df[SECTOR_COL] == sector_choice]
def compute_company_cagrs(df_input, metric_key):
    try:
        if df_input is None or df_input.empty:
            return []
        if metric_key not in METRIC_TOKENS:
            st.warning(f"Invalid metric_key '{metric_key}' in compute_company_cagrs. Available: {list(METRIC_TOKENS.keys())}")
            return []
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
    except Exception as e:
        st.error(f"Error computing CAGR for {metric_key}: {e}")
        return []
cagr_results = {}
cagr_display = []
for key in ["CA", "RE", "CP"]:
    try:
        cagr_list = compute_company_cagrs(sector_df, key)
        mean_cagr = np.mean(cagr_list) if len(cagr_list) > 0 else np.nan
        if isinstance(mean_cagr, complex):
            mean_cagr = np.nan
        cagr_results[key] = mean_cagr
        cagr_display.append((key, mean_cagr, len(cagr_list)))
    except Exception as e:
        st.error(f"Failed to compute CAGR for {key}: {e}")
        cagr_results[key] = np.nan
        cagr_display.append((key, np.nan, 0))
st.markdown("---")
def calculate_cagr(start, end, periods):
    try:
        start = float(start) if pd.notna(start) else 0.0
        end = float(end) if pd.notna(end) else 0.0
        periods = float(periods) if pd.notna(periods) else 3
        if periods <= 0:
            periods = 3
        if start <= 0:
            return np.nan
        if end <= 0:
            return np.nan
        cagr = (end / start) ** (1 / periods) - 1
        return float(cagr)
    except Exception:
        return 0.0
st.markdown("### 👥 Gestion des Groupes")
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        new_group_name = st.text_input("📝 Nom du nouveau groupe", placeholder="ex: Groupe OCP, Groupe Attijari")
    with col2:
        if st.button("➕ Créer", type="primary", disabled=not new_group_name or new_group_name.strip() == ""):
            if new_group_name not in group_manager.groups:
                group_manager.create_group(new_group_name, [])
                group_manager.save_session()
                st.success(f"✅ Groupe '{new_group_name}' créé!")
                st.rerun()
            else:
                st.error("❌ Nom de groupe déjà utilisé")
if group_manager.groups:
    for group_name in group_manager.get_all_groups():
        with st.expander(f"👥 {group_name}", expanded=False):
            if group_name not in group_manager.groups or 'companies' not in group_manager.groups[group_name]:
                group_manager.groups[group_name] = {'name': group_name, 'companies': []}
                group_manager.save_session()
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                current_companies = group_manager.groups[group_name].get('companies', [])
                if current_companies:
                    st.write(f"Entreprises ({len(current_companies)}):")
                    for company in current_companies:
                        if isinstance(company, str):
                            comp_exists = not df[df[COMPANY_COL].str.strip() == company.strip()].empty
                            status_icon = "✅" if comp_exists else "⚠️"
                            st.write(f"{status_icon} {company}")
                else:
                    st.info("Aucune entreprise dans ce groupe")
                all_companies = sorted(df[COMPANY_COL].dropna().astype(str).str.strip().unique())
                grouped_companies = set()
                for key in group_manager.company_to_group.keys():
                    if isinstance(key, str):
                        grouped_companies.add(key.strip())
                available_companies = [c for c in all_companies
                                     if c.strip() not in grouped_companies or
                                     group_manager.company_to_group.get(c.strip()) == group_name]
                if available_companies:
                    selected_to_add = st.multiselect(
                        "Ajouter des entreprises:",
                        available_companies,
                        key=f"add_{group_name}"
                    )
                    if selected_to_add and st.button(f"➕ Ajouter {len(selected_to_add)} entreprise(s)", key=f"add_btn_{group_name}"):
                        current_companies = group_manager.groups[group_name].get('companies', [])
                        for company in selected_to_add:
                            if isinstance(company, str) and company.strip() not in current_companies:
                                current_companies.append(company.strip())
                                group_manager.company_to_group[company.strip()] = group_name
                        group_manager.groups[group_name]['companies'] = current_companies
                        group_manager.save_session()
                        st.success(f"✅ {len(selected_to_add)} entreprise(s) ajoutée(s)")
                        st.rerun()
                if current_companies:
                    selected_to_remove = st.multiselect(
                        "Supprimer des entreprises:",
                        [c for c in current_companies if isinstance(c, str)],
                        key=f"remove_{group_name}"
                    )
                    if selected_to_remove and st.button(f"🗑️ Supprimer {len(selected_to_remove)} entreprise(s)", key=f"remove_btn_{group_name}", type="secondary"):
                        current_companies = [c for c in current_companies if c not in selected_to_remove]
                        for company in selected_to_remove:
                            if isinstance(company, str):
                                group_manager.company_to_group.pop(company.strip(), None)
                        group_manager.groups[group_name]['companies'] = current_companies
                        group_manager.save_session()
                        st.success(f"✅ {len(selected_to_remove)} entreprise(s) supprimée(s)")
                        st.rerun()
            with col3:
                if st.button("🗑️ Supprimer groupe", key=f"delete_{group_name}", type="secondary"):
                    current_companies = group_manager.groups[group_name].get('companies', [])
                    for company in current_companies:
                        if isinstance(company, str):
                            group_manager.company_to_group.pop(company.strip(), None)
                    if group_name in group_manager.groups:
                        del group_manager.groups[group_name]
                    group_manager.save_session()
                    st.success(f"✅ Groupe '{group_name}' supprimé!")
                    st.rerun()
grouped_companies = set()
for key in list(group_manager.company_to_group.keys()):
    if isinstance(key, str) and key.strip():
        grouped_companies.add(key.strip())
display_mode = st.radio("Mode d'affichage", ["Entreprise individuelle", "Groupe d'entreprises"], key="radio-Mode d'affichage")
if display_mode == "Entreprise individuelle":
    st.subheader("A. Vue entreprise individuelle")
    company_single = st.selectbox(
        "Sélectionner une entreprise pour détails",
        ["Aucune"] + sorted(df[COMPANY_COL].dropna().unique())
    )
    if company_single != "Aucune":
        comp_df = df[df[COMPANY_COL] == company_single]
        if comp_df.empty:
            st.write("Entreprise non trouvée.")
        else:
            ca_col = f"Chiffre d'affaires {metric_display_year} (Dhs)"
            re_col = f"Resultat d'exploitation {metric_display_year} (Dhs)"
            cp_col = f"Charges personnel {metric_display_year}"
            ebit_col = f"Marge EBIT/CA {metric_display_year}"
            ca_val = comp_df[ca_col].values[0] if ca_col in comp_df.columns else np.nan
            re_val = comp_df[re_col].values[0] if re_col in comp_df.columns else np.nan
            cp_val = comp_df[cp_col].values[0] if cp_col in comp_df.columns else np.nan
            ebit_val = comp_df[ebit_col].values[0] if ebit_col in comp_df.columns else np.nan
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(f"Chiffre d'affaires {metric_display_year}",
                       f"{ca_val/1e9:.2f} B Dhs" if pd.notna(ca_val) and ca_val>=1e9 else
                       f"{ca_val/1e6:.2f} M Dhs" if pd.notna(ca_val) and ca_val>=1e6 else
                       f"{ca_val:,.0f} Dhs" if pd.notna(ca_val) else "N/A")
            col2.metric(f"Résultat d'exploitation {metric_display_year}",
                       f"{re_val/1e9:.2f} B Dhs" if pd.notna(re_val) and re_val>=1e9 else
                       f"{re_val/1e6:.2f} M Dhs" if pd.notna(re_val) and re_val>=1e6 else
                       f"{re_val:,.0f} Dhs" if pd.notna(re_val) else "N/A")
            col3.metric(f"Charges personnel {metric_display_year}",
                       f"{cp_val/1e9:.2f} B Dhs" if pd.notna(cp_val) and cp_val>=1e9 else
                       f"{cp_val/1e6:.2f} M Dhs" if pd.notna(cp_val) and cp_val>=1e6 else
                       f"{cp_val:,.0f} Dhs" if pd.notna(cp_val) else "N/A")
            col4.metric(f"Marge EBIT/CA (%) {metric_display_year}",
                       f"{ebit_val*100:.2f}%" if pd.notna(ebit_val) else "N/A")
            ca_cols = [f"Chiffre d'affaires {y} (Dhs)" for y in YEARS]
            re_cols = [f"Resultat d'exploitation {y} (Dhs)" for y in YEARS]
            cp_cols = [f"Charges personnel {y}" for y in YEARS]
            ca_values = []
            re_values = []
            cp_values = []
            for col in ca_cols:
                if col in comp_df.columns:
                    ca_values.append(comp_df[col].iloc[0] if pd.notna(comp_df[col].iloc[0]) else 0)
                else:
                    ca_values.append(0)
            for col in re_cols:
                if col in comp_df.columns:
                    re_values.append(comp_df[col].iloc[0] if pd.notna(comp_df[col].iloc[0]) else 0)
                else:
                    re_values.append(0)
            for col in cp_cols:
                if col in comp_df.columns:
                    cp_values.append(comp_df[col].iloc[0] if pd.notna(comp_df[col].iloc[0]) else 0)
                else:
                    cp_values.append(0)
            ca_values = np.array(ca_values, dtype=float)
            re_values = np.array(re_values, dtype=float)
            n_years = len(YEARS) - 1
            cagr_ca = calculate_cagr(ca_values[0], ca_values[-1], n_years)
            cagr_re = calculate_cagr(re_values[0], re_values[-1], n_years)
            cp_values = np.array(cp_values, dtype=float)
            marge_values = np.array([re/ca if ca != 0 else np.nan for re, ca in zip(re_values, ca_values)])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=YEARS,
                y=ca_values,
                name="CA",
                marker_color="#052449",
                text=[f"{v/1e9:.2f} B" if v>=1e9 else f"{v/1e6:.2f} M" if v>=1e6 else f"{v:,.0f}" for v in ca_values],
                textposition="auto",
                textfont=dict(size=18),
                width=0.25
            ))
            fig.add_trace(go.Bar(
                x=YEARS,
                y=re_values,
                name="RE",
                marker_color="#0064EF",
                text=[f"{v/1e9:.2f} B" if v>=1e9 else f"{v/1e6:.2f} M" if v>=1e6 else f"{v:,.0f}" for v in re_values],
                textposition="auto",
                textfont=dict(size=18),
                width=0.25
            ))
            fig.add_trace(go.Scatter(
                x=YEARS,
                y=marge_values * 100,
                mode="lines+text",
                name="Marge EBIT/CA",
                line=dict(shape='spline', color="#2681FF", width=5),
                text=[f"{v*100:.2f}%" if not pd.isna(v) else "" for v in marge_values],
                textposition="top center",
                textfont=dict(size=18),
                yaxis="y2"
            ))
            fig.add_annotation(
                text=f"CAGR CA: {(cagr_ca*100):.2f}%",
                xref="paper", yref="paper", x=0.4, y=1.25,
                showarrow=False,
                font=dict(size=14, color="blue"),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="blue",
                borderwidth=1, borderpad=4,
            )
            fig.add_annotation(
                text=f"CAGR RE: {(cagr_re*100):.2f}%",
                xref="paper", yref="paper", x=0.55, y=1.25,
                showarrow=False,
                font=dict(size=14, color="red"),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="red",
                borderwidth=1, borderpad=4,
            )
            fig.update_layout(
                title=dict(text=f"CA & RE avec Marge EBIT/CA - {company_single}", font=dict(size=22)),
                xaxis=dict(tickmode="array", tickvals=YEARS, title="Année", showgrid=False, tickfont=dict(size=18)),
                yaxis=dict(title="Valeurs (Dhs)", showgrid=False, tickfont=dict(size=18), visible=False),
                yaxis2=dict(title="Marge EBIT/CA (%)", overlaying="y", side="right", showgrid=False, tickfont=dict(size=18), visible=False),
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0, font=dict(size=18)),
                height=600,
                margin=dict(l=80, r=80, t=200, b=80),
                font=dict(size=18),
                hovermode="x unified",
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            metrics_for_trends = {
                "Chiffre d'affaires": [f"Chiffre d'affaires {y} (Dhs)" for y in YEARS],
                "Résultat d'exploitation": [f"Resultat d'exploitation {y} (Dhs)" for y in YEARS],
                "Charges personnel": [f"Charges personnel {y}" for y in YEARS],
                "Marge EBIT/CA (%)": [f"Marge EBIT/CA {y}" for y in YEARS],
                "Marge EBIT/CP (%)": [f"Marge EBIT/CP {y}" for y in YEARS],
                "Marge CP/CA (%)": [f"Marge CP/CA {y}" for y in YEARS],
            }
            available_metrics = {}
            for title, cols_list in metrics_for_trends.items():
                available_cols = [col for col in cols_list if col in comp_df.columns]
                if available_cols:
                    available_metrics[title] = available_cols
            for title, cols_list in available_metrics.items():
                values = []
                for col in cols_list:
                    if col in comp_df.columns:
                        val = comp_df[col].iloc[0]
                        if pd.notna(val):
                            if "Dhs" in col:
                                values.append(float(val))
                            else:
                                values.append(float(val))
                        else:
                            values.append(np.nan)
                    else:
                        values.append(np.nan)
                fig = go.Figure()
                multi_sectors = [company_single]
                color_map = {company_single: "#0064EF"}
                fig.add_trace(go.Bar(
                    x=YEARS[:len(values)],
                    y=values,
                    name=f"{company_single} – {title}",
                    marker_color=color_map[company_single],
                    text=[format_number(v, percent=("marge" in title.lower())) for v in values],
                    textposition="auto",
                    textfont=dict(size=18),
                    width=0.25
                ))
                if len(values) > 1 and "marge" not in title.lower():
                    pct_change = np.array([np.nan] + list(np.array(values)[1:] / np.array(values)[:-1] - 1))
                    fig.add_trace(go.Scatter(
                        x=YEARS[:len(values)],
                        y=values,
                        mode="lines+text",
                        name=f"{company_single} – Var {title}",
                        line=dict(shape="spline", dash="dot", color=color_map[company_single]),
                        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct_change],
                        textposition="top center",
                        textfont=dict(size=18),
                        yaxis="y2" if "marge" in title.lower() else "y"
                    ))
                if len(values) > 1 and "marge" in title.lower():
                    pct_change = np.array([np.nan] + list(np.array(values)[1:] / np.array(values)[:-1] - 1))
                    fig.add_trace(go.Scatter(
                        x=YEARS[:len(values)],
                        y=values,
                        mode="lines+text",
                        name=f"{company_single} – Var {title}",
                        line=dict(shape="spline", dash="dot", color=color_map[company_single]),
                        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct_change],
                        textposition="top center",
                        textfont=dict(size=18),
                        yaxis="y" if "marge" in title.lower() else "y"
                    ))
                fig.add_annotation(
                    text=f"CAGR: {(calculate_cagr(values[0], values[-1], n_years)*100):.2f}%",
                    xref="paper", yref="paper", x=0.5, y=1.15,
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="black",
                    borderwidth=1, borderpad=4,
                )
                fig.update_layout(
                    title=dict(text=f"{title} - {company_single}", font=dict(size=18)),
                    barmode="group",
                    height=600,
                    font=dict(size=18),
                    xaxis=dict(tickmode="array", tickvals=YEARS[:len(values)], title=dict(text="Année", font=dict(size=18)), tickfont=dict(size=18), showgrid=False),
                    yaxis=dict(title=title, tickformat="~s" if "marge" not in title.lower() else ".1%", showgrid=False, tickfont=dict(size=18), visible=False),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="right", x=0.75, font=dict(size=18)),
                    hovermode="x unified",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
            st.subheader("📊 Données annuelles complètes")
            def format_value(val, var_name):
                if pd.isnull(val):
                    return val
                if "marge" in var_name.lower() or "variation" in var_name.lower():
                    return f"{val*100:.2f}%"
                elif isinstance(val, (int, float)):
                    return f"{val:,.0f}"
                else:
                    return val
            cols_to_show = [COMPANY_COL] + [
                col for col in comp_df.columns if any(str(y) in str(col) for y in YEARS)
            ]
            df_to_display = comp_df[cols_to_show].copy()
            for col in df_to_display.columns:
                if col != COMPANY_COL:
                    df_to_display[col] = df_to_display[col].apply(lambda v: format_value(v, str(col)))
            st.dataframe(df_to_display.T.rename(columns={comp_df.index[0]: company_single}), use_container_width=True)
elif display_mode == "Groupe d'entreprises":
    st.subheader("A. Vue groupe d'entreprises")
    existing_groups = group_manager.get_all_groups()
    if existing_groups:
        st.markdown("##### 📋 Groupes existants")
        selected_group = st.selectbox("Sélectionner un groupe", ["Aucun"] + existing_groups)
        if selected_group != "Aucun":
            group_data = group_manager.groups[selected_group]
            companies_in_group = group_data['companies']
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("#### Composition du groupe")
                company_map = {c: c for c in df[COMPANY_COL].dropna().unique()}
                html_representation = create_group_representation(companies_in_group, company_map)
                st.markdown(html_representation, unsafe_allow_html=True)
            with col2:
                if st.button(f"✏️ Modifier {selected_group}"):
                    with st.form("edit_group"):
                        new_name = st.text_input("Nouveau nom", value=selected_group)
                        edited_companies = st.multiselect(
                            "Entreprises du groupe",
                            [c for c in df[COMPANY_COL].dropna().unique() if pd.notna(c)],
                            default=companies_in_group,
                            key="edit_group_select"
                        )
                        edit_button = st.form_submit_button("Sauvegarder")
                        delete_button = st.form_submit_button("Supprimer le groupe", type="secondary")
                        if edit_button and new_name:
                            if new_name in group_manager.groups and new_name != selected_group:
                                st.error("Ce nom existe déjà!")
                            else:
                                if new_name != selected_group:
                                    del group_manager.groups[selected_group]
                                    group_manager.groups[new_name] = {'name': new_name, 'companies': edited_companies}
                                else:
                                    group_manager.groups[selected_group]['companies'] = edited_companies
                                group_manager.save_session()
                                st.success("Groupe mis à jour!")
                                st.rerun()
                        if delete_button:
                            del group_manager.groups[selected_group]
                            group_manager.save_session()
                            st.success(f"Groupe '{selected_group}' supprimé!")
                            st.rerun()
            st.markdown("---")
            if companies_in_group:
                ca_col = f"Chiffre d'affaires {metric_display_year} (Dhs)"
                re_col = f"Resultat d'exploitation {metric_display_year} (Dhs)"
                cp_col = f"Charges personnel {metric_display_year}"
                ebit_col = f"Marge EBIT/CA {metric_display_year}"
                ca_val = aggregate_group_data(companies_in_group, df, "CA", metric_display_year)
                re_val = aggregate_group_data(companies_in_group, df, "RE", metric_display_year)
                cp_val = aggregate_group_data(companies_in_group, df, "CP", metric_display_year)
                ca_val_for_ebit = aggregate_group_data(companies_in_group, df, "CA", metric_display_year)
                re_val_for_ebit = aggregate_group_data(companies_in_group, df, "RE", metric_display_year)
                ebit_val = (re_val_for_ebit / ca_val_for_ebit * 100) if ca_val_for_ebit != 0 else np.nan
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(f"Chiffre d'affaires {metric_display_year} - {selected_group}",
                           f"{ca_val/1e9:.2f} B Dhs" if pd.notna(ca_val) and ca_val>=1e9 else
                           f"{ca_val/1e6:.2f} M Dhs" if pd.notna(ca_val) and ca_val>=1e6 else
                           f"{ca_val:,.0f} Dhs" if pd.notna(ca_val) else "N/A")
                col2.metric(f"Résultat d'exploitation {metric_display_year} - {selected_group}",
                           f"{re_val/1e9:.2f} B Dhs" if pd.notna(re_val) and re_val>=1e9 else
                           f"{re_val/1e6:.2f} M Dhs" if pd.notna(re_val) and re_val>=1e6 else
                           f"{re_val:,.0f} Dhs" if pd.notna(re_val) else "N/A")
                col3.metric(f"Charges personnel {metric_display_year} - {selected_group}",
                           f"{cp_val/1e9:.2f} B Dhs" if pd.notna(cp_val) and cp_val>=1e9 else
                           f"{cp_val/1e6:.2f} M Dhs" if pd.notna(cp_val) and cp_val>=1e6 else
                           f"{cp_val:,.0f} Dhs" if pd.notna(cp_val) else "N/A")
                col4.metric(f"Marge EBIT/CA (%) {metric_display_year} - {selected_group}",
                           f"{ebit_val:.2f}%" if pd.notna(ebit_val) else "N/A")
                ca_values = [aggregate_group_data(companies_in_group, df, "CA", y) for y in YEARS]
                re_values = [aggregate_group_data(companies_in_group, df, "RE", y) for y in YEARS]
                cp_values = [aggregate_group_data(companies_in_group, df, "CP", y) for y in YEARS]
                n_years = len(YEARS) - 1
                cagr_ca = calculate_cagr(ca_values[0], ca_values[-1], n_years)
                cagr_re = calculate_cagr(re_values[0], re_values[-1], n_years)
                marge_values = np.array([aggregate_group_data(companies_in_group, df, "EBIT_CA", y) for y in YEARS])
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=YEARS,
                    y=ca_values,
                    name=f"CA - {selected_group}",
                    marker_color="#052449",
                    text=[f"{v/1e9:.2f} B" if v>=1e9 else f"{v/1e6:.2f} M" if v>=1e6 else f"{v:,.0f}" for v in ca_values],
                    textposition="auto",
                    textfont=dict(size=18),
                    width=0.25
                ))
                fig.add_trace(go.Bar(
                    x=YEARS,
                    y=re_values,
                    name=f"RE - {selected_group}",
                    marker_color="#0064EF",
                    text=[f"{v/1e9:.2f} B" if v>=1e9 else f"{v/1e6:.2f} M" if v>=1e6 else f"{v:,.0f}" for v in re_values],
                    textposition="auto",
                    textfont=dict(size=18),
                    width=0.25
                ))
                fig.add_trace(go.Scatter(
                    x=YEARS,
                    y=marge_values * 100,
                    mode="lines+text",
                    name=f"Marge EBIT/CA - {selected_group}",
                    line=dict(shape='spline', color="#2681FF", width=5),
                    text=[f"{v*100:.2f}%" if not pd.isna(v) else "" for v in marge_values],
                    textposition="top center",
                    textfont=dict(size=18),
                    yaxis="y2"
                ))
                fig.add_annotation(
                    text=f"CAGR CA: {(cagr_ca*100):.2f}%",
                    xref="paper", yref="paper", x=0.40, y=1.10,
                    showarrow=False,
                    font=dict(size=14, color="blue"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="blue",
                    borderwidth=1, borderpad=4,
                )
                fig.add_annotation(
                    text=f"CAGR RE: {(cagr_re*100):.2f}%",
                    xref="paper", yref="paper", x=0.55, y=1.10,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="red",
                    borderwidth=1, borderpad=4,
                )
                fig.update_layout(
                    title=dict(text=f"CA & RE avec Marge EBIT/CA - {selected_group}", font=dict(size=22)),
                    xaxis=dict(tickmode="array", tickvals=YEARS, title="Année", showgrid=False, tickfont=dict(size=18)),
                    yaxis=dict(title="Valeurs (Dhs)", showgrid=False, tickfont=dict(size=18), visible=False),
                    yaxis2=dict(title="Marge EBIT/CA (%)", overlaying="y", side="right", showgrid=False, tickfont=dict(size=18), visible=False),
                    barmode="group",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=18)),
                    height=600,
                    font=dict(size=18),
                    hovermode="x unified",
                )
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                metrics_for_trends = {
                    "Chiffre d'affaires": ["CA"],
                    "Résultat d'exploitation": ["RE"],
                    "Charges personnel": ["CP"],
                    "Marge EBIT/CA (%)": ["EBIT_CA"],
                    "Marge EBIT/CP (%)": ["EBIT_CP"],
                    "Marge CP/CA (%)": ["CP_CA"],
                }
                for title, metric_keys in metrics_for_trends.items():
                    metric_key = metric_keys[0]
                    values = [aggregate_group_data(companies_in_group, df, metric_key, y) for y in YEARS]
                    values = [v if pd.notna(v) else np.nan for v in values]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=YEARS[:len(values)],
                        y=values,
                        name=f"{selected_group} – {title}",
                        marker_color="#0064EF",
                        text=[format_number(v, percent=("marge" in title.lower())) for v in values],
                        textposition="auto",
                        textfont=dict(size=18),
                        width=0.25
                    ))
                    if len(values) > 1 and "marge" not in title.lower():
                        valid_values = [v for v in values if pd.notna(v)]
                        if len(valid_values) > 1:
                            pct_change = np.array([np.nan] + list(np.array(valid_values)[1:] / np.array(valid_values)[:-1] - 1))
                            fig.add_trace(go.Scatter(
                                x=YEARS[:len(values)],
                                y=values,
                                mode="lines+text",
                                name=f"{selected_group} – Var {title}",
                                line=dict(shape="spline",dash="dot", color="#0049AF"),
                                text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct_change],
                                textposition="top center",
                                textfont=dict(size=18),
                                yaxis="y2" if "marge" in title.lower() else "y"
                            ))
                    if len(values) > 1 and "marge" in title.lower():
                        pct_change = np.array([np.nan] + list(np.array(values)[1:] / np.array(values)[:-1] - 1))
                        fig.add_trace(go.Scatter(
                            x=YEARS[:len(values)],
                            y=values,
                            mode="lines+text",
                            name=f"{selected_group} – Var {title}",
                            line=dict(shape="spline", dash="dot", color="#0049AF"),
                            text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct_change],
                            textposition="top center",
                            textfont=dict(size=18),
                            yaxis="y" if "marge" in title.lower() else "y"
                        ))
                    fig.add_annotation(
                        text=f"CAGR: {(calculate_cagr(values[0], values[-1], n_years)*100):.2f}%",
                        xref="paper", yref="paper", x=0.5, y=1.15,
                        showarrow=False,
                        font=dict(size=16, color="black"),
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="black",
                        borderwidth=1, borderpad=4,
                    )
                    fig.update_layout(
                        title=dict(text=f"{title} - {selected_group}", font=dict(size=18)),
                        barmode="group",
                        height=600,
                        font=dict(size=18),
                        xaxis=dict(tickmode="array", tickvals=YEARS[:len(values)], title=dict(text="Année", font=dict(size=18)), tickfont=dict(size=18), showgrid=False),
                        yaxis=dict(title=title, tickformat="~s" if "marge" not in title.lower() else ".1%", showgrid=False, tickfont=dict(size=18), visible=False),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(size=18)),
                        hovermode="x unified",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
                st.subheader(f"📊 Données annuelles - {selected_group}")
                data_dict = {}
                for y in YEARS:
                    data_dict[y] = {
                        'CA': aggregate_group_data(companies_in_group, df, "CA", y),
                        'RE': aggregate_group_data(companies_in_group, df, "RE", y),
                        'CP': aggregate_group_data(companies_in_group, df, "CP", y),
                        'EBIT_CA': aggregate_group_data(companies_in_group, df, "EBIT_CA", y),
                        'EBIT_CP': aggregate_group_data(companies_in_group, df, "EBIT_CP", y),
                        'CP_CA': aggregate_group_data(companies_in_group, df, "CP_CA", y),
                    }
                df_group = pd.DataFrame(data_dict).T
                df_group.index.name = "Année"
                def format_cell(value, col_name):
                    if col_name in ["EBIT_CA", "EBIT_CP", "CP_CA"]:
                        return format_number(value, percent=True)
                    else:
                        return format_number(value)
                df_group_formatted = df_group.apply(lambda col: col.map(lambda v: format_cell(v, col.name)))
                st.dataframe(df_group_formatted, use_container_width=True)
    else:
        st.info("Aucun groupe créé. Utilisez le formulaire ci-dessus pour en créer un.")
st.markdown("---")
st.subheader("B. Vue entreprise comparative")
brand_colors = ["#052449", "#0064EF", "#428DF2", "#375F92", "#003967", "#FF6B6B", "#4ECDC4", "#45B7D1"]
available_individual_companies = [
    c.strip() for c in sorted(df[COMPANY_COL].dropna().astype(str).str.strip().unique())
    if c.strip() not in grouped_companies
]
st.markdown("##### 🔄 Sélection pour comparaison")
all_comparison_options = available_individual_companies + [f"👥 {g}" for g in sorted(group_manager.get_all_groups())]
selected_for_comparison = st.multiselect(
    "Entreprises ET groupes à comparer:",
    all_comparison_options,
    placeholder="Choisissez ce que vous voulez comparer...",
    help="Mélangez entreprises individuelles et groupes"
)
comparison_entities = []
for entity in selected_for_comparison:
    if entity.startswith("👥 "):
        group_name = entity[2:].strip()
        if group_name:
            comparison_entities.append(group_name)
    else:
        entity_clean = entity.strip()
        if entity_clean:
            comparison_entities.append(entity_clean)
selected_companies = [e for e in comparison_entities if e not in group_manager.groups]
selected_groups = [e for e in comparison_entities if e in group_manager.groups]
def plot_multi_companies(
    df,
    years,
    var_name,
    cols,
    yaxis_title,
    chart_title=None,
    company_list=None,
    color_map=None,
    total_width=0.8,
    template=None,
    font_size=12,
):
    if company_list is None:
        company_list = df[COMPANY_COL].unique().tolist()
    if color_map is None:
        palette = px.colors.qualitative.Plotly
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(company_list)}
    fig = go.Figure()
    n_companies = len(company_list)
    bar_width = total_width / n_companies
    for i, company in enumerate(company_list):
        if not isinstance(company, str):
            continue
        comp_df = df[df[COMPANY_COL].str.lower().str.strip() == company.lower().strip()]
        if comp_df.empty:
            continue
        vals = [float(comp_df.get(col, pd.Series(dtype=float)).mean(skipna=True)) for col in cols]
        vals = np.array(vals, dtype=float)
        texts = [format_number(v, percent=("marge" in var_name.lower())) for v in vals]
        x_offset = np.array(years) + (i - n_companies / 2 + 0.5) * bar_width
        fig.add_trace(go.Bar(
            x=x_offset,
            y=vals,
            name=company,
            marker_color=color_map[company],
            width=bar_width,
            text=texts,
            textposition="auto"
        ))
        if "marge" in var_name.lower() or "variation" in var_name.lower():
            pct_change = np.array([np.nan] + list(vals[1:] / vals[:-1] - 1)) if len(vals) > 1 else np.array([np.nan]*len(vals))
            fig.add_trace(go.Scatter(
                x=x_offset,
                y=vals * 100 if "marge" in var_name.lower() else vals,
                mode="lines+text",
                name=f"{company} – Var {var_name}",
                line=dict(shape='spline', dash="dot", color=color_map[company]),
                text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct_change],
                textposition="top center",
                yaxis="y2" if "marge" in var_name.lower() else "y"
            ))
    fig.update_layout(
        title=dict(text=chart_title or var_name, font=dict(size=font_size+4)),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=500,
        template=template,
        font=dict(size=font_size),
        xaxis=dict(tickmode="array", tickvals=years, title=dict(text="Année", font=dict(size=font_size)), showgrid=False, visible=False, showticklabels=False),
        yaxis=dict(title=yaxis_title, tickformat="~s", showgrid=False, visible=False, showticklabels=False),
        yaxis2=dict(title="Marge (%)", overlaying="y", side="right", showgrid=False, visible=False, showticklabels=False) if "marge" in var_name.lower() else {},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
def adjust_color_brightness(hex_color, factor=1.2):
    try:
        hex_color = str(hex_color).lstrip("#")
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        l = max(0, min(1, l * factor))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    except:
        return hex_color
def plot_enhanced_comparison(df, years, entities, color_map=None):
    if not entities or df is None or df.empty:
        st.warning("No valid entities or data available for comparison")
        return
    if color_map is None:
        color_map = {entity: brand_colors[i % len(brand_colors)] for i, entity in enumerate(entities)}
    fig = go.Figure()
    annotations = []
    ann_step = 1 / (len(entities) + 1)
    for i, entity in enumerate(entities):
        if not isinstance(entity, str) or not entity.strip():
            continue
        entity = entity.strip()
        is_group = entity in group_manager.groups
        display_name = f"👥 {entity}" if is_group else f"🏢 {entity}"
        ca_vals, re_vals = [], []
        for y in years:
            if is_group:
                companies_in_group = group_manager.groups[entity].get("companies", [])                
                ca_val = re_val = 0
                for comp in companies_in_group:
                    comp_df = safe_find_company(df, comp, COMPANY_COL)
                    if not comp_df.empty:
                        ca_col = get_column_for("CA", y)
                        re_col = get_column_for("RE", y)
                        ca_val += safe_to_numeric(comp_df.iloc[0][ca_col]) if ca_col and ca_col in comp_df.columns else 0
                        re_val += safe_to_numeric(comp_df.iloc[0][re_col]) if re_col and re_col in comp_df.columns else 0
                if ca_val == 0:
                    ca_val = np.nan
                if re_val == 0:
                    re_val = np.nan

            else:
                comp_df = safe_find_company(df, entity, COMPANY_COL)
                if not comp_df.empty:
                    ca_col = get_column_for("CA", y)
                    re_col = get_column_for("RE", y)
                    ca_val = safe_to_numeric(comp_df.iloc[0][ca_col]) if ca_col else np.nan
                    re_val = safe_to_numeric(comp_df.iloc[0][re_col]) if re_col else np.nan
                else:
                    ca_val = re_val = np.nan
            ca_vals.append(ca_val)
            re_vals.append(re_val)

        margin_vals = [(re / ca) if pd.notna(re) and pd.notna(ca) and ca != 0 else np.nan
                    for re, ca in zip(re_vals, ca_vals)]

        base_color = color_map.get(entity, "#636EFA")
        margin_color = adjust_color_brightness(base_color, 0.9)
        fig.add_trace(go.Bar(
            x=years, y=ca_vals,
            name=f"{display_name} - CA",
            marker_color=base_color,
            text=[format_number(v) for v in ca_vals],
            textposition="auto",
            offsetgroup=f"{entity}_CA",
            textfont=dict(size=16),
        ))
        fig.add_trace(go.Scatter(
            x=years, y=[v * 100 if pd.notna(v) else np.nan for v in margin_vals],
            mode="lines+text",
            name=f"{display_name} - Marge EBIT/CA",
            line=dict(shape='spline', color=margin_color, width=3),
            text=[format_number(v, percent=True) for v in margin_vals],
            textposition="top center",
            yaxis="y2",
            textfont=dict(size=16, color="darkblue"),
        ))
        cagr_ca = calculate_cagr(ca_vals[0], ca_vals[-1], len(years) - 1)
        annotations.append(dict(
            text=f"{display_name}<br>CAGR CA: {cagr_ca*100:.2f}%",
            xref="paper", yref="paper",
            x=0.05 + i * ann_step, y=1.2,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,1)",
            bordercolor=base_color,
            borderwidth=1,
            font=dict(size=13)
        ))
    fig.update_layout(
        title=f"Comparaison CA et Marge EBIT/CA ({years[0]}-{years[-1]}) - {len(entities)} entités",
        xaxis=dict(title="Année", tickmode="array", tickvals=years, showgrid=False),
        yaxis=dict(title="Montants (MAD)", showgrid=False, visible=False),
        yaxis2=dict(title="Marge EBIT/CA (%)", overlaying="y", side="right", showgrid=False, visible=False),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=150, b=80),
        height=600 if len(entities) > 3 else 550,
        font=dict(size=12),
        hovermode="x unified",
        annotations=annotations,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
def plot_enhanced_multi_metrics(df, years, entities, metric_key, color_map=None):
    if not entities or df is None or df.empty:
        return
    if color_map is None:
        color_map = {entity: brand_colors[i % len(brand_colors)] for i, entity in enumerate(entities)}
    fig = go.Figure()
    n_entities = len([e for e in entities if isinstance(e, str) and e.strip()])
    if n_entities == 0:
        return
    bar_width = 0.5 / max(n_entities, 1)
    if metric_key not in METRIC_TOKENS:
        st.warning(f"Invalid metric_key '{metric_key}'. Available: {list(METRIC_TOKENS.keys())}")
        return
    for i, entity in enumerate(entities):
        if not isinstance(entity, str) or not entity.strip():
            continue
        entity = entity.strip()
        is_group = entity in group_manager.groups
        values = []
        for y in years:
            if is_group:
                value = group_manager.get_group_data(entity, df, metric_key, y)
            else:
                comp_df = safe_find_company(df, entity, COMPANY_COL)
                if not comp_df.empty:
                    col = get_column_for(metric_key, y)
                    if col and col in comp_df.columns:
                        value = safe_to_numeric(comp_df.iloc[0][col])
                    else:
                        value = np.nan
                else:
                    value = np.nan
            values.append(value)
        display_name = f"👥 {entity}" if is_group else f"🏢 {entity}"
        x_offset = np.array(years) + (i - n_entities / 2 + 0.5) * bar_width
        if metric_key in ["EBIT_CA", "EBIT_CP", "CP_CA"]:
            texts = [format_number(v, percent=True) if pd.notna(v) else "" for v in values]
        else:
            texts = [format_number(v) for v in values]
        if entity in color_map:
            fig.add_trace(go.Bar(
                x=x_offset,
                y=values,
                name=display_name,
                marker_color=color_map[entity],
                width=bar_width,
                text=texts,
                textposition="auto",
                textfont=dict(size=16),
            ))
            if len(values) > 1:
                valid_values = [v for v in values if pd.notna(v)]
                if len(valid_values) > 1 and metric_key not in ["EBIT_CA", "EBIT_CP", "CP_CA"]:
                    pct_change = np.array([np.nan] + list(np.array(valid_values)[1:] / np.array(valid_values)[:-1] - 1))
                    fig.add_trace(go.Scatter(
                        x=x_offset,
                        y=values,
                        mode="lines+text",
                        name=f"{display_name} - Var",
                        line=dict(shape='spline', dash="dot", color=adjust_color_brightness(color_map[entity], 0.8)),
                        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct_change],
                        textposition="top center",
                        textfont=dict(size=16),
                    ))
                if len(valid_values) > 1 and metric_key in ["EBIT_CA", "EBIT_CP", "CP_CA"]:
                    pct_change = np.array([np.nan] + list(np.array(valid_values)[1:] / np.array(valid_values)[:-1] - 1))
                    fig.add_trace(go.Scatter(
                        x=x_offset,
                        y=values,
                        mode="lines+text",
                        name=f"{display_name} - Var",
                        line=dict(shape='spline', dash="dot", color=adjust_color_brightness(color_map[entity], 0.8)),
                        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct_change],
                        textposition="top center",
                        textfont=dict(size=16),
                    ))
    metric_display_name = {
        "CA": "Chiffre d'affaires",
        "RE": "Résultat d'exploitation", 
        "CP": "Charges personnel",
        "EBIT_CA": "Marge EBIT/CA (%)",
        "EBIT_CP": "Marge EBIT/CP (%)",
        "CP_CA": "Marge CP/CA (%)"
    }
    yaxis_title = metric_display_name.get(metric_key, metric_key)
    if metric_key in ["EBIT_CA", "EBIT_CP", "CP_CA"]:
        yaxis_title = f"{yaxis_title}"
        tickformat = ".1%"
    else:
        tickformat = "~s"
    n_years = len(YEARS) - 1
    fig.update_layout(
        title=f"{yaxis_title} - Comparaison {len(entities)} entités",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=500,
        xaxis=dict(tickmode="array", tickvals=years, title="Année", showgrid=False),
        yaxis=dict(title=yaxis_title, tickformat=tickformat, showgrid=False, visible=False, showticklabels=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        hovermode="x unified",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
if selected_for_comparison:
    st.markdown(f"### 📈 Comparaison ({len(selected_for_comparison)} entités)")
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        entity_info = []
        for entity in selected_for_comparison:
            if entity.startswith("👥 "):
                group_name = entity[2:].strip()
                if group_name and group_name in group_manager.groups:
                    count = len([c for c in group_manager.groups[group_name].get('companies', []) if isinstance(c, str)])
                    entity_info.append(f"👥 {group_name} ({count} entreprises)")
                else:
                    entity_info.append(f"👥 {group_name or 'Inconnu'} (0 entreprises)")
            else:
                entity_info.append(f"🏢 {entity.strip()}")
        st.info(" | ".join(entity_info))
    if df is not None and not df.empty and comparison_entities:
        plot_enhanced_comparison(df, YEARS, comparison_entities)
    if len(comparison_entities) <= 6:
        metrics_to_plot = ["CA", "RE", "CP", "EBIT_CA", "EBIT_CP", "CP_CA"]
        color_map = {entity: brand_colors[i % len(brand_colors)] for i, entity in enumerate(comparison_entities)}
        for metric_key in metrics_to_plot:
            plot_enhanced_multi_metrics(
                df=df,
                years=YEARS,
                entities=comparison_entities,
                metric_key=metric_key,
                color_map=color_map
            )
        st.markdown("### 📋 Récapitulatif")
        summary_data = []
        for entity in comparison_entities:
            is_group = entity in group_manager.groups
            row = {"Entité": f"👥 {entity}" if is_group else f"🏢 {entity}",
                "Type": "Groupe" if is_group else "Entreprise"}
            for y in YEARS:
                if is_group:
                    ca = aggregate_group_data(group_manager.groups[entity]['companies'], df, "CA", y)
                    re = aggregate_group_data(group_manager.groups[entity]['companies'], df, "RE", y)
                    cp = aggregate_group_data(group_manager.groups[entity]['companies'], df, "CP", y)
                    ebit_ca = (re / ca * 100) if ca != 0 else np.nan
                    ebit_cp = (re / cp * 100) if cp != 0 else np.nan
                    cp_ca = (cp / ca * 100) if ca != 0 else np.nan

                    values = {"CA": ca, "RE": re, "CP": cp,
                            "EBIT_CA": ebit_ca, "EBIT_CP": ebit_cp, "CP_CA": cp_ca}
                else:
                    comp_df = df[df[COMPANY_COL].str.strip() == entity.strip()]
                    if comp_df.empty:
                        comp_df = df[df[COMPANY_COL].str.lower().str.strip() == entity.lower().strip()]
                    if not comp_df.empty:
                        values = {}
                        for metric in ["CA", "RE", "CP"]:
                            col = get_column_for(metric, y)
                            values[metric] = safe_to_numeric(comp_df.iloc[0][col]) if col and col in comp_df.columns else np.nan
                        values["EBIT_CA"] = (values["RE"] / values["CA"] * 100) if values["CA"] else np.nan
                        values["EBIT_CP"] = (values["RE"] / values["CP"] * 100) if values["CP"] else np.nan
                        values["CP_CA"] = (values["CP"] / values["CA"] * 100) if values["CA"] else np.nan
                    else:
                        values = {metric: np.nan for metric in ["CA","RE","CP","EBIT_CA","EBIT_CP","CP_CA"]}
                for metric in ["CA","RE","CP","EBIT_CA","EBIT_CP","CP_CA"]:
                    display_name = f"{metric} {y}"
                    if metric in ["EBIT_CA","EBIT_CP","CP_CA"]:
                        row[display_name] = f"{values[metric]:.2f}%" if pd.notna(values[metric]) else "N/A"
                    else:
                        row[display_name] = format_number(values[metric]) if pd.notna(values[metric]) else "N/A"
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

else:
    st.info("👆 Sélectionnez des entreprises ou groupes pour voir la comparaison")
group_manager.save_session()