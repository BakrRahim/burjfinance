import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from difflib import get_close_matches
import colorsys
import io
import re
from typing import List, Tuple, Dict, Any

try:
    from rapidfuzz import fuzz
    def fuzzy_ratio(a: str, b: str) -> float:
        a = str(a or "").lower()
        b = str(b or "").lower()
        if not a or not b:
            return 0.0
        return fuzz.token_sort_ratio(a, b) / 100.0
except Exception:
    import difflib
    def fuzzy_ratio(a: str, b: str) -> float:
        a = str(a or "").lower()
        b = str(b or "").lower()
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

def get_largest_ca_company_sector(group_companies, df, company_col, sector_col, year=2023):
    if not group_companies or df.empty or sector_col not in df.columns:
        return None    
    max_ca = 0
    largest_sector = None
    ca_col = f"Chiffre d'affaires {year} (Dhs)"
    if ca_col not in df.columns:
        return None
    for company in group_companies:
        comp_df = safe_find_company(df, company, company_col)
        if not comp_df.empty:
            ca_value = safe_to_numeric(comp_df.iloc[0][ca_col])
            if pd.notna(ca_value) and ca_value > max_ca:
                max_ca = ca_value
                largest_sector = str(comp_df.iloc[0][sector_col]).strip() if sector_col in comp_df.columns else None
    return largest_sector

def calculate_cagr(values, n_years=None):
    try:
        valid_values = [v for v in values if pd.notna(v) and isinstance(v, (int, float)) and v > 0]
        if len(valid_values) < 2:
            return np.nan
        start_val = valid_values[0]
        end_val = valid_values[-1]

        if n_years is None:
            n_years = len(values) - 1
        else:
            n_periods = len(valid_values) - 1
            if n_periods > 0:
                n_years = n_periods        
        if start_val <= 0 or end_val <= 0 or n_years <= 2:
            return np.nan
        cagr = (end_val / start_val) ** (1 / n_years) - 1
        return float(cagr)
    except Exception:
        return np.nan

def all_entities_same_sector(entities, df, company_col, sector_col):
    if not entities:
        return False, None
    
    sectors = set()
    for entity in entities:
        if entity in group_manager.groups:
            group_companies = group_manager.groups[entity].get('companies', [])
            for company in group_companies:
                sector = get_company_sector(company, df, company_col, sector_col)
                if sector:
                    sectors.add(sector)
        else:
            sector = get_company_sector(entity, df, company_col, sector_col)
            if sector:
                sectors.add(sector)
    
    if len(sectors) == 1:
        return True, list(sectors)[0]
    elif len(sectors) > 1:
        return False, None
    else:
        return False, None

def get_company_sector(company_name, df, company_col, sector_col):
    if company_name and sector_col and sector_col in df.columns:
        comp_df = safe_find_company(df, company_name, company_col)
        if not comp_df.empty and sector_col in comp_df.columns:
            return str(comp_df.iloc[0][sector_col]).strip()
    return None

def get_group_sector(group_companies, df, company_col, sector_col):
    if not group_companies:
        return None
    sectors = []
    for company in group_companies:
        sector = get_company_sector(company, df, company_col, sector_col)
        if sector:
            sectors.append(sector)
    if sectors:
        from collections import Counter
        sector_counts = Counter(sectors)
        return sector_counts.most_common(1)[0][0]
    return None

def get_sector_margin_values(sector_name, df, margin_type, years):
    if not sector_name or df.empty or SECTOR_COL not in df.columns:
        return [np.nan] * len(years)
    
    sector_df = df[df[SECTOR_COL] == sector_name]
    if sector_df.empty:
        return [np.nan] * len(years)
    
    if margin_type == "EBIT_CA":
        cols = [f"Marge EBIT/CA {y}" for y in years]
    elif margin_type == "EBIT_CP":
        cols = [f"Marge EBIT/CP {y}" for y in years]
    elif margin_type == "CP_CA":
        cols = [f"Marge CP/CA {y}" for y in years]
    else:
        return [np.nan] * len(years)
    
    available_cols = [col for col in cols if col in sector_df.columns]
    if not available_cols:
        return [np.nan] * len(years)
    
    values = []
    for col in cols:
        if col in sector_df.columns:
            valid_values = safe_to_numeric(sector_df[col].dropna())
            if len(valid_values) > 0:
                avg_value = valid_values.mean()
                values.append(avg_value)
            else:
                values.append(np.nan)
        else:
            values.append(np.nan)
    
    return values

def get_sector_financial_values(sector_name, df, metric_key, years):
    if not sector_name or df.empty or SECTOR_COL not in df.columns:
        return [np.nan] * len(years)
    
    sector_df = df[df[SECTOR_COL] == sector_name]
    if sector_df.empty:
        return [np.nan] * len(years)
    
    values = []
    for year in years:
        if metric_key == "CA":
            col = f"Chiffre d'affaires {year} (Dhs)"
        elif metric_key == "RE":
            col = f"Resultat d'exploitation {year} (Dhs)"
        elif metric_key == "CP":
            col = f"Charges personnel {year}"
        else:
            values.append(np.nan)
            continue
        
        if col in sector_df.columns:
            valid_values = safe_to_numeric(sector_df[col].dropna())
            if len(valid_values) > 0:
                total_value = valid_values.sum()
                values.append(total_value)
            else:
                values.append(np.nan)
        else:
            values.append(np.nan)
    
    return values

def get_sector_cagr(sector_name, metric_key, years):
    if not sector_cagr_data:
        return np.nan
    
    if metric_key in ["Chiffre d'affaires", "Resultat d'exploitation", "Charges personnel"]:
        cagr_key = f"{metric_key}_CAGR"
        if cagr_key in sector_cagr_data and sector_name in sector_cagr_data[cagr_key]:
            cagr_value = sector_cagr_data[cagr_key][sector_name]
            return cagr_value if not pd.isna(cagr_value) else np.nan
        return np.nan
    
    elif metric_key in ["Marge EBIT/CA", "Marge EBIT/CP", "Marge CP/CA"]:
        cagr_key = f"{metric_key}_CAGR"
        if cagr_key in sector_cagr_data and sector_name in sector_cagr_data[cagr_key]:
            cagr_value = sector_cagr_data[cagr_key][sector_name]
            return cagr_value if not pd.isna(cagr_value) else np.nan
        return np.nan
    
    elif metric_key in ["CA", "RE", "CP"]:
        metric_mapping = {
            "CA": "Chiffre d'affaires",
            "RE": "Resultat d'exploitation", 
            "CP": "Charges personnel"
        }
        full_name = metric_mapping.get(metric_key, metric_key)
        cagr_key = f"{full_name}_CAGR"
        if cagr_key in sector_cagr_data and sector_name in sector_cagr_data[cagr_key]:
            cagr_value = sector_cagr_data[cagr_key][sector_name]
            return cagr_value if not pd.isna(cagr_value) else np.nan
        return np.nan
    
    elif metric_key in ["EBIT_CA", "EBIT_CP", "CP_CA"]:
        margin_mapping = {
            "EBIT_CA": "Marge EBIT/CA",
            "EBIT_CP": "Marge EBIT/CP",
            "CP_CA": "Marge CP/CA"
        }
        full_name = margin_mapping.get(metric_key, metric_key)
        cagr_key = f"{full_name}_CAGR"
        if cagr_key in sector_cagr_data and sector_name in sector_cagr_data[cagr_key]:
            cagr_value = sector_cagr_data[cagr_key][sector_name]
            return cagr_value if not pd.isna(cagr_value) else np.nan
        return np.nan
    
    return np.nan

def precalculate_sector_cagrs(df):
    years = [2020, 2021, 2022, 2023]
    cagr_data = {}
    
    financial_vars = {
        "Chiffre d'affaires": [f"Chiffre d'affaires {y} (Dhs)" for y in years],
        "Resultat d'exploitation": [f"Resultat d'exploitation {y} (Dhs)" for y in years],
        "Charges personnel": [f"Charges personnel {y}" for y in years]
    }
    
    margin_vars = {
        "Marge EBIT/CA": [f"Marge EBIT/CA {y}" for y in years],
        "Marge EBIT/CP": [f"Marge EBIT/CP {y}" for y in years],
        "Marge CP/CA": [f"Marge CP/CA {y}" for y in years]
    }
    
    for var_name, cols in financial_vars.items():
        if all(col in df.columns for col in cols):
            sector_sums = df.groupby("Secteur")[cols].sum()
            cagr_data[f"{var_name}_CAGR"] = {}
            for sector in sector_sums.index:
                sector_values = sector_sums.loc[sector].values
                if len(sector_values) == len(years) and not np.all(np.isnan(sector_values)):
                    cagr = calculate_cagr(sector_values, len(years)-1)
                    cagr_data[f"{var_name}_CAGR"][sector] = cagr if not pd.isna(cagr) else np.nan
                else:
                    cagr_data[f"{var_name}_CAGR"][sector] = np.nan

    for var_name, cols in margin_vars.items():
        if all(col in df.columns for col in cols):
            sector_means = df.groupby("Secteur")[cols].mean()
            cagr_data[f"{var_name}_CAGR"] = {}
            for sector in sector_means.index:
                sector_values = sector_means.loc[sector].values
                if len(sector_values) == len(years) and not np.all(np.isnan(sector_values)):
                    cagr = calculate_cagr(sector_values, len(years)-1)
                    cagr_data[f"{var_name}_CAGR"][sector] = cagr if not pd.isna(cagr) else np.nan
                else:
                    cagr_data[f"{var_name}_CAGR"][sector] = np.nan
    
    return cagr_data

@st.cache_data(ttl=300)
def get_sector_cagr_data(df):
    return precalculate_sector_cagrs(df)

def split_tokens(text: str) -> List[str]:
    if pd.isna(text):
        return []
    s = str(text)
    SEP = "<<<SPLIT>>>"
    s_chars = list(s)
    i = 0
    while i < len(s_chars):
        if s_chars[i] == ",":
            j = i + 1
            while j < len(s_chars) and s_chars[j].isspace():
                j += 1
            if j < len(s_chars) and s_chars[j].isupper():
                s_chars[i] = SEP
        i += 1
    s2 = "".join(s_chars)
    parts = s2.split(SEP)
    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p2 = re.sub(r'[\(\)\[\]\d\:.\u2022]+', '', p)
        p2 = re.sub(r'\s+', ' ', p2).strip()
        p2 = re.sub(r'^[\-\–\—\.;:]+', '', p2)
        p2 = re.sub(r'[\-\–\—\.;:]+$', '', p2)
        p2 = p2.strip().lower()
        if p2:
            cleaned.append(p2)
    seen = set()
    out = []
    for v in cleaned:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def count_similar(seed_tokens: List[str], cand_tokens: List[str], threshold: float) -> Tuple[int, int]:
    if not seed_tokens:
        return 0, 0
    matched = 0
    for s in seed_tokens:
        for c in cand_tokens:
            if fuzzy_ratio(s, c) >= threshold:
                matched += 1
                break
    return matched, len(seed_tokens)

def frac_str(m: int, t: int) -> str:
    if t == 0:
        return "0/0 (N/A)"
    ratio = m / max(1, t)
    return f"{int(m)}/{int(t)} ({ratio:.2f})"

def extract_products_column(df):
    products_candidates = ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits", "Activités", "Services"]
    for col in products_candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == 'object' and 'produ' in col.lower():
            return col
    return None

def get_company_products(df, company_name, company_col):
    if company_name not in df[company_col].values:
        return []
    
    company_row = df[df[company_col] == company_name].iloc[0]
    products_col = extract_products_column(df)
    
    if products_col and products_col in company_row.index:
        products_text = str(company_row[products_col])
        return split_tokens(products_text)
    return []

def find_closest_companies(df, seed_company, company_col, products_col, threshold=0.6, top_n=2):
    if seed_company not in df[company_col].values:
        return []
    
    seed_row = df[df[company_col] == seed_company].iloc[0]
    seed_products = split_tokens(str(seed_row[products_col])) if products_col and products_col in seed_row.index else []
    
    if not seed_products:
        return []

    similarities = []
    for _, row in df.iterrows():
        if row[company_col] == seed_company:
            continue
        
        candidate_products = split_tokens(str(row[products_col])) if products_col and products_col in row.index else []
        matched, total = count_similar(seed_products, candidate_products, threshold)
        similarity_score = matched / max(1, total) if total > 0 else 0.0
        
        similarities.append({
            'company': row[company_col],
            'similarity_score': similarity_score,
            'matched_count': matched,
            'total_count': total,
            'seed_products': seed_products,
            'candidate_products': candidate_products
        })
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similarities[:top_n]

def find_closest_companies_to_group(df, group_companies, company_col, products_col, threshold=0.6, top_n=2):
    if not group_companies:
        return []
    
    group_products = []
    for company in group_companies:
        if company in df[company_col].values:
            company_row = df[df[company_col] == company].iloc[0]
            company_prods = split_tokens(str(company_row[products_col])) if products_col and products_col in company_row.index else []
            group_products.extend(company_prods)
    group_products = list(set(group_products))
    
    if not group_products:
        return []
    
    similarities = []
    group_set = set(group_companies)
    
    for _, row in df.iterrows():
        candidate_company = row[company_col]
        if candidate_company in group_set:
            continue
        
        candidate_products = split_tokens(str(row[products_col])) if products_col and products_col in row.index else []
        matched, total = count_similar(group_products, candidate_products, threshold)
        similarity_score = matched / max(1, total) if total > 0 else 0.0
        
        similarities.append({
            'company': candidate_company,
            'similarity_score': similarity_score,
            'matched_count': matched,
            'total_count': total,
            'seed_products': group_products,
            'candidate_products': candidate_products
        })
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similarities[:top_n]

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
    .suggestion-card {
        background-color: #e8f5e8;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .similarity-score {
        background-color: #fff3cd;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-weight: bold;
        color: #856404;
    }
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

if df is not None and not df.empty:
    sector_cagr_data = get_sector_cagr_data(df)
else:
    sector_cagr_data = {}

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

metric_display_year = 2023
top_n = 10

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

sector_df = df

# def compute_company_cagrs(df_input, metric_key):
#     try:
#         if df_input is None or df_input.empty:
#             return []
#         if metric_key not in METRIC_TOKENS:
#             st.warning(f"Invalid metric_key '{metric_key}' in compute_company_cagrs. Available: {list(METRIC_TOKENS.keys())}")
#             return []
#         mat, cols_map = build_metric_matrix(df_input, metric_key)
#         cagr_list = []
#         for idx, row in mat.iterrows():
#             v0 = row[str(YEARS[0])]
#             vN = row[str(YEARS[-1])]
#             if pd.notna(v0) and pd.notna(vN) and v0 > 0 and vN > 0:
#                 try:
#                     cagr = (vN / v0) ** (1 / (len(YEARS) - 1)) - 1
#                     if isinstance(cagr, complex):
#                         continue
#                     cagr_list.append(float(cagr))
#                 except Exception:
#                     continue
#         return cagr_list
#     except Exception as e:
#         st.error(f"Error computing CAGR for {metric_key}: {e}")
#         return []

# cagr_results = {}
# cagr_display = []
# for key in ["CA", "RE", "CP"]:
#     try:
#         cagr_list = compute_company_cagrs(sector_df, key)
#         mean_cagr = np.mean(cagr_list) if len(cagr_list) > 0 else np.nan
#         if isinstance(mean_cagr, complex):
#             mean_cagr = np.nan
#         cagr_results[key] = mean_cagr
#         cagr_display.append((key, mean_cagr, len(cagr_list)))
#     except Exception as e:
#         st.error(f"Failed to compute CAGR for {key}: {e}")
#         cagr_results[key] = np.nan
#         cagr_display.append((key, np.nan, 0))

st.markdown("---")

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
    company_sector = get_company_sector(company_single, df, COMPANY_COL, SECTOR_COL)
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
            sector_ca_values = get_sector_financial_values(company_sector, df, "CA", YEARS) if company_sector else [np.nan] * len(YEARS)
            sector_re_values = get_sector_financial_values(company_sector, df, "RE", YEARS) if company_sector else [np.nan] * len(YEARS)
            sector_marge_values = get_sector_margin_values(company_sector, df, "EBIT_CA", YEARS) if company_sector else [np.nan] * len(YEARS)
            sector_cagr_ca = get_sector_cagr(company_sector, "Chiffre d'affaires", YEARS) if company_sector else np.nan
            sector_cagr_re = get_sector_cagr(company_sector, "Resultat d'exploitation", YEARS) if company_sector else np.nan
            sector_cagr_marge = get_sector_cagr(company_sector, "Marge EBIT/CA", YEARS) if company_sector else np.nan
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
            st.markdown(f"<h3 style='color: #28a745;'>Secteur : {company_sector}</h3>", unsafe_allow_html=True)
            ca_values = np.array(ca_values, dtype=float)
            re_values = np.array(re_values, dtype=float)
            n_years = len(YEARS) - 1
            cagr_ca = calculate_cagr(ca_values, n_years)
            cagr_re = calculate_cagr(re_values, n_years)
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
                xref="paper", yref="paper", x=0.4, y=1.20,
                showarrow=False,
                font=dict(size=14, color="blue"),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="blue",
                borderwidth=1, borderpad=4,
            )
            fig.add_annotation(
                text=f"CAGR RE: {(cagr_re*100):.2f}%",
                xref="paper", yref="paper", x=0.55, y=1.20,
                showarrow=False,
                font=dict(size=14, color="red"),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="red",
                borderwidth=1, borderpad=4,
            )
            if company_sector and any(pd.notna(v) for v in sector_marge_values):
                fig.add_trace(go.Scatter(
                    x=YEARS,
                    y=[v * 100 if pd.notna(v) else np.nan for v in sector_marge_values],
                    mode="lines+text",
                    name=f"{company_sector} - Marge",
                    line=dict(shape='spline', color="#28a745", width=3),
                    hovertemplate=f"{company_sector}<br>%{{x}}<br>Marge: %{{y:.1f}}%<extra></extra>",
                    textposition="top center",
                    textfont=dict(size=18),
                    yaxis="y2"
                ))
            if company_sector:
                if pd.notna(sector_cagr_ca):
                    fig.add_annotation(
                        text=f"CAGR CA Secteur: {(sector_cagr_ca*100):.2f}%",
                        xref="paper", yref="paper", x=0.38, y=1.11,
                        showarrow=False,
                        font=dict(size=14, color="#28a745"),
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="#28a745",
                        borderwidth=1, borderpad=4,
                    )
                if pd.notna(sector_cagr_re):
                    fig.add_annotation(
                        text=f"CAGR RE Secteur: {(sector_cagr_re*100):.2f}%",
                        xref="paper", yref="paper", x=0.57, y=1.11,
                        showarrow=False,
                        font=dict(size=14, color="#28a745"),
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="#28a745",
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
                margin=dict(l=80, r=80, t=150, b=80),
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
                    text=f"CAGR: {(calculate_cagr(values, n_years)*100):.2f}%",
                    xref="paper", yref="paper", x=0.4, y=1.12,
                    showarrow=False,
                    font=dict(size=16, color="black"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="black",
                    borderwidth=1, borderpad=4,
                )
                
                is_margin = "Marge" in title
                if company_sector:
                    if is_margin and "EBIT/CA" in title:
                        sector_values = get_sector_margin_values(company_sector, df, "EBIT_CA", YEARS[:len(values)])
                        sector_cagr_key = "Marge EBIT/CA"
                    elif is_margin and "EBIT/CP" in title:
                        sector_values = get_sector_margin_values(company_sector, df, "EBIT_CP", YEARS[:len(values)])
                        sector_cagr_key = "Marge EBIT/CP"
                    elif is_margin and "CP/CA" in title:
                        sector_values = get_sector_margin_values(company_sector, df, "CP_CA", YEARS[:len(values)])
                        sector_cagr_key = "Marge CP/CA"
                    elif title == "Chiffre d'affaires":
                        sector_values = get_sector_financial_values(company_sector, df, "CA", YEARS[:len(values)])
                        sector_cagr_key = "Chiffre d'affaires"
                    elif title == "Résultat d'exploitation":
                        sector_values = get_sector_financial_values(company_sector, df, "RE", YEARS[:len(values)])
                        sector_cagr_key = "Resultat d'exploitation"
                    elif title == "Charges personnel":
                        sector_values = get_sector_financial_values(company_sector, df, "CP", YEARS[:len(values)])
                        sector_cagr_key = "Charges personnel"
                    else:
                        sector_values = [np.nan] * len(values)
                        sector_cagr_key = None
                    
                    if sector_cagr_key:
                        sector_cagr = get_sector_cagr(company_sector, sector_cagr_key, YEARS[:len(values)])
                if company_sector and sector_cagr_key and pd.notna(sector_cagr):
                    fig.add_annotation(
                        text=f"CAGR Secteur: {(sector_cagr*100):.2f}%",
                        xref="paper", yref="paper", x=0.55, y=1.12,
                        showarrow=False,
                        font=dict(size=14, color="#28a745"),
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="#28a745",
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
                cagr_ca = calculate_cagr(ca_values, n_years)
                cagr_re = calculate_cagr(re_values, n_years)
                marge_values = np.array([aggregate_group_data(companies_in_group, df, "EBIT_CA", y) for y in YEARS])
                group_reference_sector = get_largest_ca_company_sector(companies_in_group, df, COMPANY_COL, SECTOR_COL)
                if not group_reference_sector:
                    group_reference_sector = get_group_sector(companies_in_group, df, COMPANY_COL, SECTOR_COL)
                sector_ca_values = get_sector_financial_values(group_reference_sector, df, "CA", YEARS) if group_reference_sector else [np.nan] * len(YEARS)
                sector_re_values = get_sector_financial_values(group_reference_sector, df, "RE", YEARS) if group_reference_sector else [np.nan] * len(YEARS)
                sector_marge_values = get_sector_margin_values(group_reference_sector, df, "EBIT_CA", YEARS) if group_reference_sector else [np.nan] * len(YEARS)
                sector_cagr_ca = get_sector_cagr(group_reference_sector, "Chiffre d'affaires", YEARS) if group_reference_sector else np.nan
                sector_cagr_re = get_sector_cagr(group_reference_sector, "Resultat d'exploitation", YEARS) if group_reference_sector else np.nan
                sector_cagr_marge = get_sector_cagr(group_reference_sector, "Marge EBIT/CA", YEARS) if group_reference_sector else np.nan
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=YEARS,
                    y=ca_values,
                    name=f"CA - {selected_group}",
                    marker_color="#052449",
                    text=[f"{v/1e9:.2f} B" if v>=1e9 else f"{v/1e6:.2f} M" if v>=1e6 else f"{v:,.0f}" for v in ca_values],
                    textposition="auto",
                    textfont=dict(size=18),
                    width=0.25,
                    offsetgroup=1
                ))
                fig.add_trace(go.Bar(
                    x=YEARS,
                    y=re_values,
                    name=f"RE - {selected_group}",
                    marker_color="#0064EF",
                    text=[f"{v/1e9:.2f} B" if v>=1e9 else f"{v/1e6:.2f} M" if v>=1e6 else f"{v:,.0f}" for v in re_values],
                    textposition="auto",
                    textfont=dict(size=18),
                    width=0.25,
                    offsetgroup=2
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
                    xref="paper", yref="paper", x=0.4, y=1.20,
                    showarrow=False,
                    font=dict(size=14, color="blue"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="blue",
                    borderwidth=1, borderpad=4,
                )
                fig.add_annotation(
                    text=f"CAGR RE: {(cagr_re*100):.2f}%",
                    xref="paper", yref="paper", x=0.55, y=1.20,
                    showarrow=False,
                    font=dict(size=14, color="red"),
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="red",
                    borderwidth=1, borderpad=4,
                )

                if group_reference_sector and any(pd.notna(v) for v in sector_marge_values):
                    fig.add_trace(go.Scatter(
                        x=YEARS,
                        y=[v * 100 if pd.notna(v) else np.nan for v in sector_marge_values],
                        mode="lines+text",
                        name=f"{group_reference_sector} - Marge",
                        line=dict(shape='spline', color="#28a745", width=3),
                        hovertemplate=f"{group_reference_sector}<br>%{{x}}<br>Marge: %{{y:.1f}}%<extra></extra>",
                        text=[f"{v*100:.2f}%" if pd.notna(v) else "" for v in sector_marge_values],
                        textposition="top center",
                        textfont=dict(size=18),
                        yaxis="y2"
                    ))
                    
                    if pd.notna(sector_cagr_ca):
                        fig.add_annotation(
                            text=f"CAGR CA : {(sector_cagr_ca*100):.2f}%",
                            xref="paper", yref="paper", x=0.40, y=1.11,
                            showarrow=False,
                            font=dict(size=14, color="#28a745"),
                            bgcolor="rgba(255,255,255,0.8)", bordercolor="#28a745",
                            borderwidth=1, borderpad=4,
                        )
                    if pd.notna(sector_cagr_re):
                        fig.add_annotation(
                            text=f"CAGR RE : {(sector_cagr_re*100):.2f}%",
                            xref="paper", yref="paper", x=0.55, y=1.11,
                            showarrow=False,
                            font=dict(size=14, color="#28a745"),
                            bgcolor="rgba(255,255,255,0.8)", bordercolor="#28a745",
                            borderwidth=1, borderpad=4,
                        )
                if group_reference_sector:
                    st.markdown(f"<h3 style='color: #28a745;'>Secteur : {group_reference_sector}</h3>", unsafe_allow_html=True)
                fig.update_layout(
                    title=dict(text=f"CA & RE avec Marge EBIT/CA - {selected_group}", font=dict(size=22)),
                    xaxis=dict(tickmode="array", tickvals=YEARS, title="Année", showgrid=False, tickfont=dict(size=18)),
                    yaxis=dict(title="Valeurs (Dhs)", showgrid=False, tickfont=dict(size=18), visible=False),
                    yaxis2=dict(title="Marge EBIT/CA (%)", overlaying="y", side="right", showgrid=False, tickfont=dict(size=18), visible=False),
                    barmode="group",
                    margin=dict(l=80, r=80, t=180 if group_reference_sector else 150, b=80),
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
                        text=f"CAGR: {(calculate_cagr(values, n_years)*100):.2f}%",
                        xref="paper", yref="paper", x=0.5, y=1.15,
                        showarrow=False,
                        font=dict(size=16, color="black"),
                        bgcolor="rgba(255,255,255,0.8)", bordercolor="black",
                        borderwidth=1, borderpad=4,
                    )
                    is_margin = "Marge" in title
                    if group_reference_sector:
                        if is_margin and "EBIT/CA" in title:
                            sector_values = get_sector_margin_values(group_reference_sector, df, "EBIT_CA", YEARS[:len(values)])
                            sector_cagr_key = "Marge EBIT/CA"
                        elif is_margin and "EBIT/CP" in title:
                            sector_values = get_sector_margin_values(group_reference_sector, df, "EBIT_CP", YEARS[:len(values)])
                            sector_cagr_key = "Marge EBIT/CP"
                        elif is_margin and "CP/CA" in title:
                            sector_values = get_sector_margin_values(group_reference_sector, df, "CP_CA", YEARS[:len(values)])
                            sector_cagr_key = "Marge CP/CA"
                        elif title == "Chiffre d'affaires":
                            sector_values = get_sector_financial_values(group_reference_sector, df, "CA", YEARS[:len(values)])
                            sector_cagr_key = "Chiffre d'affaires"
                        elif title == "Résultat d'exploitation":
                            sector_values = get_sector_financial_values(group_reference_sector, df, "RE", YEARS[:len(values)])
                            sector_cagr_key = "Resultat d'exploitation"
                        elif title == "Charges personnel":
                            sector_values = get_sector_financial_values(group_reference_sector, df, "CP", YEARS[:len(values)])
                            sector_cagr_key = "Charges personnel"
                        else:
                            sector_values = [np.nan] * len(values)
                            sector_cagr_key = None
                        
                        if sector_cagr_key:
                            sector_cagr = get_sector_cagr(group_reference_sector, sector_cagr_key, YEARS[:len(values)])
                    if group_reference_sector and sector_cagr_key and pd.notna(sector_cagr):
                        fig.add_annotation(
                            text=f"CAGR Secteur : {(sector_cagr*100):.2f}%",
                            xref="paper", yref="paper", x=0.75, y=1.12,
                            showarrow=False,
                            font=dict(size=14, color="#28a745"),
                            bgcolor="rgba(255,255,255,0.8)", bordercolor="#28a745",
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
products_col = extract_products_column(df)
if not products_col:
    st.warning("⚠️ Colonne produits/services non trouvée. Les suggestions automatiques ne seront pas disponibles.")
available_individual_companies = [
    c.strip() for c in sorted(df[COMPANY_COL].dropna().astype(str).str.strip().unique())
    if c.strip() not in grouped_companies
]
comparison_entities = []
all_comparison_options = available_individual_companies + [f"👥 {g}" for g in sorted(group_manager.get_all_groups())]
primary_selection_type = st.radio(
            "Mode d'affichage",
            ["Entreprise individuelle", "Groupe d'entreprises"],
            key="primary_selection_type"
        )
manual_selections = st.multiselect(
    "i. Entreprises et groupes à comparer:",
    all_comparison_options,
    placeholder="Choisissez manuellement...",
    key="manual_comparison"
)
with st.expander("ii. Suggestions automatiques", expanded=False):
    col1, col2 = st.columns([2, 1]) 
    primary_entity = None
    primary_entity_name = None
    primary_is_group = False
    if primary_selection_type == "Entreprise individuelle":
        if available_individual_companies:
            primary_entity = st.selectbox(
                "Sélectionnez l'entreprise principale:",
                ["Aucune"] + available_individual_companies,
                key="primary_company"
            )
            if primary_entity != "Aucune":
                primary_entity_name = primary_entity
        else:
            st.info("Aucune entreprise disponible pour sélection.")
    else:
        existing_groups = group_manager.get_all_groups()
        if existing_groups:
            primary_group = st.selectbox(
                "Sélectionnez le groupe principal:",
                ["Aucun"] + existing_groups,
                key="primary_group"
            )
            if primary_group != "Aucun":
                primary_entity_name = primary_group
                primary_is_group = True
        else:
            st.info("Aucun groupe disponible.")

    st.markdown("##### 🔍 Filtre CA 2023")
    ca_2023_col = get_column_for("CA", 2023)
    all_ca_values = safe_to_numeric(df[ca_2023_col].dropna())
    global_max_ca = float(all_ca_values.max()) if len(all_ca_values) > 0 else 10000000000
    default_min = 0
    default_max = int(global_max_ca)

    col_min, col_max = st.columns(2)
    with col_min:
        ca_min = st.number_input(
            "CA minimum (MAD)", 
            value=default_min, 
            min_value=0, 
            step=1000000, 
            key="ca_min"
        )
    with col_max:
        ca_max = st.number_input(
            "CA maximum (MAD)", 
            value=default_max, 
            min_value=0, 
            step=1000000,
            key="ca_max"
        )

    suggested_companies = []
    suggestion_threshold = 0.75
    if primary_entity_name and products_col:
        with st.container():
            if not primary_is_group:
                suggested_companies = find_closest_companies(
                    df, primary_entity_name, COMPANY_COL, products_col,
                    threshold=suggestion_threshold, top_n=5
                )
            else:
                group_companies = group_manager.groups[primary_entity_name].get('companies', [])
                suggested_companies = find_closest_companies_to_group(
                    df, group_companies, COMPANY_COL, products_col,
                    threshold=suggestion_threshold, top_n=5
                )
            
            if suggested_companies:
                ca_2023_col = get_column_for("CA", 2023)
                if ca_2023_col and ca_2023_col in df.columns:
                    filtered_suggestions = []
                    for suggestion in suggested_companies:
                        comp_df = safe_find_company(df, suggestion['company'], COMPANY_COL)
                        if not comp_df.empty:
                            ca_value_raw = comp_df.iloc[0][ca_2023_col]
                            ca_value = safe_to_numeric(ca_value_raw)
                            if pd.notna(ca_value) and ca_min <= float(ca_value) <= ca_max:
                                filtered_suggestions.append(suggestion)
                            elif pd.isna(ca_value):
                                filtered_suggestions.append(suggestion)
                    suggested_companies = filtered_suggestions
                else:
                    st.warning("⚠️ Colonne CA 2023 non trouvée. Toutes les suggestions sont affichées.")
            
            if suggested_companies:
                for i, suggestion in enumerate(suggested_companies, 1):
                    with st.container():
                        col_s1, col_s2, col_s3 = st.columns([3, 1, 2])
                        with col_s1:
                            st.markdown(f"**🏢 {suggestion['company']}**")
                        with col_s2:
                            score_class = "similarity-score"
                            st.markdown(f"""
                            <div class="{score_class}">
                                {suggestion['similarity_score']:.1%}
                                ({suggestion['matched_count']}/{suggestion['total_count']})
                            </div>
                            """, unsafe_allow_html=True)
                        with col_s3:
                            if st.button(f"➕ Ajouter à la comparaison", key=f"add_suggestion_{i}"):
                                st.session_state.comparison_suggestions = st.session_state.get('comparison_suggestions', []) + [suggestion['company']]
                                st.success(f"{suggestion['company']} ajouté à la comparaison!")
                                st.rerun()
                st.info("N.B : Les suggestions apparaitront automatiquement sur les graphes.")
            else:
                st.info("❌ Aucune entreprise similaire trouvée avec le seuil actuel.")
                if st.button("🔧 Baisser le seuil de similarité (0.3)"):
                    suggested_companies = find_closest_companies(
                        df, primary_entity_name, COMPANY_COL, products_col,
                        threshold=0.3, top_n=5
                    ) if not primary_is_group else find_closest_companies_to_group(
                        df, group_manager.groups[primary_entity_name]['companies'],
                        COMPANY_COL, products_col, threshold=0.3, top_n=5
                    )
                    
                    if suggested_companies:
                        ca_2023_col = get_column_for("CA", 2023)
                        if ca_2023_col and ca_2023_col in df.columns:
                            filtered_suggestions = []
                            for suggestion in suggested_companies:
                                comp_df = safe_find_company(df, suggestion['company'], COMPANY_COL)
                                if not comp_df.empty:
                                    ca_value_raw = comp_df.iloc[0][ca_2023_col]
                                    ca_value = safe_to_numeric(ca_value_raw)
                                    if pd.notna(ca_value) and ca_min <= float(ca_value) <= ca_max:
                                        filtered_suggestions.append(suggestion)
                                    elif pd.isna(ca_value):
                                        filtered_suggestions.append(suggestion)
                            suggested_companies = filtered_suggestions
                    
                    if suggested_companies:
                        st.rerun()
    with col2:
        st.markdown("### 📋 Suggestions ajoutées")
        if 'comparison_suggestions' in st.session_state:
            added_suggestions = st.session_state.comparison_suggestions
            for i, company in enumerate(added_suggestions):
                col_a1, col_a2 = st.columns([3, 1])
                with col_a1:
                    st.write(f"✅ {company}")
                with col_a2:
                    if st.button("❌", key=f"remove_suggestion_{i}"):
                        st.session_state.comparison_suggestions.pop(i)
                        st.rerun()
        else:
            st.info("Aucune suggestion ajoutée")

for entity in manual_selections:
    if entity.startswith("👥 "):
        group_name = entity[2:].strip()
        if group_name:
            comparison_entities.append(group_name)
    else:
        entity_clean = entity.strip()
        if entity_clean:
            comparison_entities.append(entity_clean)
comparison_entities = []

for entity in manual_selections:
    if entity.startswith("👥 "):
        group_name = entity[2:].strip()
        if group_name:
            comparison_entities.append(group_name)
    else:
        entity_clean = entity.strip()
        if entity_clean:
            comparison_entities.append(entity_clean)

if 'comparison_suggestions' in st.session_state:
    for suggestion in st.session_state.comparison_suggestions:
        if suggestion not in comparison_entities:
            comparison_entities.append(suggestion)

if primary_entity_name and primary_entity_name not in comparison_entities:
    comparison_entities.append(primary_entity_name)

if not comparison_entities:
    st.info("Aucune entité sélectionnée pour la comparaison.")

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
        vals = [int(safe_to_numeric(comp_df.get(col, pd.Series(dtype=float)).mean(skipna=True))) for col in cols]
        vals = np.array(vals, dtype=int)
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
    
    show_sector_reference = False
    reference_sector = None
    if hasattr(st.session_state, 'comparison_reference_sector'):
        show_sector_reference = True
        reference_sector = st.session_state.comparison_reference_sector
    if color_map is None:
        color_map = {entity: brand_colors[i % len(brand_colors)] for i, entity in enumerate(entities)}
    
    fig = go.Figure()
    annotations = []
    ann_step = 1 / (len(entities) + 1)
    sector_ca_vals = []
    sector_margin_vals = []
    if show_sector_reference and reference_sector:
        for y in years:
            sector_ca = get_sector_financial_values(reference_sector, df, "CA", [y])[0]
            sector_ca_vals.append(sector_ca if pd.notna(sector_ca) else np.nan)
            sector_margin = get_sector_margin_values(reference_sector, df, "EBIT_CA", [y])[0]
            sector_margin_vals.append(sector_margin if pd.notna(sector_margin) else np.nan)
    
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
                valid_companies = 0
                for comp in companies_in_group:
                    comp_df = safe_find_company(df, comp, COMPANY_COL)
                    if not comp_df.empty:
                        ca_col = get_column_for("CA", y)
                        re_col = get_column_for("RE", y)
                        ca_val_raw = comp_df.iloc[0][ca_col] if ca_col and ca_col in comp_df.columns else None
                        re_val_raw = comp_df.iloc[0][re_col] if re_col and re_col in comp_df.columns else None
                        if ca_val_raw is not None:
                            ca_numeric = safe_to_numeric(ca_val_raw)
                            if pd.notna(ca_numeric):
                                ca_val += int(float(ca_numeric))
                                valid_companies += 1
                        if re_val_raw is not None:
                            re_numeric = safe_to_numeric(re_val_raw)
                            if pd.notna(re_numeric):
                                re_val += int(float(re_numeric))
                if valid_companies == 0:
                    ca_val = np.nan
                if re_val == 0 and valid_companies == 0:
                    re_val = np.nan
            else:
                comp_df = safe_find_company(df, entity, COMPANY_COL)
                if not comp_df.empty:
                    ca_col = get_column_for("CA", y)
                    re_col = get_column_for("RE", y)
                    ca_val_raw = comp_df.iloc[0][ca_col] if ca_col else np.nan
                    re_val_raw = comp_df.iloc[0][re_col] if re_col else np.nan
                    ca_val = int(safe_to_numeric(ca_val_raw)) if pd.notna(safe_to_numeric(ca_val_raw)) else np.nan
                    re_val = int(safe_to_numeric(re_val_raw)) if pd.notna(safe_to_numeric(re_val_raw)) else np.nan
                else:
                    ca_val = re_val = np.nan
            ca_vals.append(ca_val)
            re_vals.append(re_val)
        
        margin_vals = [(re / ca * 100) if pd.notna(re) and pd.notna(ca) and ca != 0 else np.nan
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
            x=years, y=[v if pd.notna(v) else np.nan for v in margin_vals],
            mode="lines+text",
            name=f"{display_name} - Marge EBIT/CA",
            line=dict(shape='spline', color=margin_color, width=3),
            text=[f"{v:.2f}%" if pd.notna(v) else "" for v in margin_vals],
            textposition="top center",
            yaxis="y2",
            textfont=dict(size=16, color="darkblue"),
        ))
        
        cagr_ca = calculate_cagr(ca_vals, len(years) - 1)
        annotations.append(dict(
            text=f"{display_name}<br>CAGR CA: {cagr_ca*100:.2f}%",
            xref="paper", yref="paper",
            x=0.05 + i * ann_step, y = 1.3 if show_sector_reference else 1.15,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,1)",
            bordercolor=base_color,
            borderwidth=1,
            font=dict(size=13)
        ))
    if show_sector_reference and reference_sector:
        sector_display_name = f"📊 {reference_sector} - Secteur"
        sector_color = "#28a745"
        if any(pd.notna(v) for v in sector_margin_vals):
            fig.add_trace(go.Scatter(
                x=years, y=[v * 100 if pd.notna(v) else np.nan for v in sector_margin_vals],
                mode="lines+text",
                name=f"{sector_display_name} - Marge EBIT/CA",
                line=dict(color=sector_color, width=3, shape="spline"),
                yaxis="y2",
                hovertemplate=f"{reference_sector}<br>%{{x}}<br>Marge EBIT/CA: %{{y:.1f}}%<extra></extra>",
                text=[f"{v*100:.2f}%" if pd.notna(v) else "" for v in sector_margin_vals],
                textposition="top center",
                textfont=dict(size=15, color=sector_color),
            ))
        sector_cagr_ca = get_sector_cagr(reference_sector, "Chiffre d'affaires", years)
        sector_cagr_margin = get_sector_cagr(reference_sector, "Marge EBIT/CA", years)
        
        if pd.notna(sector_cagr_ca):
            annotations.append(dict(
                text=f"{sector_display_name}<br>CAGR CA: {sector_cagr_ca*100:.2f}%",
                xref="paper", yref="paper",
                x=0.02, y=1.15,
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=sector_color,
                borderwidth=1,
                font=dict(size=12, color=sector_color)
            ))
    
    fig.update_layout(
        title=f"Comparaison CA et Marge EBIT/CA ({years[0]}-{years[-1]}) - {len(entities)} entités",
        xaxis=dict(title="Année", tickmode="array", tickvals=years, showgrid=False),
        yaxis=dict(title="Montants (MAD)", showgrid=False, visible=False),
        yaxis2=dict(title="Marge EBIT/CA (%)", overlaying="y", side="right", showgrid=False, visible=False),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3 if show_sector_reference else -0.25, x=0.5, xanchor="center"),
        margin=dict(l=60, r=60, t=220 if show_sector_reference else 120, b=80),
        height=650 if show_sector_reference else (600 if len(entities) > 3 else 550),
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
    
    show_sector_reference = False
    reference_sector = None
    if hasattr(st.session_state, 'comparison_reference_sector'):
        show_sector_reference = True
        reference_sector = st.session_state.comparison_reference_sector
    
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
    
    annotations = []
    n_years = len(YEARS) - 1
    
    sector_values = []
    sector_cagr = np.nan
    if show_sector_reference and reference_sector:
        if metric_key in ["EBIT_CA", "EBIT_CP", "CP_CA"]:
            sector_values = get_sector_margin_values(reference_sector, df, metric_key, years)
            sector_cagr_key = f"Marge {metric_key}"
        else:
            sector_values = get_sector_financial_values(reference_sector, df, metric_key, years)
            sector_cagr_key = {
                "CA": "Chiffre d'affaires",
                "RE": "Resultat d'exploitation", 
                "CP": "Charges personnel"
            }.get(metric_key, metric_key)
        
        if sector_cagr_key:
            sector_cagr = get_sector_cagr(reference_sector, sector_cagr_key, years)
    
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
            y_values_for_var = values
        else:
            texts = [format_number(v) for v in values]
            y_values_for_var = values
        
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
                if len(valid_values) > 1:
                    pct_change = np.array([np.nan] + list(np.array(valid_values)[1:] / np.array(valid_values)[:-1] - 1))
                    fig.add_trace(go.Scatter(
                        x=x_offset,
                        y=y_values_for_var,
                        mode="lines+text",
                        name=f"{display_name} - Var",
                        line=dict(shape='spline', dash="dot", color=adjust_color_brightness(color_map[entity], 0.8)),
                        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct_change],
                        textposition="top center",
                        textfont=dict(size=16),
                    ))
            
            cagr_val = calculate_cagr(values, n_years)
            annotations.append(dict(
                text=f"{display_name}<br>CAGR: {cagr_val*100:.2f}%",
                xref="paper", yref="paper",
                x=0.05 + i * (1/(n_entities+1)), y=1.32 if reference_sector else 1.02,
                showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=color_map[entity],
                borderwidth=1,
                font=dict(size=12, color=color_map[entity])
            ))

    if show_sector_reference and reference_sector and any(pd.notna(v) for v in sector_values):
        sector_display_name = f"{reference_sector}"
        sector_color = "#28a745"

        annotations.append(dict(
            text=f"{sector_display_name}<br>CAGR: {sector_cagr*100:.2f}%",
            xref="paper", yref="paper",
            x=0.02, y=1.18 if reference_sector else 1.02,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=sector_color,
            borderwidth=1,
            font=dict(size=12, color=sector_color)
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
    
    legend_y = -0.3 if show_sector_reference else -0.25
    height_adjust = 50 if show_sector_reference else 0
    
    fig.update_layout(
        title=f"{yaxis_title} - Comparaison {len(entities)} entités",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=550 + height_adjust,
        margin=dict(l=80, r=80, t=200 if reference_sector else 120, b=80),
        xaxis=dict(tickmode="array", tickvals=years, title="Année", showgrid=False),
        yaxis=dict(title=yaxis_title, tickformat=tickformat, showgrid=False, visible=False, showticklabels=False),
        yaxis2=dict(title="Marge EBIT/CA (%)", overlaying="y", side="right", showgrid=False, visible=False),
        legend=dict(orientation="h", yanchor="bottom", y=legend_y, xanchor="center", x=0.5),
        hovermode="x unified",
        annotations=annotations,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

if comparison_entities:
    same_sector, reference_sector = all_entities_same_sector(
        comparison_entities, df, COMPANY_COL, SECTOR_COL
    )
    if same_sector and reference_sector:
        st.markdown(f"<h3 style='color: #28a745;'>Secteur : {reference_sector}</h3>", unsafe_allow_html=True)
        st.session_state.comparison_reference_sector = reference_sector
    else:
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 1rem; border-radius: 8px; border-left: 4px solid #ffc107;'>
            <strong>⚠️ Les entités sélectionnées ont des secteurs différents.</strong><br>
            <em>Les deux boites ont un secteur different, le benchmark secteur ne sera donc pas affiché par souci de clarté.</em>
        </div>
        """, unsafe_allow_html=True)
        if hasattr(st.session_state, 'comparison_reference_sector'):
            del st.session_state.comparison_reference_sector
    
    col1, col2, col3 = st.columns([1, 8, 1])
    if 'comparison_suggestions' in st.session_state and primary_entity_name:
        suggestion_scores = []
        products_col = extract_products_column(df)
        if products_col and primary_entity_name:
            if not primary_is_group:
                all_similarities = find_closest_companies(
                    df, primary_entity_name, COMPANY_COL, products_col,
                    threshold=0.1, top_n=10
                )
            else:
                group_companies = group_manager.groups[primary_entity_name].get('companies', [])
                all_similarities = find_closest_companies_to_group(
                    df, group_companies, COMPANY_COL, products_col,
                    threshold=0.1, top_n=10
                )
            
            if all_similarities:
                ca_2023_col = get_column_for("CA", 2023)
                if ca_2023_col and ca_2023_col in df.columns:
                    filtered_similarities = []
                    for sim in all_similarities:
                        comp_df = safe_find_company(df, sim['company'], COMPANY_COL)
                        if not comp_df.empty:
                            ca_value_raw = comp_df.iloc[0][ca_2023_col]
                            ca_value = int(safe_to_numeric(ca_value_raw)) if pd.notna(safe_to_numeric(ca_value_raw)) else np.nan
                            if pd.notna(ca_value) and ca_min <= ca_value <= ca_max:
                                filtered_similarities.append(sim)
                    all_similarities = filtered_similarities
            
            for sim in all_similarities:
                if sim['company'] in comparison_entities:
                    suggestion_scores.append({
                        'company': sim['company'],
                        'score': sim['similarity_score'],
                        'matched': sim['matched_count'],
                        'total': sim['total_count']
                    })
            if suggestion_scores:
                scores_df = pd.DataFrame(suggestion_scores)
                scores_df['Score'] = scores_df['score'].apply(lambda x: f"{x:.1%}")
                scores_df['Match'] = scores_df.apply(lambda row: f"{row['matched']}/{row['total']}", axis=1)
                st.dataframe(scores_df[['company', 'Score', 'Match']].rename(columns={'company': 'Entreprise'}), use_container_width=True)
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
        st.markdown("### 📋 Récapitulatif comparatif")
        summary_data = []
        for entity in comparison_entities:
            is_group = entity in group_manager.groups
            row = {"Entité": f"👥 {entity}" if is_group else f"🏢 {entity}",
                "Type": "Groupe" if is_group else "Entreprise"}
            for y in YEARS:
                if is_group:
                    ca_raw = aggregate_group_data(group_manager.groups[entity]['companies'], df, "CA", y)
                    re_raw = aggregate_group_data(group_manager.groups[entity]['companies'], df, "RE", y)
                    cp_raw = aggregate_group_data(group_manager.groups[entity]['companies'], df, "CP", y)
                    ca_numeric = safe_to_numeric(ca_raw)
                    re_numeric = safe_to_numeric(re_raw)
                    cp_numeric = safe_to_numeric(cp_raw)
                    ca = int(float(ca_numeric)) if pd.notna(ca_numeric) else np.nan
                    re = int(float(re_numeric)) if pd.notna(re_numeric) else np.nan
                    cp = int(float(cp_numeric)) if pd.notna(cp_numeric) else np.nan
                    ebit_ca = (re / ca * 100) if ca != 0 and pd.notna(ca) and pd.notna(re) else np.nan
                    ebit_cp = (re / cp * 100) if cp != 0 and pd.notna(cp) and pd.notna(re) else np.nan
                    cp_ca = (cp / ca * 100) if ca != 0 and pd.notna(ca) and pd.notna(cp) else np.nan
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
                            raw_value = comp_df.iloc[0][col] if col and col in comp_df.columns else np.nan
                            values[metric] = int(safe_to_numeric(raw_value)) if pd.notna(safe_to_numeric(raw_value)) else np.nan
                        values["EBIT_CA"] = (values["RE"] / values["CA"] * 100) if values["CA"] and pd.notna(values["CA"]) and pd.notna(values["RE"]) else np.nan
                        values["EBIT_CP"] = (values["RE"] / values["CP"] * 100) if values["CP"] and pd.notna(values["CP"]) and pd.notna(values["RE"]) else np.nan
                        values["CP_CA"] = (values["CP"] / values["CA"] * 100) if values["CA"] and pd.notna(values["CA"]) and pd.notna(values["CP"]) else np.nan
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
        if st.button("📥 Exporter comparaison"):
            export_data = summary_df.copy()
            if 'comparison_suggestions' in st.session_state and primary_entity_name and suggestion_scores:
                similarity_map = {s['company']: s['score'] for s in suggestion_scores}
                export_data['Score de similarité'] = export_data['Entité'].apply(
                    lambda x: similarity_map.get(x.replace('🏢 ', '').replace('👥 ', ''), 'N/A')
                )
            if primary_entity_name:
                export_data['Entité de référence'] = primary_entity_name
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                export_data.to_excel(writer, sheet_name="Comparaison", index=False)
                if suggestion_scores:
                    sim_df = pd.DataFrame(suggestion_scores)
                    sim_df.to_excel(writer, sheet_name="Scores_Similarite", index=False)
                    if products_col:
                        products_data = []
                        for entity in comparison_entities:
                            if not entity in group_manager.groups:
                                comp_df = safe_find_company(df, entity, COMPANY_COL)
                                if not comp_df.empty:
                                    products_text = str(comp_df.iloc[0][products_col]) if products_col in comp_df.columns else "N/A"
                                    products_data.append({
                                        'Entreprise': entity,
                                        'Produits/Services': products_text[:500] + "..." if len(products_text) > 500 else products_text
                                    })
                        if products_data:
                            products_df = pd.DataFrame(products_data)
                            products_df.to_excel(writer, sheet_name="Produits_Services", index=False)
            out.seek(0)
            st.download_button(
                "📥 Télécharger Excel",
                data=out,
                file_name=f"comparaison_{primary_entity_name}_{len(comparison_entities)}_entites.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
if st.button("🗑️ Vider les suggestions automatiques"):
    if 'comparison_suggestions' in st.session_state:
        del st.session_state.comparison_suggestions
    st.rerun()
group_manager.save_session()