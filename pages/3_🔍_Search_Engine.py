import streamlit as st
import pandas as pd
import numpy as np
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

st.set_page_config(page_title="Search Engine", layout="wide")
st.title("🔍 Search Engine")

def get_col(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

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

def join_products_for_group(products_list: List[str], separator: str = " | ") -> str:
    if not products_list:
        return ""
    all_tokens = []
    for products_str in products_list:
        if pd.isna(products_str) or not str(products_str).strip():
            continue
        company_tokens = split_tokens(str(products_str).strip())
        all_tokens.extend(company_tokens)
    unique_tokens = list(set(all_tokens))
    if not unique_tokens:
        return ""
    return separator.join(unique_tokens)

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

def load_or_upload(filename: str, prompt_label: str):
    try:
        df = pd.read_excel(filename)
        return df
    except Exception:
        uploaded = st.file_uploader(f"Téléversez {prompt_label} (.xlsx) si {filename} absent", type=["xlsx"], key=filename)
        if uploaded is None:
            st.info(f"Placez {filename} dans le dossier de l'app ou téléversez-le ici.")
            return None
        try:
            df = pd.read_excel(uploaded)
            st.success(f"Fichier chargé - {df.shape[0]} lignes, {df.shape[1]} colonnes.")
            return df
        except Exception as e:
            st.error(f"Erreur en lisant {filename}: {e}")
            return None

def compact_num(n) -> str:
    if pd.isna(n):
        return "N/A"
    try:
        n = float(n)
        if n < 0:
            n = 0
    except Exception:
        return str(n)
    B = 1e9
    M = 1e6
    K = 1e3
    if abs(n) >= B:
        return f"{n / B:.2f}B"
    elif abs(n) >= M:
        return f"{n / M:.1f}M"
    elif abs(n) >= K:
        return f"{n / K:.1f}K"
    else:
        return f"{n:,.0f}"

def format_range_compact(mi, ma) -> str:
    if pd.isna(mi) and pd.isna(ma):
        return "N/A"
    if pd.isna(mi) and not pd.isna(ma):
        return f"≤ {compact_num(ma)}"
    if pd.isna(ma) and not pd.isna(mi):
        return f"≥ {compact_num(mi)}"
    if mi == ma:
        return compact_num(mi)
    return f"de {compact_num(mi)} à {compact_num(ma)}"

def parse_revenue_text(s: str) -> Tuple[float, float]:
    if pd.isna(s):
        return np.nan, np.nan
    raw = str(s).strip()
    if raw == "":
        return np.nan, np.nan
    low = np.nan
    high = np.nan
    number_pattern = r'[\d]+(?:[.,]\d{3})*(?:[.,]\d+)?'
    low_candidates = re.findall(number_pattern, raw)
    norm_nums = []
    for num in low_candidates:
        if ',' in num and '.' in num:
            num = num.replace('.', '').replace(',', '.')
        elif ',' in num:
            if len(num.split(',')[1]) > 2:
                num = num.replace(',', '.')
            else:
                num = num.replace(',', '')
        elif '.' in num:
            num = num
        try:
            norm_nums.append(float(num))
        except:
            pass
    raw_lower = raw.lower()
    if len(norm_nums) >= 2 and ("à" in raw_lower or "-" in raw_lower or "to" in raw_lower or "de" in raw_lower):
        low = float(norm_nums[0])
        high = float(norm_nums[1])
        return low, high
    if "inférieur" in raw_lower or "inferieur" in raw_lower or "moins de" in raw_lower or "inf" in raw_lower:
        if len(norm_nums) >= 1:
            high = float(norm_nums[0])
            low = 0.0
            return low, high
    if "supérieur" in raw_lower or "superieur" in raw_lower or "plus de" in raw_lower or ">" in raw_lower:
        if len(norm_nums) >= 1:
            low = float(norm_nums[0])
            high = np.nan
            return low, high
    if len(norm_nums) == 1:
        val = float(norm_nums[0])
        return val, val
    return np.nan, np.nan

def find_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] or (df[col].dtype == 'object' and pd.to_numeric(df[col], errors='coerce').notna().sum() > 0):
            numeric_cols.append(col)
    return numeric_cols

def create_group_display_name(companies_list: List[str]) -> str:
    if not companies_list:
        return "Groupe Vide"
    if len(companies_list) == 1:
        return companies_list[0]
    return f"Groupe ({len(companies_list)} entreprises)"

def create_group_from_selection(selected_companies: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not selected_companies:
        return {}
    group_name = st.session_state.get('new_group_name', f"Groupe {len(st.session_state.get('groups', {})) + 1}")
    all_products_raw = []
    company_names_for_display = []
    for c in selected_companies:
        if isinstance(c, dict) and c.get('products'):
            all_products_raw.append(c['products'])
        if isinstance(c, dict) and c.get('name'):
            company_names_for_display.append(c['name'])
    group_data = {
        'name': group_name,
        'companies': selected_companies,
        'display_name': create_group_display_name(company_names_for_display) if company_names_for_display else group_name,
        'products': join_products_for_group(all_products_raw, separator=" | "),
        'revenue_min': min([c.get('revenue_min', np.inf) for c in selected_companies if isinstance(c, dict)]),
        'revenue_max': max([c.get('revenue_max', -np.inf) for c in selected_companies if isinstance(c, dict)]),
        'operating_income': sum([c.get('operating_income', 0) for c in selected_companies if isinstance(c, dict)]),
        'city': ', '.join(set([c.get('city', '') for c in selected_companies if isinstance(c, dict) and c.get('city')])),
        'sector': ', '.join(set([c.get('sector', '') for c in selected_companies if isinstance(c, dict) and c.get('sector')])),
        'metadata': selected_companies,
        'company_names': [c['name'] for c in selected_companies if isinstance(c, dict) and c.get('name')]
    }
    return group_data
df_companies = load_or_upload("companies.xlsx", "companies.xlsx")
df_kerix = load_or_upload("kerix.xlsx", "kerix.xlsx")
if df_companies is None and df_kerix is None:
    st.error("Au moins une des bases (companies.xlsx ou kerix.xlsx) doit être fournie.")
    st.stop()

def prepare_df(df: pd.DataFrame, is_companies: bool = False) -> Tuple[pd.DataFrame, dict]:
    df = df.copy()
    info = {}
    revenue_candidates = [
        "Chiffre d'affaires 2023 (Dhs)", "Chiffre d'affaires 2023 (Dhs) ", "Chiffre d'affaires 2023",
        "Chiffre d'affaires", "Chiffre d'Affaires", "Chiffre d'Affaires 2023 (Dhs)", "Chiffre d'Affaires 2023 (Dhs) "
    ]
    rev_col = get_col(df, revenue_candidates)
    info['revenue_col'] = rev_col
    if rev_col:
        df['_revenue_raw'] = df[rev_col]
    else:
        for c in df.columns:
            if 'chiffre' in c.lower():
                rev_col = c
                df['_revenue_raw'] = df[c]
                info['revenue_col'] = rev_col
                break
    if '_revenue_raw' not in df.columns:
        df['_revenue_raw'] = np.nan
    mins = []
    maxs = []
    for v in df['_revenue_raw'].fillna("").astype(str).tolist():
        mi, ma = parse_revenue_text(v)
        mins.append(mi)
        maxs.append(ma)
    df['_revenue_min'] = pd.Series(mins, dtype='float64')
    df['_revenue_max'] = pd.Series(maxs, dtype='float64')
    operating_candidates = [
        "Resultat d'exploitation 2023 (Dhs)", "Resultat d'exploitation 2023 (Dhs) ", "Resultat d'exploitation 2023",
        "Resultat d'exploitation", "Resultat d'exploitation", "Resultat d'exploitation 2023 (Dhs)", "Résultat d'exploitation 2023 (Dhs) "
    ]
    op_col = get_col(df, operating_candidates)
    if op_col:
        df['_operating_income'] = pd.to_numeric(df[op_col], errors='coerce').fillna(0)
    else:
        df['_operating_income'] = 0.0
    city_candidates = ["Ville RC", "Ville", "City", "Localisation", "Siège social"]
    city_col = get_col(df, city_candidates)
    if city_col:
        df['_city'] = df[city_col].fillna("").astype(str)
    else:
        df['_city'] = ""
    sector_candidates = ["Secteur", "Sector", "Activité principale", "Branche"]
    sector_col = get_col(df, sector_candidates)
    if sector_col:
        df['_sector'] = df[sector_col].fillna("").astype(str)
    else:
        df['_sector'] = ""
    if is_companies:
        name_candidates = ["Raison Sociale (Kerix)", "Raison Sociale (Maroc1000 Nouvelle)", "Raison Sociale (Maroc1000 ancienne)", "Raison Sociale", "Raison Sociale (Maroc1000)"]
        existing = [c for c in name_candidates if c in df.columns]
        info['name_candidates'] = existing
        def fallback_row_name(row):
            for c in existing:
                val = row.get(c)
                if pd.notna(val) and str(val).strip() != "":
                    return str(val)
            return ""
        df['_display_name'] = df.apply(fallback_row_name, axis=1)
    else:
        name_candidates_k = ["Raison Sociale", "Raison Sociale "]
        existing_k = [c for c in name_candidates_k if c in df.columns]
        info['name_candidates'] = existing_k
        if existing_k:
            primary = existing_k[0]
            df['_display_name'] = df[primary].fillna("").astype(str)
        else:
            txt_cols = [c for c in df.columns if df[c].dtype == object]
            df['_display_name'] = df[txt_cols[0]].fillna("").astype(str) if txt_cols else ""
    products_col = get_col(df, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
    if products_col:
        df['_products'] = df[products_col].fillna("").astype(str)
    else:
        df['_products'] = ""
    info['products_col'] = products_col
    return df, info
df_companies_prepared, info_companies = (prepare_df(df_companies, is_companies=True) if df_companies is not None else (None, {}))
df_kerix_prepared, info_kerix = (prepare_df(df_kerix, is_companies=False) if df_kerix is not None else (None, {}))
if 'groups' not in st.session_state:
    st.session_state.groups = {}
if 'group_exclusions' not in st.session_state:
    st.session_state.group_exclusions = {'companies': set(), 'kerix': set()}

def apply_group_exclusions(df, db_name):
    if not st.session_state.groups:
        return df
    excluded_names = st.session_state.group_exclusions.get(db_name, set())
    if '_display_name' in df.columns:
        df = df[~df['_display_name'].isin(excluded_names)].reset_index(drop=True)
    return df

df_companies_prepared = apply_group_exclusions(df_companies_prepared, 'companies') if df_companies_prepared is not None else None
df_kerix_prepared = apply_group_exclusions(df_kerix_prepared, 'kerix') if df_kerix_prepared is not None else None

def extract_available_years(df_companies, df_kerix):
    all_years = set()
    if df_companies is not None:
        for col in df_companies.columns:
            if '202' in col and any(char.isdigit() for char in col):
                year_match = re.search(r'\b(20\d{2})\b', col)
                if year_match:
                    all_years.add(int(year_match.group(1)))
    if df_kerix is not None:
        for col in df_kerix.columns:
            if '202' in col and any(char.isdigit() for char in col):
                year_match = re.search(r'\b(20\d{2})\b', col)
                if year_match:
                    all_years.add(int(year_match.group(1)))
    return sorted(list(all_years)) if all_years else [2023]
available_years = extract_available_years(df_companies, df_kerix)
st.sidebar.title("📅 Filtres Année")
if len(available_years) > 1:
    year_mode = st.sidebar.radio("Mode de filtrage année", ["Toutes les années", "Sélectionner des années"], key="year_mode")
    if year_mode == "Sélectionner des années":
        selected_years = st.sidebar.multiselect("Années", options=available_years, default=available_years[-3:] if len(available_years) >= 3 else available_years)
    else:
        selected_years = available_years
else:
    selected_years = available_years
    st.sidebar.info(f"Données disponibles pour {selected_years[0]}")

def get_year_revenue_columns(df, selected_years):
    if df is None:
        return None
    year_cols = {}
    revenue_base = ["Chiffre d'affaires", "Chiffre d'Affaires"]
    operating_base = ["Resultat d'exploitation", "Résultat d'exploitation"]
    for year in selected_years:
        rev_candidates = [f"{base} {year} (Dhs)" for base in revenue_base] + [f"{base} {year}" for base in revenue_base]
        op_candidates = [f"{base} {year} (Dhs)" for base in operating_base] + [f"{base} {year}" for base in operating_base]
        rev_col = get_col(df, rev_candidates)
        op_col = get_col(df, op_candidates)
        year_cols[year] = {'revenue': rev_col, 'operating': op_col}
    return year_cols

def extract_seed_tokens(seed_series):
    prods = ""
    for k in ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"]:
        if k in seed_series.index and pd.notna(seed_series.get(k)):
            prods = seed_series.get(k)
            break
    if prods == "":
        prods = seed_series.get("Produits / Services", "") or seed_series.get("products", "") or ""
    return split_tokens(prods)

def prepare_df_with_years(df, selected_years, is_companies=False):
    if df is None:
        return None, {}
    df = df.copy()
    info = {}
    year_cols = get_year_revenue_columns(df, selected_years)
    if not year_cols:
        return prepare_df(df, is_companies), info
    primary_year = max(selected_years)
    primary_rev_col = year_cols[primary_year]['revenue']
    if primary_rev_col:
        df['_revenue_raw'] = df[primary_rev_col]
        info['revenue_col'] = primary_rev_col
    else:
        revenue_candidates = [
            "Chiffre d'affaires 2023 (Dhs)", "Chiffre d'affaires 2023 (Dhs) ", "Chiffre d'affaires 2023",
            "Chiffre d'affaires", "Chiffre d'Affaires", "Chiffre d'Affaires 2023 (Dhs)", "Chiffre d'Affaires 2023 (Dhs) "
        ]
        rev_col = get_col(df, revenue_candidates)
        info['revenue_col'] = rev_col
        if rev_col:
            df['_revenue_raw'] = df[rev_col]
        else:
            for c in df.columns:
                if 'chiffre' in c.lower():
                    rev_col = c
                    df['_revenue_raw'] = df[c]
                    info['revenue_col'] = rev_col
                    break
    if '_revenue_raw' not in df.columns:
        df['_revenue_raw'] = np.nan
    mins = []
    maxs = []
    for v in df['_revenue_raw'].fillna("").astype(str).tolist():
        mi, ma = parse_revenue_text(v)
        mins.append(mi)
        maxs.append(ma)
    df['_revenue_min'] = pd.Series(mins, dtype='float64')
    df['_revenue_max'] = pd.Series(maxs, dtype='float64')
    primary_op_col = year_cols[primary_year]['operating']
    if primary_op_col:
        df['_operating_income'] = pd.to_numeric(df[primary_op_col], errors='coerce').fillna(0)
        info['operating_col'] = primary_op_col
    else:
        operating_candidates = [
            "Resultat d'exploitation 2023 (Dhs)", "Resultat d'exploitation 2023 (Dhs) ", "Resultat d'exploitation 2023",
            "Resultat d'exploitation", "Resultat d'exploitation", "Resultat d'exploitation 2023 (Dhs)", "Résultat d'exploitation 2023 (Dhs) "
        ]
        op_col = get_col(df, operating_candidates)
        if op_col:
            df['_operating_income'] = pd.to_numeric(df[op_col], errors='coerce').fillna(0)
            info['operating_col'] = op_col
        else:
            df['_operating_income'] = 0.0
    city_candidates = ["Ville RC", "Ville", "City", "Localisation", "Siège social"]
    city_col = get_col(df, city_candidates)
    if city_col:
        df['_city'] = df[city_col].fillna("").astype(str)
    else:
        df['_city'] = ""
    sector_candidates = ["Secteur", "Sector", "Activité principale", "Branche"]
    sector_col = get_col(df, sector_candidates)
    if sector_col:
        df['_sector'] = df[sector_col].fillna("").astype(str)
    else:
        df['_sector'] = ""
    if is_companies:
        name_candidates = ["Raison Sociale (Kerix)", "Raison Sociale (Maroc1000 Nouvelle)", "Raison Sociale (Maroc1000 ancienne)", "Raison Sociale", "Raison Sociale (Maroc1000)"]
        existing = [c for c in name_candidates if c in df.columns]
        info['name_candidates'] = existing
        def fallback_row_name(row):
            for c in existing:
                val = row.get(c)
                if pd.notna(val) and str(val).strip() != "":
                    return str(val)
            return ""
        df['_display_name'] = df.apply(fallback_row_name, axis=1)
    else:
        name_candidates_k = ["Raison Sociale", "Raison Sociale "]
        existing_k = [c for c in name_candidates_k if c in df.columns]
        info['name_candidates'] = existing_k
        if existing_k:
            primary = existing_k[0]
            df['_display_name'] = df[primary].fillna("").astype(str)
        else:
            txt_cols = [c for c in df.columns if df[c].dtype == object]
            df['_display_name'] = df[txt_cols[0]].fillna("").astype(str) if txt_cols else ""
    products_col = get_col(df, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
    if products_col:
        df['_products'] = df[products_col].fillna("").astype(str)
    else:
        df['_products'] = ""
    info['products_col'] = products_col
    info['year_cols'] = year_cols
    info['selected_years'] = selected_years
    return df, info
df_companies_prepared, info_companies = prepare_df_with_years(df_companies, selected_years, is_companies=True) if df_companies is not None else (None, {})
df_kerix_prepared, info_kerix = prepare_df_with_years(df_kerix, selected_years, is_companies=False) if df_kerix is not None else (None, {})
combined_revs_min = []
combined_revs_max = []
if df_companies_prepared is not None:
    combined_revs_min.append(df_companies_prepared['_revenue_min'].dropna())
    combined_revs_max.append(df_companies_prepared['_revenue_max'].dropna())
if df_kerix_prepared is not None:
    combined_revs_min.append(df_kerix_prepared['_revenue_min'].dropna())
    combined_revs_max.append(df_kerix_prepared['_revenue_max'].dropna())
all_mins = pd.concat(combined_revs_min) if combined_revs_min else pd.Series(dtype='float64')
all_maxs = pd.concat(combined_revs_max) if combined_revs_max else pd.Series(dtype='float64')
if len(all_mins) > 0 and len(all_maxs) > 0:
    global_min = float(all_mins.min())
    global_max = float(all_maxs.max())
elif len(all_mins) > 0:
    global_min = float(all_mins.min())
    global_max = global_min * 100 if global_min > 0 else 1_000_000.0
elif len(all_maxs) > 0:
    global_max = float(all_maxs.max())
    global_min = max(0.0, global_max / 100.0)
else:
    global_min, global_max = 0.0, 10_000_000.0
if 'current_search_target' not in st.session_state:
    st.session_state.current_search_target = 'single'
st.subheader("🏢 Gestion des Groupes d'Entreprises")
group_tab1, group_tab2 = st.tabs(["Créer un groupe", "Groupes existants"])

with group_tab1:
    new_group_name = st.text_input("Nom du groupe", value=f"Groupe {len(st.session_state.groups) + 1}")
    st.session_state.new_group_name = new_group_name
    available_companies = []
    if df_companies_prepared is not None:
        for _, row in df_companies_prepared.iterrows():
            available_companies.append({
                'name': row['_display_name'],
                'products': row.get('_products', ''),
                'revenue_min': row.get('_revenue_min', np.nan),
                'revenue_max': row.get('_revenue_max', np.nan),
                'operating_income': row.get('_operating_income', 0),
                'city': row.get('_city', ''),
                'sector': row.get('_sector', '')
            })
    if df_kerix_prepared is not None:
        for _, row in df_kerix_prepared.iterrows():
            available_companies.append({
                'name': row['_display_name'],
                'products': row.get('_products', ''),
                'revenue_min': row.get('_revenue_min', np.nan),
                'revenue_max': row.get('_revenue_max', np.nan),
                'operating_income': row.get('_operating_income', 0),
                'city': row.get('_city', ''),
                'sector': row.get('_sector', '')
            })
    selected_indices = st.multiselect(
        "Sélectionnez les entreprises pour le groupe",
        options=list(range(len(available_companies))),
        format_func=lambda i: available_companies[i]['name'],
        max_selections=20
    )
    submitted = st.button("Créer le groupe")
    selected_companies = [available_companies[i] for i in selected_indices]
    if submitted and selected_companies:
        group_data = create_group_from_selection(selected_companies)
        st.session_state.groups[group_data['name']] = group_data
        for company in selected_companies:
            if isinstance(company, dict) and company.get('name'):
                st.session_state.group_exclusions['companies'].add(company['name']) if df_companies_prepared is not None else None
                st.session_state.group_exclusions['kerix'].add(company['name']) if df_kerix_prepared is not None else None
        st.success(f"Groupe '{group_data['name']}' créé avec {len(selected_companies)} entreprises!")
        st.rerun()

with group_tab2:
    if st.session_state.groups:
        for group_name, group_data in list(st.session_state.groups.items()):
            if isinstance(group_data, dict) and 'companies' in group_data:
                with st.expander(f"🔸 {group_name} - {group_data.get('display_name', 'Groupe sans nom')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Entreprises", len(group_data['companies']))
                        st.write(f"**Ville(s):** {group_data.get('city', 'N/A')}")
                        st.write(f"**Secteur(s):** {group_data.get('sector', 'N/A')}")
                    with col2:
                        st.metric("CA Min", compact_num(group_data.get('revenue_min', 0)))
                        st.metric("CA Max", compact_num(group_data.get('revenue_max', 0)))
                    with col3:
                        st.metric("Résultat d'exploitation", compact_num(group_data.get('operating_income', 0)))
                    if st.button(f"❌ Supprimer {group_name}", key=f"delete_{group_name}"):
                        if group_name in st.session_state.groups:
                            del st.session_state.groups[group_name]
                            st.session_state.group_exclusions['companies'].clear()
                            st.session_state.group_exclusions['kerix'].clear()
                            for g_data in st.session_state.groups.values():
                                if isinstance(g_data, dict) and 'company_names' in g_data:
                                    for company in g_data['company_names']:
                                        st.session_state.group_exclusions['companies'].add(company)
                                        st.session_state.group_exclusions['kerix'].add(company)
                            df_companies_prepared = apply_group_exclusions(df_companies, 'companies') if df_companies is not None else None
                            df_kerix_prepared = apply_group_exclusions(df_kerix, 'kerix') if df_kerix is not None else None
                            st.rerun()
    else:
        st.info("Aucun groupe créé pour le moment. Utilisez l'onglet de gauche pour en créer un.")

st.subheader("🔍 Moteur de Recherche")
search_mode = st.radio("Mode de recherche", ["Entreprise unique", "Groupe d'entreprises"], horizontal=True, key="search_mode")

st.session_state.current_search_target = "group" if search_mode == "Groupe d'entreprises" else "single"
if search_mode == "Groupe d'entreprises" and not st.session_state.groups:
    st.warning("Créez d'abord un groupe dans la section Gestion des Groupes!")
    st.stop()

seed_row = None
seed_prods = None
seed_name_for_exclusion = None
if search_mode == "Entreprise unique":
    if df_kerix_prepared is None:
        st.warning("Kerix non disponible - choisissez Base totale ou Manuelle.")
    else:
        options = df_kerix_prepared['_display_name'].fillna("N/A").astype(str).tolist()
        seed_choice_idx = st.selectbox("Choisir la société seed (Kerix)", options=list(range(len(options))),
                                        format_func=lambda i: options[i])
        seed_row = df_kerix_prepared.iloc[int(seed_choice_idx)]
        seed_name_for_exclusion = seed_row.get('_display_name', "")
        seed_prods = extract_seed_tokens(seed_row)
    non_existant = st.checkbox("Entreprise non existante dans la base", False)
    if non_existant:
        with st.form("manual_seed_form_global"):
            ms_name = st.text_input("Nom de la société (seed)")
            ms_products = st.text_area("Produits / Services (séparez par , ; / )")
            submitted = st.form_submit_button("Créer seed manuel")
        if not submitted:
            st.info("Soumettez le formulaire pour créer la société seed manuelle.")
        else:
            seed_row = pd.Series({
                "name_manual": ms_name,
                "Produits / Services": ms_products
            })
            seed_name_for_exclusion = ms_name
            seed_prods = extract_seed_tokens(seed_row)
elif search_mode == "Groupe d'entreprises":
    group_options = list(st.session_state.groups.keys())
    selected_group = st.selectbox("Choisir le groupe seed", group_options)
    if selected_group:
        group_data = st.session_state.groups[selected_group]
        if isinstance(group_data, dict):
            seed_row = pd.Series({
                "name": group_data.get('name', selected_group),
                "_display_name": group_data.get('display_name', group_data.get('name', selected_group))
            })
            seed_name_for_exclusion = group_data.get('display_name', group_data.get('name', selected_group))
            joined_prods = group_data.get('products', '')
            seed_prods = [t.strip().lower() for t in joined_prods.split(' | ') if t.strip()]

st.subheader("Filtres")
use_ca_filter = st.checkbox("Activer filtre CA (Chiffre d'affaires)", value=False)
if use_ca_filter:
    st.markdown(f"##### Filtre CA - Min: {compact_num(global_min)} | Max: {compact_num(global_max)}")
    st.markdown(f"**Année(s) utilisée(s) pour CA:** {', '.join(map(str, selected_years))}")
    min_val = max(0.0, global_min)
    max_val = max(min_val + 1.0, global_max)
    range_span = max_val - min_val
    approx_step = 10 ** (int(np.floor(np.log10(max(1.0, range_span)))) - 1)
    approx_step = max(1.0, approx_step)
    col1, col2 = st.columns(2)
    with col1:
        ca_min_input = st.number_input(
            "Min (Dhs)",
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(min_val),
            step=float(approx_step)
        )
    with col2:
        ca_max_input = st.number_input(
            "Max (Dhs)",
            min_value=float(ca_min_input),
            max_value=float(max_val),
            value=float(max_val),
            step=float(approx_step)
        )
    ca_min_sel, ca_max_sel = ca_min_input, ca_max_input
    st.markdown(f"Min - Max sélectionné : **{compact_num(ca_min_sel)}** - **{compact_num(ca_max_sel)}**")
else:
    ca_min_sel, ca_max_sel = None, None
use_re_filter = st.checkbox("Activer filtre Résultat d'exploitation", value=False)
if use_re_filter:
    all_operating = []
    if df_companies_prepared is not None:
        all_operating.append(df_companies_prepared['_operating_income'])
    if df_kerix_prepared is not None:
        all_operating.append(df_kerix_prepared['_operating_income'])
    if all_operating:
        all_operating = pd.concat(all_operating)
        re_min_val = float(all_operating.min())
        re_max_val = float(all_operating.max())
    else:
        re_min_val, re_max_val = 0.0, 1_000_000.0
    st.markdown(f"##### Filtre Résultat d'exploitation - Min: {compact_num(re_min_val)} | Max: {compact_num(re_max_val)}")
    st.markdown(f"**Année(s) utilisée(s) pour RE:** {', '.join(map(str, selected_years))}")
    re_range = re_max_val - re_min_val
    re_step = 10 ** (int(np.floor(np.log10(max(1.0, abs(re_range)))) - 1))
    re_step = max(1.0, re_step)
    col1, col2 = st.columns(2)
    with col1:
        re_min_input = st.number_input(
            "Min (Dhs)",
            min_value=float(re_min_val),
            max_value=float(re_max_val),
            value=float(re_min_val),
            step=float(re_step)
        )
    with col2:
        re_max_input = st.number_input(
            "Max (Dhs)",
            min_value=float(re_min_input),
            max_value=float(re_max_val),
            value=float(re_max_val),
            step=float(re_step)
        )
    re_min_sel, re_max_sel = re_min_input, re_max_input
    st.markdown(f"Min - Max sélectionné : **{compact_num(re_min_sel)}** - **{compact_num(re_max_sel)}**")
else:
    re_min_sel, re_max_sel = None, None

st.markdown("##### Filtre Ville")
all_cities = set()
if df_companies_prepared is not None:
    all_cities.update(df_companies_prepared['_city'].dropna().unique())
if df_kerix_prepared is not None:
    all_cities.update(df_kerix_prepared['_city'].dropna().unique())
city_options = sorted(list(all_cities))
selected_cities = st.multiselect("Villes", options=city_options, default=[])

st.markdown("##### Filtre Secteur")
all_sectors = set()
if df_companies_prepared is not None:
    all_sectors.update(df_companies_prepared['_sector'].dropna().unique())
if df_kerix_prepared is not None:
    all_sectors.update(df_kerix_prepared['_sector'].dropna().unique())
sector_options = sorted(list(all_sectors))
selected_sectors = st.multiselect("Secteurs", options=sector_options, default=[])

if seed_row is None:
    st.warning("Sélectionnez ou créez une société seed pour lancer les recherches.")
    st.stop()

st.subheader("B. Algorithme Concurrentiel")
tabs = st.tabs(["Base totale", "Kerix", "Paramètres"])
with tabs[2]:
    st.markdown("### Paramètres")
    prod_threshold = st.slider("Seuil similarité produits/services (fuzzy)", min_value=0.5, max_value=0.95, value=0.75, step=0.01, key="prod_threshold")
    top_n_preview = st.number_input("Top N candidats à afficher", min_value=5, max_value=500, value=25, step=5, key="top_n_preview")
    sort_metric = st.selectbox("Trier par", ["product_fraction"], index=0, key="sort_metric")

def compute_matches_for_df(df, label):
    if df is None:
        return None, None
    candidates = df.copy().reset_index(drop=True)
    mask_city = candidates['_city'].isin(selected_cities) if selected_cities else pd.Series([True] * len(candidates))
    mask_sector = candidates['_sector'].isin(selected_sectors) if selected_sectors else pd.Series([True] * len(candidates))
    if use_ca_filter:
        sel_min = float(ca_min_sel)
        sel_max = float(ca_max_sel)
        mask_revenue = (~candidates['_revenue_min'].isna()) & (~candidates['_revenue_max'].isna())
        mask_revenue_overlap = mask_revenue & (candidates['_revenue_max'] >= sel_min) & (candidates['_revenue_min'] <= sel_max)
        candidates = candidates[mask_revenue_overlap & mask_city & mask_sector].reset_index(drop=True)
    else:
        candidates = candidates[mask_city & mask_sector].reset_index(drop=True)
    if use_re_filter:
        mask_operating = (candidates['_operating_income'] >= re_min_sel) & (candidates['_operating_income'] <= re_max_sel)
        candidates = candidates[mask_operating].reset_index(drop=True)
    try:
        if seed_name_for_exclusion and '_display_name' in candidates.columns:
            candidates = candidates[candidates['_display_name'].astype(str) != str(seed_name_for_exclusion)]
    except Exception:
        pass
    candidates = candidates.reset_index(drop=True)
    if candidates.shape[0] == 0:
        return candidates, []
    products_col = get_col(df, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
    prods_matched = []
    for _, r in candidates.iterrows():
        cand_prods = split_tokens(r.get(products_col, "")) if products_col else []
        mp, tp = count_similar(seed_prods, cand_prods, prod_threshold)
        prods_matched.append((mp, tp))
    candidates['products_common_count'] = [m for m, t in prods_matched]
    candidates['products_seed_count'] = [t for m, t in prods_matched]
    candidates['product_fraction'] = candidates.apply(lambda r: (r['products_common_count'] / max(1, r['products_seed_count'])) if r['products_seed_count']>0 else 0.0, axis=1)
    candidates['product_fraction_str'] = candidates.apply(lambda r: frac_str(r['products_common_count'], r['products_seed_count']), axis=1)
    candidates = candidates.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)
    display_cols = []
    if '_display_name' in candidates.columns:
        display_cols.append('_display_name')
    display_cols += ['product_fraction_str']
    candidates['_CA_display'] = candidates.apply(lambda r: format_range_compact(r['_revenue_min'], r['_revenue_max']), axis=1)
    display_cols.append('_CA_display')
    candidates['_OP_display'] = candidates['_operating_income'].apply(lambda x: compact_num(x))
    display_cols.append('_OP_display')
    candidates['_City_display'] = candidates['_city']
    display_cols.append('_City_display')
    candidates['_Sector_display'] = candidates['_sector']
    display_cols.append('_Sector_display')
    if 'selected_years' in info_companies if label == "Base totale" else 'selected_years' in info_kerix:
        years_str = ', '.join(map(str, selected_years))
        candidates['_Year_display'] = f"Données {years_str}"
        display_cols.append('_Year_display')
    return candidates, display_cols

with tabs[0]:
    st.markdown("### Onglet: Base totale")
    if df_companies_prepared is None:
        st.warning("Fichier companies.xlsx non chargé.")
    else:
        st.markdown("#### Seed utilisée")
        st.write({"seed_name": seed_name_for_exclusion, "seed_products_tokens": seed_prods})
        candidates, display_cols = compute_matches_for_df(df_companies_prepared, "Base totale")
        if candidates is None or candidates.shape[0] == 0:
            st.info("Aucun candidat trouvé dans la Base totale pour les filtres sélectionnés.")
        else:
            st.markdown(f"##### Top {int(top_n_preview)} candidats - Base totale")
            rename_map = {
                '_display_name': 'Raison Sociale',
                'product_fraction_str': 'Matching Score (Produits)',
                '_CA_display': "Chiffre d'affaires",
                '_OP_display': "Résultat d'exploitation",
                '_City_display': "Ville",
                '_Sector_display': "Secteur"
            }
            if '_Year_display' in display_cols:
                rename_map['_Year_display'] = "Année(s) des données"
            st.dataframe(candidates.head(int(top_n_preview))[display_cols].fillna("").rename(columns=rename_map))
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                candidates.to_excel(writer, sheet_name="base_totale_results", index=False)
            out.seek(0)
            st.download_button("📥 Télécharger résultats (Base totale)", data=out, file_name="base_totale_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[1]:
    st.markdown("### Onglet: Kerix")
    if df_kerix_prepared is None:
        st.warning("Fichier kerix.xlsx non chargé.")
    else:
        st.markdown("#### Seed utilisée")
        st.write({"seed_name": seed_name_for_exclusion, "seed_products_tokens": seed_prods})
        candidates_k, display_cols_k = compute_matches_for_df(df_kerix_prepared, "Kerix")
        if candidates_k is None or candidates_k.shape[0] == 0:
            st.info("Aucun candidat trouvé dans Kerix pour les filtres sélectionnés.")
        else:
            st.markdown(f"##### Top {int(top_n_preview)} candidats - Kerix")
            rename_map = {
                '_display_name': 'Raison Sociale',
                'product_fraction_str': 'Matching Score (Produits)',
                '_CA_display': "Chiffre d'affaires",
                '_OP_display': "Résultat d'exploitation",
                '_City_display': "Ville",
                '_Sector_display': "Secteur"
            }
            if '_Year_display' in display_cols_k:
                rename_map['_Year_display'] = "Année(s) des données"
            st.dataframe(candidates_k.head(int(top_n_preview))[display_cols_k].fillna("").rename(columns=rename_map))
            out_k = io.BytesIO()
            with pd.ExcelWriter(out_k, engine="openpyxl") as writer:
                candidates_k.to_excel(writer, sheet_name="kerix_results", index=False)
            out_k.seek(0)
            st.download_button("📥 Télécharger résultats (Kerix)", data=out_k, file_name="kerix_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")