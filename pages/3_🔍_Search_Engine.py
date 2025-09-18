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

def split_tokens(text: str) -> Tuple[List[str], List[str]]:
    if pd.isna(text):
        return [], []
    
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
    deduped = []
    for v in cleaned:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    full_with_dups = cleaned.copy()    
    return deduped, full_with_dups

def join_products_for_group(products_list: List[str], separator: str = " | ") -> str:
    if not products_list:
        return ""
    all_tokens = []
    for products_str in products_list:
        if pd.isna(products_str) or not str(products_str).strip():
            continue
        result = split_tokens(str(products_str).strip())
        if isinstance(result, tuple) and len(result) == 2:
            company_tokens = result[0]
        else:
            company_tokens = result
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

def compute_raw_fuzzy_scores(seed_tokens: List[str], cand_tokens: List[str], threshold: float) -> List[float]:
    if not seed_tokens or not cand_tokens:
        return [0.0] * len(seed_tokens)
    
    raw_scores = []
    for s in seed_tokens:
        best_score = 0.0
        for c in cand_tokens:
            score = fuzzy_ratio(s, c)
            if score > best_score:
                best_score = score
        raw_scores.append(best_score)
    
    return raw_scores

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
    except (ValueError, TypeError):
        return str(n)
    if abs(n) < 1.0:
        return f"{n:.6f}"
    
    B = 1e9
    M = 1e6
    K = 1e3    
    if abs(n) >= B:
        scaled = n / B
        return f"{scaled:.2f}B"
    elif abs(n) >= M:
        scaled = n / M
        return f"{scaled:.1f}M"
    elif abs(n) >= K:
        scaled = n / K
        return f"{scaled:.1f}K"
    else:
        if n == int(n):
            return f"{int(n):,}"
        else:
            return f"{n:,.2f}"
        
def format_range_compact(mi, ma) -> str:
    if pd.isna(mi) and pd.isna(ma):
        return "N/A"
    if (pd.isna(mi) or mi == 0.0) and not pd.isna(ma):
        return f"≤ {compact_num(ma)}"
    if pd.isna(ma) and not pd.isna(mi):
        return f"≥ {compact_num(mi)}"
    if mi == ma:
        return compact_num(mi)
    return f"de {compact_num(mi)} à {compact_num(ma)}"

import re
import numpy as np
import pandas as pd
from typing import Tuple, Union

def parse_companies_revenue(s: Union[str, int, float]) -> Tuple[Union[int, float], Union[int, float]]:
    s = s.split(".")[0]
    if pd.isna(s):
        return np.nan, np.nan
    raw = str(s).strip()
    if not raw or raw.lower() in ['n/a', 'non disponible', 'non communiqué', 'nc', '']:
        return np.nan, np.nan
    
    raw_clean = re.sub(r'\s*(dh[s]?|dhs|mad|dh|€|USD|\$)\.?\s*$', '', raw, flags=re.IGNORECASE).strip()
    number_pattern = r'\b(\d{1,3}(?:\.\d{3})+)\b'
    numbers = re.findall(number_pattern, raw_clean)
    if not numbers:
        fallback_pattern = r'\b(\d+(?:\.\d+)*)\b'
        numbers = re.findall(fallback_pattern, raw_clean)
    
    if not numbers:
        numbers = re.findall(r'\b(\d+)\b', raw_clean)
    
    norm_nums = []
    for num in numbers:
        try:
            cleaned_num = num.replace('.', '')
            cleaned_num = cleaned_num.replace(',', '')
            value = float(cleaned_num)
            norm_nums.append(int(value))
        except (ValueError, TypeError, OverflowError):
            continue
    
    if len(norm_nums) == 1:
        return norm_nums[0], norm_nums[0]
    elif len(norm_nums) > 1:
        return min(norm_nums), max(norm_nums)
    return np.nan, np.nan

def parse_kerix_revenue(s: Union[str, int, float]) -> Tuple[int, int]:
    if pd.isna(s):
        return np.nan, np.nan
    raw = str(s).strip()
    if not raw or raw.lower() in ['n/a', 'non disponible', 'non communiqué', 'nc', '', 'non défini', 'non communiqué']:
        return np.nan, np.nan
    raw_lower = raw.lower().strip()
    cleaned = re.sub(r'\s*(dh[s]?|mad|€|USD|\$)\.?\s*$', '', raw, flags=re.IGNORECASE).strip()
    comma_pattern = r'(\d{1,4}(?:,\d{3})*(?:\.\d{1,2})?)'
    comma_numbers = re.findall(comma_pattern, cleaned)
    simple_pattern = r'(\d+(?:\.\d+)?)'
    simple_numbers = re.findall(simple_pattern, cleaned)
    parsed_numbers = []
    for num_str in comma_numbers:
        try:
            clean_num = re.sub(r',', '', num_str)
            value = int(clean_num)
            parsed_numbers.append(value)
        except (ValueError, TypeError):
            continue
    # seen_values = set(parsed_numbers)
    # for num_str in simple_numbers:
    #     try:
    #         value = int(num_str)
    #         if value not in seen_values and value > 0:
    #             parsed_numbers.append(value)
    #             seen_values.add(value)
    #     except (ValueError, TypeError):
    #         continue
    valid_numbers = sorted([n for n in parsed_numbers if n > 0 and not pd.isna(n)])
    if len(valid_numbers) == 2 and valid_numbers[-1] == 1000000000:
        print(valid_numbers[0])
    if ("de" in raw_lower and "à" in raw_lower) or "entre" in raw_lower:
        if len(valid_numbers) >= 2:
            return valid_numbers[0], valid_numbers[1]
        elif len(valid_numbers) == 1:
            return valid_numbers[0], valid_numbers[0]
        else:
            return np.nan, np.nan

    elif any(word in raw_lower for word in ["inférieur", "inferieur", "moins de"]):
        if len(valid_numbers) >= 1:
            return 0, valid_numbers[0]
        else:
            return 0, np.nan

    elif any(word in raw_lower and "à" in raw_lower for word in ["supérieur", "superieur", "plus de"]):
        if len(valid_numbers) >= 1:
            return valid_numbers[0], np.nan
        else:
            return np.nan, np.nan
    else:
        if len(valid_numbers) == 1:
            return valid_numbers[0], valid_numbers[0]
        elif len(valid_numbers) >= 2:
            return valid_numbers[0], valid_numbers[-1]
        else:
            return np.nan, np.nan

def clean_operating_income(value, is_companies: bool = False) -> int:
    if pd.isna(value):
        return 0
    
    value = int(value) if value else np.nan
    raw = str(value).strip()
    if not raw or raw.lower() in ['n/a', 'non disponible', 'non communiqué', 'nc', '']:
        return 0
    
    try:
        temp_raw = re.sub(r'\s*(dh[s]?|dhs|mad|dh|€|USD|\$)\.?\s*', '', raw, flags=re.IGNORECASE).strip()
        if is_companies:
            cleaned = temp_raw.replace('.', '').replace(',', '')
            cleaned = re.sub(r'[^\d]', '', cleaned)
            if not cleaned:
                return 0
            value = int(cleaned)
            return value
        else:
            cleaned = temp_raw.replace(',', '').replace('.', '')
            cleaned = re.sub(r'[^\d]', '', cleaned)
            if not cleaned:
                return 0
            value = int(cleaned)
            return value
            
    except (ValueError, TypeError, OverflowError):
        return 0

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

def get_company_products_from_group(company_name: str, df_companies, df_kerix):
    all_deduped = []
    all_full = []
    
    if df_companies is not None and '_display_name' in df_companies.columns:
        company_row = df_companies[df_companies['_display_name'].str.strip() == company_name.strip()]
        if not company_row.empty:
            products_col = get_col(df_companies, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
            if products_col and products_col in company_row.columns:
                products_text = str(company_row.iloc[0][products_col])
                deduped, full = split_tokens(products_text)
                all_deduped.extend(deduped)
                all_full.extend(full)
    
    if not all_deduped and df_kerix is not None and '_display_name' in df_kerix.columns:
        company_row = df_kerix[df_kerix['_display_name'].str.strip() == company_name.strip()]
        if not company_row.empty:
            products_col = get_col(df_kerix, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
            if products_col and products_col in company_row.columns:
                products_text = str(company_row.iloc[0][products_col])
                deduped, full = split_tokens(products_text)
                all_deduped.extend(deduped)
                all_full.extend(full)
    return list(set(all_deduped)), all_full

def get_group_products(group_manager, group_name: str, df_companies, df_kerix):
    if group_name not in group_manager.groups:
        return [], []
    
    group_data = group_manager.groups[group_name]
    group_companies = group_data.get('companies', [])
    
    all_deduped = []
    all_full = []
    
    for company in group_companies:
        if isinstance(company, str) and company.strip():
            company_deduped, company_full = get_company_products_from_group(company.strip(), df_companies, df_kerix)
            all_deduped.extend(company_deduped)
            all_full.extend(company_full)
    group_deduped = list(set(all_deduped))
    return group_deduped, all_full

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
        "Chiffre d'affaires", "Chiffre d'Affaires", "Chiffre d'Affaires 2023 (Dhs)", "Chiffre d'Affaires 2023 (Dhs) ",
        "CA", "Chiffre d'affaire"
    ]
    rev_col = get_col(df, revenue_candidates)
    info['revenue_col'] = rev_col
    
    if not rev_col:
        for c in df.columns:
            c_lower = str(c).lower()
            if 'chiffre' in c_lower and 'affair' in c_lower:
                rev_col = c
                info['revenue_col'] = rev_col
                break
    
    if rev_col and rev_col in df.columns:
        df['_revenue_raw'] = df[rev_col]        
        if is_companies:
            parse_func = parse_companies_revenue
        else:
            parse_func = parse_kerix_revenue
        
        mins = []
        maxs = []
        for v in df['_revenue_raw'].fillna("").astype(str).tolist():
            mi, ma = parse_func(v)
            mins.append(mi)
            maxs.append(ma)
        
        df['_revenue_min'] = pd.Series(mins, dtype='float64')
        df['_revenue_max'] = pd.Series(maxs, dtype='float64')
    else:
        df['_revenue_raw'] = np.nan
        df['_revenue_min'] = np.nan
        df['_revenue_max'] = np.nan
    
    operating_candidates = [
        "Resultat d'exploitation 2023 (Dhs)", "Resultat d'exploitation 2023 (Dhs) ", "Resultat d'exploitation 2023",
        "Resultat d'exploitation", "Résultat d'exploitation 2023 (Dhs)", "Résultat d'exploitation",
        "RE", "Résultat d'Exploitation"
    ]
    op_col = get_col(df, operating_candidates)
    
    if not op_col:
        for c in df.columns:
            c_lower = str(c).lower()
            if 'resultat' in c_lower and 'exploitation' in c_lower:
                op_col = c
                break
    
    if op_col and op_col in df.columns:
        df['_operating_income'] = df[op_col].apply(lambda x: clean_operating_income(x, is_companies))
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
    
    if primary_rev_col and primary_rev_col in df.columns:
        df['_revenue_raw'] = df[primary_rev_col]
        info['revenue_col'] = primary_rev_col
    else:
        revenue_candidates = [
            "Chiffre d'affaires 2023 (Dhs)", "Chiffre d'affaires 2023 (Dhs) ", "Chiffre d'affaires 2023",
            "Chiffre d'affaires", "Chiffre d'Affaires", "Chiffre d'Affaires 2023 (Dhs)", "Chiffre d'Affaires 2023 (Dhs) ",
            "Chiffre d'Affaires", "CA", "Chiffre d'affaire"
        ]
        rev_col = get_col(df, revenue_candidates)
        info['revenue_col'] = rev_col
        
        if not rev_col:
            for c in df.columns:
                c_lower = str(c).lower()
                if 'chiffre' in c_lower and 'affair' in c_lower:
                    rev_col = c
                    info['revenue_col'] = rev_col
                    break
        
        if rev_col and rev_col in df.columns:
            df['_revenue_raw'] = df[rev_col]
        else:
            df['_revenue_raw'] = np.nan
    
    if '_revenue_raw' not in df.columns:
        df['_revenue_raw'] = np.nan
    
    if is_companies:
        parse_func = parse_companies_revenue
    else:
        parse_func = parse_kerix_revenue
    
    mins = []
    maxs = []
    for v in df['_revenue_raw'].fillna("").astype(str).tolist():
        mi, ma = parse_func(v)
        mins.append(mi)
        maxs.append(ma)
    
    df['_revenue_min'] = pd.Series(mins, dtype='float64')
    df['_revenue_max'] = pd.Series(maxs, dtype='float64')
    
    primary_op_col = year_cols[primary_year]['operating']
    if primary_op_col and primary_op_col in df.columns:
        df['_operating_income'] = df[primary_op_col].apply(lambda x: clean_operating_income(x, is_companies))
        info['operating_col'] = primary_op_col
    else:
        operating_candidates = [
            "Resultat d'exploitation 2023 (Dhs)", "Resultat d'exploitation 2023 (Dhs) ", "Resultat d'exploitation 2023",
            "Resultat d'exploitation", "Résultat d'exploitation 2023 (Dhs)", "Résultat d'exploitation",
            "RE", "Résultat d'Exploitation"
        ]
        op_col = get_col(df, operating_candidates)
        if op_col and op_col in df.columns:
            df['_operating_income'] = df[op_col].apply(lambda x: clean_operating_income(x, is_companies))
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

def apply_group_exclusions(df, db_name):
    if not st.session_state.groups:
        return df
    
    all_grouped_companies = set()
    for group_data in st.session_state.groups.values():
        for company in group_data.get('companies', []):
            if isinstance(company, str) and company.strip():
                all_grouped_companies.add(company.strip())
    
    if '_display_name' in df.columns and not all_grouped_companies:
        df = df[~df['_display_name'].isin(all_grouped_companies)].reset_index(drop=True)
    
    return df

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

df_companies_prepared, info_companies = prepare_df_with_years(df_companies, selected_years, is_companies=True) if df_companies is not None else (None, {})
df_kerix_prepared, info_kerix = prepare_df_with_years(df_kerix, selected_years, is_companies=False) if df_kerix is not None else (None, {})

def extract_seed_tokens(seed_series):
    prods = ""
    for k in ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"]:
        if k in seed_series.index and pd.notna(seed_series.get(k)):
            prods = seed_series.get(k)
            break
    if prods == "":
        prods = seed_series.get("Produits / Services", "") or seed_series.get("products", "") or ""
    
    deduped_tokens, full_tokens = split_tokens(prods)
    return deduped_tokens, full_tokens

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

if df_companies_prepared is not None and '_display_name' in df_companies_prepared.columns:
    COMPANY_COL = '_display_name'
else:
    COMPANY_COL = 'Raison Sociale'

df = pd.DataFrame()
if df_companies_prepared is not None:
    df = pd.concat([df, df_companies_prepared[['_display_name']].rename(columns={'_display_name': COMPANY_COL})], ignore_index=True)
if df_kerix_prepared is not None:
    df_kerix_temp = df_kerix_prepared[['_display_name']].rename(columns={'_display_name': COMPANY_COL})
    df = pd.concat([df, df_kerix_temp], ignore_index=True)
if df.empty:
    df[COMPANY_COL] = []

if 'current_search_target' not in st.session_state:
    st.session_state.current_search_target = 'single'

METRIC_TOKENS = {
    "CA": ["chiffre", "affaires"],
    "RE": ["resultat", "exploitation"],
    "CP": ["charges", "personnel"],
    "EBIT_CA": ["marge", "ebit", "ca"],
    "EBIT_CP": ["marge", "ebit", "cp"],
    "CP_CA": ["marge", "cp", "ca"],
}

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
    return pd.DataFrame()

def aggregate_group_data(group_entities, df, metric_key, year):
    if df is None or df.empty:
        return np.nan
    if not isinstance(group_entities, (list, tuple)):
        return np.nan
    if metric_key not in METRIC_TOKENS:
        return np.nan
    total_values = []
    for entity in group_entities:
        if isinstance(entity, str):
            try:
                company_name = str(entity).strip()
                if not company_name:
                    continue
                comp_df = safe_find_company(df, company_name, COMPANY_COL)
                if not comp_df.empty:
                    total_values.append(1.0)
                else:
                    total_values.append(np.nan)
            except Exception:
                total_values.append(np.nan)
    valid_values = [v for v in total_values if pd.notna(v)]
    return len(valid_values) if valid_values else 0

class GroupManager:
    def __init__(self):
        self.groups = {}
        self.company_to_group = {}
    def update_group_metrics(self, group_name):
        if group_name not in self.groups:
            return
        
        group_data = self.groups[group_name]
        group_companies = group_data.get('companies', [])
        
        if not group_companies:
            return
        
        all_products_raw = []
        revenue_min = np.inf
        revenue_max = -np.inf
        total_operating = 0
        all_cities = set()
        all_sectors = set()
        
        for company in group_companies:
            if isinstance(company, str) and company.strip():
                company_data = None
                if df_companies_prepared is not None and '_display_name' in df_companies_prepared.columns:
                    comp_row = df_companies_prepared[df_companies_prepared['_display_name'].str.strip() == company.strip()]
                    if not comp_row.empty:
                        company_data = comp_row.iloc[0]
                if company_data is None and df_kerix_prepared is not None and '_display_name' in df_kerix_prepared.columns:
                    comp_row = df_kerix_prepared[df_kerix_prepared['_display_name'].str.strip() == company.strip()]
                    if not comp_row.empty:
                        company_data = comp_row.iloc[0]
                
                if company_data is not None:
                    products = company_data.get('_products', '')
                    if products and str(products).strip():
                        all_products_raw.append(str(products))
                    
                    rev_min = company_data.get('_revenue_min', np.nan)
                    rev_max = company_data.get('_revenue_max', np.nan)
                    if pd.notna(rev_min) and rev_min < revenue_min:
                        revenue_min = rev_min
                    if pd.notna(rev_max) and rev_max > revenue_max:
                        revenue_max = rev_max
                    
                    operating = company_data.get('_operating_income', 0)
                    total_operating += float(operating)
                    
                    city = company_data.get('_city', '')
                    if city and str(city).strip():
                        all_cities.add(str(city).strip())
                    sector = company_data.get('_sector', '')
                    if sector and str(sector).strip():
                        all_sectors.add(str(sector).strip())
        
        group_data['products'] = join_products_for_group(all_products_raw, separator=" | ")
        group_data['revenue_min'] = revenue_min if revenue_min != np.inf else np.nan
        group_data['revenue_max'] = revenue_max if revenue_max != -np.inf else np.nan
        group_data['operating_income'] = total_operating
        group_data['city'] = ', '.join(sorted(all_cities)) if all_cities else ''
        group_data['sector'] = ', '.join(sorted(all_sectors)) if all_sectors else ''
    
    def create_group(self, name, companies=None):
        if companies is None:
            companies = []
        clean_companies = []
        for company in companies:
            if isinstance(company, str) and company.strip():
                clean_companies.append(company.strip())
        self.groups[name] = {
            'name': name, 
            'companies': clean_companies,
            'products': '',
            'revenue_min': np.inf,
            'revenue_max': -np.inf,
            'operating_income': 0,
            'city': '',
            'sector': ''
        }
        for company in clean_companies:
            self.company_to_group[company] = name
        self.update_group_metrics(name)
    
    def get_group_data(self, group_name, df, metric_key, year):
        if df is None or df.empty:
            return np.nan
        if metric_key not in METRIC_TOKENS:
            return np.nan
        group = self.groups.get(group_name)
        if not isinstance(group, dict) or 'companies' not in group:
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
        st.session_state.group_manager_data = {
            'groups': self.groups,
            'company_to_group': self.company_to_group
        }
    
    def load_session(self):
        if 'group_manager_data' in st.session_state:
            self.groups = st.session_state.group_manager_data.get('groups', {})
            self.company_to_group = st.session_state.group_manager_data.get('company_to_group', {})
            for group_name, group_data in list(self.groups.items()):
                if not isinstance(group_data, dict) or 'companies' not in group_data:
                    self.groups[group_name] = {'name': group_name, 'companies': []}
                else:
                    clean_companies = []
                    for company in group_data.get('companies', []):
                        if isinstance(company, str) and company.strip():
                            clean_companies.append(company.strip())
                    group_data['companies'] = clean_companies
                    self.update_group_metrics(group_name)

if 'group_manager' not in st.session_state:
    st.session_state.group_manager = GroupManager()
group_manager = st.session_state.group_manager
group_manager.load_session()

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

            group_data = group_manager.groups[group_name]
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                current_companies = group_data.get('companies', [])
                if current_companies:
                    st.write(f"Entreprises ({len(current_companies)}):")
                    for company in current_companies:
                        if isinstance(company, str):
                            comp_exists_companies = False
                            comp_exists_kerix = False
                            if df_companies_prepared is not None and '_display_name' in df_companies_prepared.columns:
                                comp_exists_companies = not df_companies_prepared[df_companies_prepared['_display_name'].str.strip() == company.strip()].empty
                            if df_kerix_prepared is not None and '_display_name' in df_kerix_prepared.columns:
                                comp_exists_kerix = not df_kerix_prepared[df_kerix_prepared['_display_name'].str.strip() == company.strip()].empty
                            comp_exists = comp_exists_companies or comp_exists_kerix
                            status_icon = "✅" if comp_exists else "⚠️"
                            st.write(f"{status_icon} {company}")
                else:
                    st.info("Aucune entreprise dans ce groupe")

                all_companies = []
                if df_companies_prepared is not None and '_display_name' in df_companies_prepared.columns:
                    all_companies.extend(df_companies_prepared['_display_name'].dropna().astype(str).str.strip().unique().tolist())
                if df_kerix_prepared is not None and '_display_name' in df_kerix_prepared.columns:
                    all_companies.extend(df_kerix_prepared['_display_name'].dropna().astype(str).str.strip().unique().tolist())
                all_companies = sorted(list(set(all_companies)))

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
                        current_companies = group_data.get('companies', [])
                        for company in selected_to_add:
                            if isinstance(company, str) and company.strip() not in current_companies:
                                current_companies.append(company.strip())
                                group_manager.company_to_group[company.strip()] = group_name
                        group_data['companies'] = current_companies
                        group_manager.update_group_metrics(group_name)
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
                        remaining_companies = [c for c in current_companies if c not in selected_to_remove]
                        for company in selected_to_remove:
                            if isinstance(company, str):
                                group_manager.company_to_group.pop(company.strip(), None)
                        group_data['companies'] = remaining_companies
                        group_manager.update_group_metrics(group_name)
                        group_manager.save_session()
                        st.success(f"✅ {len(selected_to_remove)} entreprise(s) supprimée(s)")
                        st.rerun()
            
            with col3:
                if st.button("🗑️ Supprimer groupe", key=f"delete_{group_name}", type="secondary"):
                    current_companies = group_data.get('companies', [])
                    for company in current_companies:
                        if isinstance(company, str):
                            group_manager.company_to_group.pop(company.strip(), None)
                    if group_name in group_manager.groups:
                        del group_manager.groups[group_name]
                    group_manager.save_session()
                    st.success(f"✅ Groupe '{group_name}' supprimé!")
                    st.rerun()
else:
    st.info("Aucun groupe créé pour le moment. Utilisez le formulaire ci-dessus pour en créer un.")

grouped_companies = set()
for key in list(group_manager.company_to_group.keys()):
    if isinstance(key, str) and key.strip():
        grouped_companies.add(key.strip())

st.subheader("🔍 Moteur de Recherche")
search_mode = st.radio("Mode de recherche", ["Entreprise unique", "Groupe d'entreprises"], horizontal=True, key="search_mode")
st.session_state.current_search_target = "group" if search_mode == "Groupe d'entreprises" else "single"

if search_mode == "Groupe d'entreprises" and not group_manager.groups:
    st.warning("Créez d'abord un groupe dans la section Gestion des Groupes!")
    st.stop()

seed_row = None
seed_prods = None
seed_prods_dedup = []
seed_prods_full = []
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
        seed_prods_dedup, seed_prods_full = extract_seed_tokens(seed_row)
    
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
            seed_prods_dedup, seed_prods_full = extract_seed_tokens(seed_row)

elif search_mode == "Groupe d'entreprises":
    group_options = list(group_manager.groups.keys())
    selected_group = st.selectbox("Choisir le groupe seed", group_options)
    if selected_group:
        group_data = group_manager.groups[selected_group]
        if isinstance(group_data, dict):
            seed_row = pd.Series({
                "name": group_data.get('name', selected_group),
                "_display_name": group_data.get('display_name', group_data.get('name', selected_group))
            })
            seed_name_for_exclusion = group_data.get('display_name', group_data.get('name', selected_group))            
            group_deduped, group_full = get_group_products(group_manager, selected_group, df_companies_prepared, df_kerix_prepared)
            seed_prods_dedup = group_deduped
            seed_prods_full = group_full

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
    
    if seed_row is None and seed_name_for_exclusion is None:
        return pd.DataFrame(), []
    
    candidates = df.copy().reset_index(drop=True)
    
    all_grouped_companies = set()
    for group_data in group_manager.groups.values():
        for company in group_data.get('companies', []):
            if isinstance(company, str) and company.strip():
                all_grouped_companies.add(company.strip())
    
    if '_display_name' in candidates.columns:
        candidates = candidates[~candidates['_display_name'].isin(all_grouped_companies)].reset_index(drop=True)
    
    mask_city = candidates['_city'].isin(selected_cities) if selected_cities else pd.Series([True] * len(candidates))
    mask_sector = candidates['_sector'].isin(selected_sectors) if selected_sectors else pd.Series([True] * len(candidates))
    
    if use_ca_filter:
        sel_min = float(ca_min_sel)
        sel_max = float(ca_max_sel)
        has_ca_data = (candidates['_revenue_min'].notna()) | (candidates['_revenue_max'].notna())
        ca_overlap = (
            ((candidates['_revenue_min'].notna()) & (candidates['_revenue_max'].notna()) & 
            (candidates['_revenue_max'] >= sel_min) & (candidates['_revenue_min'] <= sel_max)) |
            
            ((candidates['_revenue_min'].notna()) & (candidates['_revenue_max'].isna()) & 
            (candidates['_revenue_min'] <= sel_max)) |
            
            ((candidates['_revenue_max'].notna()) & (candidates['_revenue_min'].isna()) & 
            (candidates['_revenue_max'] >= sel_min))
        )
        mask_revenue_overlap = has_ca_data & ca_overlap
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
    seed_prods_dedup = []
    seed_prods_full = []
    if seed_row is not None:
        if search_mode == "Entreprise unique":
            seed_prods_dedup, seed_prods_full = extract_seed_tokens(seed_row)
        elif search_mode == "Groupe d'entreprises" and seed_name_for_exclusion:
            try:
                group_data = group_manager.groups.get(seed_name_for_exclusion, {})
                if group_data and 'companies' in group_data:
                    all_deduped = []
                    all_full = []
                    for company_name in group_data['companies']:
                        if isinstance(company_name, str) and company_name.strip():
                            company_raw = ""
                            if df_companies_prepared is not None and '_display_name' in df_companies_prepared.columns:
                                comp_row = df_companies_prepared[df_companies_prepared['_display_name'].str.strip() == company_name.strip()]
                                if not comp_row.empty:
                                    products_col = get_col(df_companies_prepared, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
                                    if products_col:
                                        company_raw = str(comp_row.iloc[0][products_col])
                            
                            if not company_raw and df_kerix_prepared is not None and '_display_name' in df_kerix_prepared.columns:
                                comp_row = df_kerix_prepared[df_kerix_prepared['_display_name'].str.strip() == company_name.strip()]
                                if not comp_row.empty:
                                    products_col = get_col(df_kerix_prepared, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
                                    if products_col:
                                        company_raw = str(comp_row.iloc[0][products_col])
                            
                            if company_raw and str(company_raw).strip():
                                company_deduped, company_full = split_tokens(company_raw)
                                all_deduped.extend(company_deduped)
                                all_full.extend(company_full)
                    seed_prods_dedup = list(set(all_deduped))
                    seed_prods_full = all_full
                else:
                    seed_prods_dedup, seed_prods_full = [], []
            except Exception:
                seed_prods_dedup, seed_prods_full = [], []
    else:
        try:
            if seed_prods is not None:
                temp_dedup, temp_full = split_tokens(" ".join(seed_prods))
                seed_prods_dedup = temp_dedup
                seed_prods_full = temp_full
        except:
            seed_prods_dedup, seed_prods_full = [], []
    
    if not seed_prods_dedup and not seed_prods_full:
        return pd.DataFrame(), []
    
    products_col = get_col(df, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
    
    prods_matched = []
    duplicate_matches = []
    product_comments = []
    
    for _, r in candidates.iterrows():
        raw_cand_products = r.get(products_col, "") if products_col else ""
        cand_prods_dedup, cand_prods_full = split_tokens(str(raw_cand_products))
        
        mp, _ = count_similar(seed_prods_dedup, cand_prods_dedup, prod_threshold)
        prods_matched.append((mp, len(seed_prods_dedup)))
        
        mp_dup, _ = count_similar(seed_prods_full, cand_prods_full, prod_threshold)
        duplicate_matches.append((mp_dup, len(seed_prods_full)))
        
        duplicate_products_found = []
        for s_token in set(seed_prods_full):
            if s_token in cand_prods_full:
                duplicate_products_found.append(s_token)
        
        if duplicate_products_found:
            if len(duplicate_products_found) == 1:
                comment = f"Forte Concentration en: {duplicate_products_found[0]}"
            elif len(duplicate_products_found) <= 3:
                comment = f"Forte Concentration en: {', '.join(sorted(duplicate_products_found))}"
            else:
                comment = f"Forte Concentration en: {', '.join(sorted(duplicate_products_found[:3]))} +{len(duplicate_products_found)-3} autres"
        else:
            comment = f"{mp_dup}/{len(seed_prods_full)} similaires"
        
        product_comments.append(comment)
    
    candidates['products_common_count'] = [m for m, t in prods_matched]
    candidates['products_seed_count'] = [t for m, t in prods_matched]
    candidates['product_fraction'] = candidates.apply(lambda r: (r['products_common_count'] / max(1, r['products_seed_count'])) if r['products_seed_count']>0 else 0.0, axis=1)
    candidates['product_fraction_str'] = candidates.apply(lambda r: frac_str(r['products_common_count'], r['products_seed_count']), axis=1)
    
    candidates['products_duplicate_count'] = [m for m, t in duplicate_matches]
    candidates['products_seed_count_dup'] = [t for m, t in duplicate_matches]
    candidates['duplicate_fraction_str'] = candidates.apply(
        lambda r: frac_str(r['products_duplicate_count'], r['products_seed_count_dup']), axis=1
    )
    
    candidates['product_comment'] = product_comments    
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
    if 'selected_years' in (info_companies if label == "Base totale" else info_kerix):
        years_str = ', '.join(map(str, selected_years))
        candidates['_Year_display'] = f"Données {years_str}"
        display_cols.append('_Year_display')
    display_cols += ['duplicate_fraction_str', 'product_comment']
    
    return candidates, display_cols

with tabs[0]:
    st.markdown("### Onglet: Base totale")
    if df_companies_prepared is None:
        st.warning("Fichier companies.xlsx non chargé.")
    else:
        st.markdown("#### Seed utilisée")
        with st.expander("Info Seed", expanded=False):
            st.write({
                "seed_name": seed_name_for_exclusion, 
                "seed_products_without_duplicates": seed_prods_dedup, 
                "seed_products_with_duplicates": seed_prods_full,
                "count_without_duplicates": len(seed_prods_dedup),
                "full_count": len(seed_prods_full)
        })
        candidates, display_cols = compute_matches_for_df(df_companies_prepared, "Base totale")
        if candidates is None or candidates.shape[0] == 0:
            st.info("Aucun candidat trouvé dans la Base totale pour les filtres sélectionnés.")
        else:
            st.markdown(f"##### Top {int(top_n_preview)} candidats - Base totale")
            rename_map = {
                '_display_name': 'Raison Sociale',
                'product_fraction_str': 'Matching Score (Produits)',
                '_City_display': "Ville",
                '_Sector_display': "Secteur",
                '_CA_display': "Chiffre d'affaires",
                '_OP_display': "Résultat d'exploitation",
                'duplicate_fraction_str': 'Matching Score (avec doublons)',
                'product_comment': 'Commentaire Produits,',
            }
            if '_Year_display' in display_cols:
                rename_map['_Year_display'] = "Année(s) des données"
            st.dataframe(candidates.head(int(top_n_preview))[display_cols].fillna("").rename(columns=rename_map))
            out = io.BytesIO()
            out.seek(0)
            st.download_button("📥 Télécharger résultats (Base totale)", data=out, file_name="base_totale_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[1]:
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        candidates.to_excel(writer, sheet_name="base_totale_results", index=False)
    st.markdown("### Onglet: Kerix")
    if df_kerix_prepared is None:
        st.warning("Fichier kerix.xlsx non chargé.")
    else:
        st.markdown("#### Seed utilisée")
        with st.expander("Info Seed", expanded=False):
            st.write({
                "seed_name": seed_name_for_exclusion, 
                "seed_products_without_duplicates": seed_prods_dedup, 
                "seed_products_with_duplicates": seed_prods_full,
                "count_without_duplicates": len(seed_prods_dedup),
                "full_count": len(seed_prods_full)
        })
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
                '_Sector_display': "Secteur",
                'duplicate_fraction_str': 'Matching Score (avec doublons)',
                'product_comment': 'Commentaire Produits',
            }
            if '_Year_display' in display_cols_k:
                rename_map['_Year_display'] = "Année(s) des données"
            st.dataframe(candidates_k.head(int(top_n_preview))[display_cols_k].fillna("").rename(columns=rename_map))
            out_k = io.BytesIO()
            with pd.ExcelWriter(out_k, engine="openpyxl") as writer:
                candidates_k.to_excel(writer, sheet_name="kerix_results", index=False)
            out_k.seek(0)
            st.download_button("📥 Télécharger résultats (Kerix)", data=out_k, file_name="kerix_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

group_manager.save_session()