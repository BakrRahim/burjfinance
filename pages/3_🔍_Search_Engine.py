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

def split_activites(text: str) -> Tuple[List[str], List[str]]:
    if pd.isna(text):
        return [], []
    
    s = str(text).strip()
    if not s:
        return [], []
    parts = [p.strip() for p in s.split(',') if p.strip()]
    cleaned = [re.sub(r'\s+', ' ', p).strip().lower() for p in parts]
    seen = set(cleaned)
    deduped = list(seen)    
    full = parts
    
    return deduped, full

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
    product_dedup = []
    product_full = []
    activity_dedup = []
    activity_full = []
    has_products = False
    using_activities = False
    found_in_df = False
    if df_companies is not None and '_display_name' in df_companies.columns:
        company_row = df_companies[df_companies['_display_name'].str.strip() == company_name.strip()]
        if not company_row.empty:
            found_in_df = True
            products_col = get_col(df_companies, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
            products_text = ""
            if products_col and products_col in company_row.columns:
                raw_value = company_row.iloc[0][products_col]
                if pd.notna(raw_value):
                    products_text = str(raw_value).strip()
                    if products_text.lower() in ['nan', 'n/a', 'non disponible', 'nc', '']:
                        products_text = ""
            if products_text:
                deduped, full = split_tokens(products_text)
                if deduped:
                    product_dedup = deduped
                    product_full = full
                    has_products = True
    if not has_products and df_kerix is not None and '_display_name' in df_kerix.columns:
        company_row = df_kerix[df_kerix['_display_name'].str.strip() == company_name.strip()]
        if not company_row.empty:
            found_in_df = True
            products_col = get_col(df_kerix, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
            products_text = ""
            if products_col and products_col in company_row.columns:
                raw_value = company_row.iloc[0][products_col]
                if pd.notna(raw_value):
                    products_text = str(raw_value).strip()
                    if products_text.lower() in ['nan', 'n/a', 'non disponible', 'nc', '']:
                        products_text = ""
            if products_text:
                deduped, full = split_tokens(products_text)
                if deduped:
                    product_dedup = deduped
                    product_full = full
                    has_products = True
    if not has_products and found_in_df:
        for df_temp in [df_companies, df_kerix]:
            if df_temp is not None and '_display_name' in df_temp.columns:
                company_row = df_temp[df_temp['_display_name'].str.strip() == company_name.strip()]
                if not company_row.empty:
                    activ_candidates = ["Activités Principales", "Activités principales", "Activité principale", "Activité principale ", "Activités"]
                    activ_col = None
                    for ac in activ_candidates:
                        if ac in company_row.columns:
                            activ_col = ac
                            break
                    if activ_col:
                        activ_raw = company_row.iloc[0][activ_col]
                        if pd.notna(activ_raw):
                            activ_text = str(activ_raw).strip()
                            if activ_text.lower() in ['nan', 'n/a', 'non disponible', 'nc', '']:
                                activ_text = ""
                        if activ_text:
                            deduped, full = split_activites(activ_text)
                            if deduped:
                                activity_dedup = deduped
                                activity_full = full
                                using_activities = True
                                break
    return product_dedup, product_full, activity_dedup, activity_full, has_products, using_activities

def get_group_products(group_manager, group_name: str, df_companies, df_kerix):
    if group_name not in group_manager.groups:
        return [], [], [], [], 0, False
    group_data = group_manager.groups[group_name]
    group_companies = group_data.get('companies', [])
    product_set = set()
    product_full_list = []
    activity_set = set()
    activity_full_list = []
    num_with_products = 0
    any_activities_used = False
    for company in group_companies:
        if isinstance(company, str) and company.strip():
            p_dedup, p_full, a_dedup, a_full, has_p, used_a = get_company_products_from_group(company.strip(), df_companies, df_kerix)
            if has_p:
                product_set.update(p_dedup)
                product_full_list.extend(p_full)
                num_with_products += 1
            else:
                activity_set.update(a_dedup)
                activity_full_list.extend(a_full)
                any_activities_used = True
    return list(product_set), product_full_list, list(activity_set), activity_full_list, num_with_products, any_activities_used

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
    
    sector_candidates = ["Secteur", "Sector", "Activité principale", "Branche", "Activités Principales", "Activités principales"]
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
    
    sector_candidates = ["Secteur", "Sector", "Activité principale", "Branche", "Activités Principales", "Activités principales"]
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
    using_activites = False
    if not deduped_tokens:
        activ_candidates = ["Activités Principales", "Activités principales", "Activité principale", "Activité principale ", "Activités"]
        activ_col = None
        for ac in activ_candidates:
            if ac in seed_series.index:
                activ_col = ac
                break
        if activ_col:
            activ = seed_series.get(activ_col)
            if pd.notna(activ) and str(activ).strip():
                deduped_tokens, full_tokens = split_activites(str(activ))
                using_activites = True
        if not deduped_tokens and '_sector' in seed_series.index:
            sector = seed_series.get('_sector', '')
            if pd.notna(sector) and str(sector).strip():
                deduped_tokens, full_tokens = split_activites(str(sector))
                using_activites = True
    
    st.session_state.using_activites_for_comparison = using_activites
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
            group_product_dedup, group_product_full, group_activity_dedup, group_activity_full, num_with_products, any_activites_used = get_group_products(group_manager, selected_group, df_companies_prepared, df_kerix_prepared)
            total_companies = len(group_data.get('companies', []))
            all_have_products = total_companies > 0 and num_with_products == total_companies
            all_lack_products = num_with_products == 0
            mixed = total_companies > 0 and not all_have_products and not all_lack_products
            st.session_state.group_all_have_products = all_have_products
            st.session_state.group_all_lack_products = all_lack_products
            st.session_state.group_mixed = mixed
            st.session_state.group_product_dedup = group_product_dedup
            st.session_state.group_product_full = group_product_full
            st.session_state.group_activity_dedup = group_activity_dedup
            st.session_state.group_activity_full = group_activity_full
            st.session_state.group_num_with_products = num_with_products
            st.session_state.group_total_companies = total_companies
            st.session_state.using_activites_for_comparison = any_activites_used

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
st.markdown("#### Seed utilisée")
with st.expander("Détails Seed", expanded=False):
    if search_mode == "Entreprise unique":
        using_activ = st.session_state.get('using_activites_for_comparison', False)
        term = "Activités" if using_activ else "Produits/Services"
        tokens_key = "seed_activities_without_duplicates" if using_activ else "seed_products_without_duplicates"
        full_key = "seed_activities_with_duplicates" if using_activ else "seed_products_with_duplicates"
        st.write({
            "seed_name": seed_name_for_exclusion,
            tokens_key: seed_prods_dedup,
            full_key: seed_prods_full,
            "count_unique": len(seed_prods_dedup),
            "full_count": len(seed_prods_full)
        })
    else:
        total = st.session_state.get('group_total_companies', 0)
        num_p = st.session_state.get('group_num_with_products', 0)
        st.write(f"Nombre d'entreprises avec produits/services: {num_p}/{total}")
        if st.session_state.get('group_all_have_products', False):
            st.write("Cas: Toutes les entreprises ont produits/services")
            st.write(f"Tokens produits uniques: {len(st.session_state.group_product_dedup)}")
            st.write(f"Tokens produits complets: {len(st.session_state.group_product_full)}")
            st.json({"product_tokens_unique": st.session_state.group_product_dedup})
        elif st.session_state.get('group_all_lack_products', False):
            st.write("Cas: Aucune entreprise n'a produits/services (uniquement activités)")
            st.write(f"Tokens activités uniques: {len(st.session_state.group_activity_dedup)}")
            st.write(f"Tokens activités complets: {len(st.session_state.group_activity_full)}")
            st.json({"activity_tokens_unique": st.session_state.group_activity_dedup})
        else:
            st.write("Cas: Mixte (produits pour celles qui en ont, activités pour les autres)")
            st.write(f"Tokens produits uniques (de {num_p} entreprises): {len(st.session_state.group_product_dedup)}")
            st.write(f"Tokens activités uniques (de {total - num_p} entreprises): {len(st.session_state.group_activity_dedup)}")
            st.json({
                "product_tokens_unique": st.session_state.group_product_dedup,
                "activity_tokens_unique": st.session_state.group_activity_dedup
            })
    
tabs = st.tabs(["Base totale", "Kerix", "Paramètres"])

with tabs[2]:
    st.markdown("### Paramètres")
    prod_threshold = st.slider("Seuil similarité produits/services (fuzzy)", min_value=0.5, max_value=0.95, value=0.8, step=0.01, key="prod_threshold")
    top_n_preview = st.number_input("Top N candidats à afficher", min_value=5, max_value=500, value=25, step=5, key="top_n_preview")
    sort_options = ["product_fraction", "activity_fraction"]
    sort_metric = st.selectbox("Trier par", sort_options, index=0, key="sort_metric")

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
    is_group = st.session_state.get('current_search_target') == 'group'
    products_col = get_col(candidates, ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"])
    activ_candidates = ["Activités Principales", "Activités principales", "Activité principale", "Activité principale ", "Activités"]
    activ_col = get_col(candidates, activ_candidates)
    show_product = False
    show_activity = False
    using_activites = False
    has_product_match = False
    has_activity_match = False
    all_have = False
    all_lack = False
    mixed = False
    if is_group:
        all_have = st.session_state.get('group_all_have_products', False)
        all_lack = st.session_state.get('group_all_lack_products', False)
        mixed = st.session_state.get('group_mixed', False)
        show_product = all_have or mixed
        show_activity = all_lack or mixed
        using_activites = all_lack or mixed
        group_p_dedup = st.session_state.get('group_product_dedup', [])
        group_a_dedup = st.session_state.get('group_activity_dedup', [])
        group_p_full = st.session_state.get('group_product_full', [])
        group_a_full = st.session_state.get('group_activity_full', [])
        has_product_match = bool(group_p_dedup)
        has_activity_match = bool(group_a_dedup)
    else:
        seed_prods_dedup, seed_prods_full = extract_seed_tokens(seed_row)
        using_activites = st.session_state.get('using_activites_for_comparison', False)
        show_product = not using_activites
        show_activity = using_activites
        seed_prods_dedup = seed_prods_dedup if show_product else seed_prods_dedup
        seed_prods_full = seed_prods_full
        has_product_match = bool(seed_prods_dedup) and show_product
        has_activity_match = bool(seed_prods_dedup) and show_activity
    candidates['_CA_display'] = candidates.apply(lambda r: format_range_compact(r['_revenue_min'], r['_revenue_max']), axis=1)
    candidates['_OP_display'] = candidates['_operating_income'].apply(lambda x: compact_num(x))
    candidates['_City_display'] = candidates['_city']
    candidates['_Sector_display'] = candidates['_sector']
    if 'selected_years' in (info_companies if label == "Base totale" else info_kerix):
        years_str = ', '.join(map(str, selected_years))
        candidates['_Year_display'] = f"Données {years_str}"
    display_cols = ['_display_name']
    if has_product_match:
        display_cols.insert(1, 'product_fraction_str')
    if has_activity_match and not has_product_match:
        display_cols.insert(1, 'activity_fraction_str')
    if has_product_match and has_activity_match:
        display_cols.insert(1, 'product_fraction_str')
        display_cols.insert(2, 'activity_fraction_str')
    display_cols += ['_CA_display', '_OP_display', '_City_display', '_Sector_display']
    if '_Year_display' in candidates.columns:
        display_cols.append('_Year_display')
        for idx, r in candidates.iterrows():
            cand_p_dedup = []
            cand_p_full = []
            cand_a_dedup = []
            cand_a_full = []
            if products_col and pd.notna(r.get(products_col)):
                raw_prod = str(r[products_col])
                if raw_prod.strip():
                    cand_p_dedup, cand_p_full = split_tokens(raw_prod)
            if activ_col and pd.notna(r.get(activ_col)):
                raw_activ = str(r[activ_col])
                if raw_activ.strip():
                    cand_a_dedup, cand_a_full = split_activites(raw_activ)
            if show_product:
                if is_group:
                    seed_p_dedup = group_p_dedup
                    seed_p_full_for_dup = group_p_full
                else:
                    seed_p_dedup = seed_prods_dedup
                    seed_p_full_for_dup = seed_prods_full
                mp_p, sp_p = count_similar(seed_p_dedup, cand_p_dedup, prod_threshold)
                candidates.at[idx, 'product_common_count'] = mp_p
                candidates.at[idx, 'product_seed_count'] = sp_p
                candidates.at[idx, 'product_fraction'] = mp_p / max(1, sp_p) if sp_p > 0 else 0.0
                candidates.at[idx, 'product_fraction_str'] = frac_str(mp_p, sp_p)
                mp_dup_p, _= count_similar(seed_p_full_for_dup, cand_p_full, prod_threshold)
                candidates.at[idx, 'product_dup_count'] = mp_dup_p
                candidates.at[idx, 'product_dup_seed_count'] = len(seed_p_full_for_dup)
                candidates.at[idx, 'product_dup_fraction_str'] = frac_str(mp_dup_p, len(seed_p_full_for_dup))
                duplicate_tokens_found_p = []
                for s_token in set(seed_p_full_for_dup):
                    for c_token in cand_p_full:
                        if fuzzy_ratio(s_token, c_token) >= prod_threshold:
                            duplicate_tokens_found_p.append(c_token)
                            break
                term_p = "Produits/Services"
                if duplicate_tokens_found_p:
                    if len(duplicate_tokens_found_p) == 1:
                        comment_p = f"Forte Concentration en: {duplicate_tokens_found_p[0]} ({term_p})"
                    elif len(duplicate_tokens_found_p) <= 3:
                        comment_p = f"Forte Concentration en: {', '.join(sorted(duplicate_tokens_found_p))} ({term_p})"
                    else:
                        comment_p = f"Forte Concentration en: {', '.join(sorted(duplicate_tokens_found_p[:3]))} +{len(duplicate_tokens_found_p)-3} autres ({term_p})"
                else:
                    comment_p = f"{mp_dup_p}/{len(seed_p_full_for_dup)} similaires ({term_p})"
                candidates.at[idx, 'product_comment'] = comment_p

            if show_activity:
                if is_group:
                    seed_a_dedup = group_a_dedup
                    seed_a_full_for_dup = group_a_full
                else:
                    seed_a_dedup = seed_prods_dedup
                    seed_a_full_for_dup = seed_prods_full
                mp_a, sp_a = count_similar(seed_a_dedup, cand_a_dedup, prod_threshold)
                candidates.at[idx, 'activity_common_count'] = mp_a
                candidates.at[idx, 'activity_seed_count'] = sp_a
                candidates.at[idx, 'activity_fraction'] = mp_a / max(1, sp_a) if sp_a > 0 else 0.0
                candidates.at[idx, 'activity_fraction_str'] = frac_str(mp_a, sp_a)
                mp_dup_a, _= count_similar(seed_a_full_for_dup, cand_a_full, prod_threshold)
                candidates.at[idx, 'activity_dup_count'] = mp_dup_a
                candidates.at[idx, 'activity_dup_seed_count'] = len(seed_a_full_for_dup)
                candidates.at[idx, 'activity_dup_fraction_str'] = frac_str(mp_dup_a, len(seed_a_full_for_dup))
                duplicate_tokens_found_a = []
                for s_token in set(seed_a_full_for_dup):
                    for c_token in cand_a_full:
                        if fuzzy_ratio(s_token, c_token) >= prod_threshold:
                            duplicate_tokens_found_a.append(c_token)
                            break
                term_a = "Activités"
                if duplicate_tokens_found_a:
                    if len(duplicate_tokens_found_a) == 1:
                        comment_a = f"Forte Concentration en: {duplicate_tokens_found_a[0]} ({term_a})"
                    elif len(duplicate_tokens_found_a) <= 3:
                        comment_a = f"Forte Concentration en: {', '.join(sorted(duplicate_tokens_found_a))} ({term_a})"
                    else:
                        comment_a = f"Forte Concentration en: {', '.join(sorted(duplicate_tokens_found_a[:3]))} +{len(duplicate_tokens_found_a)-3} autres ({term_a})"
                else:
                    comment_a = f"{mp_dup_a}/{len(seed_a_full_for_dup)} similaires ({term_a})"
                candidates.at[idx, 'activity_comment'] = comment_a

        if is_group and mixed:
            total_seed_matched = candidates['product_common_count'] + candidates['activity_common_count']
            total_seed_count = candidates['product_seed_count'] + candidates['activity_seed_count']
            candidates['combined_fraction'] = (candidates['product_fraction'].fillna(0) + candidates['activity_fraction'].fillna(0)) / 2
            candidates['combined_fraction_str'] = total_seed_matched.astype(int).astype(str) + '/' + total_seed_count.astype(int).astype(str) + ' (' + candidates['combined_fraction'].round(2).astype(str) + ')'
            if 'product_fraction_str' in display_cols:
                display_cols.remove('product_fraction_str')
            if 'activity_fraction_str' in display_cols:
                display_cols.remove('activity_fraction_str')
            display_cols.insert(1, 'combined_fraction_str')

        if show_product and has_product_match and 'product_dup_fraction_str' not in display_cols:
            display_cols += ['product_dup_fraction_str', 'product_comment']
        if show_activity and has_activity_match and (mixed or all_lack) and (not mixed):
            display_cols += ['activity_dup_fraction_str', 'activity_comment']

        if is_group:
            if all_have:
                candidates = candidates.sort_values(by='product_fraction', ascending=False).reset_index(drop=True)
            elif all_lack:
                candidates = candidates.sort_values(by='activity_fraction', ascending=False).reset_index(drop=True)
            elif mixed:
                candidates['avg_fraction'] = (candidates['product_fraction'].fillna(0) + candidates['activity_fraction'].fillna(0)) / 2
                candidates = candidates.sort_values(by='avg_fraction', ascending=False).reset_index(drop=True)
                candidates = candidates.drop('avg_fraction', axis=1)
        else:
            sort_col = 'activity_fraction' if using_activites else 'product_fraction'
            candidates = candidates.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
        if is_group and (mixed or all_have or all_lack):
            display_cols = list(dict.fromkeys(display_cols))
        return candidates, display_cols, (using_activites or mixed if is_group else using_activites)

out = None
with tabs[0]:
    st.markdown("### Onglet: Base totale")
    if df_companies_prepared is None:
        st.warning("Fichier companies.xlsx non chargé.")
    else:
        candidates, display_cols, using_activites = compute_matches_for_df(df_companies_prepared, "Base totale")
        is_group = st.session_state.get('current_search_target') == 'group'
        all_have = st.session_state.get('group_all_have_products', False)
        all_lack = st.session_state.get('group_all_lack_products', False)
        mixed = st.session_state.get('group_mixed', False)
        if is_group:
            if all_have:
                term = "Produits/Services"
                match_col = 'product_fraction_str'
                dup_col = 'product_dup_fraction_str'
                comment_col = 'product_comment'
                sort_note = "Trié par similarité produits"
            elif all_lack:
                term = "Activités"
                match_col = 'activity_fraction_str'
                dup_col = 'activity_dup_fraction_str'
                comment_col = 'activity_comment'
                sort_note = "Trié par similarité activités"
            else:
                term = "Produits/Services et Activités (moyenne)"
                match_col = None
                sort_note = "Trié par moyenne des similarités"
            st.info(f"Recherche groupe: {sort_note}")
        else:
            term = "Activités" if using_activites else "Produits/Services"
            match_col = 'activity_fraction_str' if using_activites else 'product_fraction_str'
            dup_col = 'activity_dup_fraction_str' if using_activites else 'product_dup_fraction_str'
            comment_col = 'activity_comment' if using_activites else 'product_comment'
            sort_note = None
        if candidates is None or candidates.shape[0] == 0:
            st.info(f"Aucun candidat trouvé dans la Base totale pour les options sélectionnés ({term}).")
        else:
            st.markdown(f"##### Top {int(top_n_preview)} candidats - Base totale ({term})")
            rename_map = {
                '_display_name': 'Raison Sociale',
                '_CA_display': "Chiffre d'affaires",
                '_OP_display': "Résultat d'exploitation",
                '_City_display': "Ville",
                '_Sector_display': "Secteur",
            }
            if is_group and mixed:
                rename_map['combined_fraction_str'] = 'Matching Score Total (Produits/Services + Activités)'
                rename_map['product_dup_fraction_str'] = 'Détail doublons Produits/Services'
                rename_map['product_comment'] = 'Commentaire Produits/Services'
            else:
                if match_col in candidates.columns:
                    rename_map[match_col] = f'Matching Score ({term})'
                if dup_col in candidates.columns:
                    dup_label = f'Matching Score (avec doublons)' if not using_activites else f'Matching Score ({term} complet)'
                    rename_map[dup_col] = dup_label
                if comment_col in candidates.columns:
                    rename_map[comment_col] = f'Commentaire {term}'
            if '_Year_display' in display_cols:
                rename_map['_Year_display'] = "Année(s) des données"
            df_to_show = candidates.head(int(top_n_preview))[display_cols].fillna("")
            st.dataframe(df_to_show.rename(columns=rename_map))
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                df_to_show.rename(columns=rename_map).to_excel(writer, sheet_name="base_totale_results", index=False)
            out.seek(0)
            dl_filename = "base_totale_results.xlsx" if not all_lack else "base_totale_results_activites.xlsx"
            if mixed:
                dl_filename = "base_totale_results_mixed.xlsx"
            st.download_button(f"📥 Télécharger résultats (Base totale - {term})", data=out, file_name=dl_filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[1]:
    st.markdown("### Onglet: Kerix")
    if df_kerix_prepared is None:
        st.warning("Fichier kerix.xlsx non chargé.")
    else:
        candidates_k, display_cols_k, using_activites_k = compute_matches_for_df(df_kerix_prepared, "Kerix")
        is_group_k = st.session_state.get('current_search_target') == 'group'
        all_have_k = st.session_state.get('group_all_have_products', False)
        all_lack_k = st.session_state.get('group_all_lack_products', False)
        mixed_k = st.session_state.get('group_mixed', False)
        if is_group_k:
            if all_have_k:
                term_k = "Produits/Services"
                match_col_k = 'product_fraction_str'
                dup_col_k = 'product_dup_fraction_str'
                comment_col_k = 'product_comment'
            elif all_lack_k:
                term_k = "Activités"
                match_col_k = 'activity_fraction_str'
                dup_col_k = 'activity_dup_fraction_str'
                comment_col_k = 'activity_comment'
            else:
                term_k = "Produits/Services et Activités (moyenne)"
                match_col_k = None
            st.info(f"Recherche groupe: Trié par moyenne des similarités" if mixed_k else f"Trié par similarité {term_k.lower()}")
        else:
            term_k = "Activités" if using_activites_k else "Produits/Services"
            match_col_k = 'activity_fraction_str' if using_activites_k else 'product_fraction_str'
            dup_col_k = 'activity_dup_fraction_str' if using_activites_k else 'product_dup_fraction_str'
            comment_col_k = 'activity_comment' if using_activites_k else 'product_comment'
        if candidates_k is None or candidates_k.shape[0] == 0:
            st.info(f"Aucun candidat trouvé dans Kerix pour les options sélectionnés ({term_k}).")
        else:
            st.markdown(f"##### Top {int(top_n_preview)} candidats - Kerix ({term_k})")
            rename_map_k = {
                '_display_name': 'Raison Sociale',
                '_CA_display': "Chiffre d'affaires",
                '_OP_display': "Résultat d'exploitation",
                '_City_display': "Ville",
                '_Sector_display': "Secteur",
            }
            if is_group_k and mixed_k:
                rename_map_k['combined_fraction_str'] = 'Matching Score Total (Produits/Services + Activités)'
                rename_map_k['product_dup_fraction_str'] = 'Détail doublons Produits/Services'
                rename_map_k['product_comment'] = 'Commentaire Produits/Services'
            else:
                if match_col_k in candidates_k.columns:
                    rename_map_k[match_col_k] = f'Matching Score ({term_k})'
                if dup_col_k in candidates_k.columns:
                    dup_label_k = f'Matching Score (avec doublons)' if not using_activites_k else f'Matching Score ({term_k} complet)'
                    rename_map_k[dup_col_k] = dup_label_k
                if comment_col_k in candidates_k.columns:
                    rename_map_k[comment_col_k] = f'Commentaire {term_k}'
            if '_Year_display' in display_cols_k:
                rename_map_k['_Year_display'] = "Année(s) des données"
            st.dataframe(candidates_k.head(int(top_n_preview))[display_cols_k].fillna("").rename(columns=rename_map_k))
            out_k = io.BytesIO()
            with pd.ExcelWriter(out_k, engine="openpyxl") as writer:
                candidates_k.to_excel(writer, sheet_name="kerix_results", index=False)
            out_k.seek(0)
            dl_filename_k = "kerix_results.xlsx" if not all_lack_k else "kerix_results_activites.xlsx"
            if mixed_k:
                dl_filename_k = "kerix_results_mixed.xlsx"
            st.download_button(f"📥 Télécharger résultats (Kerix - {term_k})", data=out_k, file_name=dl_filename_k, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

group_manager.save_session()