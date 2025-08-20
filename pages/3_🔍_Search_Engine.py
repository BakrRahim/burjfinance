import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from typing import List, Tuple

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
st.title("Search Engine")


def get_col(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def split_tokens(text: str) -> List[str]:
    """Tokenize activities/products with the special comma rule described."""
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
        n = int(float(n))
    except Exception:
        return str(n)
    parts = []
    B = 1_000_000_000
    M = 1_000_000
    K = 1_000
    if n >= B:
        b = n // B
        parts.append(f"{b}B")
        n = n % B
    if n >= M:
        m = n // M
        parts.append(f"{m}M")
        n = n % M
    if n >= K and not parts:
        k = n // K
        parts.append(f"{k}K")
    if not parts:
        return "0"
    return " ".join(parts)

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
    low_candidates = re.findall(r'[\d\.,]+', raw)
    norm_nums = []
    for num in low_candidates:
        digits = re.sub(r'[^\d]', '', num)
        if digits != "":
            try:
                norm_nums.append(int(digits))
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
    return df, info

df_companies_prepared, info_companies = (prepare_df(df_companies, is_companies=True) if df_companies is not None else (None, {}))
df_kerix_prepared, info_kerix = (prepare_df(df_kerix, is_companies=False) if df_kerix is not None else (None, {}))

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


st.subheader("Filtres")

seed_row = None
seed_name_for_exclusion = None

if df_kerix_prepared is None:
    st.warning("Kerix non disponible - choisissez Base totale ou Manuelle.")
else:
    options = df_kerix_prepared['_display_name'].fillna("N/A").astype(str).tolist()
    seed_choice_idx = st.selectbox("Choisir la société seed (Kerix)", options=list(range(len(options))),
                                    format_func=lambda i: options[i])
    seed_row = df_kerix_prepared.iloc[int(seed_choice_idx)]
    seed_name_for_exclusion = seed_row.get('_display_name', "")

non_existant = st.checkbox("Entreprise non existante dans la base", False)
if non_existant:
    with st.form("manual_seed_form_global"):
        ms_name = st.text_input("Nom de la société (seed)")
        ms_activities = st.text_area("Activités Principales (séparez par , ; / )")
        ms_products = st.text_area("Produits / Services (séparez par , ; / )")
        submitted = st.form_submit_button("Créer seed manuel")
    if not submitted:
        st.info("Soumettez le formulaire pour créer la société seed manuelle.")
    else:
        seed_row = pd.Series({
            "name_manual": ms_name,
            "Activités Principales": ms_activities,
            "Produits / Services": ms_products
        })
        seed_name_for_exclusion = ms_name

st.markdown("### Filtre CA (Chiffre d'affaires)")

min_val = max(0.0, global_min)
max_val = max(min_val + 1.0, global_max)
range_span = max_val - min_val

approx_step = 10 ** (int(np.floor(np.log10(max(1.0, range_span)))) - 1)
approx_step = max(1.0, approx_step)


# --- Number inputs aligned under slider edges ---
col1, col2 = st.columns([1, 1])
with col1:
    ca_min_input = st.number_input(
        "Min (Dhs)", 
        min_value=float(min_val), 
        max_value=float(max_val),
        value=min_val,
        step=float(approx_step)
    )
with col2:
    ca_max_input = st.number_input(
        "Max (Dhs)", 
        min_value=float(ca_min_input), 
        max_value=float(max_val),
        value=max_val,
        step=float(approx_step)
    )

# Keep slider and inputs in sync
ca_min_sel, ca_max_sel = ca_min_input, ca_max_input

st.markdown(f"Min - Max sélectionné : **{compact_num(ca_min_sel)}** - **{compact_num(ca_max_sel)}**")

if seed_row is None:
    st.warning("Sélectionnez ou créez une société seed pour lancer les recherches.")
    st.stop()

def extract_seed_tokens(seed_series):
    acts = ""
    prods = ""
    for k in ["Activités Principales", "Activités Principales ", "Activités", "Activités Principales (Kerix)", "activities"]:
        if k in seed_series.index and pd.notna(seed_series.get(k)):
            acts = seed_series.get(k)
            break
    if acts == "":
        acts = seed_series.get("Activités Principales", "") or seed_series.get("activities", "") or ""
    for k in ["Produits / Services", "Produits/Services", "Produits / Services ", "products", "Produits"]:
        if k in seed_series.index and pd.notna(seed_series.get(k)):
            prods = seed_series.get(k)
            break
    if prods == "":
        prods = seed_series.get("Produits / Services", "") or seed_series.get("products", "") or ""
    return split_tokens(acts), split_tokens(prods)

seed_acts, seed_prods = extract_seed_tokens(seed_row)


tabs = st.tabs(["Base totale", "Kerix", "Paramètres"])

with tabs[2]:
    st.header("Paramètres")
    act_threshold = st.slider("Seuil similarité activités (fuzzy)", min_value=0.5, max_value=0.95, value=0.75, step=0.01, key="act_threshold")
    prod_threshold = st.slider("Seuil similarité produits/services (fuzzy)", min_value=0.5, max_value=0.95, value=0.75, step=0.01, key="prod_threshold")
    top_n_preview = st.number_input("Top N candidats à afficher", min_value=5, max_value=500, value=25, step=5, key="top_n_preview")
    sort_metric = st.selectbox("Trier par", ["product_fraction", "activity_fraction", "average_fraction"], index=0, key="sort_metric")

def compute_matches_for_df(df, label):
    if df is None:
        return None, None

    candidates = df.copy().reset_index(drop=True)

    sel_min = float(ca_min_sel)
    sel_max = float(ca_max_sel)
    mask_parsed = (~candidates['_revenue_min'].isna()) & (~candidates['_revenue_max'].isna())
    mask_overlap = mask_parsed & (candidates['_revenue_max'] >= sel_min) & (candidates['_revenue_min'] <= sel_max)
    candidates = candidates[mask_overlap].reset_index(drop=True)

    try:
        if seed_name_for_exclusion and '_display_name' in candidates.columns:
            candidates = candidates[candidates['_display_name'].astype(str) != str(seed_name_for_exclusion)]
    except Exception:
        pass
    candidates = candidates.reset_index(drop=True)

    if candidates.shape[0] == 0:
        return candidates, []

    activities_col = get_col(df, ["Activités Principales", "Activités Principales ", "Activités"])
    products_col = get_col(df, ["Produits / Services", "Produits/Services", "Produits / Services "])

    acts_matched = []
    prods_matched = []
    for _, r in candidates.iterrows():
        cand_acts = split_tokens(r.get(activities_col, "")) if activities_col else []
        cand_prods = split_tokens(r.get(products_col, "")) if products_col else []
        ma, ta = count_similar(seed_acts, cand_acts, act_threshold)
        mp, tp = count_similar(seed_prods, cand_prods, prod_threshold)
        acts_matched.append((ma, ta))
        prods_matched.append((mp, tp))

    candidates['activities_common_count'] = [m for m, t in acts_matched]
    candidates['activities_seed_count'] = [t for m, t in acts_matched]
    candidates['activity_fraction'] = candidates.apply(lambda r: (r['activities_common_count'] / max(1, r['activities_seed_count'])) if r['activities_seed_count']>0 else 0.0, axis=1)
    candidates['activity_fraction_str'] = candidates.apply(lambda r: frac_str(r['activities_common_count'], r['activities_seed_count']), axis=1)

    candidates['products_common_count'] = [m for m, t in prods_matched]
    candidates['products_seed_count'] = [t for m, t in prods_matched]
    candidates['product_fraction'] = candidates.apply(lambda r: (r['products_common_count'] / max(1, r['products_seed_count'])) if r['products_seed_count']>0 else 0.0, axis=1)
    candidates['product_fraction_str'] = candidates.apply(lambda r: frac_str(r['products_common_count'], r['products_seed_count']), axis=1)

    candidates['combined_matched'] = candidates['activities_common_count'] + candidates['products_common_count']
    candidates['combined_seed_total'] = candidates['activities_seed_count'] + candidates['products_seed_count']
    candidates['average_fraction'] = candidates.apply(lambda r: (r['combined_matched'] / max(1, r['combined_seed_total'])) if r['combined_seed_total']>0 else 0.0, axis=1)
    candidates['average_fraction_str'] = candidates.apply(lambda r: frac_str(r['combined_matched'], r['combined_seed_total']), axis=1)

    candidates = candidates.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)

    display_cols = []
    if '_display_name' in candidates.columns:
        display_cols.append('_display_name')
    display_cols += ['average_fraction_str']

    candidates['_CA_display'] = candidates.apply(lambda r: format_range_compact(r['_revenue_min'], r['_revenue_max']), axis=1)
    display_cols.append('_CA_display')
    return candidates, display_cols


with tabs[0]:
    st.header("Onglet: Base totale")
    if df_companies_prepared is None:
        st.warning("Fichier companies.xlsx non chargé.")
    else:
        st.subheader("Seed utilisée")
        st.write({"seed_name": seed_name_for_exclusion})

        candidates, display_cols = compute_matches_for_df(df_companies_prepared, "Base totale")

        if candidates is None or candidates.shape[0] == 0:
            st.info("Aucun candidat trouvé dans la Base totale pour la plage CA et la seed sélectionnées.")
        else:
            st.subheader(f"Top {int(top_n_preview)} candidats - Base totale")
            rename_map = {
                '_display_name': 'Raison Sociale',
                'average_fraction_str': 'Matching Score',
                '_CA_display': "Chiffre d'affaires"
            }
            st.dataframe(candidates.head(int(top_n_preview))[display_cols].fillna("").rename(columns=rename_map))

            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="openpyxl") as writer:
                candidates.to_excel(writer, sheet_name="base_totale_results", index=False)
            out.seek(0)
            st.download_button("📥 Télécharger résultats (Base totale)", data=out, file_name="base_totale_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


with tabs[1]:
    st.header("Onglet: Kerix")
    if df_kerix_prepared is None:
        st.warning("Fichier kerix.xlsx non chargé.")
    else:
        st.subheader("Seed utilisée")
        st.write({"seed_name": seed_name_for_exclusion})

        candidates_k, display_cols_k = compute_matches_for_df(df_kerix_prepared, "Kerix")

        if candidates_k is None or candidates_k.shape[0] == 0:
            st.info("Aucun candidat trouvé dans Kerix pour la plage CA et la seed sélectionnées.")
        else:
            st.subheader(f"Top {int(top_n_preview)} candidats - Kerix")
            rename_map = {
                '_display_name': 'Raison Sociale',
                'average_fraction_str': 'Matching Score',
                '_CA_display': "Chiffre d'affaires (compact)"
            }
            st.dataframe(candidates_k.head(int(top_n_preview))[display_cols_k].fillna("").rename(columns=rename_map))

            out_k = io.BytesIO()
            with pd.ExcelWriter(out_k, engine="openpyxl") as writer:
                candidates_k.to_excel(writer, sheet_name="kerix_results", index=False)
            out_k.seek(0)
            st.download_button("📥 Télécharger résultats (Kerix)", data=out_k, file_name="kerix_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")