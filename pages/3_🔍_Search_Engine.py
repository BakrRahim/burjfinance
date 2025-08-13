import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import time
from typing import List, Tuple

# fuzzy helper (rapidfuzz if available)
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

# Helpers
def get_col(df: pd.DataFrame, candidates: List[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def split_tokens(text: str) -> List[str]:
    """Tokenize an activities/products string into normalized tokens (deduplicated, order preserved).

    Special comma rule: split ONLY on a comma if the first non-space character AFTER the comma is an uppercase letter.
    All other commas remain inside the same token (e.g. "Fruits secs, fruits séchés" stays one token).
    """
    if pd.isna(text):
        return []
    s = str(text)

    # Marker to replace commas that should ACT as separators
    SEP = "<<<SPLIT>>>"
    s_chars = list(s)
    i = 0
    while i < len(s_chars):
        if s_chars[i] == ",":
            # look ahead for the next non-space character
            j = i + 1
            while j < len(s_chars) and s_chars[j].isspace():
                j += 1
            # split only if the next non-space character exists and is uppercase
            if j < len(s_chars) and s_chars[j].isupper():
                s_chars[i] = SEP
        i += 1

    s2 = "".join(s_chars)

    # Split only on our SEP marker
    parts = s2.split(SEP)

    cleaned = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # remove parentheses, brackets, digits, colons and dots (keep commas if they were not separators)
        p2 = re.sub(r'[\(\)\[\]\d\:.\u2022]+', '', p)
        # normalize whitespace
        p2 = re.sub(r'\s+', ' ', p2).strip()
        # remove some leading/trailing punctuation except keep internal commas
        p2 = re.sub(r'^[\-\–\—\.;:]+', '', p2)
        p2 = re.sub(r'[\-\–\—\.;:]+$', '', p2)
        p2 = p2.strip().lower()
        if p2:
            cleaned.append(p2)

    # deduplicate while preserving order
    seen = set()
    out = []
    for v in cleaned:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def count_similar(seed_tokens: List[str], cand_tokens: List[str], threshold: float) -> Tuple[int, int]:
    """
    Return (matched_seed_count, total_seed_count).
    A seed token counts as matched if any candidate token has fuzzy_ratio >= threshold.
    """
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
    return f"{int(m)}/{int(t)} ({m/max(1,t):.2f})"

def load_or_upload(filename: str, prompt_label: str):
    try:
        df = pd.read_excel(filename)
        st.success(f"Chargé depuis {filename} — {df.shape[0]} lignes, {df.shape[1]} colonnes.")
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

# Sidebar controls
act_threshold = st.sidebar.slider("Seuil similarité activités (fuzzy)", min_value=0.5, max_value=0.95, value=0.75, step=0.01)
prod_threshold = st.sidebar.slider("Seuil similarité produits/services (fuzzy)", min_value=0.5, max_value=0.95, value=0.75, step=0.01)
top_n_preview = st.sidebar.number_input("Top N candidats à afficher", min_value=5, max_value=500, value=25, step=5)
sort_metric = st.sidebar.selectbox("Trier par", ["activity_fraction", "product_fraction", "average_fraction"], index=0)

tabs = st.tabs(["Base totale", "Kerix"])

with tabs[0]:
    st.header("Onglet: Base totale (companies.xlsx)")
    df_companies = load_or_upload("companies.xlsx", "companies.xlsx")
    if df_companies is None:
        st.stop()

    st.subheader("Aperçu (5 lignes)")
    st.dataframe(df_companies.head().fillna(""))

    # automatic column detection for companies.xlsx
    name_col = get_col(df_companies, ["Raison Sociale (Kerix)", "Raison Sociale (Maroc1000 Nouvelle)", "Raison Sociale (Maroc1000 ancienne)", "Raison Sociale", "Raison Sociale (Maroc1000)"])
    activities_col = get_col(df_companies, ["Activités Principales", "Activités Principales ", "Activités"])
    products_col = get_col(df_companies, ["Produits / Services", "Produits/Services", "Produits / Services "])
    revenue_col = get_col(df_companies, ["Chiffre d'affaires 2023 (Dhs)", "Chiffre d'affaires 2023 (Dhs) ", "Chiffre d'affaires 2023", "Chiffre d'affaires"])
    city_col = get_col(df_companies, ["Ville RC", "Ville"])

    st.markdown("Colonnes détectées (automatique):")
    st.write({
        "name": name_col, "activities": activities_col, "products": products_col,
        "revenue": revenue_col, "city": city_col
    })

    # Seed selection
    st.subheader("Sélection de la société cible (seed)")
    seed_mode = st.radio("Mode seed", ["Choisir depuis la base", "Saisir manuellement"], index=0)
    if seed_mode == "Choisir depuis la base":
        if name_col:
            seed_choice = st.selectbox("Choisir la société seed (par nom)", options=df_companies[name_col].fillna("N/A").astype(str).tolist())
            seed_row = df_companies[df_companies[name_col].astype(str) == seed_choice].iloc[0]
        else:
            idx = st.number_input("Index de la ligne seed", min_value=0, max_value=len(df_companies)-1, value=0)
            seed_row = df_companies.iloc[int(idx)]
    else:
        with st.form("manual_seed_form_companies"):
            ms_name = st.text_input("Nom de la société (seed)")
            ms_activities = st.text_area("Activités Principales (séparez par , ; / )")
            ms_products = st.text_area("Produits / Services (séparez par , ; / )")
            ms_city = st.text_input("Ville")
            submitted = st.form_submit_button("Créer seed manuel")
        if not submitted:
            st.info("Soumettez le formulaire pour créer la société seed.")
            st.stop()
        seed_row = pd.Series({
            name_col if name_col else "name": ms_name,
            activities_col if activities_col else "Activités Principales": ms_activities,
            products_col if products_col else "Produits / Services": ms_products,
            city_col if city_col else "Ville": ms_city
        })

    st.markdown("Seed sélectionnée (aperçu):")
    st.write(seed_row[[c for c in [name_col, activities_col, products_col, city_col] if c is not None]].to_dict())

    # Tokenize seed
    seed_acts = split_tokens(seed_row.get(activities_col, "") if activities_col else seed_row.get("Activités Principales", ""))
    seed_prods = split_tokens(seed_row.get(products_col, "") if products_col else seed_row.get("Produits / Services", ""))

    # Build candidates (exclude seed if matched by name)
    candidates = df_companies.copy()
    try:
        if name_col:
            candidates = candidates[candidates[name_col].astype(str) != str(seed_row.get(name_col, ""))]
    except Exception:
        pass
    candidates = candidates.reset_index(drop=True)

    # Compute match counts for activities and products
    acts_matched = []
    prods_matched = []
    for _, r in candidates.iterrows():
        cand_acts = split_tokens(r.get(activities_col, "")) if activities_col else []
        cand_prods = split_tokens(r.get(products_col, "")) if products_col else []
        ma, ta = count_similar(seed_acts, cand_acts, act_threshold)
        mp, tp = count_similar(seed_prods, cand_prods, prod_threshold)
        # total seed counts are len(seed_acts) and len(seed_prods)
        acts_matched.append((ma, len(seed_acts)))
        prods_matched.append((mp, len(seed_prods)))

    candidates["activities_common_count"] = [m for m, t in acts_matched]
    candidates["activities_seed_count"] = [t for m, t in acts_matched]
    candidates["activity_fraction_str"] = candidates.apply(lambda r: frac_str(r["activities_common_count"], r["activities_seed_count"]), axis=1)
    candidates["activity_fraction"] = candidates.apply(lambda r: (r["activities_common_count"] / max(1, r["activities_seed_count"])) if r["activities_seed_count"]>0 else 0.0, axis=1)

    candidates["products_common_count"] = [m for m, t in prods_matched]
    candidates["products_seed_count"] = [t for m, t in prods_matched]
    candidates["product_fraction_str"] = candidates.apply(lambda r: frac_str(r["products_common_count"], r["products_seed_count"]), axis=1)
    candidates["product_fraction"] = candidates.apply(lambda r: (r["products_common_count"] / max(1, r["products_seed_count"])) if r["products_seed_count"]>0 else 0.0, axis=1)

    candidates["average_fraction"] = candidates[["activity_fraction", "product_fraction"]].mean(axis=1)

    # Sorting
    candidates = candidates.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)

    # Display columns
    display_cols = []
    if name_col:
        display_cols.append(name_col)
    display_cols += [
        "activities_common_count", "activities_seed_count", "activity_fraction", "activity_fraction_str",
        "products_common_count", "products_seed_count", "product_fraction", "product_fraction_str"
    ]
    if "Chiffre d'affaires 2023 (Dhs)" in candidates.columns:
        display_cols.append("Chiffre d'affaires 2023 (Dhs)")
    elif "Chiffre d'affaires" in candidates.columns:
        display_cols.append("Chiffre d'affaires")

    st.subheader(f"Top {top_n_preview} candidats — Base totale")
    st.dataframe(candidates.head(top_n_preview)[display_cols].fillna(""))

    # Download
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        candidates.to_excel(writer, sheet_name="base_totale_results", index=False)
    out.seek(0)
    st.download_button("📥 Télécharger résultats", data=out, file_name="base_totale_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with tabs[1]:
    st.header("Onglet: Kerix (kerix.xlsx)")
    df_kerix = load_or_upload("kerix.xlsx", "kerix.xlsx")
    if df_kerix is None:
        st.stop()

    st.subheader("Aperçu (5 lignes)")
    st.dataframe(df_kerix.head().fillna(""))

    # automatic column detection for kerix
    name_col_k = get_col(df_kerix, ["Raison Sociale", "Raison Sociale "])
    activities_col_k = get_col(df_kerix, ["Activités Principales", "Activités Principales "])
    products_col_k = get_col(df_kerix, ["Produits / Services", "Produits/Services", "Produits / Services "])
    revenue_col_k = get_col(df_kerix, ["Chiffre d'Affaires", "Chiffre d'Affaires 2023 (Dhs)", "Chiffre d'Affaires 2023 (Dhs) "])
    city_col_k = get_col(df_kerix, ["Ville RC", "Ville"])

    st.markdown("Colonnes détectées (Kerix):")
    st.write({
        "name": name_col_k, "activities": activities_col_k, "products": products_col_k,
        "revenue": revenue_col_k, "city": city_col_k
    })

    # Seed selection for Kerix
    st.subheader("Sélection seed (Kerix)")
    seed_mode_k = st.radio("Mode seed (Kerix)", ["Choisir depuis Kerix", "Saisir manuellement"], key="seed_mode_k", index=0)
    if seed_mode_k == "Choisir depuis Kerix":
        if name_col_k:
            seed_choice_k = st.selectbox("Choisir seed (par nom) - Kerix", options=df_kerix[name_col_k].fillna("N/A").astype(str).tolist(), key="seed_choice_k")
            seed_row_k = df_kerix[df_kerix[name_col_k].astype(str) == seed_choice_k].iloc[0]
        else:
            idx = st.number_input("Index seed Kerix", min_value=0, max_value=len(df_kerix)-1, value=0, key="idx_k")
            seed_row_k = df_kerix.iloc[int(idx)]
    else:
        with st.form("manual_seed_form_kerix"):
            ms_name_k = st.text_input("Nom de la société (seed) - Kerix")
            ms_activities_k = st.text_area("Activités Principales (séparez par , ; / ) - Kerix")
            ms_products_k = st.text_area("Produits / Services (séparez par , ; / ) - Kerix")
            submitted_k = st.form_submit_button("Créer seed manuel - Kerix")
        if not submitted_k:
            st.info("Soumettez le formulaire pour créer la société seed (Kerix).")
            st.stop()
        seed_row_k = pd.Series({
            name_col_k if name_col_k else "name": ms_name_k,
            activities_col_k if activities_col_k else "Activités Principales": ms_activities_k,
            products_col_k if products_col_k else "Produits / Services": ms_products_k
        })

    st.markdown("Seed Kerix (aperçu):")
    st.write(seed_row_k[[c for c in [name_col_k, activities_col_k, products_col_k, city_col_k] if c is not None]].to_dict())

    # Tokenize seed for kerix
    seed_acts_k = split_tokens(seed_row_k.get(activities_col_k, "") if activities_col_k else seed_row_k.get("Activités Principales", ""))
    seed_prods_k = split_tokens(seed_row_k.get(products_col_k, "") if products_col_k else seed_row_k.get("Produits / Services", ""))

    # Build candidates
    candidates_k = df_kerix.copy()
    try:
        if name_col_k:
            candidates_k = candidates_k[candidates_k[name_col_k].astype(str) != str(seed_row_k.get(name_col_k, ""))]
    except Exception:
        pass
    candidates_k = candidates_k.reset_index(drop=True)

    acts_matched_k = []
    prods_matched_k = []
    for _, r in candidates_k.iterrows():
        cand_acts = split_tokens(r.get(activities_col_k, "")) if activities_col_k else []
        cand_prods = split_tokens(r.get(products_col_k, "")) if products_col_k else []
        ma, ta = count_similar(seed_acts_k, cand_acts, act_threshold)
        mp, tp = count_similar(seed_prods_k, cand_prods, prod_threshold)
        acts_matched_k.append((ma, len(seed_acts_k)))
        prods_matched_k.append((mp, len(seed_prods_k)))

    candidates_k["activities_common_count"] = [m for m, t in acts_matched_k]
    candidates_k["activities_seed_count"] = [t for m, t in acts_matched_k]
    candidates_k["activity_fraction_str"] = candidates_k.apply(lambda r: frac_str(r["activities_common_count"], r["activities_seed_count"]), axis=1)
    candidates_k["activity_fraction"] = candidates_k.apply(lambda r: (r["activities_common_count"] / max(1, r["activities_seed_count"])) if r["activities_seed_count"]>0 else 0.0, axis=1)

    candidates_k["products_common_count"] = [m for m, t in prods_matched_k]
    candidates_k["products_seed_count"] = [t for m, t in prods_matched_k]
    candidates_k["product_fraction_str"] = candidates_k.apply(lambda r: frac_str(r["products_common_count"], r["products_seed_count"]), axis=1)
    candidates_k["product_fraction"] = candidates_k.apply(lambda r: (r["products_common_count"] / max(1, r["products_seed_count"])) if r["products_seed_count"]>0 else 0.0, axis=1)

    candidates_k["average_fraction"] = candidates_k[["activity_fraction", "product_fraction"]].mean(axis=1)

    candidates_k = candidates_k.sort_values(by=sort_metric, ascending=False).reset_index(drop=True)

    display_cols_k = []
    if name_col_k:
        display_cols_k.append(name_col_k)
    display_cols_k += [
        "activities_common_count", "activities_seed_count", "activity_fraction", "activity_fraction_str",
        "products_common_count", "products_seed_count", "product_fraction", "product_fraction_str"
    ]
    if revenue_col_k:
        display_cols_k.append(revenue_col_k)
    if city_col_k:
        display_cols_k.append(city_col_k)

    st.subheader(f"Top {top_n_preview} candidats — Kerix")
    st.dataframe(candidates_k.head(top_n_preview)[display_cols_k].fillna(""))

    out_k = io.BytesIO()
    with pd.ExcelWriter(out_k, engine="openpyxl") as writer:
        candidates_k.to_excel(writer, sheet_name="kerix_results", index=False)
    out_k.seek(0)
    st.download_button("📥 Télécharger résultats", data=out_k, file_name="kerix_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")