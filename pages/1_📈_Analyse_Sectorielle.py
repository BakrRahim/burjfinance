import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re

@st.cache_data
def load_data():
    df = pd.read_excel("companies.xlsx")
    return df

df = load_data()
columns_to_keep = [
    "Raison Sociale (Maroc1000 Nouvelle)", "Raison Sociale (Maroc1000 ancienne)", "Raison Sociale (Kerix)", "Secteur",
    "Chiffre d'affaires 2020 (Dhs)", "Chiffre d'affaires 2021 (Dhs)", "Chiffre d'affaires 2022 (Dhs)", "Chiffre d'affaires 2023 (Dhs)",
    "Resultat d'exploitation 2020 (Dhs)", "Resultat d'exploitation 2021 (Dhs)", "Resultat d'exploitation 2022 (Dhs)", "Resultat d'exploitation 2023 (Dhs)",
    "Charges personnel 2020", "Charges personnel 2021", "Charges personnel 2022", "Charges personnel 2023",
    'Marge EBIT/CA 2020', 'Marge EBIT/CA 2021', 'Marge EBIT/CA 2022', 'Marge EBIT/CA 2023',
    'Marge EBIT/CP 2020', 'Marge EBIT/CP 2021', 'Marge EBIT/CP 2022', 'Marge EBIT/CP 2023',
    'Marge CP/CA 2020', 'Marge CP/CA 2021', 'Marge CP/CA 2022', 'Marge CP/CA 2023',
    "Variation CA 2020/2021", "Variation CA 2021/2022", "Variation CA 2022/2023",
    "Variation RE 2020/2021", "Variation RE 2021/2022", "Variation RE 2022/2023",
    "Variation CP 2020/2021", "Variation CP 2021/2022", "Variation Charges 2022/2023",
]

def format_dirhams(value):
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f} Md Dhs"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f} M Dhs"
    elif value >= 1_000:
        return f"{value/1_000:.1f} K Dhs"
    else:
        return f"{value:.0f} Dhs"

df = df[columns_to_keep]

df["Entreprise"] = df[
    ["Raison Sociale (Maroc1000 Nouvelle)", "Raison Sociale (Maroc1000 ancienne)", "Raison Sociale (Kerix)"]
].bfill(axis=1).iloc[:, 0].str.strip()

num_cols = [col for col in df.columns if col not in ["Secteur", "Raison Sociale (Maroc1000 Nouvelle)", "Raison Sociale (Maroc1000 ancienne)", "Raison Sociale (Kerix)", "Entreprise"]]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=["Secteur"])
st.title("📊 Analyse Sectorielle")

num_cols = [col for col in df.columns if col not in ["Secteur", "Raison Sociale (Maroc1000 Nouvelle)", "Raison Sociale (Maroc1000 ancienne)", "Raison Sociale (Kerix)", "Entreprise"]]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=["Secteur"])

available_years = sorted([
    int(re.search(r"\b(20\d{2})\b", col).group(1))
    for col in df.columns
    if "Chiffre d'affaires" in col and "Dhs" in col and re.search(r"\b(20\d{2})\b", col)
])
selected_year = st.selectbox("Sélectionnez l'année", available_years, index=len(available_years)-1)

ca_column = f"Chiffre d'affaires {selected_year} (Dhs)"

st.header(f"1. Composition des Secteurs par CA Total ({selected_year})")

def format_dirhams(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} B Dhs"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f} M Dhs"
    else:
        return f"{value:.0f} Dhs"

sector_sum = df.groupby("Secteur")[ca_column].sum().reset_index()
sector_sum = sector_sum.sort_values(by=ca_column, ascending=False)
total_market = sector_sum[ca_column].sum()
sector_stats = df.groupby("Secteur").agg(
    ca_sum=(ca_column, "sum"),
    company_count=("Secteur", "count")
).reset_index()

sector_stats = sector_stats.sort_values(by="ca_sum", ascending=False)
total_market = sector_stats["ca_sum"].sum()
total_companies = sector_stats["company_count"].sum()
sector_stats["percentage"] = (sector_stats["ca_sum"] / total_market * 100).round(2)
sector_stats["label"] = sector_stats["Secteur"] + "<br>" + sector_stats["percentage"].astype(str) + "%"
st.markdown(f"### Total CA marché : **{format_dirhams(total_market)} pour {total_companies} entreprises**")

fig_treemap = px.treemap(
    sector_stats,
    path=["label"],
    values="ca_sum",
    hover_data={
        "ca_sum": ":,.0f",
        "company_count": True,
        "percentage": True
    },
    title=f"Treemap: Répartition du CA {selected_year} par Secteur"
)

st.plotly_chart(fig_treemap)


fig_pie = px.pie(sector_sum, names='Secteur', values=ca_column, hover_data={ca_column: ":,.0f"},
                 title=f"Répartition CA {selected_year} par Secteur")

fig_pie.update_layout(
    annotations=[dict(
        text=f"Total CA {selected_year}:<br><b>{total_market / 1e9:.2f} B Dhs</b>",
        x=0.5, y=1.15,
        xref='paper', yref='paper',
        font_size=14,
        showarrow=False
    )]
)

st.plotly_chart(fig_pie)

st.header("2. Évolution d'un Secteur Sélectionné avec CAGR")
def calculate_cagr(start, end, periods):
    try:
        return ((end / start) ** (1 / periods)) - 1
    except:
        return np.nan

selected_sector = st.selectbox("Choisir un secteur à visualiser:", sorted(df['Secteur'].unique()))
filtered_df = df[df['Secteur'] == selected_sector]

# variables = {
#     "Chiffre d'affaires": [f"Chiffre d'affaires {y} (Dhs)" for y in range(2020, 2024)],
#     "Resultat d'exploitation": [f"Resultat d'exploitation {y} (Dhs)" for y in range(2020, 2024)],
#     "Charges personnel": [f"Charges personnel {y}" for y in range(2020, 2024)]
# }

# for var_name, cols in variables.items():
#     sector_yearly = filtered_df[cols].sum().reset_index()
#     sector_yearly.columns = ["Année", var_name]
#     sector_yearly["Année"] = sector_yearly["Année"].str.extract(r'(\d{4})').astype(int)

#     start_value = sector_yearly[var_name].iloc[0]
#     end_value = sector_yearly[var_name].iloc[-1]
#     n_years = sector_yearly["Année"].iloc[-1] - sector_yearly["Année"].iloc[0]
#     cagr = ((end_value / start_value) ** (1 / n_years) - 1) if start_value != 0 else 0
#     cagr_text = f"CAGR: {cagr*100:.2f}%"

#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=sector_yearly["Année"],
#         y=sector_yearly[var_name],
#         name=f"{var_name} Total",
#         text=[f"{y/1e9:.2f} B Dhs" for y in sector_yearly[var_name]],
#         textposition="outside"
#     ))

#     sector_yearly["Variation"] = sector_yearly[var_name].pct_change()
#     fig.add_trace(go.Scatter(
#         x=sector_yearly["Année"],
#         y=sector_yearly[var_name],
#         mode="lines+markers+text",
#         name="Variation",
#         text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in sector_yearly["Variation"]],
#         textposition="bottom center",
#         line=dict(dash="dot")
#     ))

#     fig.add_annotation(
#         text=cagr_text,
#         xref="paper", yref="paper",
#         x=0.5, y=1.15,
#         showarrow=False,
#         font=dict(size=16, color="black", family="Arial Black"),
#         bgcolor="rgba(255,255,255,0.8)",
#         bordercolor="black",
#         borderwidth=1,
#         borderpad=4,
#     )

#     fig.update_layout(
#         title=f"Évolution de {var_name} pour {selected_sector}",
#         yaxis_title=var_name,
#         xaxis_title="Année",
#         margin=dict(t=100)
#     )

#     st.plotly_chart(fig)

variables = {
    "Chiffre d'affaires": [f"Chiffre d'affaires {y} (Dhs)" for y in range(2020, 2024)],
    "Resultat d'exploitation": [f"Resultat d'exploitation {y} (Dhs)" for y in range(2020, 2024)],
    "Charges personnel": [f"Charges personnel {y}" for y in range(2020, 2024)]
}


ca_cols = variables["Chiffre d'affaires"]
re_cols = variables["Resultat d'exploitation"]


ca_yearly = filtered_df[ca_cols].sum().reset_index()
ca_yearly.columns = ["Année", "CA"]
ca_yearly["Année"] = ca_yearly["Année"].str.extract(r'(\d{4})').astype(int)

re_yearly = filtered_df[re_cols].sum().reset_index()
re_yearly.columns = ["Année", "RE"]
re_yearly["Année"] = re_yearly["Année"].str.extract(r'(\d{4})').astype(int)


merged_df = ca_yearly.merge(re_yearly, on="Année")


def compute_cagr(series):
    start_value, end_value = series.iloc[0], series.iloc[-1]
    n_years = merged_df["Année"].iloc[-1] - merged_df["Année"].iloc[0]
    return ((end_value / start_value) ** (1 / n_years) - 1) if start_value != 0 else 0

cagr_ca = compute_cagr(merged_df["CA"])
cagr_re = compute_cagr(merged_df["RE"])

merged_df["CA_Var"] = merged_df["CA"].pct_change()
merged_df["RE_Var"] = merged_df["RE"].pct_change()

fig = go.Figure()

fig.add_trace(go.Bar(
    x=merged_df["Année"],
    y=merged_df["CA"],
    name="Chiffre d'affaires (CA)",
    text=[f"{y/1e9:.2f} B Dhs" for y in merged_df["CA"]],
    textposition="outside"
))
fig.add_trace(go.Bar(
    x=merged_df["Année"],
    y=merged_df["RE"],
    name="Résultat d’exploitation (RE)",
    text=[f"{y/1e9:.2f} B Dhs" for y in merged_df["RE"]],
    textposition="outside"
))

fig.add_trace(go.Scatter(
    x=merged_df["Année"],
    y=merged_df["CA"],
    mode="lines+markers+text",
    name="Variation CA",
    text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in merged_df["CA_Var"]],
    textposition="bottom center",
    line=dict(dash="dot", color="blue")
))
fig.add_trace(go.Scatter(
    x=merged_df["Année"],
    y=merged_df["RE"],
    mode="lines+markers+text",
    name="Variation RE",
    text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in merged_df["RE_Var"]],
    textposition="top center",
    line=dict(dash="dot", color="red")
))


fig.add_annotation(
    text=f"CAGR CA: {cagr_ca*100:.2f}%",
    xref="paper", yref="paper", x=0.25, y=1.15,
    showarrow=False,
    font=dict(size=14, color="blue", family="Arial Black"),
    bgcolor="rgba(255,255,255,0.8)", bordercolor="blue",
    borderwidth=1, borderpad=4,
)
fig.add_annotation(
    text=f"CAGR RE: {cagr_re*100:.2f}%",
    xref="paper", yref="paper", x=0.75, y=1.15,
    showarrow=False,
    font=dict(size=14, color="red", family="Arial Black"),
    bgcolor="rgba(255,255,255,0.8)", bordercolor="red",
    borderwidth=1, borderpad=4,
)


fig.update_layout(
    title=f"Évolution de CA et RE pour {selected_sector}",
    yaxis_title="Montants (Dhs)",
    xaxis_title="Année",
    barmode="group",
    margin=dict(t=120),
    legend=dict(orientation="h", y=-0.2)
)

st.plotly_chart(fig)


charges_cols = variables["Charges personnel"]
charges_yearly = filtered_df[charges_cols].sum().reset_index()
charges_yearly.columns = ["Année", "Charges"]
charges_yearly["Année"] = charges_yearly["Année"].str.extract(r'(\d{4})').astype(int)


start_value = charges_yearly["Charges"].iloc[0]
end_value = charges_yearly["Charges"].iloc[-1]
n_years = charges_yearly["Année"].iloc[-1] - charges_yearly["Année"].iloc[0]
cagr_charges = ((end_value / start_value) ** (1 / n_years) - 1) if start_value != 0 else 0


charges_yearly["Variation"] = charges_yearly["Charges"].pct_change()


fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=charges_yearly["Année"],
    y=charges_yearly["Charges"],
    name="Charges personnel",
    text=[f"{y/1e9:.2f} B Dhs" for y in charges_yearly["Charges"]],
    textposition="outside",
    width=0.3
))
fig2.add_trace(go.Scatter(
    x=charges_yearly["Année"],
    y=charges_yearly["Charges"],
    mode="lines+markers+text",
    name="Variation",
    text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in charges_yearly["Variation"]],
    textposition="bottom center",
    line=dict(dash="dot")
))

fig2.add_annotation(
    text=f"CAGR: {cagr_charges*100:.2f}%",
    xref="paper", yref="paper", x=0.5, y=1.15,
    showarrow=False,
    font=dict(size=16, color="black", family="Arial Black"),
    bgcolor="rgba(255,255,255,0.8)", bordercolor="black",
    borderwidth=1, borderpad=4,
)

fig2.update_layout(
    title=f"Évolution des Charges de personnel pour {selected_sector}",
    yaxis_title="Charges (Dhs)",
    xaxis_title="Année",
    margin=dict(t=100)
)

st.plotly_chart(fig2)

marge_variables = {
    "Marge EBIT/CA": [f"Marge EBIT/CA {y}" for y in range(2020, 2024)],
    "Marge EBIT/CP": [f"Marge EBIT/CP {y}" for y in range(2020, 2024)],
    "Marge CP/CA": [f"Marge CP/CA {y}" for y in range(2020, 2024)]
}

for var_name, cols in marge_variables.items():
    sector_yearly = filtered_df[cols].mean().reset_index()
    sector_yearly.columns = ["Année", var_name]
    sector_yearly["Année"] = sector_yearly["Année"].str.extract(r'(\d{4})').astype(int)

    start_value = sector_yearly[var_name].iloc[0]
    end_value = sector_yearly[var_name].iloc[-1]
    n_years = sector_yearly["Année"].iloc[-1] - sector_yearly["Année"].iloc[0]
    cagr = ((end_value / start_value) ** (1 / n_years) - 1) if start_value != 0 else np.nan
    cagr_text = f"CAGR: {cagr*100:.2f}%" if not np.isnan(cagr) else "CAGR: N/A"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sector_yearly["Année"],
        y=sector_yearly[var_name],
        name=f"{var_name} Moyen",
        text=[f"{v*100:.1f}%" for v in sector_yearly[var_name]],
        textposition="outside",
        width=0.3
    ))

    sector_yearly["Variation"] = sector_yearly[var_name].pct_change()
    fig.add_trace(go.Scatter(
        x=sector_yearly["Année"],
        y=sector_yearly[var_name],
        mode="lines+markers+text",
        name="Variation",
        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in sector_yearly["Variation"]],
        textposition="bottom center",
        line=dict(dash="dot")
    ))

    fig.add_annotation(
        text=cagr_text,
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(size=16, color="black", family="Arial Black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )

    fig.update_layout(
        title=f"Évolution de {var_name} pour {selected_sector}",
        yaxis_title=var_name,
        xaxis_title="Année",
        margin=dict(t=100)
    )

    st.plotly_chart(fig)

st.header("3. CAGR pour Tous les Indicateurs par Secteur")

cagr_variables = {
    "Chiffre d'affaires": {
        "label": "CAGR CA",
        "start_col": "Chiffre d'affaires 2020 (Dhs)",
        "end_col": "Chiffre d'affaires 2023 (Dhs)"
    },
    "Resultat d'exploitation": {
        "label": "CAGR RE",
        "start_col": "Resultat d'exploitation 2020 (Dhs)",
        "end_col": "Resultat d'exploitation 2023 (Dhs)"
    },
    "Charges personnel": {
        "label": "CAGR CP",
        "start_col": "Charges personnel 2020",
        "end_col": "Charges personnel 2023"
    },
    "Marge EBIT/CA": {
        "label": "CAGR EBIT/CA",
        "start_col": "Marge EBIT/CA 2020",
        "end_col": "Marge EBIT/CA 2023"
    },
    "Marge EBIT/CP": {
        "label": "CAGR EBIT/CP",
        "start_col": "Marge EBIT/CP 2020",
        "end_col": "Marge EBIT/CP 2023"
    },
    "Marge CP/CA": {
        "label": "CAGR CP/CA",
        "start_col": "Marge CP/CA 2020",
        "end_col": "Marge CP/CA 2023"
    }
}

def calculate_cagr(start, end, years):
    if start <= 0 or end <= 0:
        return np.nan
    try:
        return ((end / start) ** (1 / years)) - 1
    except Exception:
        return np.nan

for var_name, cols in variables.items():
    sector_yearly = filtered_df[cols].sum().reset_index()
    sector_yearly.columns = ["Année", var_name]
    sector_yearly["Année"] = sector_yearly["Année"].str.extract(r'(\d{4})').astype(int)

    start_value = sector_yearly[var_name].iloc[0]
    end_value = sector_yearly[var_name].iloc[-1]
    n_years = sector_yearly["Année"].iloc[-1] - sector_yearly["Année"].iloc[0]

    cagr = calculate_cagr(start_value, end_value, n_years)
    cagr_text = f"CAGR: {cagr*100:.2f}%" if not np.isnan(cagr) else "CAGR: N/A"


for var, config in cagr_variables.items():
    label = config["label"]
    start_col = config["start_col"]
    end_col = config["end_col"]

    temp_data = df.dropna(subset=[start_col, end_col]).copy()
    if temp_data.empty:
        st.info(f"Aucune donnée pour calculer {label}")
        continue

    try:
        temp_data[label] = temp_data.apply(
            lambda row: calculate_cagr(row[start_col], row[end_col], 3),
            axis=1
        )
        temp_data = temp_data.dropna(subset=[label])
        sector_avg = temp_data.groupby("Secteur")[label].mean().reset_index()
        fig = px.bar(sector_avg.sort_values(label, ascending=False), x="Secteur", y=label,
                     title=f"{label} Moyen par Secteur (2020–2023)", text=label)
        fig.update_layout(yaxis_tickformat=".2%")
        st.plotly_chart(fig)
    except Exception as e:
        st.warning(f"Erreur lors du calcul du CAGR pour {var}: {e}")


st.header("4. Détail par Secteur")
selected_sector = st.selectbox("Choisir un secteur:", sorted(df['Secteur'].unique()))
filtered_df = df[df['Secteur'] == selected_sector]
cols_to_show = ['Entreprise'] + [col for col in df.columns if "Chiffre d'affaires" in col]
st.dataframe(filtered_df[cols_to_show])

st.header("5. Analyse d'une Entreprise")
selected_company = st.selectbox("Choisir une entreprise:", sorted(df['Entreprise'].dropna().unique()))

company_df = df[df['Entreprise'] == selected_company]
if not company_df.empty:
    company_series = company_df.iloc[0]
    company_data = pd.DataFrame({
        'Variable': company_series.index,
        'Valeur': company_series.values
    })
    st.dataframe(company_data)
else:
    st.write("Aucune donnée disponible pour cette entreprise.")