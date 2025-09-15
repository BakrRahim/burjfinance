import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re
import plotly.colors as pc

@st.cache_data
def load_data():
    df = pd.read_excel("companies.xlsx")
    return df

st.set_page_config(
    page_title="Analyse Des Secteurs",
    page_icon="📊",
    layout="wide"
)
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
df = df[columns_to_keep]
df["Entreprise"] = df[
    ["Raison Sociale (Maroc1000 Nouvelle)", "Raison Sociale (Maroc1000 ancienne)", "Raison Sociale (Kerix)"]
].bfill(axis=1).iloc[:, 0].astype(str).str.strip()
non_num = ["Secteur", "Raison Sociale (Maroc1000 Nouvelle)", "Raison Sociale (Maroc1000 ancienne)",
           "Raison Sociale (Kerix)", "Entreprise"]
num_cols = [col for col in df.columns if col not in non_num]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=["Secteur"])

def format_dirhams(value):
    if pd.isna(value):
        return "0.0"
    value = float(value)
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} B Dhs"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f} M Dhs"
    elif value >= 1_000:
        return f"{value / 1_000:.1f} K Dhs"
    else:
        return f"{value:.0f} Dhs"

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
        sector_sums = df.groupby("Secteur")[cols].sum()
        cagr_data[f"{var_name}_CAGR"] = {}
        for sector in sector_sums.index:
            sector_values = sector_sums.loc[sector].values
            if len(sector_values) == len(years) and not np.all(np.isnan(sector_values)):
                start_val = sector_values[0]
                end_val = sector_values[-1]
                if pd.notna(start_val) and pd.notna(end_val) and start_val > 0:
                    cagr = calculate_cagr(start_val, end_val, len(years)-1)
                    cagr_data[f"{var_name}_CAGR"][sector] = cagr
                else:
                    cagr_data[f"{var_name}_CAGR"][sector] = np.nan
            else:
                cagr_data[f"{var_name}_CAGR"][sector] = np.nan
    
    for var_name, cols in margin_vars.items():
        sector_means = df.groupby("Secteur")[cols].mean()
        cagr_data[f"{var_name}_CAGR"] = {}
        for sector in sector_means.index:
            sector_values = sector_means.loc[sector].values
            if len(sector_values) == len(years) and not np.all(np.isnan(sector_values)):
                start_val = sector_values[0]
                end_val = sector_values[-1]
                if pd.notna(start_val) and pd.notna(end_val):
                    cagr = calculate_cagr(start_val, end_val, len(years)-1)
                    cagr_data[f"{var_name}_CAGR"][sector] = cagr
                else:
                    cagr_data[f"{var_name}_CAGR"][sector] = np.nan
            else:
                cagr_data[f"{var_name}_CAGR"][sector] = np.nan
    
    return cagr_data
cagr_data = precalculate_sector_cagrs(df)
st.title("📊 Analyse Des Secteurs")
available_years = sorted([
    int(re.search(r"\b(20\d{2})\b", col).group(1))
    for col in df.columns
    if "Chiffre d'affaires" in col and re.search(r"\b(20\d{2})\b", col)
])
selected_year = st.selectbox("Sélectionnez l'année", available_years, index=len(available_years)-1)
ca_column = f"Chiffre d'affaires {selected_year} (Dhs)"
sector_sum = df.groupby("Secteur")[ca_column].sum().reset_index().fillna(0)
sector_sum = sector_sum.sort_values(by=ca_column, ascending=False)
sector_order = sector_sum["Secteur"].tolist()
st.header(f"A. Overview Marché")
sector_stats = df.groupby("Secteur").agg(
    ca_sum=(ca_column, "sum"),
    company_count=("Secteur", "count")
).reset_index().fillna(0)
sector_stats = sector_stats.sort_values(by="ca_sum", ascending=False)
total_market = sector_stats["ca_sum"].sum()
total_companies = sector_stats["company_count"].sum()
sector_stats["percentage"] = (sector_stats["ca_sum"] / total_market * 100).round(2)
sector_stats["label"] = sector_stats["Secteur"] + "<br>" + sector_stats["percentage"].astype(str) + "%"
st.markdown(f"### Total CA marché : **{format_dirhams(total_market)} pour {int(total_companies)} entreprises**")
fig_treemap = px.treemap(
    sector_stats,
    path=["label"],
    values="ca_sum",
    hover_data={"ca_sum": ":,.0f", "company_count": True, "percentage": True},
    title=f"Treemap: Répartition du CA {selected_year} par Secteur"
)
fig_treemap.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', zeroline=True, zerolinecolor="black",
        zerolinewidth=1), 
    yaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False, showticklabels=False),
    font=dict(color='black', size=18),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig_treemap.update_traces(marker=dict(cornerradius=5))
st.plotly_chart(fig_treemap, use_container_width=True)
top_n = 15
sector_sum_sorted = sector_sum.sort_values(ca_column, ascending=False)
top_df = sector_sum_sorted.head(top_n)
others_sum = sector_sum_sorted.iloc[top_n:][ca_column].sum()
if others_sum > 0:
    top_df = pd.concat([top_df, pd.DataFrame({"Secteur": ["Autres"], ca_column: [others_sum]})])
fig_pie = px.pie(
    top_df,
    names="Secteur",
    values=ca_column,
    title=f"Répartition CA {selected_year} par Secteur",
    hole=0.4,
)
fig_pie.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    annotations=[dict(
        text=f"Total CA {selected_year}:<br><b>{total_market / 1e9:.2f} B Dhs</b>",
        x=0.5, y=0.5,
        xref='paper', yref='paper',
        font_size=20,
        showarrow=False,
        bgcolor='rgba(0,0,0,0)'
    )],
    xaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False,zeroline=True,zerolinecolor="black",
        zerolinewidth=1),
    yaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False, showticklabels=False),
    font=dict(color='black', size=20),
    margin=dict(l=0, r=0, t=40, b=0)
)
fig_pie.update_traces(marker=dict(line=dict(color='white', width=2)))
st.plotly_chart(fig_pie, use_container_width=True)

def plot_sector_evolution(sector_name, df_sector, ca_cols, re_cols, title_suffix=None):
    ca_yearly = df_sector[ca_cols].sum().reset_index()
    ca_yearly.columns = ["Année", "CA"]
    ca_yearly["Année"] = ca_yearly["Année"].str.extract(r'(\d{4})').astype(int)
    re_yearly = df_sector[re_cols].sum().reset_index()
    re_yearly.columns = ["Année", "RE"]
    re_yearly["Année"] = re_yearly["Année"].str.extract(r'(\d{4})').astype(int)
    merged_df = ca_yearly.merge(re_yearly, on="Année").sort_values("Année")
    if merged_df.shape[0] < 2:
        st.info(f"Pas assez de données pour {sector_name}")
        return
    n_years = merged_df["Année"].iloc[-1] - merged_df["Année"].iloc[0]
    cagr_ca = cagr_data.get("Chiffre d'affaires_CAGR", {}).get(sector_name, np.nan)
    cagr_re = cagr_data.get("Resultat d'exploitation_CAGR", {}).get(sector_name, np.nan)
    
    merged_df["CA_Var"] = merged_df["CA"].pct_change()
    merged_df["RE_Var"] = merged_df["RE"].pct_change()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged_df["Année"],
        y=merged_df["CA"],
        name="Chiffre d'affaires (CA)",
        text=[format_dirhams(y) for y in merged_df["CA"]],
        textposition="auto",
        yaxis="y1",
        offsetgroup=1,
        textfont=dict(size=16),
        width=0.26,
        marker=dict(line=dict(width=0))
    ))
    fig.add_trace(go.Bar(
        x=merged_df["Année"],
        y=merged_df["RE"],
        name="Résultat d'exploitation (RE)",
        text=[format_dirhams(y) for y in merged_df["RE"]],
        textposition="auto",
        yaxis="y2",
        offsetgroup=2,
        textfont=dict(size=16),
        width=0.26,
        marker=dict(line=dict(width=0))
    ))
    fig.add_trace(go.Scatter(
        x=merged_df["Année"] - 0.20,
        y=merged_df["CA"],
        mode="lines+text",
        name="Variation CA",
        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in merged_df["CA_Var"]],
        textposition="top center",
        line=dict(shape='spline', dash="dot", color="blue"),
        textfont=dict(size=16),
        yaxis="y1",
        marker=dict(size=10, line=dict(width=0))
    ))
    fig.add_trace(go.Scatter(
        x=merged_df["Année"] + 0.20,
        y=merged_df["RE"],
        mode="lines+text",
        name="Variation RE",
        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in merged_df["RE_Var"]],
        textposition="top center",
        line=dict(shape='spline', dash="dot", color="red"),
        yaxis="y2",
        textfont=dict(size=16),
        marker=dict(size=10, line=dict(width=0))
    ))
    marge_cols = [f"Marge EBIT/CA {y}" for y in range(2020, 2024)]
    marge_yearly = df_sector[marge_cols].mean().reset_index()
    marge_yearly.columns = ["Année", "Marge_EBIT_CA"]
    marge_yearly["Année"] = marge_yearly["Année"].str.extract(r'(\d{4})').astype(int)
    merged_df = merged_df.merge(marge_yearly, on="Année", how="left")
    merged_df["Marge_EBIT_CA_Var"] = merged_df["Marge_EBIT_CA"].pct_change()
    fig.add_trace(go.Scatter(
        x=merged_df["Année"],
        y=merged_df["Marge_EBIT_CA"],
        mode="lines+text",
        name="Marge EBIT/CA",
        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in merged_df["Marge_EBIT_CA"]],
        textposition="top center",
        textfont=dict(size=16),
        line=dict(shape='spline', color="#052449"),
        yaxis="y3",
        marker=dict(size=10, line=dict(width=0))
    ))
    fig.add_annotation(
        text=f"CAGR CA: {(cagr_ca*100):.2f}%" if not pd.isna(cagr_ca) else "CAGR CA: N/A",
        xref="paper", yref="paper", x=0.25, y=1.15,
        showarrow=False,
        font=dict(size=14, color="blue"),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="blue",
        borderwidth=1, borderpad=4,
    )
    fig.add_annotation(
        text=f"CAGR RE: {(cagr_re*100):.2f}%" if not pd.isna(cagr_re) else "CAGR RE: N/A",
        xref="paper", yref="paper", x=0.75, y=1.15,
        showarrow=False,
        font=dict(size=14, color="red"),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="red",
        borderwidth=1, borderpad=4,
    )    
    
    max_ca = merged_df["CA"].max() if merged_df["CA"].max() > 0 else 1
    fig.update_layout(
        title=f"i. Évolution de CA, RE et Marge EBIT/CA pour {sector_name}" + (f" - {title_suffix}" if title_suffix else ""),
        xaxis_title="Année",
        xaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', zeroline=True,zerolinecolor="black",
        zerolinewidth=1),
        yaxis=dict(title="Montants CA (Dhs)", side="left", showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False, showticklabels=False),
        yaxis2=dict(
            title="Montants RE (Dhs)",
            side="right",
            overlaying="y",
            range=[0, max_ca / 8],
            showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)',
            visible=False, showticklabels=False
        ),
        yaxis3=dict(
            title="Marge EBIT/CA (%)",
            side="right",
            overlaying="y",
            range=[-0.05, 0.15],
            showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)',
            tickformat=".0%", visible=False, showticklabels=False
        ),
        barmode="group",
        margin=dict(t=94),
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black')
    )
    fig.update_xaxes(
        dtick=1,
        tick0=2020,
        tickformat="d"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("B. Vue sectorielle individuelle")
selected_sector = st.selectbox("Choisir un secteur à visualiser:", sorted(df['Secteur'].unique()))
filtered_df = df[df['Secteur'] == selected_sector]
variables = {
    "Chiffre d'affaires": [f"Chiffre d'affaires {y} (Dhs)" for y in range(2020, 2024)],
    "Resultat d'exploitation": [f"Resultat d'exploitation {y} (Dhs)" for y in range(2020, 2024)],
    "Charges personnel": [f"Charges personnel {y}" for y in range(2020, 2024)]
}
ca_cols = variables["Chiffre d'affaires"]
re_cols = variables["Resultat d'exploitation"]
charges_cols = variables["Charges personnel"]
plot_sector_evolution(selected_sector, filtered_df, ca_cols, re_cols)
charges_yearly = filtered_df[charges_cols].sum().reset_index()
charges_yearly.columns = ["Année", "Charges"]
charges_yearly["Année"] = charges_yearly["Année"].str.extract(r'(\d{4})').astype(int)
if charges_yearly.shape[0] >= 2:
    cagr_charges = cagr_data.get("Charges personnel_CAGR", {}).get(selected_sector, np.nan)
    charges_yearly["Variation"] = charges_yearly["Charges"].pct_change()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=charges_yearly["Année"],
        y=charges_yearly["Charges"],
        name="Charges personnel",
        text=[format_dirhams(y) for y in charges_yearly["Charges"]],
        textposition="auto",
        width=0.3,
        textfont=dict(size=16),
        marker=dict(line=dict(width=0))
    ))
    fig2.add_trace(go.Scatter(
        x=charges_yearly["Année"],
        y=charges_yearly["Charges"],
        mode="lines+text",
        name="Variation",
        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in charges_yearly["Variation"]],
        textposition="top center",
        line=dict(shape='spline', dash='dot'),
        textfont=dict(size=16),
        marker=dict(size=10, line=dict(width=0))
    ))
    fig2.add_annotation(
        text=f"CAGR: {(cagr_charges*100):.2f}%" if not pd.isna(cagr_charges) else "CAGR: N/A",
        xref="paper", yref="paper", x=0.5, y=1.15,
        showarrow=False,
        font=dict(size=16, color="black"),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="black",
        borderwidth=1, borderpad=4,
    )
    fig2.update_layout(
        title=f"ii. Évolution des Charges de personnel pour {selected_sector}",
        yaxis_title="Charges (Dhs)",
        xaxis_title="Année",
        xaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', zeroline=True, zerolinecolor="black",
        zerolinewidth=1),
        yaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False, showticklabels=False),
        margin=dict(t=90),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black')
    )
    fig2.update_xaxes(
        dtick=1,
        tick0=2020,
        tickformat="d"
    )
    st.plotly_chart(fig2, use_container_width=True)
marge_variables = {
    "Marge EBIT/CA": [f"Marge EBIT/CA {y}" for y in range(2020, 2024)],
    "Marge CP/CA": [f"Marge CP/CA {y}" for y in range(2020, 2024)],
    "Marge EBIT/CP": [f"Marge EBIT/CP {y}" for y in range(2020, 2024)]
}
label = {
    "Marge EBIT/CA": "iii",
    "Marge CP/CA": "iv",
    "Marge EBIT/CP": "v"
}
for var_name, cols in marge_variables.items():
    sector_yearly = filtered_df[cols].mean().reset_index()
    sector_yearly.columns = ["Année", var_name]
    sector_yearly["Année"] = sector_yearly["Année"].str.extract(r'(\d{4})').astype(int)
    if sector_yearly.shape[0] < 2:
        continue
    cagr = cagr_data.get(f"{var_name}_CAGR", {}).get(selected_sector, np.nan)
    sector_yearly["Variation"] = sector_yearly[var_name].pct_change()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sector_yearly["Année"],
        y=sector_yearly[var_name],
        name=f"{var_name}",
        text=[f"{v*100:.1f}%" if not pd.isna(v) else "" for v in sector_yearly[var_name]],
        textfont=dict(size=16),
        textposition="auto",
        width=0.3,
        marker=dict(line=dict(width=0))
    ))
    fig.add_trace(go.Scatter(
        x=sector_yearly["Année"],
        y=sector_yearly[var_name],
        mode="lines+text",
        name="Variation",
        text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in sector_yearly["Variation"]],
        textposition="top center",
        line=dict(shape='spline', dash="dot"),
        textfont=dict(size=16),
        marker=dict(size=10, line=dict(width=0))
    ))
    fig.add_annotation(
        text=f"CAGR: {(cagr*100):.2f}%" if not pd.isna(cagr) else "CAGR: N/A",
        xref="paper", yref="paper",
        x=0.5, y=1.15,
        showarrow=False,
        font=dict(size=16, color="black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )
    fig.update_layout(
        title=f"{label[var_name]}. Évolution de {var_name} pour {selected_sector}",
        yaxis_title=f"{var_name} (%)",
        xaxis_title="Année",
        xaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', zeroline=True, zerolinecolor="black",
        zerolinewidth=1),
        yaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', tickformat=".0%", visible=False, showticklabels=False),
        margin=dict(t=150),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black')
    )
    fig.update_xaxes(
        dtick=1,
        tick0=2020,
        tickformat="d"
    )
    st.plotly_chart(fig, use_container_width=True)
st.markdown(
    f"<h6 style='font-weight:700;'>vi. Répartition {selected_sector}</h6>",
    unsafe_allow_html=True
)
top_n_single = st.number_input("Nombre d'entreprises à afficher (secteur sélectionné)", 3, 100, 10, key="top_n_single")
sector_df = df[df['Secteur'] == selected_sector].dropna(subset=[ca_column])
sorted_sector_df = sector_df.sort_values(by=ca_column, ascending=False)
top_companies = sorted_sector_df.head(top_n_single).copy()
others = sorted_sector_df.iloc[top_n_single:]
if not others.empty:
    autres_sum = others[ca_column].sum()
    autres_row = pd.DataFrame({"Entreprise": ["Autres"], ca_column: [autres_sum], "Secteur": [selected_sector]})
    top_companies = pd.concat([top_companies, autres_row], ignore_index=True)
fig_sector_pie = px.pie(
    top_companies,
    names="Entreprise",
    values=ca_column,
    title=f"{selected_sector} (Top {top_n_single} + Autres)",
    hover_data={ca_column: ":,.0f"}
)
fig_sector_pie.update_traces(textinfo="percent+label", pull=[0.05] * len(top_companies))
fig_sector_pie.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', zeroline=True, zerolinecolor="black",
        zerolinewidth=1),
    yaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False, showticklabels=False),
    font=dict(color='black'),
    margin=dict(l=0, r=0, t=40, b=0),
)
fig_sector_pie.update_traces(textfont=dict(size=13))
st.plotly_chart(fig_sector_pie, use_container_width=True)

def plot_multi_metric(multi_sectors, df, years, var_name, cols, yaxis_title, chart_title=None):
    bar_count = len(multi_sectors)
    bar_width = 0.6 / max(1, bar_count)
    fig = go.Figure()
    palette = ["#052449", "#0064EF"]
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(multi_sectors)}
    annotations = []
    for i, sector in enumerate(multi_sectors):
        s_df = df[df["Secteur"] == sector]
        if s_df.empty:
            continue
        vals = [float(s_df.get(col, pd.Series(dtype=float)).mean(skipna=True)) for col in cols]
        vals = np.array(vals, dtype=float)
        color = color_map[sector]
        scatter_offset = (i * bar_width) - (bar_count/2 - 0.5) * bar_width
        fig.add_trace(go.Bar(
            x=years,
            y=vals,
            name=f"{sector} - {var_name}",
            marker_color=color,
            width=bar_width,
            offsetgroup=f"{sector}_{var_name}",
            alignmentgroup="all",
            text=[f"{v*100:.1f}%" if "Marge" in var_name and not pd.isna(v) else format_dirhams(v) for v in vals],
            textposition="auto",
            marker=dict(line=dict(width=0)),
            textfont=dict(size=16),
        ))
        pct = np.array([np.nan] + list((vals[1:] / vals[:-1] - 1))) if len(vals) > 1 else np.array([np.nan]*len(vals))
        fig.add_trace(go.Scatter(
            x=[y + scatter_offset for y in years],
            y=vals,
            mode="lines+text",
            name=f"{sector} - Var {var_name}",
            line=dict(shape='spline', dash="dot", color=color),
            text=["" if pd.isna(v) else f"{v*100:.1f}%" for v in pct],
            textposition="top center",
            hovertemplate="%{x}<br>%{fullData.name}: %{text}<extra></extra>",
            marker=dict(size=10, line=dict(width=0)),
            textfont=dict(size=16),
        ))
        cagr_key = f"{var_name}_CAGR" if var_name in ["Chiffre d'affaires", "Resultat d'exploitation", "Charges personnel"] else f"{var_name}_CAGR"
        cagr = cagr_data.get(cagr_key, {}).get(sector, np.nan)
        annotations.append(dict(
            text=f"CAGR: {(cagr*100):.2f}%" if not pd.isna(cagr) else "CAGR: N/A",
            x=0.27 + i * (0.55 / max(1, bar_count)),
            y=1.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14, color=color),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=color,
            borderwidth=1,
            borderpad=4
        ))
    fig.update_layout(
        title=dict(text=chart_title or var_name),
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        height=500,
        margin=dict(t=95),
        xaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', zeroline=True, zerolinecolor="black",
        zerolinewidth=1),
        yaxis=dict(title=yaxis_title, showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False, showticklabels=False),
        annotations=annotations,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black'),
        legend=dict(orientation="h", y=-0.2),
    )
    fig.update_xaxes(
        dtick=1,
        tick0=2020,
        tickformat="d"
    )
    all_y = []
    for trace in fig.data:
        if hasattr(trace, "y") and trace.y is not None:
            all_y.extend(trace.y)
    if all_y:
        max_y = np.nanmax(all_y)
        fig.update_yaxes(range=[0, max_y * 1.2])
    st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.header("C. Vue sectorielle comparative")
multi_sectors = st.multiselect(
    "Choisir plusieurs secteurs à comparer",
    sorted(df["Secteur"].unique()),
    default=sorted(df["Secteur"].unique())[-2:]
)
if multi_sectors:
    years = [2020, 2021, 2022, 2023]
    n_years = len(years) - 1
    combined_fig = go.Figure()
    palette = ["rgba(0,100,239,1)", "rgba(0,57,103,1)", "rgba(66,141,242,1)"]
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(multi_sectors)}
    annotations = []
    ann_x = 0.02
    ann_step = 0.95 / max(1, len(multi_sectors))
    n_sectors = len(multi_sectors)
    bar_count = len(multi_sectors)
    bar_width = 0.3
    min_marge = 0
    max_marge = 0.15
    padding = 0.05
    for i, sector in enumerate(multi_sectors):
        s_df = df[df['Secteur'] == sector]
        if s_df.empty:
            continue
        ca_vals = [float(s_df.get(f"Chiffre d'affaires {y} (Dhs)", pd.Series(dtype=float)).sum(skipna=True)) for y in years]
        re_vals = [float(s_df.get(f"Resultat d'exploitation {y} (Dhs)", pd.Series(dtype=float)).sum(skipna=True)) for y in years]
        ca_vals = np.array(ca_vals, dtype=float)
        re_vals = np.array(re_vals, dtype=float)
        
        marge_cols = [f"Marge EBIT/CA {y}" for y in years]
        marge_vals = s_df[marge_cols].mean().values
        
        ca_color = color_map[sector]
        
        combined_fig.add_trace(go.Bar(
            x=years,
            y=ca_vals,
            name=f"{sector} - CA",
            marker_color=ca_color,
            width=bar_width,
            offsetgroup=f"{sector}_CA",
            text=[format_dirhams(v) for v in ca_vals],
            textposition="auto",
            textfont=dict(size=16)
        ))
        
        combined_fig.add_trace(go.Scatter(
            x=years,
            y=marge_vals,
            name=f"{sector} - Marge EBIT/CA",
            line=dict(shape='spline', color=ca_color, width=2),
            mode="lines+text",
            text=[f"{v*100:.1f}%" if pd.notna(v) else "" for v in marge_vals],
            textposition="top center",
            yaxis="y3",
            textfont=dict(size=16)
        ))

        cagr_ca = cagr_data.get("Chiffre d'affaires_CAGR", {}).get(sector, np.nan)
        cagr_marge = cagr_data.get("Marge EBIT/CA_CAGR", {}).get(sector, np.nan)
        
        annotations.append(dict(
            text=f"{sector}<br>CAGR CA: {(cagr_ca*100):.2f}%" if not pd.isna(cagr_ca) else f"{sector}<br>CAGR CA: N/A",
            xref="paper",
            yref="paper",
            x=ann_x + i * ann_step,
            y=1.05,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor=ca_color,
            borderwidth=1,
            font=dict(size=11)
        ))
    max_ca = max([df[df['Secteur']==s][f"Chiffre d'affaires {y} (Dhs)"].sum() for s in multi_sectors for y in years if len(df[df['Secteur']==s]) > 0]) if multi_sectors else 1
    
    combined_fig.update_layout(
        title=f"i. Comparaison CA, Marge EBIT/CA et CAGR pour {len(multi_sectors)} Secteurs (2020-2023)",
        xaxis_title="Année",
        xaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)'),
        height = 700,
        yaxis=dict(
            title="Montants CA (Dhs)", 
            side="left",
            showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False, showticklabels=False, zeroline=True, zerolinecolor="black",
        zerolinewidth=1
        ),
        yaxis2=dict(
            title="Montants RE (Dhs)",
            side="right", 
            overlaying="y", 
            range=[0, max_ca / 8],
            showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False, showticklabels=False,
        ),
        yaxis3=dict(
            title="Marge EBIT/CA (%)",
            overlaying="y",
            side="right",
            anchor="x",
            showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', visible=False, showticklabels=False,
            tickformat=".0%",
            range=[min_marge - padding, max_marge + padding]
        ),
        barmode="group",
        margin=dict(t=160),
        legend=dict(orientation="h", y=-0.2),
        annotations=annotations,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black')
    )
    combined_fig.update_xaxes(
        dtick=1,
        tick0=2020,
        tickformat="d"
    )
    st.plotly_chart(combined_fig, use_container_width=True, key="combined_multi_evolution")
    
    plot_multi_metric(multi_sectors, df, years, "Chiffre d'affaires", [f"Chiffre d'affaires {y} (Dhs)" for y in years], "CA (Dhs)", chart_title=f"ii. Comparaison CA et Variations pour {len(multi_sectors)} Secteurs")
    plot_multi_metric(multi_sectors, df, years, "Résultat d'exploitation", [f"Resultat d'exploitation {y} (Dhs)" for y in years], "RE (Dhs)", chart_title=f"iii. Comparaison RE et Variations pour {len(multi_sectors)} Secteurs")
    plot_multi_metric(multi_sectors, df, years, "Charges personnel", [f"Charges personnel {y}" for y in years], "Charges (Dhs)", chart_title=f"iv. Comparaison Charges et Variations pour {len(multi_sectors)} Secteurs")
    plot_multi_metric(multi_sectors, df, years, "Marge EBIT/CA", [f"Marge EBIT/CA {y}" for y in years], "Marge EBIT/CA (%)", chart_title=f"v. Comparaison Marge EBIT/CA pour {len(multi_sectors)} Secteurs")
    plot_multi_metric(multi_sectors, df, years, "Marge EBIT/CP", [f"Marge EBIT/CP {y}" for y in years], "Marge EBIT/CP (%)", chart_title=f"vi. Comparaison Marge EBIT/CP pour {len(multi_sectors)} Secteurs")
    plot_multi_metric(multi_sectors, df, years, "Marge CP/CA", [f"Marge CP/CA {y}" for y in years], "Marge CP/CA (%)", chart_title=f"vii. Comparaison Marge CP/CA pour {len(multi_sectors)} Secteurs")
st.markdown("---")
st.header("D. Classement CAGR")
cagr_variables = {
    "Chiffre d'affaires": {
        "label": "CAGR CA",
        "cagr_key": "Chiffre d'affaires_CAGR"
    },
    "Resultat d'exploitation": {
        "label": "CAGR RE",
        "cagr_key": "Resultat d'exploitation_CAGR"
    },
    "Charges personnel": {
        "label": "CAGR CP",
        "cagr_key": "Charges personnel_CAGR"
    },
    "Marge EBIT/CA": {
        "label": "CAGR EBIT/CA",
        "cagr_key": "Marge EBIT/CA_CAGR"
    },
    "Marge EBIT/CP": {
        "label": "CAGR EBIT/CP",
        "cagr_key": "Marge EBIT/CP_CAGR"
    },
    "Marge CP/CA": {
        "label": "CAGR CP/CA",
        "cagr_key": "Marge CP/CA_CAGR"
    }
}
cagr_plot_idx = 0
for var, config in cagr_variables.items():
    label = config["label"]
    cagr_key = config["cagr_key"]
    
    if cagr_key not in cagr_data or not cagr_data[cagr_key]:
        st.info(f"Aucune donnée pour calculer {label}")
        continue
    
    sector_cagrs = pd.DataFrame.from_dict(cagr_data[cagr_key], orient='index', columns=[label]).reset_index()
    sector_cagrs.columns = ["Secteur", label]
    sector_cagrs = sector_cagrs.set_index("Secteur").reindex(sector_order).reset_index().fillna(np.nan)
    sector_cagrs = sector_cagrs.sort_values(label, ascending=False).dropna(subset=[label])
    
    if sector_cagrs.empty:
        st.info(f"Aucune valeur CAGR calculable pour {label}")
        continue
        
    fig = px.bar(sector_cagrs, x="Secteur", y=label,
                 title=f"{label} des Secteurs (2020–2023)")
    fig.update_traces(texttemplate='%{y:.2%}', textposition='auto', textfont=dict(size=20))
    fig.update_layout(
        height=550,
        margin=dict(t=80, b=30, l=40, r=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', tickangle=45, zeroline=True),
        yaxis=dict(showgrid=False, gridwidth=0, gridcolor='rgba(0,0,0,0)', tickformat=".2%", visible=False, showticklabels=False),
        font=dict(color='black')
    )
    fig.update_xaxes(
        dtick=1,
        tick0=2020,
        tickformat="d"
    )
    fig.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True, key=f"cagr_{cagr_plot_idx}")
    cagr_plot_idx += 1
st.markdown("---")
st.header("E. Détail par Secteur")
selected_sector_detail = st.selectbox(
    "Choisir un secteur:", 
    sorted(df['Secteur'].unique()), 
    key="detail_sector"
)
filtered_df_detail = df[df['Secteur'] == selected_sector_detail]
cols_to_show = ['Entreprise'] + [col for col in df.columns if "Chiffre d'affaires" in col]
df_detail_display = filtered_df_detail[cols_to_show].copy()
for col in df_detail_display.columns:
    if col != "Entreprise":
        df_detail_display[col] = df_detail_display[col].apply(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else x
        )
st.dataframe(df_detail_display)