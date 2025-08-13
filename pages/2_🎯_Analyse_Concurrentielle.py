import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Analyse Concurrentielle",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f5f7fa;
        color: #333333;
    }
    .block-container {
        padding: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton > button {
        border-radius: 10px;
        padding: 10px 20px;
        background: linear-gradient(90deg, #2980b9, #6dd5fa);
        border: none;
        transition: background 0.3s ease;
        color: white;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #6dd5fa, #2980b9);
    }
    .stDataFrame, .stPlotlyChart {
        transition: transform 0.2s ease-in-out;
    }
    .stDataFrame:hover, .stPlotlyChart:hover {
        transform: scale(1.01);
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Analyse Concurrentielle")
st.markdown("""
Analysez facilement la performance des entreprises par secteur, année et d'autres métriques.
Téléchargez votre fichier Excel et utilisez les filtres pour trouver les entreprises les plus ou les moins performantes.
""")

uploaded_file = st.file_uploader("📂 Téléversez le fichier Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.success("✅ Fichier téléversé et données chargées avec succès !")

    st.sidebar.header("🔍 Filtres")
    sector_list = df['Secteur'].dropna().unique().tolist()
    sector = st.sidebar.selectbox("🏣 Sélectionner un secteur", ["Tous"] + sorted(sector_list))

    year = st.sidebar.selectbox("📅 Sélectionner une année", [2020, 2021, 2022, 2023])
    metric_options = {
        "Chiffre d'affaires": f"Chiffre d'affaires {year} (Dhs)",
        "Résultat d'exploitation": f"Resultat d'exploitation {year} (Dhs)",
        "Stock": f"Stock {year}",
        "Charges personnel": f"Charges personnel {year}",
        "Marge EBIT/CA": f"Marge EBIT/CA {year}",
        "Marge EBIT/CP": f"Marge EBIT/CP {year}",
        "Marge CP/CA": f"Marge CP/CA {year}"
    }
    metric = st.sidebar.selectbox("📈 Sélectionner une métrique", list(metric_options.keys()))
    metric_col = metric_options[metric]
    top_n = st.sidebar.slider("🏆 Nombre d'entreprises à afficher", 5, 50, 10)
    ascending = st.sidebar.checkbox("🔽 Afficher les moins performantes", value=False)

    st.sidebar.markdown("---")

    st.sidebar.subheader("📈 Analyse sectorielle")

    sector_for_evolution = st.sidebar.selectbox("🏭 Sélectionnez un secteur pour analyse d'évolution", ["Aucun"] + sorted(sector_list), index=0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🏢 Analyse d'une entreprise spécifique")
    company_names = df['Raison Sociale (Kerix)'].dropna().unique().tolist()
    selected_company = st.sidebar.selectbox("🔎 Rechercher une entreprise", ["Aucune"] + sorted(company_names))

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Comparaison multi-entreprises")
    selected_companies = st.sidebar.multiselect("Sélectionner plusieurs entreprises", sorted(company_names), default=company_names[:2])
    selected_years = st.sidebar.multiselect("Sélectionner les années", [2020, 2021, 2022, 2023], default=[2020, 2021, 2022, 2023])
    selected_metrics = st.sidebar.multiselect("Sélectionner les critères", list(metric_options.keys()), default=["Chiffre d'affaires"])

    filtered_df = df.copy()
    if sector != "Tous":
        filtered_df = filtered_df[filtered_df['Secteur'] == sector]

    filtered_df = filtered_df[["Raison Sociale (Kerix)", metric_col, "Secteur"]].dropna()
    filtered_df[metric_col] = pd.to_numeric(filtered_df[metric_col], errors='coerce')
    filtered_df = filtered_df.dropna(subset=[metric_col])
    filtered_df = filtered_df.sort_values(by=metric_col, ascending=ascending).head(top_n)

    filtered_df[f"{metric_col}_formatted"] = filtered_df[metric_col].apply(lambda x: f"{int(x):,}".replace(",", "."))

    st.subheader(f"{'🔻 Moins performantes' if ascending else '📁 Top'} {top_n} entreprises par {metric} en {year}")
    display_df = filtered_df[["Raison Sociale (Kerix)", f"{metric_col}_formatted", "Secteur"]]
    display_df.columns = ["Raison Sociale", metric, "Secteur"]
    st.dataframe(display_df, use_container_width=True)

    fig = px.bar(
        filtered_df,
        x=metric_col,
        y="Raison Sociale (Kerix)",
        orientation='h',
        color="Secteur",
        title=f"{'Moins performantes' if ascending else 'Top'} {top_n} entreprises par {metric} ({year})",
        labels={metric_col: metric, "Raison Sociale (Kerix)": "Entreprise"},
        height=600
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>" + f"{metric} : %{{customdata[0]}}<extra></extra>",
        customdata=filtered_df[[f"{metric_col}_formatted"]]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.sidebar.markdown("---")
    if sector_for_evolution != "Aucun":
        st.markdown("---")
        st.subheader(f"📊 Évolution moyenne : {sector_for_evolution}")
        years_all = [2020, 2021, 2022, 2023]
        base_metrics = [
            "Chiffre d'affaires",
        ]

        variation_metrics = [
            ("Variation CA 2020/2021", "Variation CA 2020/2021"),
            ("Variation CA 2021/2022", "Variation CA 2021/2022"),
            ("Variation CA 2022/2023", "Variation CA 2022/2023"),
        ]
        all_metrics_to_plot = []
        for m in base_metrics:
            for y in years_all:
                all_metrics_to_plot.append( (f"{m} {y}", f"{m} {y} (Dhs)") )

        all_metrics_to_plot.extend(variation_metrics)

        sector_df = df[df["Secteur"] == sector_for_evolution]
        avg_metrics = {}
        for label, col in all_metrics_to_plot:
            if col in sector_df.columns:
                values = pd.to_numeric(sector_df[col], errors='coerce')
                avg_metrics[label] = values.mean()
            else:
                avg_metrics[label] = None

        evolution_data = {}

        for label in avg_metrics:
            parts = label.rsplit(" ", 1)
            metric_name = parts[0]
            year = parts[1]

            if "Variation" in metric_name and "/" in year:
                if metric_name not in evolution_data:
                    evolution_data[metric_name] = {}
                evolution_data[metric_name][year] = avg_metrics[label]
            else:
                if metric_name not in evolution_data:
                    evolution_data[metric_name] = {}
                try:
                    evolution_data[metric_name][int(year)] = avg_metrics[label]
                except ValueError:
                    pass

        for metric_name, year_vals in evolution_data.items():
            years_sorted = sorted(year_vals.keys())
            values_sorted = [year_vals[y] if year_vals[y] is not None else 0 for y in years_sorted]

            fig = px.line(
                x=years_sorted,
                y=values_sorted,
                labels={"x": "Année", "y": metric_name},
                title=f"Évolution moyenne de la variable \"{metric_name}\" pour le secteur {sector_for_evolution}<br>Étude basée sur les {len(sector_df)} boites les plus grandes du Maroc en terme de CA - Données disponibles",
                markers=True
            )
            fig.update_layout(
                xaxis=dict(tickmode="array", tickvals=years_sorted, tickformat='d'),
                yaxis_title=metric_name,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    if selected_companies and selected_metrics and selected_years:
        st.markdown("---")
        st.subheader("🔍 Comparaison de plusieurs entreprises")

        for metric_name in selected_metrics:
            metric_code = metric_options[metric_name].split()[0]
            metric_label = metric_name
            fig = go.Figure()

            for company in selected_companies:
                company_df = df[df["Raison Sociale (Kerix)"] == company]
                values = []
                for y in selected_years:
                    possible_cols = [col for col in df.columns if metric_code in col and str(y) in col]
                    if possible_cols:
                        if not company_df.empty and possible_cols[0] in company_df.columns:
                            val = company_df[possible_cols[0]].values[0]
                        else:
                            val = None
                    else:
                        val = 0
                    values.append(val if pd.notna(val) else 0)

                fig.add_trace(go.Scatter(
                    x=selected_years,
                    y=values,
                    mode='lines+markers',
                    name=company,
                    hovertemplate=f"<b>{company}</b><br>Année: %{{x}}<br>{metric_label}: %{{y:,.0f}}".replace(",", ".") + "<extra></extra>",
                ))

            fig.update_layout(
                title=f"Comparaison - {metric_label}",
                xaxis_title="Année",
                yaxis_title=metric_label,
                xaxis=dict(tickmode='array', tickvals=selected_years),
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    if selected_company != "Aucune":
        st.markdown("---")
        st.subheader(f"📈 Évolution des performances de l'entreprise : {selected_company}")
        years = [2020, 2021, 2022, 2023]

        metrics_to_plot = [
            (f"Chiffre d'affaires {{year}} (Dhs)", "Chiffre d'affaires"),
            ("Resultat d'exploitation {year} (Dhs)", "Résultat d'exploitation"),
            ("Charges personnel {year}", "Charges personnel"),
            ("Marge EBIT/CA {year}", "Marge EBIT/CA")
        ]

        company_data = df[df['Raison Sociale (Kerix)'] == selected_company]
        for template, label in metrics_to_plot:
            values = []
            for y in years:
                col = template.format(year=y)
                val = company_data[col].values[0] if col in company_data.columns else None
                values.append(val if pd.notna(val) else 0)

            line_fig = px.line(
                x=years,
                y=values,
                labels={'x': 'Année', 'y': label},
                markers=True,
                title=f"{label} de {selected_company} (2020 - 2023)"
            )
            line_fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=years,
                    tickformat='d'
                )
            )
            st.plotly_chart(line_fig, use_container_width=True)

    with st.expander("🔎 Afficher les données brutes"):
        st.dataframe(df, use_container_width=True)
else:
    st.info("📁 Veuillez téléverser un fichier Excel pour commencer l'analyse.")
