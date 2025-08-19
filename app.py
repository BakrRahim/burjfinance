import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="BurjFinance - Tableau de Bord",
    page_icon="💼",
    layout="centered"
)

st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            font-size: 3rem;
            font-weight: 700;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
        }
        .nav-button {
            display: block;
            width: 100%;
            padding: 1.2rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 1rem;
            background: linear-gradient(90deg, #283E51 0%, #485563 100%);
            color: white;
            text-decoration: none;
            transition: transform 0.2s ease-in-out;
        }
        .nav-button:hover {
            transform: scale(1.03);
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>📊 Tableau de Bord BurjFinance</div>", unsafe_allow_html=True)

pages = [
    {"title": "📈 Analyse Sectorielle", "file": "Analyse_Sectorielle"},
    {"title": "🎯 Analyse Concurrentielle", "file": "Analyse_Concurrentielle"},
    {"title": "🔍 Moteur de Recherche", "file": "Search_Engine"}
]

for page in pages:
    st.markdown(
        f"<a class='nav-button' href='/{page['file']}' target='_self'>{page['title']}</a>",
        unsafe_allow_html=True
    )
    
st.markdown("---")
st.markdown("💡 *Sélectionnez une page pour commencer votre analyse.*")
