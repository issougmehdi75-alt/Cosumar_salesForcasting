"""
app.py — Application Streamlit operationnelle pour le groupe COSUMAR.
Permet a un utilisateur non technique de choisir une societe (ou le Groupe consolide),
un horizon de prevision (en mois, via un curseur) et d'obtenir la prevision du nombre
d'articles vendus, avec visualisation, tableau de donnees et export CSV.

Lancement : streamlit run app.py
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Configuration generale de la page
# ------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "models"
ASSETS_DIR = APP_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "logo.jpg"

NAVY = "#00428C"
NAVY_DARK = "#00285A"
GOLD = "#F5BC2D"
GOLD_DARK = "#D89B00"
BG = "#F5F7FA"
TEXT = "#0A2540"

st.set_page_config(
    page_title="COSUMAR — Prévision des ventes",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Styles (identite visuelle COSUMAR : bleu marine + or)
# ------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

.stApp {{
    background-color: {BG};
}}

/* ---- Forcer la visibilite du texte quel que soit le theme systeme
   (corrige le texte blanc-sur-blanc en mode sombre) ---- */
.stApp, .stApp p, .stApp span, .stApp label, .stApp li,
.stApp .stMarkdown, .stApp .stCaption, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
    color: {TEXT} !important;
}}
[data-testid="stDataFrame"] * {{
    color: {TEXT} !important;
}}
/* Les menus deroulants (selectbox) sont rendus dans un calque a part :
   il faut les cibler explicitement pour eviter le texte invisible */
[data-baseweb="popover"], [data-baseweb="menu"] {{
    background-color: #ffffff !important;
}}
[data-baseweb="popover"] *, [data-baseweb="menu"] * {{
    color: {TEXT} !important;
    -webkit-text-fill-color: {TEXT} !important;
    opacity: 1 !important;
}}
/* Le champ ferme du selectbox (valeur choisie, ex: "SUTA") a un fond
   clair : le texte doit rester NOIR ici. On booste artificiellement la
   specificite avec ":not(#____jamais____)" (astuce CSS classique) pour
   etre certain de gagner face au CSS interne de Streamlit, quel que soit
   son ordre d'injection — les tentatives precedentes perdaient ce
   conflit de specificite/ordre. */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background-color: #ffffff !important;
    border-radius: 8px !important;
}}
section[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] *:not(#____jamais____) {{
    color: {TEXT} !important;
    -webkit-text-fill-color: {TEXT} !important;
    opacity: 1 !important;
}}
/* Titre du selectbox ("Société a analyser") : reste blanc, plus grand */
section[data-testid="stSidebar"] [data-testid="stSelectbox"] label:not(#____jamais____),
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p:not(#____jamais____) {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
}}
/* Contenu des expander (ex: "Comment utiliser cet outil ?") dans la sidebar :
   fond fonce translucide + texte blanc, pour ne pas avoir de blanc sur blanc */
section[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background-color: rgba(255,255,255,0.08) !important;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.15);
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] * {{
    color: #ffffff !important;
}}
section[data-testid="stSidebar"] [data-testid="stExpander"] code {{
    background-color: rgba(255,255,255,0.15) !important;
    color: {GOLD} !important;
}}

/* Bandeau d'en-tete */
.cosumar-header {{
background: linear-gradient(135deg, {NAVY} 0%, {NAVY_DARK} 100%);
padding: 2rem 2.5rem;
border-radius: 0 0 18px 18px;
margin: -1rem -1rem 2rem -1rem;
display: flex;
align-items: center;
gap: 1.5rem;
box-shadow: 0 4px 18px rgba(0,40,90,0.25);
}}
.cosumar-header img {{
height: 68px;
border-radius: 6px;
}}
.cosumar-header .title-block h1 {{
font-family: 'Playfair Display', serif;
font-weight: 800;
color: white !important;
font-size: 1.9rem;
margin: 0;
letter-spacing: 0.3px;
}}
.cosumar-header .title-block p {{
color: {GOLD} !important;
font-family: 'Inter', sans-serif;
font-weight: 500;
margin: 0.2rem 0 0 0;
font-size: 1rem;
}}

/* Cartes KPI */
.kpi-card {{
background: white;
border-radius: 14px;
padding: 1.1rem 1.3rem;
border-left: 5px solid {GOLD};
box-shadow: 0 2px 10px rgba(0,40,90,0.07);
height: 100%;
}}
.kpi-card .kpi-label {{
font-size: 0.78rem;
text-transform: uppercase;
letter-spacing: 0.6px;
color: #5b6b82 !important;
font-weight: 600;
margin-bottom: 0.35rem;
}}
.kpi-card .kpi-value {{
font-family: 'Playfair Display', serif;
font-size: 1.7rem;
font-weight: 700;
color: {NAVY} !important;
}}
.kpi-card .kpi-sub {{
font-size: 0.78rem;
color: #8896a8 !important;
margin-top: 0.2rem;
}}

/* Section titres */
.section-title {{
font-family: 'Playfair Display', serif;
font-weight: 700;
color: {NAVY} !important;
font-size: 1.35rem;
margin: 1.6rem 0 0.6rem 0;
border-bottom: 3px solid {GOLD};
display: inline-block;
padding-bottom: 0.2rem;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
background: linear-gradient(180deg, {NAVY_DARK} 0%, {NAVY} 100%);
}}
section[data-testid="stSidebar"] * {{
color: white !important;
}}
section[data-testid="stSidebar"] .stButton button {{
background: {GOLD};
color: {NAVY_DARK} !important;
font-weight: 700;
border: none;
border-radius: 8px;
padding: 0.6rem 1rem;
width: 100%;
transition: transform 0.1s ease;
}}
section[data-testid="stSidebar"] .stButton button:hover {{
transform: translateY(-1px);
background: {GOLD_DARK};
}}
/* Curseur (slider) : le rendre bien visible sur fond navy */
section[data-testid="stSidebar"] [data-baseweb="slider"] div[role="slider"] {{
background-color: {GOLD} !important;
border-color: {GOLD} !important;
}}
section[data-testid="stSidebar"] [data-testid="stTickBar"] {{
color: white !important;
}}

/* Badge modele */
.model-badge {{
display: inline-block;
background: {GOLD};
color: {NAVY_DARK} !important;
font-weight: 700;
padding: 0.25rem 0.75rem;
border-radius: 20px;
font-size: 0.85rem;
}}

.footer-note {{
color: #8896a8 !important;
font-size: 0.8rem;
margin-top: 2rem;
text-align: center;
}}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Chargement des modeles (mise en cache pour un chargement instantane)
# ------------------------------------------------------------------
@st.cache_resource
def load_manifest():
    with open(MODELS_DIR / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_bundle(filename):
    with open(MODELS_DIR / filename, "rb") as f:
        return pickle.load(f)


MODEL_LABELS_FR = {
    "SARIMA": "SARIMA (modèle statistique saisonnier)",
    "Holt_Winters": "Holt-Winters (lissage exponentiel)",
    "Prophet": "Prophet (Meta/Facebook)",
    "Random_Forest": "Random Forest",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "Naive": "Modèle naïf",
}


import math


def months_from_horizon(unit, value):
    """Convertit un horizon exprime en jours/mois/annees en nombre de mois (granularite du modele)."""
    if unit == "Jours":
        return max(1, math.ceil(value / 30.44))
    if unit == "Années":
        return max(1, int(value) * 12)
    return max(1, int(value))


def forecast_bundle(bundle, horizon_months):
    """Genere une prevision a horizon_months mois avec le modele serialise dans bundle."""
    ts = bundle["ts"]
    freq = bundle["freq"]
    model_type = bundle["model_type"]
    future_index = pd.date_range(ts.index[-1], periods=horizon_months + 1, freq=freq)[1:]

    if model_type == "SARIMA":
        mf = bundle["statsmodels_result"]
        fc = mf.get_forecast(steps=horizon_months)
        mean = pd.Series(fc.predicted_mean.values, index=future_index)
        ci = fc.conf_int(alpha=0.05)
        lower = pd.Series(ci.iloc[:, 0].values, index=future_index)
        upper = pd.Series(ci.iloc[:, 1].values, index=future_index)
        return mean, lower, upper

    if model_type == "Holt_Winters":
        mf = bundle["statsmodels_result"]
        mean = pd.Series(mf.forecast(horizon_months).values, index=future_index)
        rmse = bundle["results_df"].iloc[0]["RMSE"]
        return mean, mean - 1.96 * rmse, mean + 1.96 * rmse

    if model_type == "Prophet":
        from prophet.serialize import model_from_json
        mf = model_from_json(bundle["prophet_json"])
        fut = mf.make_future_dataframe(periods=horizon_months, freq=freq)
        pred = mf.predict(fut).iloc[-horizon_months:]
        mean = pd.Series(pred["yhat"].values, index=future_index)
        lower = pd.Series(pred["yhat_lower"].values, index=future_index)
        upper = pd.Series(pred["yhat_upper"].values, index=future_index)
        return mean, lower, upper

    if model_type in ("Random_Forest", "XGBoost", "LightGBM"):
        mf = bundle["ml_model"]
        feature_cols = bundle["feature_cols"]
        n_lags = bundle["n_lags"]

        def build_features(series, n_lags):
            d = pd.DataFrame({"y": series})
            for lag in range(1, n_lags + 1):
                d[f"lag_{lag}"] = d["y"].shift(lag)
            d["rolling_mean_3"] = d["y"].shift(1).rolling(3).mean()
            d["rolling_mean_6"] = d["y"].shift(1).rolling(6).mean()
            d["rolling_std_3"] = d["y"].shift(1).rolling(3).std()
            d["month"] = d.index.month
            d["quarter"] = d.index.quarter
            d["year"] = d.index.year
            d["time_index"] = np.arange(len(d))
            return d

        history = ts.copy()
        preds = []
        for step in range(horizon_months):
            feat = build_features(history, n_lags)
            row = feat.iloc[[-1]][feature_cols].copy()
            nd = future_index[step]
            row["month"], row["quarter"], row["year"] = nd.month, nd.quarter, nd.year
            row["time_index"] = feat["time_index"].iloc[-1] + 1
            pv = max(0, mf.predict(row)[0])
            preds.append(pv)
            history.loc[nd] = pv
        mean = pd.Series(preds, index=future_index)
        rmse = bundle["results_df"].iloc[0]["RMSE"]
        return mean, (mean - 1.96 * rmse).clip(lower=0), mean + 1.96 * rmse

    # Fallback naif
    mean = pd.Series([bundle.get("last_value", ts.iloc[-1])] * horizon_months, index=future_index)
    return mean, mean, mean


# ------------------------------------------------------------------
# En-tete
# ------------------------------------------------------------------
import base64

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = img_to_base64(LOGO_PATH) if LOGO_PATH.exists() else None
logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}"/>' if logo_b64 else ""

# NB : le HTML ci-dessous commence en colonne 0 (pas d'indentation) car
# Streamlit/Markdown interprete un bloc indente de 4 espaces comme du code
# et l'affiche tel quel au lieu de le rendre — c'est ce qui causait le bug
# du <div>...</div> affiche en texte brut dans l'application.
st.markdown(f"""
<div class="cosumar-header">
{logo_html}
<div class="title-block">
<h1>Prévision des ventes — Groupe COSUMAR</h1>
<p>Outil d'aide à la décision · Prévision du nombre d'articles vendus</p>
</div>
</div>
""", unsafe_allow_html=True)

manifest = load_manifest()
societe_options = list(manifest.keys())

# ------------------------------------------------------------------
# Barre laterale — parametres
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Paramètres de la prévision")
    st.markdown("")

    societe = st.selectbox(
        "Société à analyser",
        societe_options,
        index=societe_options.index("GROUPE (toutes societes)") if "GROUPE (toutes societes)" in societe_options else 0,
        help="Choisissez une entité du groupe, ou 'GROUPE' pour la vision consolidée des 4 sociétés.",
    )

    st.markdown("")
    st.markdown("**Horizon de prévision**")
    horizon_unit = st.radio(
        "Unité",
        ["Jours", "Mois", "Années"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
    )

    if horizon_unit == "Jours":
        horizon_value = st.slider("Valeur (jours)", min_value=7, max_value=1095, value=180, step=1, label_visibility="collapsed")
    elif horizon_unit == "Années":
        horizon_value = st.slider("Valeur (années)", min_value=1, max_value=5, value=1, step=1, label_visibility="collapsed")
    else:
        horizon_value = st.slider("Valeur (mois)", min_value=1, max_value=36, value=6, step=1, label_visibility="collapsed")

    horizon_months = months_from_horizon(horizon_unit, horizon_value)
    years_eq = horizon_months / 12
    st.caption(
        f"📅 **{horizon_value} {horizon_unit.lower()}** "
        f"→ ≈ **{horizon_months} mois** de prévision (≈ {years_eq:.1f} an{'s' if years_eq >= 2 else ''})"
    )

    st.markdown("")
    generate = st.button("🔮 Générer la prévision", use_container_width=True)

    st.markdown("---")
    with st.expander("📖 Comment utiliser cet outil ?"):
        st.markdown("""
1. **Choisissez une société** dans la liste (ou "GROUPE" pour la vue consolidée).
2. **Choisissez l'unité** (jours, mois ou années) puis **faites glisser le curseur** pour fixer l'horizon souhaité.
3. Cliquez sur **Générer la prévision**.
4. Consultez le graphique, les indicateurs clés, et **téléchargez** les chiffres en CSV
   pour les intégrer à vos propres tableaux de bord.

Le modèle utilisé est **sélectionné automatiquement** parmi plusieurs approches
(SARIMA, Holt-Winters, Prophet, Random Forest, XGBoost, LightGBM) : celui affichant
la meilleure précision historique est retenu pour chaque société.

*Les modèles sont ré-entraînés périodiquement (voir `train_models.py`) à mesure que
de nouvelles données de vente sont disponibles.*
        """)

# ------------------------------------------------------------------
# Corps principal
# ------------------------------------------------------------------
info = manifest[societe]
bundle = load_bundle(info["file"])
ts = bundle["ts"]

st.markdown('<div class="section-title">Vue d\'ensemble</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
<div class="kpi-card">
<div class="kpi-label">Société</div>
<div class="kpi-value" style="font-size:1.3rem">{societe.replace(" (toutes societes)","")}</div>
<div class="kpi-sub">Dernières données : {info['last_date']}</div>
</div>
""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
<div class="kpi-card">
<div class="kpi-label">Modèle retenu</div>
<div class="kpi-value" style="font-size:1.2rem">{MODEL_LABELS_FR.get(info['model_type'], info['model_type'])}</div>
<div class="kpi-sub">Sélectionné automatiquement (meilleur RMSE)</div>
</div>
""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
<div class="kpi-card">
<div class="kpi-label">Précision estimée (MAPE)</div>
<div class="kpi-value">{info['mape']:.1f}%</div>
<div class="kpi-sub">Erreur moyenne sur données historiques de test</div>
</div>
""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
<div class="kpi-card">
<div class="kpi-label">Dernier volume observé</div>
<div class="kpi-value">{int(ts.iloc[-1]):,}</div>
<div class="kpi-sub">Articles — {ts.index[-1].strftime('%B %Y')}</div>
</div>
""".replace(",", " "), unsafe_allow_html=True)

st.write("")

if generate or True:  # la prevision par defaut s'affiche aussi sans clic (meilleure UX)
    mean, lower, upper = forecast_bundle(bundle, horizon_months)

    st.markdown('<div class="section-title">Prévision</div>', unsafe_allow_html=True)

    label_horizon = f"{horizon_value} {horizon_unit.lower()}" + (f" (≈ {horizon_months} mois)" if horizon_unit != "Mois" else "")
    st.markdown(f"Prévision demandée : **{label_horizon}** — à partir de {ts.index[-1].strftime('%B %Y')}")

    # ---- Graphique ----
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values, mode="lines+markers", name="Historique",
        line=dict(color=NAVY, width=2.5), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=mean.index, y=mean.values, mode="lines+markers", name="Prévision",
        line=dict(color=GOLD_DARK, width=2.5, dash="solid"), marker=dict(size=6, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=list(mean.index) + list(mean.index[::-1]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself", fillcolor="rgba(245,188,45,0.18)",
        line=dict(color="rgba(255,255,255,0)"), name="Intervalle de confiance (95%)",
        showlegend=True,
    ))
    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title="Nombre d'articles",
        hovermode="x unified",
        font=dict(family="Inter, sans-serif", color=TEXT),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Tableau + export ----
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown('<div class="section-title" style="font-size:1.1rem">Détail des valeurs prévues</div>', unsafe_allow_html=True)
        table = pd.DataFrame({
            "Période": mean.index.strftime("%B %Y"),
            "Prévision (nb articles)": mean.round(0).astype(int),
            "Borne basse (95%)": lower.round(0).clip(lower=0).astype(int),
            "Borne haute (95%)": upper.round(0).astype(int),
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<div class="section-title" style="font-size:1.1rem">Export</div>', unsafe_allow_html=True)
        csv = table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Télécharger la prévision (CSV)",
            data=csv,
            file_name=f"prevision_{societe.replace(' ', '_')}_{horizon_months}mois.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown(f"""
<div class="kpi-card" style="margin-top:0.8rem">
<div class="kpi-label">Total prévu sur la période</div>
<div class="kpi-value">{int(mean.sum()):,}</div>
<div class="kpi-sub">articles, cumulés sur {horizon_months} mois</div>
</div>
""".replace(",", " "), unsafe_allow_html=True)

    # ---- Details modele (pour utilisateurs curieux) ----
    with st.expander("🔍 Détail de la comparaison des modèles testés"):
        results_display = bundle["results_df"].copy()
        results_display.columns = ["Modèle", "MAE", "RMSE", "MAPE (%)", "SMAPE (%)"]
        for c in ["MAE", "RMSE", "MAPE (%)", "SMAPE (%)"]:
            results_display[c] = results_display[c].round(2)
        st.dataframe(results_display, use_container_width=True, hide_index=True)
        st.caption(
            "RMSE = erreur quadratique moyenne (plus bas = meilleur). "
            "Le modèle en tête de ce classement est celui utilisé pour la prévision ci-dessus."
        )

st.markdown(
    '<div class="footer-note">Outil interne de prévision — Groupe COSUMAR · '
    'Modèles ré-entraînables via train_models.py</div>',
    unsafe_allow_html=True,
)
