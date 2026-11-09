"""
COSUMAR — Prévision des ventes (tonnage net) par jour, semaine, mois et période.

Lancement : streamlit run app.py
"""

from __future__ import annotations

import warnings

import streamlit as st

warnings.filterwarnings("ignore")
st.set_page_config(page_title="COSUMAR — Prévision des ventes", page_icon="📈", layout="wide")

from ui import commun as U  # noqa: E402
from ui import page_calendrier, page_donnees, page_fiabilite, page_historique, page_prevoir  # noqa: E402

st.markdown(U.CSS, unsafe_allow_html=True)

pages = st.navigation({
    "Prévisions": [
        st.Page(page_prevoir.page, title="Prévoir", icon="🔮", default=True),
        st.Page(page_historique.page, title="Historique", icon="📊", url_path="historique"),
        st.Page(page_fiabilite.page, title="Fiabilité", icon="🎯", url_path="fiabilite"),
    ],
    "Administration": [
        st.Page(page_donnees.page, title="Mettre à jour les données", icon="📥", url_path="donnees"),
        st.Page(page_calendrier.page, title="Calendrier", icon="🗓️", url_path="calendrier"),
    ],
})

with st.sidebar:
    st.markdown("### 📈 COSUMAR\nPrévision des ventes")
    st.caption("Tonnage net = poids net vendu, retours déduits (tonnes).")
    with st.expander("❓ Aide rapide"):
        st.markdown("""
**Prévoir** — choisissez la société, le marché, le produit, puis un jour, une semaine, un mois ou une période.
La barre ambre est la prévision, le trait vertical l'intervalle à 95 % ; le losange gris le réalisé un an plus tôt.

**Fiabilité** — l'erreur habituelle des prévisions, mesurée sur le passé.

**Mettre à jour les données** — importez les nouvelles extractions SAP pour prévoir la période actuelle.

**Calendrier** — jours fériés et Ramadan (à corriger dès l'annonce officielle des fêtes religieuses).
""")
    if U.source_modifiee():
        st.info("Données ou calendrier modifiés dans cette session.")

pages.run()

st.markdown('<div class="pied">Projet de stage — Groupe COSUMAR · prévisions statistiques indicatives, à recouper '
            "avec l'expertise des équipes commerciales et logistiques.</div>", unsafe_allow_html=True)
