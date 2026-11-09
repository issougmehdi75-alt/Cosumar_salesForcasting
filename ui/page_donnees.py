"""Page « Mettre à jour les données » : import d'extractions SAP et recalcul."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from prevision import donnees as D
from ui import commun as U


def page() -> None:
    U.entete("Groupe COSUMAR · données", "Mettre à jour les données",
             "Importez les dernières extractions de ventes pour prévoir la période actuelle")
    df = U.donnees()
    societes = D.societes(df)
    k1, k2, k3 = st.columns(3)
    k1.markdown(U.kpi("Données utilisées", "importées (session)" if "donnees" in st.session_state else "du projet",
                      f"{U.nb(len(df))} lignes agrégées"), unsafe_allow_html=True)
    k2.markdown(U.kpi("Période couverte", f"{U.date_fr(df['date'].min())} → {U.date_fr(df['date'].max())}"),
                unsafe_allow_html=True)
    k3.markdown(U.kpi("Sociétés", str(len(societes)), ", ".join(societes)), unsafe_allow_html=True)

    st.subheader("1 · Importer une extraction")
    with st.expander("Format attendu", expanded=False):
        st.markdown(f"""
Fichier(s) **CSV** exporté(s) de SAP, une ligne par ligne de facturation, avec au minimum les colonnes :
`{'`, `'.join(D.COLONNES_SAP)}` (séparateur `,` ou `;`).

- `Jour calendaire` au format `M/J/AA` (ex. `1/31/21`) ; les formats `JJ/MM/AAAA` et `AAAA-MM-JJ` sont aussi acceptés.
- `Poids net` en tonnes (négatif pour un retour ou un avoir).
- Les autres colonnes (dont les numéros de clients) sont **ignorées dès la lecture** : seuls des totaux par jour,
  société, marché et produit sont conservés.
""")
    fichiers = st.file_uploader("Extraction(s) SAP (.csv)", type=["csv"], accept_multiple_files=True, key="maj_fichiers")
    mode = st.radio("Mode", ["Ajouter les jours postérieurs aux données actuelles", "Remplacer toutes les données"],
                    key="maj_mode", help="« Ajouter » ignore les jours déjà présents pour chaque société : recharger "
                                         "une période déjà importée ne crée pas de double comptage.")
    if not fichiers:
        st.info("Déposez un ou plusieurs fichiers pour continuer.")
    else:
        try:
            nouveau, rapport = D.agreger_extraction([f for f in fichiers])
        except Exception as e:
            st.error(f"Lecture impossible : {e}")
            return
        mode_c = "remplacer" if mode.startswith("Remplacer") else "ajouter"
        fusion, info = D.fusionner(df, nouveau, mode_c)
        st.subheader("2 · Contrôle")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(U.kpi("Lignes lues", U.nb(rapport["lignes_lues"]),
                          f"{U.nb(rapport['lignes_invalides'])} illisibles (date ou poids)"), unsafe_allow_html=True)
        c2.markdown(U.kpi("Période du fichier", f"{U.date_fr(rapport['debut'])} → {U.date_fr(rapport['fin'])}"),
                    unsafe_allow_html=True)
        c3.markdown(U.kpi("Tonnage net du fichier", U.tonnes(rapport["tonnage"]), ", ".join(rapport["societes"])),
                    unsafe_allow_html=True)
        c4.markdown(U.kpi("Après fusion", f"{U.date_fr(fusion['date'].min())} → {U.date_fr(fusion['date'].max())}",
                          f"{U.nb(info['ajoutees'])} lignes agrégées ajoutées · {U.nb(info['ignorees'])} ignorées "
                          "(jours déjà présents)"), unsafe_allow_html=True)
        inconnues = sorted(set(rapport["societes"]) - set(societes))
        if inconnues and mode_c == "ajouter":
            st.warning(f"Nouvelle(s) société(s) : {', '.join(inconnues)}. Elles seront modélisées si leur historique "
                       "est suffisant (au moins 30 mois pour l'évaluation).")
        if info["ajoutees"] == 0:
            st.warning("Aucun jour nouveau : toutes les dates du fichier sont déjà couvertes. Utilisez « Remplacer » "
                       "pour corriger des données existantes.")
        recap = (nouveau.assign(mois=nouveau["date"].dt.to_period("M").astype(str))
                 .pivot_table(index="mois", columns="societe", values="tonnage_net", aggfunc="sum", fill_value=0).round(0))
        with st.expander("Tonnage par mois et par société dans le fichier"):
            st.dataframe(recap, use_container_width=True)
        if st.button("✅ Appliquer et recalculer les prévisions", type="primary", disabled=info["ajoutees"] == 0):
            st.session_state["donnees"] = fusion
            U.resultats()
            st.success("Données mises à jour : toutes les pages utilisent maintenant ces données.")

    st.subheader("Conserver les données")
    st.markdown("Les données importées ne sont conservées que **pendant cette session**. Pour les rendre "
                "permanentes, téléchargez le fichier agrégé (anonyme) et remplacez `data/ventes_journalieres.csv` "
                "dans le dépôt GitHub, puis lancez `python scripts/calculer_previsions.py` pour un démarrage instantané.")
    c1, c2 = st.columns(2)
    c1.download_button("📥 Télécharger ventes_journalieres.csv", D.exporter(U.donnees()),
                       file_name="ventes_journalieres.csv", mime="text/csv", use_container_width=True)
    if "donnees" in st.session_state and c2.button("↩️ Revenir aux données du projet", use_container_width=True):
        del st.session_state["donnees"]
        st.rerun()
