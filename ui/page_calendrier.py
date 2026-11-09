"""Page « Calendrier » : jours fériés et Ramadan, modifiables."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from prevision import calendrier as C
from ui import commun as U


def page() -> None:
    U.entete("Groupe COSUMAR · paramètres", "Calendrier des événements",
             "Jours fériés et Ramadan utilisés par les prévisions")
    ev = U.calendrier()
    couv = C.couverture(ev)
    U.note(f"Les prévisions utilisent ce calendrier : <b>jours fériés</b> (ventes quasi nulles), <b>Ramadan</b> et "
           f"<b>mois précédant le Ramadan</b> (achats anticipés). Le Ramadan est renseigné jusqu'au "
           f"<b>{U.date_fr(couv)}</b> : les prévisions ne peuvent pas aller au-delà. Les fêtes religieuses dépendent "
           "de l'observation du croissant lunaire : corrigez les dates dès leur annonce officielle.")

    annee = st.segmented_control("Afficher", ["Toutes", *sorted({str(d.year) for d in ev["debut"]})],
                                 default="Toutes", key="cal_annee")
    vue = ev if annee in (None, "Toutes") else ev[ev["debut"].dt.year == int(annee)]
    edite = st.data_editor(
        vue.assign(debut=vue["debut"].dt.date, fin=vue["fin"].dt.date), num_rows="dynamic", hide_index=True,
        use_container_width=True, key=f"cal_editeur_{annee}_{st.session_state.get('_cal_v', 0)}",
        column_config={
            "debut": st.column_config.DateColumn("Début", format="DD/MM/YYYY", required=True),
            "fin": st.column_config.DateColumn("Fin", format="DD/MM/YYYY", required=True),
            "evenement": st.column_config.TextColumn("Événement", required=True),
            "type": st.column_config.SelectboxColumn("Type", options=["ferie", "ramadan"], required=True),
            "fiabilite": st.column_config.SelectboxColumn("Fiabilité", options=["officielle", "estimée", "± 1 jour"]),
        })
    c1, c2, c3 = st.columns(3)
    if c1.button("✅ Appliquer et recalculer", type="primary", use_container_width=True):
        try:
            nouveau = C.fusionner_edition(ev, edite, None if annee in (None, "Toutes") else int(annee))
            if nouveau.equals(ev.sort_values(["debut", "evenement"]).reset_index(drop=True)):
                st.info("Aucune modification à appliquer.")
            else:
                st.session_state["calendrier"] = nouveau
                st.session_state["_cal_ok"] = True
                st.session_state["_cal_v"] = st.session_state.get("_cal_v", 0) + 1
                st.rerun()
        except Exception as e:
            st.error(f"Calendrier invalide : {e}")
    if st.session_state.pop("_cal_ok", False):
        U.resultats()
        st.success(f"Calendrier appliqué ({len(U.calendrier())} événements) : les prévisions ont été recalculées.")
    c2.download_button("📥 Télécharger le calendrier (CSV)",
                       U.calendrier().assign(debut=lambda x: x["debut"].dt.strftime("%Y-%m-%d"),
                                             fin=lambda x: x["fin"].dt.strftime("%Y-%m-%d")).to_csv(index=False).encode("utf-8"),
                       file_name="calendrier_evenements.csv", mime="text/csv", use_container_width=True)
    if "calendrier" in st.session_state and c3.button("↩️ Calendrier du projet", use_container_width=True):
        del st.session_state["calendrier"]
        st.session_state["_cal_v"] = st.session_state.get("_cal_v", 0) + 1
        st.rerun()
    st.caption("Pour rendre une modification permanente : téléchargez le CSV et remplacez "
               "`data/calendrier_evenements.csv` dans le dépôt.")
