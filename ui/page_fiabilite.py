"""Page « Fiabilité » : erreurs mesurées sur le passé, modèles retenus, calibration des intervalles."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prevision import donnees as D
from prevision import moteur as M
from ui import commun as U


def page() -> None:
    U.entete("Groupe COSUMAR · transparence", "Fiabilité des prévisions",
             "Erreurs mesurées en rejouant le passé : chaque modèle prévoit des mois qu'il n'a jamais vus")
    res = U.resultats()
    options = [D.GROUPE, *res.societes()]
    societe = st.selectbox("Société", options, key="fiab_soc")
    fiab = res.fiabilite(societe)
    comps = res._comp(societe)

    U.note(f"<b>Méthode.</b> Les {M.PLIS} dernières périodes de {M.HORIZON_TEST} mois sont rejouées : pour chacune, "
           "tous les modèles sont réentraînés sur les données antérieures puis prévoient les 6 mois suivants, comparés "
           "ensuite aux ventes réelles. L'<b>erreur attendue</b> d'une composante est mesurée par <i>sélection "
           "imbriquée</i> : le modèle est choisi sur les périodes précédentes puis jugé sur la suivante, ce qui évite "
           "de surestimer la précision. WAPE = erreur absolue totale / volume total.")

    if fiab:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(U.kpi("Erreur par mois", f"± {U.nb(fiab['mensuel']['WAPE'], 1)} %",
                          f"soit ± {U.tonnes(fiab['mensuel']['MAE'])} par mois · {fiab['mois']} mois testés"),
                    unsafe_allow_html=True)
        if "jour" in fiab:
            k2.markdown(U.kpi("Erreur par semaine", f"± {U.nb(fiab['semaine'], 1)} %", "total de la semaine"),
                        unsafe_allow_html=True)
            k3.markdown(U.kpi("Erreur par jour", f"± {U.nb(fiab['jour'], 1)} %",
                              f"contre ± {U.nb(fiab['jour_uniforme'], 1)} % en répartissant le mois uniformément"),
                        unsafe_allow_html=True)
            k4.markdown(U.kpi("Jours dans l'intervalle 95 %", f"{U.nb(fiab['couverture_jour'] * 100)} %",
                              "un intervalle bien calibré contient ≈ 95 % des valeurs réelles"), unsafe_allow_html=True)

        s = fiab["serie"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=s["ds"], y=s["reel"], name="Réalisé", marker=dict(color=U.BLEU, cornerradius=3),
                             hovertemplate="%{x|%m/%Y} réalisé : %{y:,.0f} t<extra></extra>"))
        fig.add_trace(go.Scatter(x=s["ds"], y=s["prevu"], name="Prévu sans connaître la période", mode="lines+markers",
                                 line=dict(color=U.AMBRE, width=2.5), marker=dict(size=8, color=U.AMBRE),
                                 hovertemplate="prévu : %{y:,.0f} t<extra></extra>"))
        for i in range(1, len(s) // M.HORIZON_TEST):
            x = s["ds"].iloc[i * M.HORIZON_TEST] - pd.Timedelta(days=15)
            fig.add_vline(x=x, line=dict(color=U.MUET, dash="dot", width=1))
        U.mise_en_forme(fig, 360)
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(tickformat="%m/%Y", hoverformat="%m/%Y")
        st.plotly_chart(fig, use_container_width=True, config=U.CFG)
        st.caption("Chaque bloc de 6 mois (séparé par un pointillé) a été prévu d'un seul coup, à partir des seules "
                   "données antérieures et avec le modèle qui était le meilleur AVANT ce bloc. La période inclut 2020 "
                   "(COVID-19).")

    st.subheader("Par composante")
    lignes = []
    for c in comps:
        e, ej = c.erreur, c.erreur_jour
        lignes.append({
            "Composante": f"{c.societe} · {'marché local' if c.segment == 'Local' else 'export'}",
            "Modèle retenu": c.modele,
            "Erreur mois (%)": e.get("WAPE"), "Erreur semaine (%)": ej.get("semaine"),
            "Erreur jour (%)": ej.get("jour"),
            "Mois dans IC 95 % (%)": 100 * ej["couverture_mois"] if "couverture_mois" in ej else None,
            "Jours dans IC 95 % (%)": 100 * ej["couverture_jour"] if "couverture_jour" in ej else None,
        })
    t = pd.DataFrame(lignes)
    st.dataframe(t, hide_index=True, use_container_width=True, column_config={
        **{k: st.column_config.NumberColumn(format="%.1f") for k in ["Erreur mois (%)", "Erreur semaine (%)", "Erreur jour (%)"]},
        **{k: st.column_config.NumberColumn(format="%.0f") for k in ["Mois dans IC 95 % (%)", "Jours dans IC 95 % (%)"]},
    })
    st.caption("Les exportations sont facturées par cargaisons : leur erreur journalière est structurellement élevée. "
               "Les petites sucreries (SUNABEL, SURAC) ont des ventes irrégulières ; leurs prévisions sont à "
               "considérer comme des ordres de grandeur.")

    for c in comps:
        with st.expander(f"Comparaison des modèles — {c.societe} · {'marché local' if c.segment == 'Local' else 'export'}"):
            tb = pd.DataFrame(c.tableau)
            if tb.empty:
                st.info(c.erreur.get("methode", ""))
                continue
            b = tb.sort_values("WAPE", ascending=False)
            coul = [U.BLEU if m == c.modele else (U.GRIS if r else U.BLEU_CLAIR) for m, r in zip(b["modele"], b["reference"])]
            f = go.Figure(go.Bar(y=b["modele"], x=b["WAPE"], orientation="h", marker=dict(color=coul, cornerradius=3),
                                 text=[f"{U.nb(v, 1)} %" for v in b["WAPE"]], textposition="outside", cliponaxis=False,
                                 customdata=b["description"], hovertemplate="%{y}<br>%{customdata}<extra></extra>"))
            U.mise_en_forme(f, 34 * len(b) + 60, "", legende=False)
            f.update_xaxes(showgrid=True, gridcolor=U.GRILLE, range=[0, b["WAPE"].max() * 1.2],
                           title=dict(text="WAPE (%) — plus bas = meilleur", font=dict(color=U.MUET)))
            f.update_yaxes(showgrid=False, tickfont=dict(color=U.ENCRE), title=None)
            st.plotly_chart(f, use_container_width=True, config=U.CFG)
            st.caption("Gris : références simples · bleu foncé : modèle retenu. Survolez une barre pour sa description.")
