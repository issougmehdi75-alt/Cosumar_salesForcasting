"""Page « Historique » : ventes passées, saisonnalité, jours de la semaine, produits."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prevision import donnees as D
from ui import commun as U


def _filtrer(df, societe, segment, produit):
    d = df if societe == D.GROUPE else df[df["societe"] == societe]
    if segment:
        d = d[d["segment"] == segment]
    if produit != D.TOUS:
        d = d[d["produit"] == produit]
    return d


def page() -> None:
    U.entete("Groupe COSUMAR · analyse", "Historique des ventes",
             "Tonnage net réalisé, saisonnalité, jours de la semaine et répartition par produit")
    res, df = U.resultats(), U.donnees()
    societe, segment, produit = U.selecteurs(res, "hist")
    d = _filtrer(df, societe, segment, produit)
    if d.empty:
        st.info("Aucune vente pour ce périmètre.")
        return

    d0, d1 = d["date"].min(), d["date"].max()
    c1, c2 = st.columns([2, 1])
    with c1:
        per = st.date_input("Période affichée", value=(d0.date(), d1.date()), min_value=d0.date(),
                            max_value=d1.date(), format="DD/MM/YYYY", key="hist_per")
    with c2:
        gran = st.segmented_control("Granularité", ["Jour", "Semaine", "Mois"], default="Mois", key="hist_gran")
    if not isinstance(per, (tuple, list)) or len(per) < 2:
        st.info("Sélectionnez la date de fin.")
        return
    a, b = pd.Timestamp(per[0]), pd.Timestamp(per[1])
    freq = {"Jour": "D", "Semaine": "W-SUN", "Mois": "MS"}[gran or "Mois"]
    sel = d[(d["date"] >= a) & (d["date"] <= b)]
    serie = sel.groupby("date")["tonnage_net"].sum().reindex(pd.date_range(a, b), fill_value=0).resample(freq).sum()
    if freq == "W-SUN":
        serie.index = serie.index - pd.Timedelta(days=6)

    tot, n_j = serie.sum(), (b - a).days + 1
    k1, k2, k3 = st.columns(3)
    k1.markdown(U.kpi("Tonnage net sur la période", U.tonnes(tot), f"{U.date_fr(a)} → {U.date_fr(b)}"),
                unsafe_allow_html=True)
    k2.markdown(U.kpi("Moyenne par mois", U.tonnes(tot / max(1, n_j / 30.44))), unsafe_allow_html=True)
    k3.markdown(U.kpi("Jours avec ventes", f"{U.nb((sel.groupby('date')['tonnage_net'].sum() > 0).sum())}",
                      f"sur {n_j} jours"), unsafe_allow_html=True)

    fig = go.Figure(go.Bar(x=serie.index, y=serie.to_numpy(), marker=dict(color=U.BLEU, cornerradius=2),
                           hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:,.0f} t</b><extra></extra>", name="Réalisé"))
    U.mise_en_forme(fig, 360, legende=False)
    fig.update_xaxes(hoverformat="%d/%m/%Y", tickformat="%m/%Y" if freq != "D" else "%d/%m/%y")
    hm = res.historique_mensuel(societe, segment) if produit == D.TOUS else None
    if hm is not None and freq == "MS" and hm["manquant"].any():
        m = hm[hm["manquant"] & hm["ds"].between(a, b)]
        if len(m):
            fig.add_trace(go.Scatter(x=m["ds"], y=[0] * len(m), mode="markers", name="Mois incomplet dans l'extraction",
                                     marker=dict(symbol="x", size=10, color="#d03b3b"),
                                     hovertemplate="%{x|%m/%Y} : extraction incomplète<extra></extra>"))
            fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True, config=U.CFG)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Saisonnalité** — moyenne par mois de l'année (indice 100 = mois moyen)")
        m = d.groupby(d["date"].dt.to_period("M").dt.to_timestamp())["tonnage_net"].sum()
        if hm is not None:
            m = m[~m.index.isin(hm.loc[hm["manquant"], "ds"])]
        prof = m.groupby(m.index.month).mean()
        prof = prof / prof.mean() * 100
        f = go.Figure(go.Bar(x=[U.MOIS_C[i - 1] for i in prof.index], y=prof.to_numpy(),
                             marker=dict(color=[U.BLEU if v >= 100 else U.BLEU_CLAIR for v in prof], cornerradius=3),
                             hovertemplate="%{x} : indice %{y:.0f}<extra></extra>"))
        f.add_hline(y=100, line=dict(color=U.MUET, dash="dash", width=1))
        U.mise_en_forme(f, 280, "Indice", legende=False)
        f.update_yaxes(tickformat=".0f", rangemode="normal")
        st.plotly_chart(f, use_container_width=True, config=U.CFG)
        st.caption("Le Ramadan avance d'environ 11 jours par an : son effet se déplace d'un mois à l'autre "
                   "et se lit mal dans une moyenne par mois calendaire. Les modèles le prennent en compte directement.")
    with c2:
        st.markdown("**Jour de la semaine** — part moyenne d'un jour (indice 100 = jour moyen)")
        q = d.groupby("date")["tonnage_net"].sum().reindex(pd.date_range(d0, d1), fill_value=0)
        q = q / q.groupby(q.index.to_period("M")).transform("mean")
        pj = q.groupby(q.index.dayofweek).mean() * 100
        f = go.Figure(go.Bar(x=[U.JOURS_C[i] for i in pj.index], y=pj.to_numpy(),
                             marker=dict(color=[U.BLEU if v >= 100 else U.BLEU_CLAIR for v in pj], cornerradius=3),
                             hovertemplate="%{x} : indice %{y:.0f}<extra></extra>"))
        f.add_hline(y=100, line=dict(color=U.MUET, dash="dash", width=1))
        U.mise_en_forme(f, 280, "Indice", legende=False)
        f.update_yaxes(tickformat=".0f")
        st.plotly_chart(f, use_container_width=True, config=U.CFG)

    if produit == D.TOUS:
        st.markdown("**Répartition par produit et par année** (tonnes)")
        t = d.pivot_table(index="produit", columns=d["date"].dt.year, values="tonnage_net", aggfunc="sum", fill_value=0)
        t = t.loc[t.sum(axis=1).sort_values(ascending=False).index]
        t.columns = [str(c) for c in t.columns]
        st.dataframe(t.round(0), use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.0f") for c in t.columns})
    st.download_button("📥 Télécharger la série affichée (Excel)",
                       U.excel({"Historique": pd.DataFrame({"Période": serie.index.strftime("%d/%m/%Y"),
                                                            "Tonnage net (t)": serie.round(1).to_numpy()})}),
                       file_name=f"historique_{U.ascii_slug(U.libelle(societe, segment, produit))}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
