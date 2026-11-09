"""Page « Prévoir » : un jour, une semaine, un mois ou une période libre."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from prevision import calendrier as Cal
from prevision import donnees as D
from ui import commun as U


def _repartition_produits(res, societe, segment, debut, fin) -> pd.DataFrame:
    lignes = []
    for p in res.produits(societe):
        t = res.agreger(res.journalier(societe, p, segment), debut, fin, "M")
        if len(t) and t["yhat"].sum() > 0.5:
            lignes.append({"Produit": p, "Prévision (t)": t["yhat"].sum()})
    out = pd.DataFrame(lignes)
    if len(out):
        out["Part"] = out["Prévision (t)"] / out["Prévision (t)"].sum()
        out = out.sort_values("Prévision (t)", ascending=False)
    return out


def _bloc_resultat(res, df, t, freq, societe, segment, produit, debut, fin, titre_export, surligne=None,
                   hauteur=360):
    n1 = U.realise_n1(df, t, freq, societe, produit, segment)
    st.plotly_chart(U.figure_prevision(t, freq, n1, surligne=surligne, hauteur=hauteur),
                    use_container_width=True, config=U.CFG)
    tab = U.tableau_export(t, freq, n1)
    c1, c2 = st.columns([4, 1], vertical_alignment="top")
    with c1:
        with st.expander(f"Tableau détaillé ({len(tab)} lignes)", expanded=len(tab) <= 7):
            st.dataframe(tab, hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.0f") for c in tab.columns[1:]})
        if produit == D.TOUS:
            rep = _repartition_produits(res, societe, segment, debut, fin)
            if len(rep) > 1:
                with st.expander("Répartition par produit"):
                    st.dataframe(rep, hide_index=True, use_container_width=True,
                                 column_config={"Prévision (t)": st.column_config.NumberColumn(format="%.0f"),
                                                "Part": st.column_config.ProgressColumn(format="percent", min_value=0,
                                                                                        max_value=1)})
                    st.caption("Répartition selon les parts de chaque produit sur les 12 derniers mois observés.")
    with c2:
        params = pd.DataFrame({"Paramètre": ["Périmètre", "Du", "Au", "Granularité", "Données réelles jusqu'au",
                                             "Unité"],
                               "Valeur": [U.libelle(societe, segment, produit), U.date_fr(debut), U.date_fr(fin),
                                          {"D": "jour", "W": "semaine", "M": "mois"}[freq],
                                          U.date_fr(res.fin_donnees), "tonnes (poids net, retours déduits)"]})
        st.download_button("📥 Excel", U.excel({"Prévision": tab, "Paramètres": params}),
                           file_name=f"prevision_{titre_export}.xlsx", use_container_width=True,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("📄 CSV", tab.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                           file_name=f"prevision_{titre_export}.csv", mime="text/csv", use_container_width=True)


def page() -> None:
    U.entete("Groupe COSUMAR · prévision des ventes", "Prévoir les ventes",
             "Tonnage net prévu par jour, semaine, mois ou période — par société, marché et produit")
    res = U.resultats()
    df = U.donnees()
    U.bandeau_fraicheur(res)
    debut_w, fin_w = U.fenetre(res)
    if fin_w < debut_w:
        return

    societe, segment, produit = U.selecteurs(res, "prev")
    question = st.segmented_control("Que voulez-vous prévoir ?", ["Un jour", "Une semaine", "Un mois", "Une période"],
                                    default="Un mois", key="prev_question")
    question = question or "Un mois"
    j = res.journalier(societe, produit, segment)
    if j.empty:
        st.info("Aucune vente de ce produit pour ce périmètre sur les 12 derniers mois.")
        return
    ev = U.calendrier()
    libelle = U.libelle(societe, segment, produit)
    slug = U.ascii_slug(libelle)

    aujourd_hui = pd.Timestamp.today().normalize()
    defaut = aujourd_hui if debut_w <= aujourd_hui <= fin_w else debut_w

    # ── Un jour ──
    if question == "Un jour":
        jour = pd.Timestamp(st.date_input("Date", value=defaut.date(), min_value=debut_w.date(),
                                          max_value=fin_w.date(), format="DD/MM/YYYY", key="prev_jour"))
        r = res.agreger(j, jour, jour, "D").iloc[0]
        evt = Cal.nom_evenement(ev, jour)
        n1_jour = jour - pd.Timedelta(days=364)
        n1 = M_realise(df, societe, produit, n1_jour, segment)
        mois = jour.to_period("M").to_timestamp()
        tm = res.agreger(j, mois, mois + pd.offsets.MonthEnd(0), "M").iloc[0]

        c1, c2, c3 = st.columns([1.3, 1, 1])
        c1.markdown(U.kpi(f"Prévision · {U.jour_fr(jour)}", U.tonnes(r["yhat"]),
                          f"intervalle 95 % : {U.nb(r['bas'])} – {U.nb(r['haut'])} t", vedette=True),
                    unsafe_allow_html=True)
        sous = (U.jour_fr(n1_jour) + (f" · écart {U.ecart(r['yhat'], n1)}" if n1 and n1 > 0 else "")
                if n1 is not None else f"{U.jour_fr(n1_jour)} : hors des données disponibles")
        c2.markdown(U.kpi("Réalisé le même jour de semaine, un an plus tôt",
                          U.tonnes(n1) if n1 is not None else "—", sous), unsafe_allow_html=True)
        c3.markdown(U.kpi(f"Total prévu en {U.mois_fr(mois)}", U.tonnes(tm["yhat"]),
                          f"ce jour = {U.nb(r['yhat'] / tm['yhat'] * 100 if tm['yhat'] else 0, 1)} % du mois"),
                    unsafe_allow_html=True)
        puces = []
        if jour.dayofweek == 6:
            puces.append('<span class="puce">dimanche : activité quasi nulle</span>')
        if evt:
            genre = "ram" if "Ramadan" in evt and "Aïd" not in evt else "ferie"
            puces.append(f'<span class="puce {genre}">{evt}</span>')
        if segment != "Local" and "Export" in res.segments(societe) and (jour + pd.offsets.MonthEnd(0) - jour).days <= 4:
            puces.append('<span class="puce">fin de mois : pic habituel des facturations export</span>')
        if puces:
            st.markdown(" ".join(puces), unsafe_allow_html=True)
        if segment is None and "Export" in res.segments(societe):
            U.note("Ce total inclut les <b>exportations</b>, facturées par cargaisons (surtout en fin de mois) : "
                   "leur montant pour un jour donné est très incertain. Choisissez <b>Marché local</b> pour une "
                   "prévision journalière plus précise.", "alerte")
        st.markdown("##### Les jours autour")
        d0, d1 = max(debut_w, jour - pd.Timedelta(days=10)), min(fin_w, jour + pd.Timedelta(days=10))
        t = res.agreger(j, d0, d1, "D")
        _bloc_resultat(res, df, t, "D", societe, segment, produit, d0, d1, f"{slug}_{jour:%Y-%m-%d}", surligne=jour)

    # ── Une semaine ──
    elif question == "Une semaine":
        jour = pd.Timestamp(st.date_input("Choisissez un jour de la semaine voulue", value=defaut.date(),
                                          min_value=debut_w.date(), max_value=fin_w.date(), format="DD/MM/YYYY",
                                          key="prev_sem"))
        lundi = jour - pd.Timedelta(days=jour.dayofweek)
        d0, d1 = max(lundi, debut_w), min(lundi + pd.Timedelta(days=6), fin_w)
        tw = res.agreger(j, d0, d1, "W").iloc[0]
        n1 = M_realise(df, societe, produit, d0 - pd.Timedelta(days=364), segment, d1 - pd.Timedelta(days=364))
        c1, c2, c3 = st.columns([1.3, 1, 1])
        c1.markdown(U.kpi(f"Semaine du {U.date_fr(d0)} au {U.date_fr(d1)}", U.tonnes(tw["yhat"]),
                          f"intervalle 95 % : {U.nb(tw['bas'])} – {U.nb(tw['haut'])} t", vedette=True),
                    unsafe_allow_html=True)
        c2.markdown(U.kpi("Même semaine un an plus tôt", U.tonnes(n1) if n1 is not None else "—",
                          f"écart {U.ecart(tw['yhat'], n1)}" if n1 else ""), unsafe_allow_html=True)
        feries = [f"{U.jour_fr(d, False)} : {Cal.nom_evenement(ev, d)}" for d in pd.date_range(d0, d1)
                  if d in Cal.jours_feries(ev)]
        c3.markdown(U.kpi("Jours fériés dans la semaine", str(len(feries)), "<br>".join(feries) or "aucun"),
                    unsafe_allow_html=True)
        if d1 - d0 < pd.Timedelta(days=6):
            st.caption("Semaine incomplète : limitée à la fenêtre de prévision disponible.")
        t = res.agreger(j, d0, d1, "D")
        _bloc_resultat(res, df, t, "D", societe, segment, produit, d0, d1, f"{slug}_semaine_{d0:%Y-%m-%d}")

    # ── Un mois ──
    elif question == "Un mois":
        mois_dispo = pd.date_range(debut_w.to_period("M").to_timestamp(), fin_w, freq="MS")
        mois = st.selectbox("Mois", list(mois_dispo), format_func=lambda d: U.mois_fr(d).capitalize(), key="prev_mois")
        d0, d1 = max(mois, debut_w), min(mois + pd.offsets.MonthEnd(0), fin_w)
        tm = res.agreger(j, d0, d1, "M").iloc[0]
        n1 = M_realise(df, societe, produit, d0 - pd.DateOffset(years=1), segment, d1 - pd.DateOffset(years=1))
        fiab = res.fiabilite(societe)
        c1, c2, c3 = st.columns([1.3, 1, 1])
        c1.markdown(U.kpi(f"Prévision · {U.mois_fr(mois)}", U.tonnes(tm["yhat"]),
                          f"intervalle 95 % : {U.nb(tm['bas'])} – {U.nb(tm['haut'])} t", vedette=True),
                    unsafe_allow_html=True)
        c2.markdown(U.kpi(f"Réalisé en {U.mois_fr(mois - pd.DateOffset(years=1))}",
                          U.tonnes(n1) if n1 is not None else "—",
                          f"écart {U.ecart(tm['yhat'], n1)}" if n1 else ""), unsafe_allow_html=True)
        if fiab:
            c3.markdown(U.kpi("Erreur mensuelle habituelle", f"± {U.nb(fiab['mensuel']['WAPE'], 1)} %",
                              f"mesurée sur {fiab['mois']} mois passés · {societe.replace(' (consolidé)', '')}"),
                        unsafe_allow_html=True)
        ram = [d for d in pd.date_range(d0, d1) if d in Cal.jours_ramadan(ev)]
        feries = sorted({Cal.nom_evenement(ev, d) for d in pd.date_range(d0, d1) if d in Cal.jours_feries(ev)})
        puces = ([f'<span class="puce ram">{len(ram)} jours de Ramadan</span>'] if ram else []) + \
                [f'<span class="puce ferie">{f}</span>' for f in feries]
        if puces:
            st.markdown(" ".join(puces), unsafe_allow_html=True)
        vue = st.segmented_control("Détail", ["Par jour", "Par semaine"], default="Par jour", key="prev_mois_vue")
        freq = "W" if vue == "Par semaine" else "D"
        t = res.agreger(j, d0, d1, freq)
        _bloc_resultat(res, df, t, freq, societe, segment, produit, d0, d1, f"{slug}_{mois:%Y-%m}")

    # ── Une période ──
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            per = st.date_input("Période", value=(debut_w.date(), min(fin_w, debut_w + pd.DateOffset(months=3)
                                                                      - pd.Timedelta(days=1)).date()),
                                min_value=debut_w.date(), max_value=fin_w.date(), format="DD/MM/YYYY",
                                key="prev_periode")
        with c2:
            gran = st.segmented_control("Granularité", ["Jour", "Semaine", "Mois"], default="Semaine", key="prev_gran")
        if not isinstance(per, (tuple, list)) or len(per) < 2:
            st.info("Sélectionnez la date de fin de la période dans le calendrier.")
            return
        d0, d1 = pd.Timestamp(per[0]), pd.Timestamp(per[1])
        freq = {"Jour": "D", "Semaine": "W", "Mois": "M"}[gran or "Semaine"]
        t = res.agreger(j, d0, d1, freq)
        tot = res.agreger(j, d0, d1, "T").iloc[0]
        n1 = M_realise(df, societe, produit, d0 - pd.Timedelta(days=364), segment, d1 - pd.Timedelta(days=364))
        k1, k2, k3 = st.columns([1.3, 1, 1])
        k1.markdown(U.kpi(f"Total du {U.date_fr(d0)} au {U.date_fr(d1)} ({(d1 - d0).days + 1} jours)",
                          U.tonnes(tot["yhat"]), f"intervalle 95 % : {U.nb(tot['bas'])} – {U.nb(tot['haut'])} t",
                          vedette=True), unsafe_allow_html=True)
        k2.markdown(U.kpi("Même période un an plus tôt", U.tonnes(n1) if n1 is not None else "—",
                          f"écart {U.ecart(tot['yhat'], n1)}" if n1 else ""), unsafe_allow_html=True)
        k3.markdown(U.kpi("Moyenne par jour ouvré",
                          U.tonnes(tot["yhat"] / max(1, sum(1 for d in pd.date_range(d0, d1)
                                                                   if d.dayofweek < 6 and d not in Cal.jours_feries(ev)))),
                          "lundi–samedi hors jours fériés"), unsafe_allow_html=True)
        _bloc_resultat(res, df, t, freq, societe, segment, produit, d0, d1, f"{slug}_{d0:%Y%m%d}-{d1:%Y%m%d}",
                       hauteur=400)


def M_realise(df, societe, produit, d0, segment, d1=None):
    """Réalisé historique entre d0 et d1 (None si la période n'est pas couverte par les données)."""
    d1 = d0 if d1 is None else d1
    if d0 < df["date"].min() or d1 > df["date"].max():
        return None
    from prevision.moteur import Resultats
    return Resultats.realise(df, societe, produit, d0, d1, segment)
