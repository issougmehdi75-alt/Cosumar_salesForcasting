"""État de session, formatage français, graphiques et composants partagés par les pages."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from prevision import calendrier as C
from prevision import donnees as D
from prevision import moteur as M

RACINE = Path(__file__).resolve().parent.parent
PRECALCUL = RACINE / "results" / "previsions.json"

# ── Palette (paire bleu/ambre validée pour le daltonisme) ──
BLEU, BLEU_CLAIR, AMBRE, AMBRE_CLAIR = "#1c5cab", "#9ec5f4", "#c98500", "#f2d9a6"
GRIS, ENCRE, ENCRE_2, MUET, GRILLE = "#b9b8ae", "#0a2540", "#4a5568", "#8896a8", "#e6e9ef"
NAVY, OR = "#00428C", "#F5BC2D"
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre",
        "novembre", "décembre"]
MOIS_C = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
JOURS_C = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
MARCHES = {"Total": None, "Marché local": "Local", "Export": "Export"}

CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; color: {ENCRE}; }}
.entete {{ background: linear-gradient(120deg, #00285A 0%, {NAVY} 100%); border-radius: 14px;
          padding: 20px 28px; margin-bottom: 14px; border-bottom: 4px solid {OR}; }}
.entete .sur {{ color: {OR}; font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; font-weight: 650; }}
.entete h1 {{ color: #fff; margin: 3px 0 2px 0; padding: 0; font-size: 1.65rem; font-weight: 750; }}
.entete p {{ color: #c9d6e8; margin: 0; font-size: .95rem; }}
.kpi {{ background: #fff; border: 1px solid #dfe4ec; border-radius: 12px; padding: 14px 18px; height: 100%; }}
.kpi .l {{ font-size: .78rem; color: {ENCRE_2}; }}
.kpi .v {{ font-size: 1.6rem; font-weight: 720; color: {ENCRE}; line-height: 1.25; margin-top: 2px; }}
.kpi .s {{ font-size: .78rem; color: {MUET}; margin-top: 3px; line-height: 1.4; }}
.kpi.vedette {{ border: 2px solid {AMBRE}; background: #fffaf0; }}
.kpi.vedette .v {{ font-size: 2.2rem; }}
.note {{ border-left: 3px solid {NAVY}; background: #eef3fa; padding: 11px 15px; border-radius: 0 10px 10px 0;
         color: {ENCRE}; font-size: .9rem; margin: 6px 0 12px 0; }}
.note.alerte {{ border-left-color: {AMBRE}; background: #fff7e6; color: #5c3d00; }}
.note.danger {{ border-left-color: #d03b3b; background: #fdecec; color: #6b1414; }}
.puce {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600;
         background: #eef3fa; color: {NAVY}; margin-right: 6px; }}
.puce.ferie {{ background: #fdecec; color: #8a1c1c; }}
.puce.ram {{ background: #f1ecfb; color: #4b2a8a; }}
.pied {{ text-align: center; color: {MUET}; font-size: .78rem; padding: 18px 0 4px 0; }}
</style>
"""


# ──────────────────────────────────────────────────────────────
#  Formatage
# ──────────────────────────────────────────────────────────────
def nb(x: float, dec: int = 0) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{dec}f}".replace(",", " ").replace(".", ",")


def tonnes(x: float) -> str:
    return f"{nb(x)} t" if abs(x) < 10_000 else f"{nb(x / 1000, 1)} kt"


def jour_fr(d, annee: bool = True) -> str:
    d = pd.Timestamp(d)
    return f"{JOURS[d.dayofweek]} {d.day} {MOIS[d.month - 1]}" + (f" {d.year}" if annee else "")


def date_fr(d) -> str:
    return pd.Timestamp(d).strftime("%d/%m/%Y")


def mois_fr(d, court: bool = False) -> str:
    d = pd.Timestamp(d)
    return f"{(MOIS_C if court else MOIS)[d.month - 1]} {d.year}"


def ascii_slug(texte: str) -> str:
    """Nom de fichier sûr (sans accents ni espaces) : certains navigateurs ignorent sinon le nom proposé."""
    import re
    import unicodedata
    t = unicodedata.normalize("NFKD", texte).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")


def kpi(label: str, valeur: str, sous: str = "", vedette: bool = False) -> str:
    return (f'<div class="kpi{" vedette" if vedette else ""}"><div class="l">{label}</div>'
            f'<div class="v">{valeur}</div><div class="s">{sous}</div></div>')


def entete(sur: str, titre: str, texte: str) -> None:
    st.markdown(f'<div class="entete"><div class="sur">{sur}</div><h1>{titre}</h1><p>{texte}</p></div>',
                unsafe_allow_html=True)


def note(texte: str, genre: str = "") -> None:
    st.markdown(f'<div class="note {genre}">{texte}</div>', unsafe_allow_html=True)


def ecart(prevu: float, ref: float) -> str:
    if not ref or not np.isfinite(ref) or ref <= 0:
        return ""
    v = (prevu / ref - 1) * 100
    return f"{'+' if v >= 0 else '−'}{nb(abs(v), 1)} %"


# ──────────────────────────────────────────────────────────────
#  État : données, calendrier, résultats
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _donnees_projet() -> pd.DataFrame:
    return D.charger()


@st.cache_data(show_spinner=False)
def _calendrier_projet() -> pd.DataFrame:
    return C.charger_evenements()


@st.cache_data(show_spinner=False)
def _precalcul() -> dict | None:
    return json.loads(PRECALCUL.read_text(encoding="utf-8")) if PRECALCUL.exists() else None


def donnees() -> pd.DataFrame:
    return st.session_state.get("donnees", _donnees_projet())


def calendrier() -> pd.DataFrame:
    return st.session_state.get("calendrier", _calendrier_projet())


def source_modifiee() -> bool:
    return "donnees" in st.session_state or "calendrier" in st.session_state


def resultats() -> M.Resultats:
    df, ev = donnees(), calendrier()
    cle = M.empreinte(df, ev)
    cache = st.session_state.setdefault("_resultats", {})
    if cle in cache:
        return cache[cle]
    pre = _precalcul()
    if pre and pre.get("empreinte") == cle:
        res = M.Resultats.depuis_dict(pre, ev)
    else:
        with st.status("Recalcul des prévisions avec les nouvelles données…", expanded=True) as statut:
            barre = st.progress(0.0)
            n_total = len(D.composantes(df)) * len(M.MODELES)
            compteur = {"i": 0}

            def progression(msg: str) -> None:
                compteur["i"] += 1
                barre.progress(min(1.0, compteur["i"] / n_total), text=msg)

            res = M.calculer(df, ev, progression)
            statut.update(label="Prévisions recalculées", state="complete", expanded=False)
    cache.clear()
    cache[cle] = res
    return res


def fenetre(res: M.Resultats) -> tuple[pd.Timestamp, pd.Timestamp]:
    return res.fin_donnees + pd.Timedelta(days=1), res.fin_prevision


def bandeau_fraicheur(res: M.Resultats) -> None:
    debut, fin = fenetre(res)
    aujourd_hui = pd.Timestamp.today().normalize()
    src = "importées dans cette session" if "donnees" in st.session_state else "du projet"
    if fin < debut:
        note("Le calendrier ne couvre pas la période à prévoir (Ramadan non renseigné). "
             "Complétez-le dans la page <b>Calendrier</b>.", "danger")
    elif aujourd_hui > fin:
        note(f"📅 Données {src} : ventes réelles jusqu'au <b>{date_fr(res.fin_donnees)}</b>. "
             f"Prévisions disponibles du <b>{date_fr(debut)}</b> au <b>{date_fr(fin)}</b>. "
             f"Pour prévoir la période actuelle, importez les ventes récentes dans la page "
             f"<b>Mettre à jour les données</b>.", "alerte")
    else:
        note(f"📅 Ventes réelles jusqu'au <b>{date_fr(res.fin_donnees)}</b> · prévisions disponibles jusqu'au "
             f"<b>{date_fr(fin)}</b>.")


# ──────────────────────────────────────────────────────────────
#  Sélecteurs communs
# ──────────────────────────────────────────────────────────────
def selecteurs(res: M.Resultats, cle: str, produit: bool = True) -> tuple[str, str | None, str]:
    options = [D.GROUPE, *res.societes()]
    c1, c2, c3 = st.columns(3)
    with c1:
        societe = st.selectbox("Société", options, key=f"{cle}_soc",
                               help="« GROUPE (consolidé) » additionne les quatre sociétés.")
    segs = res.segments(societe)
    marches = ["Total", "Marché local", "Export"] if "Export" in segs else ["Marché local"]
    with c2:
        marche = st.selectbox("Marché", marches, key=f"{cle}_mar",
                              help="Les exportations (COSUMAR SA) sont facturées par cargaisons, surtout en fin "
                                   "de mois : elles sont prévues séparément du marché local.")
    segment = MARCHES[marche] if len(marches) > 1 else None
    prod = D.TOUS
    if produit:
        with c3:
            prods = [D.TOUS, *[p for p in res.produits(societe) if p != "Autre"]]
            prod = st.selectbox("Produit", prods, key=f"{cle}_prod")
    return societe, segment, prod


def libelle(societe: str, segment: str | None, produit: str) -> str:
    s = societe.replace(" (consolidé)", "")
    if segment:
        s += f" · {'marché local' if segment == 'Local' else 'export'}"
    if produit != D.TOUS:
        s += f" · {produit}"
    return s


# ──────────────────────────────────────────────────────────────
#  Graphiques
# ──────────────────────────────────────────────────────────────
def mise_en_forme(fig: go.Figure, hauteur: int, y_titre: str = "Tonnes", legende: bool = True) -> go.Figure:
    fig.update_layout(
        height=hauteur, margin=dict(l=8, r=8, t=34 if legende else 12, b=8), plot_bgcolor="#fff",
        paper_bgcolor="#fff", font=dict(family=FONT, size=12, color=ENCRE_2), separators=", ",
        hoverlabel=dict(bgcolor="white", bordercolor=GRILLE, font=dict(family=FONT, color=ENCRE)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        showlegend=legende, bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRIS, tickfont=dict(color=MUET))
    fig.update_yaxes(showgrid=True, gridcolor=GRILLE, zeroline=False, tickfont=dict(color=MUET),
                     tickformat=",d", title=dict(text=y_titre, font=dict(color=MUET)), rangemode="tozero")
    return fig


def etiquettes(periodes: pd.Series, freq: str) -> list[str]:
    if freq == "D":
        return [f"{JOURS_C[d.dayofweek]} {d.day:02d}/{d.month:02d}" for d in periodes]
    if freq == "W":
        return [f"sem. du {d.day:02d}/{d.month:02d}" for d in periodes]
    return [mois_fr(d, court=True) for d in periodes]


def figure_prevision(t: pd.DataFrame, freq: str, n1: pd.Series | None = None, surligne=None,
                     hauteur: int = 380) -> go.Figure:
    """Barres = prévision (ambre) avec intervalle 95 % ; losanges gris = réalisé un an plus tôt."""
    x = etiquettes(t["periode"], freq)
    choisi = [surligne is not None and p == surligne for p in t["periode"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=t["yhat"], name="Prévision", marker=dict(
            color=AMBRE, cornerradius=3,
            line=dict(color=[ENCRE if c else "rgba(0,0,0,0)" for c in choisi], width=[3 if c else 0 for c in choisi])),
        error_y=dict(type="data", symmetric=False, array=(t["haut"] - t["yhat"]).to_numpy(),
                     arrayminus=(t["yhat"] - t["bas"]).to_numpy(), color=ENCRE_2, thickness=1, width=3),
        customdata=np.c_[t["bas"], t["haut"]],
        hovertemplate="%{x}<br><b>%{y:,.0f} t</b><br>intervalle 95 % : %{customdata[0]:,.0f} – %{customdata[1]:,.0f} t"
                      "<extra></extra>"))
    if n1 is not None and n1.notna().any():
        fig.add_trace(go.Scatter(x=x, y=n1.to_numpy(), name="Réalisé un an plus tôt", mode="markers",
                                 marker=dict(symbol="diamond", size=9, color=GRIS, line=dict(color=ENCRE_2, width=1)),
                                 hovertemplate="un an plus tôt : %{y:,.0f} t<extra></extra>"))
    mise_en_forme(fig, hauteur)
    fig.update_layout(hovermode="closest")
    if any(choisi):
        i = choisi.index(True)
        fig.add_annotation(x=x[i], y=float(t["haut"].iloc[i]), text="jour choisi", showarrow=False, yshift=12,
                           font=dict(color=ENCRE, size=12))
    if len(x) > 20:
        fig.update_xaxes(tickangle=-45)
    return fig


def realise_n1(df: pd.DataFrame, t: pd.DataFrame, freq: str, societe, produit, segment) -> pd.Series:
    """Réalisé sur la même période un an plus tôt (364 jours pour jours/semaines : même jour de semaine)."""
    vals = []
    for _, r in t.iterrows():
        if freq == "M":
            d0, d1 = r["debut"] - pd.DateOffset(years=1), r["fin"] - pd.DateOffset(years=1)
        else:
            d0, d1 = r["debut"] - pd.Timedelta(days=364), r["fin"] - pd.Timedelta(days=364)
        if d0 < df["date"].min() or d1 > df["date"].max():
            vals.append(np.nan)
            continue
        vals.append(M.Resultats.realise(df, societe, produit, d0, d1, segment))
    return pd.Series(vals, index=t.index)


# ──────────────────────────────────────────────────────────────
#  Exports
# ──────────────────────────────────────────────────────────────
def excel(feuilles: dict[str, pd.DataFrame]) -> bytes:
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        for nom, f in feuilles.items():
            f.to_excel(w, index=False, sheet_name=nom[:31])
            ws = w.sheets[nom[:31]]
            for i, col in enumerate(f.columns, 1):
                larg = max(len(str(col)), *(len(str(v)) for v in f[col].head(200))) if len(f) else len(str(col))
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(45, larg + 2)
    return out.getvalue()


def tableau_export(t: pd.DataFrame, freq: str, n1: pd.Series | None) -> pd.DataFrame:
    if freq == "D":
        per = [jour_fr(d) for d in t["periode"]]
    elif freq == "W":
        per = [f"Semaine du {date_fr(a)} au {date_fr(b)}" for a, b in zip(t["debut"], t["fin"])]
    else:
        per = [mois_fr(d).capitalize() for d in t["periode"]]
    out = pd.DataFrame({
        "Période": per,
        "Prévision (t)": t["yhat"].round(1).to_numpy(),
        "Borne basse 95 % (t)": t["bas"].round(1).to_numpy(),
        "Borne haute 95 % (t)": t["haut"].round(1).to_numpy(),
    })
    if n1 is not None:
        out["Réalisé un an plus tôt (t)"] = n1.round(1).to_numpy()
    return out
