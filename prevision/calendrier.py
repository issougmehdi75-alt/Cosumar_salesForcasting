"""Calendrier marocain : jours fériés, Ramadan et variables calendaires mensuelles.

Les événements sont lus dans data/calendrier_evenements.csv (modifiable depuis
l'application, page « Paramètres »). Les fêtes religieuses dépendent de
l'observation du croissant lunaire : les dates futures sont des estimations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FICHIER = Path(__file__).resolve().parent.parent / "data" / "calendrier_evenements.csv"
COLONNES = ["debut", "fin", "evenement", "type", "fiabilite"]


def charger_evenements(source=FICHIER) -> pd.DataFrame:
    ev = pd.read_csv(source)
    manquantes = set(COLONNES) - set(ev.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans le calendrier : {', '.join(sorted(manquantes))}")
    ev["debut"] = pd.to_datetime(ev["debut"])
    ev["fin"] = pd.to_datetime(ev["fin"])
    if (ev["fin"] < ev["debut"]).any():
        raise ValueError("Une date de fin est antérieure à la date de début.")
    return ev[COLONNES].sort_values("debut").reset_index(drop=True)


def _jours(ev: pd.DataFrame, type_: str) -> pd.DatetimeIndex:
    parts = [pd.date_range(d, f, freq="D") for d, f in zip(ev.loc[ev["type"] == type_, "debut"],
                                                             ev.loc[ev["type"] == type_, "fin"])]
    return pd.DatetimeIndex(np.concatenate([p.values for p in parts])) if parts else pd.DatetimeIndex([])


def jours_feries(ev: pd.DataFrame) -> set[pd.Timestamp]:
    return set(_jours(ev, "ferie"))


def jours_ramadan(ev: pd.DataFrame) -> set[pd.Timestamp]:
    return set(_jours(ev, "ramadan"))


def nom_evenement(ev: pd.DataFrame, jour: pd.Timestamp) -> str | None:
    m = ev[(ev["debut"] <= jour) & (ev["fin"] >= jour)]
    return " · ".join(m["evenement"]) if len(m) else None


def couverture(ev: pd.DataFrame) -> pd.Timestamp:
    """Dernière date pour laquelle Ramadan est renseigné (au-delà, les variables sont incomplètes)."""
    r = ev[ev["type"] == "ramadan"]
    return r["fin"].max() if len(r) else pd.Timestamp("1900-01-01")


def variables_mensuelles(mois: pd.DatetimeIndex, ev: pd.DataFrame) -> pd.DataFrame:
    """Variables connues à l'avance pour chaque mois (début de mois) :

    ramadan       part des jours du mois en Ramadan
    pre_ramadan   part des jours du mois SUIVANT en Ramadan (achats anticipés)
    jours_ouvres  lundi–vendredi + ½ samedi, hors jours fériés
    aid_adha      1 si l'Aïd al-Adha tombe dans le mois
    """
    feries, ram = jours_feries(ev), jours_ramadan(ev)
    adha = set(_jours(ev[ev["evenement"].str.contains("Adha", case=False)], "ferie"))
    lignes = []
    for m in mois:
        jours = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D")
        m_suiv = m + pd.offsets.MonthBegin(1)
        suivant = pd.date_range(m_suiv, m_suiv + pd.offsets.MonthEnd(0), freq="D")
        ouv = sum((1.0 if j.dayofweek < 5 else 0.5 if j.dayofweek == 5 else 0.0)
                  for j in jours if j not in feries)
        lignes.append({
            "ramadan": np.mean([j in ram for j in jours]),
            "pre_ramadan": np.mean([j in ram for j in suivant]),
            "jours_ouvres": ouv,
            "aid_adha": float(any(j in adha for j in jours)),
        })
    return pd.DataFrame(lignes, index=mois)


def fusionner_edition(ev: pd.DataFrame, edite: pd.DataFrame, annee: int | None) -> pd.DataFrame:
    """Calendrier complet après édition dans l'application.

    `edite` contient les lignes affichées dans l'éditeur : toutes les lignes si `annee` est None,
    sinon celles de l'année `annee` (les autres années sont conservées telles quelles).
    """
    autres = ev.iloc[0:0] if annee is None else ev[ev["debut"].dt.year != annee]
    e = edite.dropna(subset=["debut", "fin", "evenement", "type"]).copy()
    e["debut"], e["fin"] = pd.to_datetime(e["debut"]), pd.to_datetime(e["fin"])
    if (e["fin"] < e["debut"]).any():
        raise ValueError("une date de fin est antérieure à sa date de début")
    if not set(e["type"]) <= {"ferie", "ramadan"}:
        raise ValueError("type inconnu (attendu : ferie ou ramadan)")
    e["fiabilite"] = e["fiabilite"].fillna("estimée")
    res = pd.concat([autres, e[COLONNES]], ignore_index=True).drop_duplicates()
    return res.sort_values(["debut", "evenement"]).reset_index(drop=True)
