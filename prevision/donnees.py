"""Données de ventes : import des extractions SAP, agrégats journaliers anonymes, séries mensuelles.

Format publié (data/ventes_journalieres.csv) — aucune information client :
    date, societe, segment (Local | Export), produit, tonnage_net, nb_lignes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
FICHIER = RACINE / "data" / "ventes_journalieres.csv"
GROUPE = "GROUPE (consolidé)"
TOUS = "Tous les produits"
SEUIL_LACUNE = 0.05     # mois avec < 5 % du nombre habituel de lignes = lacune d'extraction

COLONNES_SAP = ["Société", "Groupe de clients", "Groupe de marchandises", "Jour calendaire", "Poids net"]
PRODUITS = {
    "GR-GCDT": "Granulé — grand cond.",
    "GR-PCDT": "Granulé — petit cond.",
    "Pain": "Pain de sucre",
    "Lingot": "Lingot",
    "Morceau": "Morceau",
}


# ──────────────────────────────────────────────────────────────
#  Import des extractions SAP (ligne par ligne)
# ──────────────────────────────────────────────────────────────
def _lire_csv(source) -> pd.DataFrame:
    for sep in (",", ";"):
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, sep=sep, encoding="utf-8-sig", dtype=str)
        df.columns = [c.strip() for c in df.columns]
        if set(COLONNES_SAP) <= set(df.columns):
            return df
    raise ValueError("Colonnes attendues introuvables : " + ", ".join(COLONNES_SAP))


def _nombre(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.strip().str.replace(" ", "").str.replace(" ", "").str.replace(",", "."),
                         errors="coerce")


def agreger_extraction(sources) -> tuple[pd.DataFrame, dict]:
    """Une ou plusieurs extractions SAP → agrégats journaliers anonymes + rapport de contrôle.

    Seules 5 colonnes sont lues ; les numéros de clients sont ignorés dès la lecture.
    """
    if not isinstance(sources, (list, tuple)):
        sources = [sources]
    brut = pd.concat([_lire_csv(s)[COLONNES_SAP] for s in sources], ignore_index=True)
    n = len(brut)
    brut["date"] = pd.to_datetime(brut["Jour calendaire"].str.strip(), format="%m/%d/%y", errors="coerce")
    sans_date = brut["date"].isna()
    if sans_date.any():  # autres formats courants d'export
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y"):
            m = brut["date"].isna()
            brut.loc[m, "date"] = pd.to_datetime(brut.loc[m, "Jour calendaire"].str.strip(), format=fmt, errors="coerce")
    brut["poids"] = _nombre(brut["Poids net"])
    invalides = brut["date"].isna() | brut["poids"].isna()
    brut = brut[~invalides]
    brut["segment"] = np.where(brut["Groupe de clients"].str.strip().str.lower() == "export", "Export", "Local")
    brut["produit"] = brut["Groupe de marchandises"].str.strip().map(PRODUITS).fillna("Autre")
    brut["societe"] = brut["Société"].str.strip()
    agg = (brut.groupby(["date", "societe", "segment", "produit"])
           .agg(tonnage_net=("poids", "sum"), nb_lignes=("poids", "size")).reset_index())
    agg["tonnage_net"] = agg["tonnage_net"].round(3)
    rapport = {
        "lignes_lues": int(n), "lignes_invalides": int(invalides.sum()),
        "debut": agg["date"].min(), "fin": agg["date"].max(),
        "societes": sorted(agg["societe"].unique()),
        "tonnage": float(agg["tonnage_net"].sum()),
    }
    return agg, rapport


def fusionner(base: pd.DataFrame, nouveau: pd.DataFrame, mode: str = "ajouter") -> tuple[pd.DataFrame, dict]:
    """mode « ajouter » : ne garde du nouveau fichier que les jours POSTÉRIEURS aux données
    existantes de chaque société (pas de double comptage si l'on recharge une période).
    mode « remplacer » : les nouvelles données remplacent tout."""
    if mode == "remplacer":
        return nouveau.sort_values(["date", "societe"]).reset_index(drop=True), {"ajoutees": len(nouveau), "ignorees": 0}
    fin = base.groupby("societe")["date"].max()
    limite = nouveau["societe"].map(fin)
    garder = limite.isna() | (nouveau["date"] > limite)
    res = pd.concat([base, nouveau[garder]], ignore_index=True).sort_values(["date", "societe"]).reset_index(drop=True)
    return res, {"ajoutees": int(garder.sum()), "ignorees": int((~garder).sum())}


# ──────────────────────────────────────────────────────────────
#  Lecture du jeu publié
# ──────────────────────────────────────────────────────────────
def charger(source=FICHIER) -> pd.DataFrame:
    df = pd.read_csv(source, parse_dates=["date"])
    attendu = {"date", "societe", "segment", "produit", "tonnage_net", "nb_lignes"}
    if not attendu <= set(df.columns):
        raise ValueError("Fichier de ventes agrégées invalide : colonnes " + ", ".join(sorted(attendu)))
    return df


def exporter(df: pd.DataFrame) -> bytes:
    return df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_csv(index=False).encode("utf-8")


def societes(df: pd.DataFrame) -> list[str]:
    tot = df.groupby("societe")["tonnage_net"].sum().sort_values(ascending=False)
    return list(tot.index)


def composantes(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Séries modélisées séparément : (société, segment) représentant ≥ 1 % du tonnage de la société."""
    t = df.groupby(["societe", "segment"])["tonnage_net"].sum()
    part = t / t.groupby(level=0).transform("sum")
    return [k for k, v in part.items() if v >= 0.01]


# ──────────────────────────────────────────────────────────────
#  Séries mensuelles
# ──────────────────────────────────────────────────────────────
def debut_complet(df: pd.DataFrame, societe: str) -> pd.Timestamp:
    """Premier mois complet (un mois commencé après le 5 est incomplet)."""
    d0 = df.loc[df["societe"] == societe, "date"].min()
    m0 = d0.to_period("M").to_timestamp()
    return m0 if d0.day <= 5 else m0 + pd.offsets.MonthBegin(1)


def fin_extraction(df: pd.DataFrame) -> pd.Timestamp:
    """Date de fin de l'extraction (la même pour toutes les sociétés : une société peut ne
    rien vendre les derniers jours du mois sans que les données soient incomplètes)."""
    return df["date"].max()


def fin_complete(df: pd.DataFrame, societe: str | None = None) -> pd.Timestamp:
    """Dernier mois complet de l'extraction (fin à moins de 2 jours de la fin du mois)."""
    d1 = fin_extraction(df)
    fin_mois = d1 + pd.offsets.MonthEnd(0)
    m1 = d1.to_period("M").to_timestamp()
    return m1 if (fin_mois - d1).days <= 2 else m1 - pd.offsets.MonthBegin(1)


def serie_mensuelle(df: pd.DataFrame, societe: str, segment: str) -> pd.DataFrame:
    """ds (début de mois), y (tonnage net), lignes, manquant (lacune interpolée)."""
    d = df[(df["societe"] == societe) & (df["segment"] == segment)]
    m0, m1 = debut_complet(df, societe), fin_complete(df, societe)
    mois = pd.date_range(m0, m1, freq="MS")
    g = d.groupby(d["date"].dt.to_period("M").dt.to_timestamp()).agg(y=("tonnage_net", "sum"), lignes=("nb_lignes", "sum"))
    g = g.reindex(mois, fill_value=0)
    # les lacunes sont détectées sur l'ensemble de la société (tous segments)
    toutes = df[df["societe"] == societe]
    lignes_soc = toutes.groupby(toutes["date"].dt.to_period("M").dt.to_timestamp())["nb_lignes"].sum().reindex(mois, fill_value=0)
    manquant = lignes_soc < SEUIL_LACUNE * lignes_soc.median()
    y = g["y"].astype(float).where(~manquant).interpolate(limit_direction="both")
    return pd.DataFrame({"ds": mois, "y": y.to_numpy(), "lignes": g["lignes"].to_numpy(), "manquant": manquant.to_numpy()})


def parts_produits(df: pd.DataFrame, societe: str, segment: str, fin: pd.Timestamp, mois: int = 12) -> pd.Series:
    """Répartition du tonnage par produit sur les `mois` derniers mois (somme = 1)."""
    d = df[(df["societe"] == societe) & (df["segment"] == segment)
           & (df["date"] > fin - pd.DateOffset(months=mois)) & (df["date"] <= fin)]
    t = d.groupby("produit")["tonnage_net"].sum().clip(lower=0)
    return t / t.sum() if t.sum() > 0 else t
