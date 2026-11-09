"""Tests : python -m pytest -q"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from prevision import calendrier as C  # noqa: E402
from prevision import donnees as D  # noqa: E402
from prevision import moteur as M  # noqa: E402
from prevision import modeles as Mo  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return D.charger()


@pytest.fixture(scope="module")
def ev():
    return C.charger_evenements()


@pytest.fixture(scope="module")
def res(ev):
    d = json.loads((RACINE / "results" / "previsions.json").read_text(encoding="utf-8"))
    return M.Resultats.depuis_dict(d, ev)


def _extraction(lignes, sep=",", fmt="%m/%d/%y"):
    """Fabrique une extraction SAP minimale."""
    t = pd.DataFrame(lignes, columns=["Société", "Groupe de clients", "Groupe de marchandises", "date", "Poids net"])
    brut = t.pop("date")
    dates = pd.to_datetime(brut, errors="coerce")
    t["Jour calendaire"] = [d.strftime(fmt) if pd.notna(d) else b for d, b in zip(dates, brut)]
    t["Numéro de client"] = "123456"
    b = io.BytesIO(t.to_csv(index=False, sep=sep).encode("utf-8-sig"))
    return b


# ── Données ──
def test_donnees_publiees_anonymes(df):
    assert set(df.columns) == {"date", "societe", "segment", "produit", "tonnage_net", "nb_lignes"}
    assert not list((RACINE / "data").glob("*_[0-9].csv")), "extractions brutes présentes dans data/"


def test_import_formats_et_anonymisation():
    lignes = [("SUTA", "Grossistes", "Pain", "2021-01-04", "1,5"), ("SUTA", "Export", "GR-GCDT", "2021-01-04", "10"),
              ("SUTA", "GMS", "Pain", "2021-01-04", "-0,5"), ("SUTA", "GMS", "Pain", "pas une date", "3")]
    for sep, fmt in [(",", "%m/%d/%y"), (";", "%d/%m/%Y"), (";", "%Y-%m-%d")]:
        agg, r = D.agreger_extraction(_extraction(lignes, sep, fmt))
        assert "Numéro de client" not in agg.columns
        assert r["lignes_invalides"] == 1
        loc = agg[(agg["segment"] == "Local") & (agg["produit"] == "Pain de sucre")]
        assert loc["tonnage_net"].sum() == pytest.approx(1.0) and loc["nb_lignes"].sum() == 2
        assert agg.loc[agg["segment"] == "Export", "tonnage_net"].sum() == pytest.approx(10)


def test_fusion_sans_double_comptage(df):
    nouveau = df[df["date"] >= "2020-12-01"].assign(date=lambda x: x["date"] + pd.Timedelta(days=31))
    f, info = D.fusionner(df, pd.concat([df[df["date"] >= "2020-12-15"], nouveau]), "ajouter")
    assert info["ignorees"] == (df["date"] >= "2020-12-15").sum()      # jours déjà présents ignorés
    assert f["tonnage_net"].sum() == pytest.approx(df["tonnage_net"].sum() + nouveau["tonnage_net"].sum())
    r, _ = D.fusionner(df, nouveau, "remplacer")
    assert len(r) == len(nouveau)


def test_mois_incomplets(df):
    s = D.serie_mensuelle(df, "SURAC", "Local")
    assert s["manquant"].sum() == 3 and s["y"].notna().all()
    assert D.serie_mensuelle(df, "SUTA", "Local")["ds"].iloc[0] == pd.Timestamp("2017-02-01")  # janv. 2017 incomplet


# ── Calendrier ──
def test_calendrier(ev):
    X = C.variables_mensuelles(pd.date_range("2026-01-01", periods=4, freq="MS"), ev)
    assert X.loc["2026-01-01", "pre_ramadan"] > 0.3          # Ramadan 2026 commence le 19 février
    assert X.loc["2026-02-01", "ramadan"] > 0.3
    assert X.notna().all().all()
    assert pd.Timestamp("2026-03-20") in C.jours_feries(ev)  # Aïd al-Fitr 2026
    assert C.couverture(ev) >= pd.Timestamp("2028-01-01")


def test_edition_calendrier(ev):
    tout = ev.assign(debut=ev["debut"].dt.date, fin=ev["fin"].dt.date)
    assert len(C.fusionner_edition(ev, tout, None)) == len(ev.drop_duplicates())       # pas de doublons
    an = ev[ev["debut"].dt.year == 2027].copy()
    an.loc[an["evenement"] == "Aïd al-Fitr", ["debut", "fin"]] = [pd.Timestamp("2027-03-11"), pd.Timestamp("2027-03-12")]
    nouveau = C.fusionner_edition(ev, an, 2027)
    assert len(nouveau) == len(ev)
    assert pd.Timestamp("2027-03-12") in C.jours_feries(nouveau) and pd.Timestamp("2027-03-10") not in C.jours_feries(nouveau)
    with pytest.raises(ValueError):
        C.fusionner_edition(ev, an.assign(fin=pd.Timestamp("2000-01-01")), 2027)


# ── Modèles ──
def test_backtest_sans_fuite():
    y = pd.Series(np.arange(48.0), index=pd.date_range("2017-01-01", periods=48, freq="MS"))
    X = pd.DataFrame(0.0, index=y.index, columns=Mo.EXOG)
    vus = []

    def espion(hist, h, Xp, Xf):
        vus.append(hist.index[-1])
        assert len(Xf) == h and Xp.index[-1] == hist.index[-1]
        return np.repeat(hist.iloc[-1], h)

    b = Mo.backtest(y, X, espion, 6, 3)
    assert [d.strftime("%Y-%m") for d in vus] == ["2019-06", "2019-12", "2020-06"]
    assert np.allclose(b.erreurs[:, 0], 1) and np.allclose(b.erreurs[:, -1], 6)


def test_wape():
    assert Mo.metriques([100, 100], [90, 110])["WAPE"] == 10


# ── Prévisions ──
def test_precalcul_a_jour(df, ev):
    d = json.loads((RACINE / "results" / "previsions.json").read_text(encoding="utf-8"))
    assert d["empreinte"] == M.empreinte(df, ev), "relancer scripts/calculer_previsions.py"


def test_jours_somment_aux_mois(res):
    for soc in [D.GROUPE, *res.societes()]:
        j = res.journalier(soc)
        m = res.agreger(j, j["date"].min(), j["date"].max(), "M")
        attendu = sum(c.prevision["yhat"].sum() for c in res._comp(soc))
        assert m["yhat"].sum() == pytest.approx(attendu, rel=1e-9)
        sem = res.agreger(j, j["date"].min(), j["date"].max(), "W")
        assert sem["yhat"].sum() == pytest.approx(attendu, rel=1e-9)


def test_produits_somment_au_total(res):
    soc = "COSUMAR SA"
    d0, d1 = pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-31")
    total = res.agreger(res.journalier(soc), d0, d1, "T")["yhat"].sum()
    parts = sum(res.agreger(res.journalier(soc, p), d0, d1, "T")["yhat"].sum() for p in res.produits(soc))
    assert parts == pytest.approx(total, rel=1e-6)


def test_dimanches_et_feries_faibles(res):
    j = res.journalier("SUTA").set_index("date")["yhat"]
    ouvres = j[(j.index.dayofweek < 5)].mean()
    assert j[j.index.dayofweek == 6].mean() < 0.2 * ouvres
    assert j.loc["2021-05-13"] < 0.2 * ouvres                 # Aïd al-Fitr 2021


def test_intervalles(res):
    j = res.journalier(D.GROUPE)
    t = res.agreger(j, j["date"].min(), j["date"].max(), "M")
    assert ((t["bas"] <= t["yhat"]) & (t["yhat"] <= t["haut"])).all()
    largeur = (t["haut"] - t["yhat"]) / t["yhat"]
    assert largeur.iloc[-1] > largeur.iloc[0]                 # l'incertitude augmente avec l'horizon
    total = res.agreger(j, j["date"].min(), j["date"].max(), "T").iloc[0]
    assert total["yhat"] == pytest.approx(t["yhat"].sum())


def test_fiabilite_calibree(res):
    f = res.fiabilite(D.GROUPE)
    assert f["mois"] >= 12 and 0.85 <= f["couverture_jour"] <= 1.0
    assert f["jour"] < f["jour_uniforme"]                     # le profil journalier apporte de l'information
