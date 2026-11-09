"""Moteur de prévision : mensuel → journalier → semaine / mois / période, par produit.

Étapes, pour chaque composante (société × segment Local / Export) :
  1. série mensuelle + variables calendaires ;
  2. backtest de tous les modèles (3 fenêtres de 6 mois), sélection du meilleur (MAE) ;
     l'« erreur attendue » affichée est mesurée par SÉLECTION IMBRIQUÉE (le modèle est
     choisi sur les fenêtres précédentes puis évalué sur la suivante) ;
  3. prévision mensuelle sur l'horizon + incertitude σ(h) issue du backtest ;
  4. profil journalier (jour de semaine, fin de mois, jours fériés, Ramadan) estimé par
     régression de Poisson sur l'historique → répartition de chaque mois sur ses jours ;
  5. répartition par produit (parts des 12 derniers mois).
Les composantes sont additionnées pour obtenir la société et le groupe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import calendrier as C
from . import donnees as D
from .modeles import MODELES, backtest, metriques, plis_possibles, sigma_horizon

HORIZON_TEST, PLIS, HORIZON_MAX = 6, 3, 12
Z95 = 1.96


# ──────────────────────────────────────────────────────────────
#  Profil journalier
# ──────────────────────────────────────────────────────────────
def _rang_ouvre_fin_mois(jours: pd.DatetimeIndex, feries: set) -> np.ndarray:
    """Position parmi les jours ouvrés (lundi–samedi hors fériés) en partant de la fin du mois :
    0 = dernier jour ouvré du mois, 1 = avant-dernier… ; -1 pour un jour non ouvré.
    Les facturations de fin de mois (export) tombent sur les derniers jours OUVRÉS, pas sur
    le dernier jour calendaire (qui peut être un dimanche)."""
    rang = np.full(len(jours), -1)
    ouvre = np.array([(d.dayofweek < 6) and (d not in feries) for d in jours])
    mois = jours.to_period("M")
    for m in np.unique(mois):
        idx = np.where((mois == m) & ouvre)[0]
        rang[idx] = np.arange(len(idx))[::-1]
    return rang


def _variables_jour(jours: pd.DatetimeIndex, feries: set, ram: set) -> np.ndarray:
    """Variables du profil journalier (effets multiplicatifs, estimés par segment) :
    jour de semaine ; position dans les 5 derniers jours CALENDAIRES du mois (clôture comptable,
    facturations export, même un dimanche) ; position dans les 5 derniers jours OUVRÉS (marché
    local) ; fin de mois tombant un jour non ouvré ; jour férié ; lendemain de fête ; Ramadan."""
    fin_cal = (jours + pd.offsets.MonthEnd(0) - jours).days.to_numpy()
    rang = _rang_ouvre_fin_mois(jours, feries)
    non_ouvre = np.array([(d.dayofweek == 6) or (d in feries) for d in jours])
    X = np.zeros((len(jours), 6 + 5 + 5 + 4))
    for k in range(6):                                   # lundi … samedi (dimanche = référence)
        X[:, k] = jours.dayofweek == k
    for k in range(5):
        X[:, 6 + k] = fin_cal == k
        X[:, 11 + k] = rang == k
    X[:, 16] = (fin_cal <= 1) & non_ouvre
    X[:, 17] = [j in feries for j in jours]
    X[:, 18] = [(j - pd.Timedelta(days=1)) in feries and j not in feries for j in jours]  # lendemain de fête
    X[:, 19] = [j in ram for j in jours]
    return X


@dataclass
class Profil:
    coef: list
    intercept: float
    phi: float                     # surdispersion : Var(jour) ≈ φ · moyenne

    def poids(self, jours: pd.DatetimeIndex, feries: set, ram: set) -> np.ndarray:
        X = _variables_jour(jours, feries, ram)
        return np.exp(np.clip(X @ np.asarray(self.coef) + self.intercept, -20, 5))


def ajuster_profil(quotidien: pd.Series, feries: set, ram: set) -> Profil:
    """quotidien : tonnage par jour (index = jours, 0 inclus)."""
    from sklearn.linear_model import PoissonRegressor

    q = quotidien.clip(lower=0)
    moy_mois = q.groupby(q.index.to_period("M")).transform("mean")
    ok = moy_mois > 0
    r = (q[ok] / moy_mois[ok]).to_numpy()
    X = _variables_jour(q.index[ok], feries, ram)
    m = PoissonRegressor(alpha=1e-3, max_iter=1000).fit(X, r)
    prof = Profil(list(map(float, m.coef_)), float(m.intercept_), 1.0)
    # surdispersion estimée en tonnes : résidus jour / (total du mois réparti selon le profil)
    attendu = repartir(q.groupby(q.index.to_period("M").to_timestamp()).sum(), prof, feries, ram)
    a = attendu.reindex(q.index).fillna(0).to_numpy()
    x = q.to_numpy()
    sel = a > 0
    prof.phi = float(np.sum((x[sel] - a[sel]) ** 2) / max(np.sum(a[sel]), 1e-9))
    return prof


def repartir(mensuel: pd.Series, prof: Profil, feries: set, ram: set) -> pd.Series:
    """Répartit chaque total mensuel (index = début de mois) sur ses jours selon le profil."""
    parts = []
    for m, total in mensuel.items():
        jours = pd.date_range(m, m + pd.offsets.MonthEnd(0), freq="D")
        w = prof.poids(jours, feries, ram)
        parts.append(pd.Series(total * w / w.sum(), index=jours))
    return pd.concat(parts) if parts else pd.Series(dtype=float)


# ──────────────────────────────────────────────────────────────
#  Composante
# ──────────────────────────────────────────────────────────────
@dataclass
class Composante:
    societe: str
    segment: str
    historique: pd.DataFrame          # ds, y, lignes, manquant
    tableau: list                     # comparaison des modèles
    modele: str
    erreur: dict                      # erreur attendue (sélection imbriquée)
    prevision: pd.DataFrame           # ds, yhat, sigma (mensuel)
    profil: Profil
    parts: dict                       # produit -> part
    backtest_mensuel: pd.DataFrame    # ds, reel, prevu (modèle retenu, 18 mois)
    erreur_jour: dict = field(default_factory=dict)
    backtest_imbrique: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ds", "reel", "prevu"]))
    backtest_jour: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ds", "reel", "prevu", "sd"]))


def _selection_imbriquee(resultats: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Pour les plis 2..n : modèle choisi sur les plis précédents (MAE), évalué sur le pli courant."""
    noms = list(resultats)
    n = next(iter(resultats.values())).reels.shape[0]
    reels, prevus = [], []
    for k in range(1, n):
        mae = {nm: np.mean(np.abs(resultats[nm].erreurs[:k])) for nm in noms}
        choix = min(mae, key=mae.get)
        reels.append(resultats[choix].reels[k])
        prevus.append(resultats[choix].prevus[k])
    return np.array(reels), np.array(prevus)


def traiter_composante(df: pd.DataFrame, ev: pd.DataFrame, societe: str, segment: str,
                       progression: Callable | None = None) -> Composante:
    s = D.serie_mensuelle(df, societe, segment)
    y = pd.Series(s["y"].to_numpy(float), index=pd.DatetimeIndex(s["ds"], freq="MS"))
    futur = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=HORIZON_MAX, freq="MS")
    X = C.variables_mensuelles(y.index.append(futur), ev)
    Xp, Xf = X.loc[y.index], X.loc[futur]
    feries, ram = C.jours_feries(ev), C.jours_ramadan(ev)

    plis = plis_possibles(len(y), HORIZON_TEST, PLIS)
    tableau, res = [], {}
    if plis > 0:
        for nom, (f, ref, desc) in MODELES.items():
            if progression:
                progression(f"{societe} · {segment} · {nom}")
            b = backtest(y, X.loc[y.index], f, HORIZON_TEST, plis)
            res[nom] = b
            tableau.append({"modele": nom, "reference": ref, "description": desc,
                            **{k: round(v, 2) for k, v in b.scores.items()}})
        tableau.sort(key=lambda r: r["MAE"])
        retenu = tableau[0]["modele"]
        bt_imb = pd.DataFrame(columns=["ds", "reel", "prevu"])
        if plis >= 2:
            r_, p_ = _selection_imbriquee(res)
            erreur = {**metriques(r_.ravel(), p_.ravel()), "methode": f"sélection imbriquée, {r_.size} mois"}
            bt_imb = pd.DataFrame({"ds": y.index[-r_.size:], "reel": r_.ravel(), "prevu": p_.ravel()})
        else:
            erreur = {**res[retenu].scores, "methode": f"backtest, {res[retenu].reels.size} mois"}
        erreurs = res[retenu].erreurs
        n_bt = plis * HORIZON_TEST
        bt = pd.DataFrame({"ds": y.index[-n_bt:], "reel": res[retenu].reels.ravel(), "prevu": res[retenu].prevus.ravel()})
    else:  # historique trop court : moyenne 12 mois, incertitude = écart-type
        retenu = "Moyenne 12 mois"
        erreurs = np.full((1, HORIZON_TEST), float(y.std()))
        erreur = {"MAE": float("nan"), "WAPE": float("nan"), "RMSE": float("nan"), "biais": float("nan"),
                  "methode": "historique trop court pour une évaluation"}
        bt = pd.DataFrame(columns=["ds", "reel", "prevu"])
        bt_imb = bt.copy()

    yhat = np.maximum(0, np.asarray(MODELES[retenu][0](y, HORIZON_MAX, Xp, Xf), float))
    sigma = sigma_horizon(erreurs, HORIZON_MAX, plafond=float(y.std()) * 1.5)
    prev = pd.DataFrame({"ds": futur, "yhat": yhat, "sigma": sigma})

    # profil journalier (sur l'historique complet de la composante)
    d = df[(df["societe"] == societe) & (df["segment"] == segment)]
    jours = pd.date_range(y.index[0], y.index[-1] + pd.offsets.MonthEnd(0), freq="D")
    quotidien = d.groupby("date")["tonnage_net"].sum().reindex(jours, fill_value=0.0)
    manq = set(s.loc[s["manquant"], "ds"])
    quotidien = quotidien[~quotidien.index.to_period("M").to_timestamp().isin(manq)]
    profil = ajuster_profil(quotidien, feries, ram)

    parts = D.parts_produits(df, societe, segment, jours[-1]).to_dict()
    comp = Composante(societe, segment, s, tableau, retenu, erreur, prev, profil, parts, bt,
                      backtest_imbrique=bt_imb)

    # précision journalière : prévisions mensuelles du backtest réparties avec un profil
    # estimé AVANT la période de test, comparées aux tonnages réels de chaque jour
    if len(bt):
        debut_test = bt["ds"].iloc[0]
        prof_av = ajuster_profil(quotidien[quotidien.index < debut_test], feries, ram)
        prevu_j = repartir(bt.set_index("ds")["prevu"], prof_av, feries, ram)
        uniforme = pd.concat([pd.Series(v / (m + pd.offsets.MonthEnd(0)).day,
                                        index=pd.date_range(m, m + pd.offsets.MonthEnd(0)))
                              for m, v in bt.set_index("ds")["prevu"].items()])
        reel_j = quotidien.reindex(prevu_j.index)
        ok = reel_j.notna()
        sem = lambda x: x[ok].groupby(x[ok].index.to_period("W")).sum()  # noqa: E731
        # couverture empirique des intervalles 95 % sur la période de test
        sig_h = sigma_horizon(erreurs, HORIZON_TEST, plafond=float(y.std()) * 1.5)
        h_idx = np.tile(np.arange(HORIZON_TEST), len(bt) // HORIZON_TEST)
        sig_m = pd.Series(sig_h[h_idx], index=bt["ds"].to_numpy())
        couv_m = float(np.mean(np.abs(bt["reel"] - bt["prevu"]).to_numpy() <= Z95 * sig_m.to_numpy()))
        jours_test = prevu_j.index
        mois_j = jours_test.to_period("M").to_timestamp()
        M_j = bt.set_index("ds")["prevu"].reindex(mois_j).to_numpy()
        s_j = (prevu_j / np.maximum(M_j, 1e-9)).to_numpy()
        sd_j = np.sqrt(s_j**2 * sig_m.reindex(mois_j).to_numpy() ** 2 + prof_av.phi * M_j * s_j * (1 - s_j))
        e_j = np.abs(reel_j.to_numpy() - prevu_j.to_numpy())
        couv_j = float(np.mean(e_j[ok.to_numpy()] <= Z95 * sd_j[ok.to_numpy()]))
        comp.backtest_jour = pd.DataFrame({"ds": prevu_j.index[ok.to_numpy()], "reel": reel_j[ok].to_numpy(),
                                           "prevu": prevu_j[ok].to_numpy(), "sd": sd_j[ok.to_numpy()]})
        comp.erreur_jour = {
            "couverture_mois": couv_m, "couverture_jour": couv_j,
            "jour": metriques(reel_j[ok], prevu_j[ok])["WAPE"],
            "jour_uniforme": metriques(reel_j[ok], uniforme.reindex(prevu_j.index)[ok])["WAPE"],
            "semaine": metriques(sem(reel_j), sem(prevu_j))["WAPE"],
        }
    return comp


# ──────────────────────────────────────────────────────────────
#  Résultats complets
# ──────────────────────────────────────────────────────────────
@dataclass
class Resultats:
    composantes: list[Composante]
    fin_donnees: pd.Timestamp
    fin_prevision: pd.Timestamp
    calendrier: pd.DataFrame

    # --- sélection ---
    def _comp(self, societe: str) -> list[Composante]:
        return self.composantes if societe == D.GROUPE else [c for c in self.composantes if c.societe == societe]

    def societes(self) -> list[str]:
        vus = []
        for c in self.composantes:
            if c.societe not in vus:
                vus.append(c.societe)
        return vus

    def produits(self, societe: str) -> list[str]:
        p = {}
        for c in self._comp(societe):
            for k, v in c.parts.items():
                p[k] = p.get(k, 0) + v * c.historique["y"].tail(12).sum()
        return [k for k, v in sorted(p.items(), key=lambda kv: -kv[1]) if v > 0]

    def segments(self, societe: str) -> list[str]:
        return [c.segment for c in self._comp(societe)]

    # --- historique ---
    def historique_mensuel(self, societe: str, segment: str | None = None) -> pd.DataFrame:
        cs = [c for c in self._comp(societe) if segment in (None, c.segment)]
        idx = cs[0].historique["ds"]
        for c in cs[1:]:
            idx = idx[idx.isin(c.historique["ds"])]
        y = sum(c.historique.set_index("ds").loc[idx, "y"].to_numpy() for c in cs)
        manq = np.any([c.historique.set_index("ds").loc[idx, "manquant"].to_numpy() for c in cs], axis=0)
        return pd.DataFrame({"ds": idx.to_numpy(), "y": y, "manquant": manq})

    # --- prévision ---
    def journalier(self, societe: str, produit: str = D.TOUS, segment: str | None = None) -> pd.DataFrame:
        """Prévision jour par jour sur tout l'horizon : date, yhat, var_mois (par mois), var_bruit, mois, segment."""
        feries, ram = C.jours_feries(self.calendrier), C.jours_ramadan(self.calendrier)
        lignes = []
        for c in self._comp(societe):
            if segment not in (None, c.segment):
                continue
            p = 1.0 if produit == D.TOUS else c.parts.get(produit, 0.0)
            if p <= 0:
                continue
            for _, r in c.prevision.iterrows():
                jours = pd.date_range(r["ds"], r["ds"] + pd.offsets.MonthEnd(0), freq="D")
                w = c.profil.poids(jours, feries, ram)
                s = w / w.sum()
                mu = p * r["yhat"] * s
                lignes.append(pd.DataFrame({
                    "date": jours, "yhat": mu, "s": s, "p": p, "M": r["yhat"], "sigma_mois": r["sigma"],
                    "phi": c.profil.phi, "mois": r["ds"],
                    "composante": f"{c.societe}|{c.segment}", "segment": c.segment,
                }))
        return pd.concat(lignes, ignore_index=True) if lignes else pd.DataFrame()

    def agreger(self, j: pd.DataFrame, debut: pd.Timestamp, fin: pd.Timestamp, freq: str) -> pd.DataFrame:
        """freq : 'D' (jour), 'W' (semaine lundi–dimanche), 'M' (mois), 'T' (total). Intervalle 95 %."""
        j = j[(j["date"] >= debut) & (j["date"] <= fin)].copy()
        if j.empty:
            return pd.DataFrame(columns=["periode", "debut", "fin", "yhat", "bas", "haut"])
        if freq == "T":                       # total de la période
            j["periode"] = pd.Timestamp(debut)
        elif freq == "D":
            j["periode"] = j["date"]
        elif freq == "W":
            j["periode"] = j["date"] - pd.to_timedelta(j["date"].dt.dayofweek, unit="D")
        else:
            j["periode"] = j["date"].dt.to_period("M").dt.to_timestamp()
        # Variance, par composante et par mois touché par la période (q = part du mois couverte) :
        #   incertitude du total mensuel : (p·q)² · σ_mois²
        #   répartition jour à jour      : p · φ · M · q(1−q)   (nulle sur un mois complet : les
        #                                  écarts journaliers se compensent dans le total du mois)
        k = j.groupby(["periode", "composante", "mois"]).agg(
            q=("s", "sum"), p=("p", "first"), M=("M", "first"), sigma=("sigma_mois", "first"), phi=("phi", "first"))
        q = k["q"].clip(0, 1)
        var = (k["p"] * q) ** 2 * k["sigma"] ** 2 + k["p"] * k["phi"] * k["M"] * q * (1 - q)
        var = var.groupby(level=0).sum()
        g = j.groupby("periode").agg(yhat=("yhat", "sum"), debut=("date", "min"), fin=("date", "max"))
        sd = np.sqrt(var.reindex(g.index).fillna(0).clip(lower=0))
        g["bas"] = (g["yhat"] - Z95 * sd).clip(lower=0)
        g["haut"] = g["yhat"] + Z95 * sd
        return g.reset_index()[["periode", "debut", "fin", "yhat", "bas", "haut"]]

    @staticmethod
    def realise(df: pd.DataFrame, societe: str, produit: str, debut, fin, segment: str | None = None) -> float:
        d = df[(df["date"] >= debut) & (df["date"] <= fin)]
        if societe != D.GROUPE:
            d = d[d["societe"] == societe]
        if produit != D.TOUS:
            d = d[d["produit"] == produit]
        if segment:
            d = d[d["segment"] == segment]
        return float(d["tonnage_net"].sum())

    # --- fiabilité agrégée ---
    def fiabilite(self, societe: str) -> dict:
        """Erreur mensuelle de la société / du groupe : somme des composantes, chaque composante étant
        prévue par le modèle choisi SANS connaître la période évaluée (sélection imbriquée)."""
        bts = [c.backtest_imbrique.set_index("ds") for c in self._comp(societe) if len(c.backtest_imbrique)]
        if not bts:
            return {}
        idx = bts[0].index
        for b in bts[1:]:
            idx = idx.intersection(b.index)
        reel = sum(b.loc[idx, "reel"].to_numpy(float) for b in bts)
        prevu = sum(b.loc[idx, "prevu"].to_numpy(float) for b in bts)
        out = {"mensuel": metriques(reel, prevu), "mois": len(idx),
               "serie": pd.DataFrame({"ds": idx, "reel": reel, "prevu": prevu})}
        # précision journalière / hebdomadaire de la somme des composantes
        bj = [c.backtest_jour.set_index("ds") for c in self._comp(societe) if len(c.backtest_jour)]
        if bj:
            jidx = bj[0].index
            for b in bj[1:]:
                jidx = jidx.intersection(b.index)
            r = pd.Series(sum(b.loc[jidx, "reel"].to_numpy(float) for b in bj), index=jidx)
            p = pd.Series(sum(b.loc[jidx, "prevu"].to_numpy(float) for b in bj), index=jidx)
            sd = np.sqrt(sum(b.loc[jidx, "sd"].to_numpy(float) ** 2 for b in bj))
            sem = lambda x: x.groupby(x.index.to_period("W")).sum()  # noqa: E731
            uni = pd.concat([pd.Series(v / len(g), index=g.index) for _, g in p.groupby(p.index.to_period("M"))
                             for v in [g.sum()]])
            out.update({"jour": metriques(r, p)["WAPE"], "semaine": metriques(sem(r), sem(p))["WAPE"],
                        "jour_uniforme": metriques(r, uni.reindex(r.index))["WAPE"],
                        "couverture_jour": float(np.mean(np.abs(r - p).to_numpy() <= Z95 * sd))})
        return out

    # --- sérialisation (cache JSON) ---
    def vers_dict(self) -> dict:
        def comp(c: Composante) -> dict:
            return {
                "societe": c.societe, "segment": c.segment,
                "historique": c.historique.assign(ds=c.historique["ds"].dt.strftime("%Y-%m-%d")).to_dict("list"),
                "tableau": c.tableau, "modele": c.modele, "erreur": c.erreur,
                "prevision": c.prevision.assign(ds=c.prevision["ds"].dt.strftime("%Y-%m-%d")).to_dict("list"),
                "profil": {"coef": c.profil.coef, "intercept": c.profil.intercept, "phi": c.profil.phi},
                "parts": c.parts,
                "backtest_mensuel": c.backtest_mensuel.assign(
                    ds=pd.to_datetime(c.backtest_mensuel["ds"]).dt.strftime("%Y-%m-%d")).to_dict("list"),
                "erreur_jour": c.erreur_jour,
                "backtest_jour": c.backtest_jour.assign(
                    ds=pd.to_datetime(c.backtest_jour["ds"]).dt.strftime("%Y-%m-%d")).to_dict("list"),
                "backtest_imbrique": c.backtest_imbrique.assign(
                    ds=pd.to_datetime(c.backtest_imbrique["ds"]).dt.strftime("%Y-%m-%d")).to_dict("list"),
            }
        return {"composantes": [comp(c) for c in self.composantes],
                "fin_donnees": self.fin_donnees.strftime("%Y-%m-%d"),
                "fin_prevision": self.fin_prevision.strftime("%Y-%m-%d")}

    @classmethod
    def depuis_dict(cls, d: dict, ev: pd.DataFrame) -> "Resultats":
        def df_(x):
            t = pd.DataFrame(x)
            if "ds" in t:
                t["ds"] = pd.to_datetime(t["ds"])
            return t
        cs = [Composante(c["societe"], c["segment"], df_(c["historique"]), c["tableau"], c["modele"], c["erreur"],
                         df_(c["prevision"]), Profil(**c["profil"]), c["parts"], df_(c["backtest_mensuel"]),
                         c["erreur_jour"], df_(c["backtest_imbrique"]), df_(c["backtest_jour"]))
              for c in d["composantes"]]
        return cls(cs, pd.Timestamp(d["fin_donnees"]), pd.Timestamp(d["fin_prevision"]), ev)


def empreinte(df: pd.DataFrame, ev: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    h.update(pd.util.hash_pandas_object(ev.astype(str), index=False).values.tobytes())
    h.update(b"moteur-v3.5")
    return h.hexdigest()[:16]


def calculer(df: pd.DataFrame, ev: pd.DataFrame, progression: Callable | None = None) -> Resultats:
    comps = [traiter_composante(df, ev, s, seg, progression) for s, seg in D.composantes(df)]
    fin_donnees = D.fin_complete(df) + pd.offsets.MonthEnd(0)
    fin_prev = min(fin_donnees + pd.offsets.MonthEnd(HORIZON_MAX), C.couverture(ev))
    return Resultats(comps, fin_donnees, fin_prev, ev)


def sauvegarder(res: Resultats, chemin, cle: str) -> None:
    chemin.write_text(json.dumps({"empreinte": cle, **res.vers_dict()}, ensure_ascii=False), encoding="utf-8")
