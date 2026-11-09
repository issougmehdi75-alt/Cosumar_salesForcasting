"""Modèles mensuels et évaluation hors-échantillon.

Chaque modèle a la signature  f(y, h, X_passe, X_futur) -> np.ndarray  où
    y         série mensuelle observée (index = début de mois)
    h         nombre de mois à prévoir
    X_passe   variables calendaires des mois observés   (mêmes index que y)
    X_futur   variables calendaires des h mois à prévoir

Les variables calendaires (Ramadan, mois pré-Ramadan, jours ouvrés, Aïd al-Adha) sont
connues à l'avance : les utiliser n'introduit aucune fuite d'information.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

SAISON = 12
EXOG = ["ramadan", "pre_ramadan", "jours_ouvres"]


def metriques(reel, prevu) -> dict[str, float]:
    reel, prevu = np.asarray(reel, float), np.asarray(prevu, float)
    e = reel - prevu
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))),
            "WAPE": float(np.sum(np.abs(e)) / max(np.sum(np.abs(reel)), 1e-9) * 100),
            "biais": float(np.sum(-e) / max(np.sum(np.abs(reel)), 1e-9) * 100)}


# ──────────────────────────────────────────────────────────────
#  Références
# ──────────────────────────────────────────────────────────────
def naif(y, h, Xp=None, Xf=None):
    return np.repeat(float(y.iloc[-1]), h)


def naif_saisonnier(y, h, Xp=None, Xf=None):
    der = y.iloc[-SAISON:].to_numpy(float)
    return np.array([der[i % SAISON] for i in range(h)])


def moyenne_12(y, h, Xp=None, Xf=None):
    return np.repeat(float(y.iloc[-SAISON:].mean()), h)


# ──────────────────────────────────────────────────────────────
#  Statistiques
# ──────────────────────────────────────────────────────────────
_GRILLE = [((1, 0, 0), (0, 0, 0)), ((1, 0, 1), (0, 0, 0)), ((0, 1, 1), (0, 0, 0)), ((1, 1, 1), (0, 0, 0)),
           ((1, 0, 0), (1, 0, 0)), ((0, 1, 1), (1, 0, 0)), ((1, 0, 1), (1, 0, 0))]


def _sarimax(y, h, Xp, Xf, exog: bool):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    meilleur, aic = None, np.inf
    ex_p = Xp[EXOG].to_numpy(float) if exog else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order, sorder in _GRILLE:
            try:
                r = SARIMAX(y.to_numpy(float), exog=ex_p, order=order, seasonal_order=(*sorder, SAISON),
                            trend="c" if order[1] == 0 else None,
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            except Exception:
                continue
            if np.isfinite(r.aic) and r.aic < aic:
                meilleur, aic = r, r.aic
        if meilleur is None:
            return moyenne_12(y, h)
        return meilleur.forecast(h, exog=Xf[EXOG].to_numpy(float) if exog else None)


def sarima(y, h, Xp=None, Xf=None):
    return _sarimax(y, h, Xp, Xf, exog=False)


def sarimax_calendrier(y, h, Xp, Xf):
    return _sarimax(y, h, Xp, Xf, exog=True)


def holt_winters(y, h, Xp=None, Xf=None):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    meilleur, aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tendance, saison, amorti in [(None, None, False), ("add", None, True),
                                         (None, "add", False), ("add", "add", True)]:
            if saison and len(y) < 2 * SAISON + 2:
                continue
            try:
                r = ExponentialSmoothing(y.to_numpy(float), trend=tendance, seasonal=saison,
                                         seasonal_periods=SAISON if saison else None, damped_trend=amorti,
                                         initialization_method="estimated").fit()
            except Exception:
                continue
            if np.isfinite(r.aic) and r.aic < aic:
                meilleur, aic = r, r.aic
    return meilleur.forecast(h) if meilleur is not None else moyenne_12(y, h)


# ──────────────────────────────────────────────────────────────
#  Niveau 12 mois × effets calendaires
# ──────────────────────────────────────────────────────────────
def moyenne_12_calendrier(y, h, Xp, Xf):
    """Prévision = niveau (moyenne des 12 derniers mois) × coefficient calendaire.

    Le coefficient r_t = y_t / moyenne(y_{t-12..t-1}) est régressé (Ridge) sur les
    variables calendaires du mois. C'est la référence « Moyenne 12 mois » enrichie de
    ce qui est connu à l'avance : Ramadan, mois pré-Ramadan, jours ouvrés.
    """
    from sklearn.linear_model import RidgeCV

    v = y.to_numpy(float)
    niveaux = np.array([v[i - SAISON:i].mean() for i in range(SAISON, len(v))])
    r = v[SAISON:] / np.maximum(niveaux, 1e-9)
    X = Xp[EXOG].to_numpy(float)[SAISON:]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    m = RidgeCV(alphas=np.logspace(-1, 3, 25)).fit((X - mu) / sd, r)
    passe, preds = list(v), []
    for i in range(h):
        niveau = np.mean(passe[-SAISON:])
        p = max(0.0, niveau * float(m.predict(((Xf[EXOG].to_numpy(float)[i] - mu) / sd)[None])[0]))
        preds.append(p)
        passe.append(p)
    return np.array(preds)


# ──────────────────────────────────────────────────────────────
#  Machine learning (récursif)
# ──────────────────────────────────────────────────────────────
LAGS = (1, 2, 3, 12)


def _features(passe: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.concatenate([[passe[-k] for k in LAGS], [passe[-3:].mean(), passe[-SAISON:].mean()], x])


def _ml(fabrique: Callable) -> Callable:
    def prevoir(y, h, Xp, Xf):
        v = y.to_numpy(float)
        ex_p, ex_f = Xp[EXOG].to_numpy(float), Xf[EXOG].to_numpy(float)
        X = np.array([_features(v[:i], ex_p[i]) for i in range(SAISON, len(v))])
        modele = fabrique().fit(X, v[SAISON:])
        passe, preds = list(v), []
        for i in range(h):
            p = max(0.0, float(modele.predict(_features(np.asarray(passe), ex_f[i])[None])[0]))
            preds.append(p)
            passe.append(p)
        return np.array(preds)
    return prevoir


def _ridge():
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))


def _foret():
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.6, random_state=42, n_jobs=1)


ridge = _ml(_ridge)
foret = _ml(_foret)


def combinaison(y, h, Xp, Xf):
    """Moyenne simple, liste fixée à l'avance (pas choisie d'après les résultats)."""
    return np.mean([moyenne_12_calendrier(y, h, Xp, Xf), sarimax_calendrier(y, h, Xp, Xf), ridge(y, h, Xp, Xf)], axis=0)


# nom : (fonction, est_reference, description)
MODELES: dict[str, tuple[Callable, bool, str]] = {
    "Naïf (mois précédent)": (naif, True, "Répète le dernier mois observé."),
    "Naïf saisonnier (M-12)": (naif_saisonnier, True, "Répète le même mois de l'année précédente."),
    "Moyenne 12 mois": (moyenne_12, True, "Moyenne des 12 derniers mois."),
    "SARIMA": (sarima, False, "ARIMA saisonnier, ordre choisi par AIC sur l'entraînement."),
    "SARIMAX + calendrier": (sarimax_calendrier, False, "SARIMA avec Ramadan, pré-Ramadan et jours ouvrés."),
    "Holt-Winters": (holt_winters, False, "Lissage exponentiel, configuration choisie par AIC."),
    "Moyenne 12 mois + calendrier": (moyenne_12_calendrier, False,
                                     "Niveau des 12 derniers mois × effet Ramadan / pré-Ramadan / jours ouvrés."),
    "Ridge + calendrier": (ridge, False, "Régression régularisée sur retards et calendrier, récursive."),
    "Random Forest + calendrier": (foret, False, "Forêt aléatoire sur retards et calendrier, récursive."),
    "Combinaison": (combinaison, False, "Moyenne de Moyenne 12 mois + calendrier, SARIMAX et Ridge."),
}


# ──────────────────────────────────────────────────────────────
#  Backtest
# ──────────────────────────────────────────────────────────────
@dataclass
class Backtest:
    reels: np.ndarray          # (plis, horizon)
    prevus: np.ndarray
    scores: dict = field(default_factory=dict)

    @property
    def erreurs(self) -> np.ndarray:
        return self.reels - self.prevus


def backtest(y: pd.Series, X: pd.DataFrame, fonction: Callable, horizon: int, plis: int) -> Backtest:
    reels, prevus = [], []
    for k in range(plis, 0, -1):
        o = len(y) - k * horizon
        reels.append(y.iloc[o:o + horizon].to_numpy(float))
        prevus.append(np.asarray(fonction(y.iloc[:o], horizon, X.iloc[:o], X.iloc[o:o + horizon]), float))
    r, p = np.array(reels), np.array(prevus)
    return Backtest(r, p, metriques(r.ravel(), p.ravel()))


def plis_possibles(n: int, horizon: int, voulu: int = 3, min_train: int = 2 * SAISON + 5) -> int:
    return int(max(0, min(voulu, (n - min_train) // horizon)))


def sigma_horizon(erreurs: np.ndarray, h_max: int, plafond: float) -> np.ndarray:
    """σ(h) = c·√h (c estimé sur les erreurs du backtest), plafonné."""
    H = erreurs.shape[1]
    c = np.sqrt(np.sum(erreurs**2) / (erreurs.shape[0] * np.arange(1, H + 1).sum()))
    return np.minimum(c * np.sqrt(np.arange(1, h_max + 1)), plafond)
