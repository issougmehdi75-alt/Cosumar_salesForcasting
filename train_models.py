"""
train_models.py — Entraine les modeles de prevision pour les 4 societes du groupe COSUMAR
et exporte le dossier models/ (manifest.json + un bundle .pkl par societe) attendu par app.py.

A executer une seule fois (ou a chaque nouvel exercice de donnees) AVANT de deployer l'app
Streamlit, car le dossier models/ n'est pas fourni : app.py ne fait que LIRE des modeles
deja entraines, il ne les entraine jamais lui-meme.

Usage :
    python train_models.py
"""
import glob
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from prophet import Prophet
from prophet.serialize import model_to_json

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb

import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# Configuration (identique au notebook)
# ------------------------------------------------------------------
APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
MODELS_DIR = APP_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FREQ = "ME"
N_TRIALS = 40          # identique au notebook original (pipeline_soutenance.ipynb)
TEST_SIZE = 6
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SOCIETES = ["COSUMAR SA", "SUNABEL", "SURAC", "SUTA"]


# ------------------------------------------------------------------
# Chargement / nettoyage des donnees (identique au notebook)
# ------------------------------------------------------------------
def load_data():
    files = glob.glob(str(DATA_DIR / "*.csv"))
    dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Jour calendaire"], format="%m/%d/%y", errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates()
    return df


def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1, denom)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE(%)": mape(y_true, y_pred), "SMAPE(%)": smape(y_true, y_pred)}


def build_features(series, n_lags):
    d = pd.DataFrame({"y": series})
    for lag in range(1, n_lags + 1):
        d[f"lag_{lag}"] = d["y"].shift(lag)
    d["rolling_mean_3"] = d["y"].shift(1).rolling(3).mean()
    d["rolling_mean_6"] = d["y"].shift(1).rolling(6).mean()
    d["rolling_std_3"] = d["y"].shift(1).rolling(3).std()
    d["month"] = d.index.month
    d["quarter"] = d.index.quarter
    d["year"] = d.index.year
    d["time_index"] = np.arange(len(d))
    return d


# ------------------------------------------------------------------
# Pipeline complet pour une societe -> bundle pret pour l'app Streamlit
# ------------------------------------------------------------------
def train_company(df_all, societe, freq=FREQ, n_trials=N_TRIALS, test_size=TEST_SIZE):
    print(f"\n=== {societe} ===")
    data_soc = df_all[df_all["Société"] == societe].copy().set_index("date").sort_index()
    ts = data_soc["Article"].resample(freq).count()
    ts.name = "nb_articles"
    ts = ts.asfreq(freq, fill_value=0)
    period = 12

    n_test = min(test_size, max(3, len(ts) // 5))
    train, test = ts.iloc[:-n_test], ts.iloc[-n_test:]

    results = []

    def add_result(name, pred):
        results.append(evaluate(test, pred, name))

    # ---- Baselines ----
    add_result("Naive", pd.Series([train.iloc[-1]] * n_test, index=test.index))
    w = min(period, len(train))
    add_result("Moyenne_mobile", pd.Series([train.iloc[-w:].mean()] * n_test, index=test.index))
    if len(train) >= period:
        sv = train.iloc[-period:].values
        add_result("Naif_saisonnier", pd.Series([sv[i % period] for i in range(n_test)], index=test.index))

    # ---- SARIMA ----
    def obj_sarima(trial):
        p = trial.suggest_int("p", 0, 3); d_ = trial.suggest_int("d", 0, 1); q = trial.suggest_int("q", 0, 3)
        P = trial.suggest_int("P", 0, 2); D_ = trial.suggest_int("D", 0, 1); Q = trial.suggest_int("Q", 0, 2)
        try:
            m = SARIMAX(train, order=(p, d_, q), seasonal_order=(P, D_, Q, period),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            return np.sqrt(mean_squared_error(test, m.forecast(n_test)))
        except Exception:
            return float("inf")

    study_sarima = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_sarima.optimize(obj_sarima, n_trials=n_trials, show_progress_bar=False)
    bp = study_sarima.best_params
    m_sarima = SARIMAX(train, order=(bp["p"], bp["d"], bp["q"]), seasonal_order=(bp["P"], bp["D"], bp["Q"], period),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    add_result("SARIMA", m_sarima.forecast(n_test))

    # ---- Holt-Winters ----
    def obj_ets(trial):
        trend = trial.suggest_categorical("trend", ["add", "mul", None])
        seasonal = trial.suggest_categorical("seasonal", ["add", "mul", None])
        damped = trial.suggest_categorical("damped_trend", [True, False]) if trend is not None else False
        try:
            m = ExponentialSmoothing(train, trend=trend, seasonal=seasonal,
                                      seasonal_periods=period if seasonal else None,
                                      damped_trend=damped, initialization_method="estimated").fit()
            return np.sqrt(mean_squared_error(test, m.forecast(n_test)))
        except Exception:
            return float("inf")

    study_ets = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_ets.optimize(obj_ets, n_trials=n_trials, show_progress_bar=False)
    bp = study_ets.best_params
    m_ets = ExponentialSmoothing(train, trend=bp["trend"], seasonal=bp["seasonal"],
                                  seasonal_periods=period if bp["seasonal"] else None,
                                  damped_trend=bp.get("damped_trend", False),
                                  initialization_method="estimated").fit()
    add_result("Holt_Winters", m_ets.forecast(n_test))

    # ---- Prophet ----
    p_train = train.reset_index(); p_train.columns = ["ds", "y"]
    p_test = test.reset_index(); p_test.columns = ["ds", "y"]

    def obj_prophet(trial):
        cps = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
        sps = trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True)
        mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
        try:
            m = Prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps, seasonality_mode=mode,
                        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            m.fit(p_train)
            fut = m.make_future_dataframe(periods=n_test, freq=freq)
            fc = m.predict(fut).set_index("ds")["yhat"].reindex(p_test["ds"]).values
            return np.sqrt(mean_squared_error(p_test["y"], fc))
        except Exception:
            return float("inf")

    study_prophet = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_prophet.optimize(obj_prophet, n_trials=n_trials, show_progress_bar=False)
    bp = study_prophet.best_params
    m_prophet = Prophet(changepoint_prior_scale=bp["changepoint_prior_scale"],
                         seasonality_prior_scale=bp["seasonality_prior_scale"],
                         seasonality_mode=bp["seasonality_mode"], yearly_seasonality=True,
                         weekly_seasonality=False, daily_seasonality=False)
    m_prophet.fit(p_train)
    fut = m_prophet.make_future_dataframe(periods=n_test, freq=freq)
    pred_prophet = pd.Series(m_prophet.predict(fut).set_index("ds")["yhat"].reindex(p_test["ds"]).values, index=test.index)
    add_result("Prophet", pred_prophet)

    # ---- ML : RF / XGBoost / LightGBM ----
    n_lags = min(12, max(2, len(train) // 3))
    full_feat = build_features(ts, n_lags=n_lags).dropna()
    feature_cols = [c for c in full_feat.columns if c != "y"]
    X, y = full_feat[feature_cols], full_feat["y"]
    n_test_ml = min(n_test, len(X) - 1)
    X_train, y_train = X.iloc[:-n_test_ml], y.iloc[:-n_test_ml]
    X_test, y_test = X.iloc[-n_test_ml:], y.iloc[-n_test_ml:]

    def obj_rf(trial):
        params = dict(n_estimators=trial.suggest_int("n_estimators", 50, 400),
                       max_depth=trial.suggest_int("max_depth", 2, 12),
                       min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                       min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                       max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                       random_state=RANDOM_STATE)
        m = RandomForestRegressor(**params).fit(X_train, y_train)
        return np.sqrt(mean_squared_error(y_test, m.predict(X_test)))

    study_rf = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(obj_rf, n_trials=n_trials, show_progress_bar=False)
    m_rf = RandomForestRegressor(**study_rf.best_params, random_state=RANDOM_STATE).fit(X_train, y_train)
    add_result("Random_Forest", pd.Series(m_rf.predict(X_test), index=y_test.index))

    def obj_xgb(trial):
        params = dict(n_estimators=trial.suggest_int("n_estimators", 50, 500),
                       max_depth=trial.suggest_int("max_depth", 2, 10),
                       learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                       subsample=trial.suggest_float("subsample", 0.5, 1.0),
                       colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                       reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                       reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                       random_state=RANDOM_STATE, verbosity=0)
        m = xgb.XGBRegressor(**params).fit(X_train, y_train)
        return np.sqrt(mean_squared_error(y_test, m.predict(X_test)))

    study_xgb = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(obj_xgb, n_trials=n_trials, show_progress_bar=False)
    m_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=RANDOM_STATE, verbosity=0).fit(X_train, y_train)
    add_result("XGBoost", pd.Series(m_xgb.predict(X_test), index=y_test.index))

    def obj_lgbm(trial):
        params = dict(n_estimators=trial.suggest_int("n_estimators", 50, 500),
                       max_depth=trial.suggest_int("max_depth", 2, 10),
                       learning_rate=trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                       num_leaves=trial.suggest_int("num_leaves", 7, 100),
                       subsample=trial.suggest_float("subsample", 0.5, 1.0),
                       colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                       reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                       reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
                       random_state=RANDOM_STATE, verbosity=-1)
        m = lgb.LGBMRegressor(**params).fit(X_train, y_train)
        return np.sqrt(mean_squared_error(y_test, m.predict(X_test)))

    study_lgbm = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgbm.optimize(obj_lgbm, n_trials=n_trials, show_progress_bar=False)
    m_lgbm = lgb.LGBMRegressor(**study_lgbm.best_params, random_state=RANDOM_STATE, verbosity=-1).fit(X_train, y_train)
    add_result("LightGBM", pd.Series(m_lgbm.predict(X_test), index=y_test.index))

    # ---- Classement + choix du meilleur modele ----
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = results_df.iloc[0]["model"]
    print(results_df.to_string(index=False))
    print(f">>> Meilleur modele retenu pour {societe} : {best_model_name}")

    # ---- Reentrainement du meilleur modele sur TOUT l'historique ----
    bundle = {"ts": ts, "freq": freq, "model_type": best_model_name, "results_df": results_df}

    if best_model_name == "SARIMA":
        bp = study_sarima.best_params
        mf = SARIMAX(ts, order=(bp["p"], bp["d"], bp["q"]), seasonal_order=(bp["P"], bp["D"], bp["Q"], period),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        bundle["statsmodels_result"] = mf

    elif best_model_name == "Holt_Winters":
        bp = study_ets.best_params
        mf = ExponentialSmoothing(ts, trend=bp["trend"], seasonal=bp["seasonal"],
                                   seasonal_periods=period if bp["seasonal"] else None,
                                   damped_trend=bp.get("damped_trend", False),
                                   initialization_method="estimated").fit()
        bundle["statsmodels_result"] = mf

    elif best_model_name == "Prophet":
        bp = study_prophet.best_params
        full_p = ts.reset_index(); full_p.columns = ["ds", "y"]
        mf = Prophet(changepoint_prior_scale=bp["changepoint_prior_scale"],
                     seasonality_prior_scale=bp["seasonality_prior_scale"],
                     seasonality_mode=bp["seasonality_mode"], yearly_seasonality=True,
                     weekly_seasonality=False, daily_seasonality=False)
        mf.fit(full_p)
        bundle["prophet_json"] = model_to_json(mf)

    elif best_model_name in ("Random_Forest", "XGBoost", "LightGBM"):
        study_map = {"Random_Forest": study_rf, "XGBoost": study_xgb, "LightGBM": study_lgbm}
        model_map = {"Random_Forest": RandomForestRegressor, "XGBoost": xgb.XGBRegressor, "LightGBM": lgb.LGBMRegressor}
        bp = study_map[best_model_name].best_params
        extra = {"random_state": RANDOM_STATE}
        if best_model_name == "XGBoost":
            extra["verbosity"] = 0
        if best_model_name == "LightGBM":
            extra["verbosity"] = -1
        mf = model_map[best_model_name](**bp, **extra)
        full_feat_all = build_features(ts, n_lags=n_lags).dropna()
        Xf, yf = full_feat_all[feature_cols], full_feat_all["y"]
        mf.fit(Xf, yf)
        bundle["ml_model"] = mf
        bundle["feature_cols"] = feature_cols
        bundle["n_lags"] = n_lags

    else:
        bundle["last_value"] = float(ts.iloc[-1])

    return bundle, results_df.iloc[0]["MAPE(%)"]


def main():
    df = load_data()
    manifest = {}

    for societe in SOCIETES:
        bundle, best_mape = train_company(df, societe)
        slug = societe.replace(" ", "_")
        filename = f"{slug}.pkl"
        with open(MODELS_DIR / filename, "wb") as f:
            pickle.dump(bundle, f)

        manifest[societe] = {
            "file": filename,
            "model_type": bundle["model_type"],
            "mape": float(best_mape),
            "last_date": bundle["ts"].index[-1].strftime("%d/%m/%Y"),
        }

    with open(MODELS_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\nTermine. Dossier models/ genere :")
    for p in sorted(MODELS_DIR.glob("*")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
