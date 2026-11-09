# Prévision du volume d'articles — COSUMAR SA / SUNABEL / SURAC / SUTA

## Rapport de soutenance de stage — Analyse de séries temporelles

Ce dossier contient l'analyse complète des ventes 2017-2020 des quatre entités du groupe sucrier,
avec sélection automatique du meilleur modèle de prévision pour chacune.

## Contenu

```
soutenance_stage/
├── data/                         10 fichiers CSV bruts (fournis)
├── notebooks/
│   └── pipeline_soutenance.ipynb  <- NOTEBOOK PRINCIPAL, déjà exécuté (voir résultats ci-dessous)
├── outputs/
│   ├── 00_vue_ensemble_4_societes.png
│   ├── {SOCIETE}_01_eda.png                  (par société : distribution, saisonnalité)
│   ├── {SOCIETE}_02_decomposition.png        (par société : STL tendance/saisonnalité/résidu)
│   ├── {SOCIETE}_03_acf_pacf.png             (par société : autocorrélations)
│   ├── {SOCIETE}_04_comparaison_modeles.png  (par société : tous les modèles superposés)
│   ├── {SOCIETE}_05_prevision_finale.png     (par société : prévision à 6 mois + IC 95%)
│   ├── {SOCIETE}_resultats_modeles.csv       (par société : classement complet des modèles)
│   ├── {SOCIETE}_prevision_finale.csv        (par société : chiffres de prévision)
│   ├── 99_synthese_comparative.png
│   ├── 99_previsions_4_societes.png
│   └── synthese_comparative_4_societes.csv   <- résumé exécutif des 4 sociétés
├── requirements.txt
└── README.md
```

## Résultat de synthèse (meilleur modèle par société)

| Société    | Meilleur modèle | RMSE (test) | MAPE (%) | Série stationnaire (ADF) |
|------------|------------------|-------------|----------|---------------------------|
| COSUMAR SA | SARIMA           | 259.2       | 4.4 %    | Oui                        |
| SUNABEL    | SARIMA           | 141.9       | 17.6 %   | Oui                        |
| SURAC      | XGBoost          | 80.0        | 13.4 %   | Non (série irrégulière)    |
| SUTA       | XGBoost          | 106.0       | 6.3 %    | Oui                        |

*(Détail complet des métriques par modèle dans `outputs/{SOCIETE}_resultats_modeles.csv`)*

## Points méthodologiques clés (pour la soutenance)

- **Nettoyage** : la colonne `Mois/Année` a des formats incohérents selon les fichiers sources
  (`4/2017` vs `201909`) — la période est donc reconstruite exclusivement à partir de `Jour calendaire`
  (format homogène `M/J/AA`), plus fiable.
- **Cible** : nombre de lignes d'articles vendus par mois et par société (comptage), agrégé sur `Article`.
- **Stationnarité** : COSUMAR SA, SUNABEL et SUTA sont stationnaires (test ADF, p < 0.05) ; SURAC ne l'est
  pas, ce qui explique pourquoi un modèle SARIMA n'a pas été retenu pour cette dernière (série irrégulière,
  avec des mois à zéro article).
- **8 modèles comparés** pour chaque société : Naïf, Moyenne mobile, Naïf saisonnier, SARIMA, Holt-Winters,
  Prophet, Random Forest, XGBoost, LightGBM — tous les modèles avancés optimisés via **Optuna** (40 essais
  chacun, TPE sampler).
- **Sélection** : le modèle avec le RMSE le plus bas sur le jeu de test (6 derniers mois) est retenu et
  réentraîné sur l'historique complet pour produire la prévision finale à 6 mois.

## Reproduire l'analyse

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/pipeline_soutenance.ipynb
```

Pour ajuster la granularité ou l'horizon, modifier la cellule "1. Configuration générale" en tête du notebook
(`FREQ`, `HORIZON`, `N_TRIALS`, `TEST_SIZE`).
