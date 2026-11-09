"""Précalcule les prévisions pour les données du dépôt → results/previsions.json.

L'application charge ce fichier au démarrage (ouverture instantanée). Si les données ou
le calendrier changent (import depuis l'application, modification du calendrier), elle
recalcule elle-même. Relancez ce script après avoir mis à jour data/ dans le dépôt.

Usage : python scripts/calculer_previsions.py      (~20 s)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from prevision import calendrier as C, donnees as D, moteur as M  # noqa: E402

warnings.filterwarnings("ignore")
SORTIE = RACINE / "results" / "previsions.json"


def main() -> None:
    df, ev = D.charger(), C.charger_evenements()
    t0 = time.time()
    res = M.calculer(df, ev, progression=lambda m: print("  ·", m))
    SORTIE.parent.mkdir(exist_ok=True)
    M.sauvegarder(res, SORTIE, M.empreinte(df, ev))
    print(f"\nTerminé en {time.time() - t0:.0f} s → {SORTIE.relative_to(RACINE)}")
    for c in res.composantes:
        print(f"  {c.societe:<11} {c.segment:<7} {c.modele:<30} erreur attendue {c.erreur['WAPE']:.1f} %")


if __name__ == "__main__":
    main()
