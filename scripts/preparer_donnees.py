"""Extractions SAP brutes → data/ventes_journalieres.csv (agrégats journaliers anonymes).

Entrée  : donnees_brutes/*.csv — extractions ligne par ligne, CONFIDENTIELLES
          (numéros de clients) ; dossier exclu par .gitignore, ne jamais publier.
Sortie  : data/ventes_journalieres.csv — une ligne par jour × société × segment × produit,
          sans aucune information client.

La même conversion est disponible dans l'application (page « Mettre à jour les données »).

Usage : python scripts/preparer_donnees.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prevision.donnees import FICHIER, agreger_extraction, exporter  # noqa: E402

BRUTES = Path(__file__).resolve().parent.parent / "donnees_brutes"


def main() -> None:
    fichiers = sorted(BRUTES.glob("*.csv"))
    if not fichiers:
        raise SystemExit(f"Aucun CSV dans {BRUTES}")
    agg, r = agreger_extraction(fichiers)
    FICHIER.write_bytes(exporter(agg))
    print(f"{r['lignes_lues']} lignes lues ({r['lignes_invalides']} invalides) · "
          f"{r['debut']:%d/%m/%Y} → {r['fin']:%d/%m/%Y} · {len(agg)} lignes agrégées → data/{FICHIER.name}")


if __name__ == "__main__":
    main()
