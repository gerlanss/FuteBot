"""
Define um novo marco visivel da banca para os relatorios enviados ao usuario.

Nao apaga historico interno do bot. So reposiciona o ponto de partida exibido em
relatorios e resumos publicos/operacionais.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reinicia apenas o historico visivel da banca para o usuario."
    )
    parser.add_argument(
        "--inicio",
        dest="inicio",
        help="ISO datetime para usar como inicio visivel. Default: agora.",
    )
    args = parser.parse_args()

    inicio = args.inicio or datetime.now().astimezone().isoformat()
    db = Database()
    salvo = db.definir_inicio_banca_visivel(inicio)
    print(f"inicio_banca_visivel={salvo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
