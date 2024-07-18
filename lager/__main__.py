"""python -m lager  →  Meldebestand-Methoden vergleichen, Kostenoptimum suchen, ABC/XYZ-Analyse."""

import sys

import pandas as pd

from .compare import EXAMPLES, compare_methods
from .model import eoq, reorder_point_normal
from .portfolio import classify, matrix, sample_assortment
from .simulate import Scenario, optimize_sq


def main() -> int:
    pd.set_option("display.width", 160, "display.float_format", "{:.3f}".format)

    print("Meldebestand für 95 % Servicegrad (α, Anteil Lieferzyklen ohne Fehlmenge), 30 Jahre simuliert\n")
    print(compare_methods().to_string(index=False))

    item = EXAMPLES[0]
    Q, (s, _) = round(eoq(item)), reorder_point_normal(item, 0.95)
    print(f"\nKostenoptimum für '{item.name}' (Raster um s={s}, Q={Q})")
    result = optimize_sq(Scenario.draw(item, days=365 * 10, seed=4), range(s - 30, s + 41, 5), range(Q - 150, Q + 251, 25))
    print(result.head(5)[["s", "Q", "kosten", "csl", "fill_rate", "bestand"]].to_string(index=False))

    print("\nABC/XYZ-Matrix (60 Artikel, Anzahl / Anteil am Verbrauchswert)")
    print(matrix(classify(sample_assortment())).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
