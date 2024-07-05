"""ABC/XYZ-Analyse für ein Artikelsortiment.

ABC nach Verbrauchswert (Jahresmenge × Preis): A = die Artikel, die zusammen
80 % des Werts ausmachen, B = die nächsten 15 %, C = der Rest.
XYZ nach Schwankung der Wochennachfrage (Variationskoeffizient):
X < 0,5 (gut planbar), Y bis 1,0, Z > 1,0 (sporadisch).

Daraus folgt ein Ziel-Servicegrad pro Feld: Bei teuren, gut planbaren
A-Artikeln lohnt ein hoher Servicegrad mit wenig Sicherheitsbestand. Bei
billigen, sporadischen C-Z-Artikeln ist ein niedrigerer Servicegrad oft wirtschaftlicher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_LEVELS = {
    "AX": 0.98, "AY": 0.97, "AZ": 0.95,
    "BX": 0.97, "BY": 0.95, "BZ": 0.92,
    "CX": 0.95, "CY": 0.92, "CZ": 0.90,
}


def sample_assortment(n: int = 60, weeks: int = 52, seed: int = 3) -> pd.DataFrame:
    """Wochennachfrage für ein Sortiment: wenige Renner, viele Langsamdreher."""
    rng = np.random.default_rng(seed)
    mean = rng.lognormal(np.log(15), 1.3, n)
    price = rng.lognormal(np.log(25), 1.0, n).round(2)
    dispersion = rng.choice([50.0, 5.0, 0.4, 0.1], n, p=[0.35, 0.3, 0.2, 0.15])
    demand = rng.negative_binomial(dispersion[:, None], dispersion[:, None] / (dispersion[:, None] + mean[:, None]), (n, weeks))
    return pd.DataFrame({
        "artikel": [f"A-{i + 1:03d}" for i in range(n)],
        "preis": price,
        **{f"w{w + 1:02d}": demand[:, w] for w in range(weeks)},
    })


def classify(assortment: pd.DataFrame) -> pd.DataFrame:
    weekly = assortment.filter(regex=r"^w\d+$")
    out = assortment[["artikel", "preis"]].copy()
    out["jahresmenge"] = weekly.sum(axis=1)
    out["verbrauchswert"] = out["jahresmenge"] * out["preis"]
    mean, std = weekly.mean(axis=1), weekly.std(axis=1, ddof=1)
    out["variationskoeffizient"] = np.where(mean > 0, std / mean.where(mean > 0, 1), np.inf)

    out = out.sort_values("verbrauchswert", ascending=False).reset_index(drop=True)
    share = out["verbrauchswert"].cumsum() / out["verbrauchswert"].sum()
    previous = share.shift(fill_value=0)  # Anteil vor diesem Artikel: so landet der Artikel, der die 80 % überschreitet, noch in A
    out["abc"] = np.select([previous < 0.80, previous < 0.95], ["A", "B"], default="C")
    out["xyz"] = np.select([out["variationskoeffizient"] < 0.5, out["variationskoeffizient"] <= 1.0], ["X", "Y"], default="Z")
    out["klasse"] = out["abc"] + out["xyz"]
    out["ziel_servicegrad"] = out["klasse"].map(SERVICE_LEVELS)
    return out


def matrix(classified: pd.DataFrame) -> pd.DataFrame:
    """Anzahl Artikel und Wertanteil je ABC/XYZ-Feld."""
    total = classified["verbrauchswert"].sum()
    count = classified.pivot_table(index="abc", columns="xyz", values="artikel", aggfunc="count", fill_value=0)
    value = classified.pivot_table(index="abc", columns="xyz", values="verbrauchswert", aggfunc="sum", fill_value=0) / total
    return count.astype(str) + " Art. / " + (value * 100).round(1).astype(str) + " %"
