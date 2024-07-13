"""Vergleich der drei Wege zum Meldebestand für typische Artikel."""

from __future__ import annotations

import pandas as pd

from .model import Item, eoq, reorder_point_daily_review, reorder_point_normal
from .simulate import SQ, Scenario, calibrate_reorder_point, simulate

EXAMPLES = [
    Item("Schnelldreher"),
    Item("Lieferzeit schwankt", lead_time_spread=3),
    Item("Ersatzteil sporadisch", mean_demand=0.8, dispersion=0.15, lead_time=10, unit_cost=120,
         order_cost=40, shortage_cost=60),
]


def compare_methods(items: list[Item] = EXAMPLES, target: float = 0.95, years: int = 30, seed: int = 11) -> pd.DataFrame:
    """Simulierter Servicegrad und Kosten für Lehrbuchformel, korrigierte Formel und Kalibrierung.

    Spalte 'neue_daten': Servicegrad des kalibrierten s auf frisch gezogenen
    Zufallszahlen. Zeigt, ob die Kalibrierung nur auf ihr Szenario passt.
    """
    rows = []
    for item in items:
        Q = max(1, round(eoq(item)))
        scenario = Scenario.draw(item, days=365 * years, seed=seed)
        fresh = Scenario.draw(item, days=365 * years, seed=seed + 1000)
        candidates = {
            "Lehrbuch (laufende Prüfung)": reorder_point_normal(item, target)[0],
            "mit Undershoot-Korrektur": reorder_point_daily_review(item, target),
            "per Simulation kalibriert": calibrate_reorder_point(scenario, Q, target, start=int(item.mean_demand * item.lead_time)),
        }
        for method, s in candidates.items():
            r = simulate(scenario, SQ(s, Q))
            rows.append({
                "artikel": item.name, "methode": method, "s": s, "Q": Q,
                "servicegrad": r.cycle_service_level, "fill_rate": r.fill_rate, "kosten_jahr": r.cost_per_year,
                "neue_daten": simulate(fresh, SQ(s, Q)).cycle_service_level,
            })
    return pd.DataFrame(rows)
