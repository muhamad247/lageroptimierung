"""Artikel, Nachfrage und die klassischen Formeln (EOQ, Sicherheitsbestand).

Zeiteinheit ist der Tag. Kosten:
- Bestellkosten K pro Bestellung (Bearbeitung, Transport)
- Lagerkosten: Jahreszins h auf den Einstandspreis c, pro Stück und Tag also h·c/365
- Fehlmengenkosten p pro Stück, das nicht sofort geliefert werden kann
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Item:
    name: str = "Artikel"
    mean_demand: float = 20.0  # Stück pro Tag
    dispersion: float | None = None  # None = Poisson, sonst negative Binomialverteilung mit Var = μ + μ²/k
    lead_time: int = 5  # Tage
    lead_time_spread: int = 0  # Lieferzeit gleichverteilt in lead_time ± spread
    unit_cost: float = 20.0
    holding_rate: float = 0.25  # pro Jahr
    order_cost: float = 50.0
    shortage_cost: float = 5.0  # pro fehlendem Stück

    @property
    def demand_variance(self) -> float:
        mu = self.mean_demand
        return mu if self.dispersion is None else mu + mu ** 2 / self.dispersion

    @property
    def lead_time_variance(self) -> float:
        n = 2 * self.lead_time_spread + 1  # diskrete Gleichverteilung
        return (n ** 2 - 1) / 12

    @property
    def holding_cost_per_day(self) -> float:
        return self.holding_rate * self.unit_cost / 365

    def sample_demand(self, days: int, rng: np.random.Generator) -> np.ndarray:
        if self.dispersion is None:
            return rng.poisson(self.mean_demand, days)
        k = self.dispersion
        return rng.negative_binomial(k, k / (k + self.mean_demand), days)

    def sample_lead_times(self, n: int, rng: np.random.Generator) -> np.ndarray:
        spread = self.lead_time_spread
        return self.lead_time + rng.integers(-spread, spread + 1, n)


def eoq(item: Item) -> float:
    """Wirtschaftliche Bestellmenge: sqrt(2·D·K / H) mit Jahresbedarf D und Lagerkosten H pro Stück und Jahr."""
    annual_demand = item.mean_demand * 365
    return float(np.sqrt(2 * annual_demand * item.order_cost / (item.holding_rate * item.unit_cost)))


def lead_time_demand_sd(item: Item) -> float:
    """σ der Nachfrage während der Lieferzeit, mit Unsicherheit in Nachfrage und Lieferzeit."""
    return float(np.sqrt(item.lead_time * item.demand_variance + item.mean_demand ** 2 * item.lead_time_variance))


def reorder_point_normal(item: Item, service_level: float) -> tuple[int, float]:
    """Meldebestand s = μ·L + z·σ_L unter Normalverteilungsannahme. Gibt (s, Sicherheitsbestand) zurück."""
    safety = norm.ppf(service_level) * lead_time_demand_sd(item)
    return int(np.ceil(item.mean_demand * item.lead_time + safety)), float(safety)


def reorder_point_empirical(item: Item, service_level: float, samples: int = 200_000, seed: int = 0) -> int:
    """Meldebestand als Quantil der simulierten Nachfrage während der Lieferzeit.

    Braucht keine Normalverteilungsannahme und funktioniert deshalb auch bei
    sporadischer Nachfrage, wo die Normalverteilung zu wenig Sicherheitsbestand liefert.
    """
    rng = np.random.default_rng(seed)
    lead_times = item.sample_lead_times(samples, rng)
    total = np.zeros(samples)
    for L in np.unique(lead_times):
        mask = lead_times == L
        total[mask] = item.sample_demand(int(mask.sum()) * int(L), rng).reshape(-1, int(L)).sum(axis=1)
    return int(np.quantile(total, service_level, method="higher"))
