"""Tagesgenaue Lagersimulation für (s, Q)- und (R, S)-Politik mit Rückständen.

(s, Q): Fällt die Bestandsposition auf den Meldebestand s oder darunter, wird Q bestellt.
(R, S): Alle R Tage wird auf den Bestellbestand S aufgefüllt.

Bestandsposition = Lagerbestand + offene Bestellungen − Rückstände.

Gemeinsame Zufallszahlen: Nachfrage und Lieferzeiten werden einmal aus dem
Seed gezogen und für jede Politik wiederverwendet. Unterschiede in den
Kosten kommen dann von der Politik und nicht vom Zufall, was den Vergleich
und die Optimierung deutlich stabiler macht.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import Item


@dataclass(frozen=True)
class SQ:
    s: int
    Q: int

    def order(self, day: int, position: float) -> int:
        n = 0
        while position + n * self.Q <= self.s:
            n += 1
        return n * self.Q


@dataclass(frozen=True)
class RS:
    R: int
    S: int

    def order(self, day: int, position: float) -> int:
        return max(0, int(self.S - position)) if day % self.R == 0 else 0


@dataclass
class Result:
    days: pd.DataFrame
    cycle_service_level: float  # Anteil der Lieferzyklen ohne Fehlmenge (α-Servicegrad)
    fill_rate: float  # Anteil der Nachfrage, die sofort aus dem Lager bedient wurde
    avg_on_hand: float
    orders_per_year: float
    cost_per_year: float
    cost_breakdown: dict[str, float]


@dataclass
class Scenario:
    """Fest gezogene Nachfrage und Lieferzeiten (gemeinsame Zufallszahlen)."""
    item: Item
    demand: np.ndarray
    lead_times: np.ndarray

    @classmethod
    def draw(cls, item: Item, days: int = 3650, seed: int = 0) -> Scenario:
        rng = np.random.default_rng(seed)
        return cls(item, item.sample_demand(days, rng), item.sample_lead_times(days, rng))
