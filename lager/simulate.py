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


def simulate(scenario: Scenario, policy, warmup: int = 60) -> Result:
    item, demand = scenario.item, scenario.demand
    days = len(demand)
    on_hand = float(policy.s + policy.Q if isinstance(policy, SQ) else policy.S)
    backorder = 0.0
    arrivals = np.zeros(days + int(scenario.lead_times.max()) + 2)
    on_order = 0.0
    unmet = np.zeros(days)
    rows, orders = [], []

    for day in range(days):
        received = arrivals[day]
        on_hand += received
        on_order -= received
        served_backlog = min(backorder, on_hand)
        backorder -= served_backlog
        on_hand -= served_backlog

        d = demand[day]
        served = min(d, on_hand)
        on_hand -= served
        unmet[day] = d - served
        backorder += d - served

        qty = policy.order(day, on_hand + on_order - backorder)
        if qty > 0:
            lead = int(scenario.lead_times[len(orders)])
            arrivals[day + lead + 1] += qty  # bestellt am Abend, verfügbar am Morgen nach L Tagen
            on_order += qty
            orders.append((day, day + lead + 1))
        rows.append((day, on_hand, backorder, qty, d, served))

    df = pd.DataFrame(rows, columns=["tag", "lagerbestand", "rueckstand", "bestellung", "nachfrage", "bedient"])
    measured = df[df["tag"] >= warmup]
    years = len(measured) / 365

    # Lieferzyklus = Zeit zwischen zwei Wareneingängen. Fehlmengen entstehen immer kurz vor dem nächsten Eingang.
    arrival_days = sorted(a for _, a in orders if a <= days)
    cycles = [(a, b) for a, b in zip(arrival_days, arrival_days[1:]) if a >= warmup]
    stockout = [unmet[a:b].sum() > 0 for a, b in cycles]
    holding = measured["lagerbestand"].sum() * item.holding_cost_per_day
    ordering = (measured["bestellung"] > 0).sum() * item.order_cost
    shortage = (measured["nachfrage"] - measured["bedient"]).sum() * item.shortage_cost

    return Result(
        days=df,
        cycle_service_level=1 - float(np.mean(stockout)) if stockout else float("nan"),
        fill_rate=float(measured["bedient"].sum() / max(measured["nachfrage"].sum(), 1)),
        avg_on_hand=float(measured["lagerbestand"].mean()),
        orders_per_year=float((measured["bestellung"] > 0).sum() / years),
        cost_per_year=float((holding + ordering + shortage) / years),
        cost_breakdown={"Lager": holding / years, "Bestellung": ordering / years, "Fehlmenge": shortage / years},
    )


def optimize_sq(scenario: Scenario, s_values, q_values) -> pd.DataFrame:
    """Kosten für alle Kombinationen von s und Q, sortiert nach Jahreskosten."""
    rows = []
    for s in s_values:
        for q in q_values:
            r = simulate(scenario, SQ(int(s), int(q)))
            rows.append({"s": int(s), "Q": int(q), "kosten": r.cost_per_year, "csl": r.cycle_service_level,
                         "fill_rate": r.fill_rate, "bestand": r.avg_on_hand, **{f"kosten_{k.lower()}": v for k, v in r.cost_breakdown.items()}})
    return pd.DataFrame(rows).sort_values("kosten").reset_index(drop=True)


def calibrate_reorder_point(scenario: Scenario, Q: int, target: float, start: int = 0, limit: int = 100_000) -> int:
    """Kleinster Meldebestand, mit dem die Simulation den Ziel-Servicegrad erreicht.

    Braucht keine Verteilungsannahme und berücksichtigt alles, was die Formeln
    vereinfachen: Undershoot, schiefe Nachfrage, schwankende Lieferzeiten.
    """
    s = max(0, start)
    while s < limit:
        if simulate(scenario, SQ(s, Q)).cycle_service_level >= target:
            return s
        s += 1
    raise RuntimeError("Ziel-Servicegrad nicht erreichbar")
