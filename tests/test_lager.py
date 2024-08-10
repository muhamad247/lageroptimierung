from pathlib import Path

import numpy as np
import pytest

from lager.compare import EXAMPLES, compare_methods
from lager.model import Item, eoq, expected_undershoot, lead_time_demand_sd, reorder_point_empirical, reorder_point_normal
from lager.portfolio import classify, sample_assortment
from lager.simulate import RS, SQ, Scenario, optimize_sq, simulate

ROOT = Path(__file__).resolve().parent.parent


def test_eoq_formula():
    item = Item(mean_demand=10, unit_cost=20, holding_rate=0.25, order_cost=50)
    assert eoq(item) == pytest.approx(np.sqrt(2 * 3650 * 50 / 5))


def test_lead_time_variance_increases_safety_stock():
    fixed, varying = Item(), Item(lead_time_spread=3)
    assert lead_time_demand_sd(varying) > lead_time_demand_sd(fixed)
    assert reorder_point_normal(varying, 0.95)[0] > reorder_point_normal(fixed, 0.95)[0]


def test_simulation_is_reproducible():
    sc = Scenario.draw(Item(), days=500, seed=3)
    a, b = simulate(sc, SQ(120, 380)), simulate(sc, SQ(120, 380))
    assert a.cost_per_year == b.cost_per_year


def test_inventory_balance():
    """Nettobestand am Ende = Anfangsbestand + eingegangene Ware − gesamte Nachfrage."""
    days, lead = 2000, 5
    sc = Scenario.draw(Item(lead_time=lead), days=days, seed=4)
    d = simulate(sc, SQ(120, 380), warmup=0).days
    received = d.loc[d["tag"] + lead + 1 <= days - 1, "bestellung"].sum()  # Eingang noch im Zeitraum
    net_end = d["lagerbestand"].iloc[-1] - d["rueckstand"].iloc[-1]
    assert net_end == pytest.approx(120 + 380 + received - d["nachfrage"].sum())
    assert (d["lagerbestand"] >= 0).all() and (d["rueckstand"] >= 0).all()
    assert (d["bedient"] <= d["nachfrage"]).all()


def test_higher_reorder_point_means_better_service():
    sc = Scenario.draw(Item(), days=365 * 10, seed=5)
    levels = [simulate(sc, SQ(s, 382)).cycle_service_level for s in (100, 120, 140)]
    assert levels[0] < levels[1] < levels[2]


def test_textbook_formula_misses_target_with_daily_review():
    """Die Lehrbuchformel nimmt laufende Prüfung an. Täglich geprüft fehlt der Undershoot."""
    table = compare_methods(EXAMPLES[:1], years=30)
    book = table[table["methode"].str.startswith("Lehrbuch")].iloc[0]
    corrected = table[table["methode"].str.startswith("mit Undershoot")].iloc[0]
    assert book["servicegrad"] < 0.85
    assert corrected["servicegrad"] >= 0.94
    assert expected_undershoot(Item()) == pytest.approx(10.5)


def test_calibrated_reorder_point_hits_target_for_every_example():
    table = compare_methods(years=30)
    calibrated = table[table["methode"] == "per Simulation kalibriert"]
    assert (calibrated["servicegrad"] >= 0.95).all()
    assert (calibrated["neue_daten"] >= 0.92).all()  # hält auch auf neuen Zufallszahlen ungefähr


def test_normal_approximation_is_too_low_for_sporadic_demand():
    sporadic = EXAMPLES[2]
    normal_s = reorder_point_normal(sporadic, 0.95)[0]
    empirical_s = reorder_point_empirical(sporadic, 0.95)
    assert empirical_s > normal_s


def test_fill_rate_exceeds_cycle_service_level():
    r = simulate(Scenario.draw(Item(), days=365 * 10, seed=6), SQ(117, 382))
    assert r.fill_rate > r.cycle_service_level


def test_periodic_review_order_up_to_level():
    sc = Scenario.draw(Item(), days=365 * 10, seed=7)
    fill = [simulate(sc, RS(14, S)).fill_rate for S in (360, 400, 440)]
    assert fill[0] < fill[1] < fill[2]


def test_optimum_is_inside_the_grid():
    sc = Scenario.draw(Item(), days=365 * 5, seed=8)
    table = optimize_sq(sc, range(90, 171, 10), [382])
    best = table.iloc[0]["s"]
    assert 90 < best < 170  # Kostenkurve ist U-förmig: zu wenig Bestand kostet Fehlmengen, zu viel Lagerkosten


def test_abc_classes_follow_value_shares():
    c = classify(sample_assortment())
    share = c.groupby("abc")["verbrauchswert"].sum() / c["verbrauchswert"].sum()
    assert 0.8 <= share["A"] < 0.9
    assert share["C"] < 0.06
    assert set(c["xyz"]) == {"X", "Y", "Z"}
    assert c["ziel_servicegrad"].notna().all()


def test_streamlit_app_runs():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception
