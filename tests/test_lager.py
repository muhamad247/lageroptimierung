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
