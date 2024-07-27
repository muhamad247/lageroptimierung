"""Streamlit-Oberfläche: streamlit run app.py"""

import streamlit as st

from lager.model import Item, eoq, expected_undershoot, lead_time_demand_sd, reorder_point_daily_review, reorder_point_normal
from lager.portfolio import classify, matrix, sample_assortment
from lager.simulate import SQ, Scenario, calibrate_reorder_point, optimize_sq, simulate

st.set_page_config(page_title="Lagerbestand optimieren", layout="wide")
st.title("Lagerbestand optimieren")

with st.sidebar:
    st.header("Artikel")
    mean = st.number_input("Ø Nachfrage pro Tag", 0.1, 500.0, 20.0)
    sporadic = st.toggle("Sporadische Nachfrage (negativ binomial)")
    dispersion = st.slider("Dispersion k (kleiner = sprunghafter)", 0.05, 5.0, 0.3) if sporadic else None
    lead = st.slider("Lieferzeit (Tage)", 1, 30, 5)
    spread = st.slider("Lieferzeit schwankt um ± Tage", 0, 10, 0)
    st.header("Kosten")
    unit = st.number_input("Einstandspreis (EUR)", 0.1, 10_000.0, 20.0)
    rate = st.slider("Lagerkostensatz p. a.", 0.05, 0.5, 0.25)
    order = st.number_input("Kosten pro Bestellung (EUR)", 1.0, 5000.0, 50.0)
    short = st.number_input("Fehlmengenkosten pro Stück (EUR)", 0.0, 1000.0, 5.0)
    target = st.slider("Ziel-Servicegrad", 0.80, 0.995, 0.95, 0.005)

item = Item("Artikel", mean, dispersion, lead, spread, unit, rate, order, short)
scenario = Scenario.draw(item, days=365 * 20, seed=1)
Q = max(1, round(eoq(item)))
s_book, safety = reorder_point_normal(item, target)
s_corr = reorder_point_daily_review(item, target)

tab_policy, tab_opt, tab_abc = st.tabs(["Bestellpolitik", "Kostenoptimum", "ABC/XYZ"])

with tab_policy:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bestellmenge (EOQ)", Q)
    c2.metric("σ Nachfrage in Lieferzeit", f"{lead_time_demand_sd(item):.1f}")
    c3.metric("Meldebestand Lehrbuch", s_book, help=f"Sicherheitsbestand {safety:.1f}")
    c4.metric("mit Undershoot-Korrektur", s_corr, help=f"+ {expected_undershoot(item):.1f} Stück")

    if st.button("Meldebestand per Simulation kalibrieren"):
        st.session_state["s_cal"] = calibrate_reorder_point(scenario, Q, target, start=int(mean * lead))
    options = {"Lehrbuch": s_book, "mit Korrektur": s_corr}
    if "s_cal" in st.session_state:
        options["kalibriert"] = st.session_state["s_cal"]
    choice = st.radio("Meldebestand verwenden", list(options), horizontal=True)
    result = simulate(scenario, SQ(options[choice], Q))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Servicegrad α (simuliert)", f"{result.cycle_service_level:.1%}", f"{result.cycle_service_level - target:+.1%} zum Ziel")
    k2.metric("Fill Rate β", f"{result.fill_rate:.2%}")
    k3.metric("Ø Lagerbestand", f"{result.avg_on_hand:.0f}")
    k4.metric("Kosten pro Jahr", f"{result.cost_per_year:,.0f} EUR")
    st.caption("α zählt Lieferzyklen ohne jede Fehlmenge, β den Anteil der Nachfrage, der sofort bedient wurde. "
               "β ist fast immer deutlich höher als α; welches Maß man steuert, ist eine Geschäftsentscheidung.")
    st.subheader("Lagerbestand im ersten Jahr")
    st.line_chart(result.days.set_index("tag").loc[:365, ["lagerbestand", "rueckstand"]])

with tab_opt:
    st.write("Gesamtkosten (Lager + Bestellung + Fehlmenge) für ein Raster aus Meldebestand und Bestellmenge, "
             "mit denselben Zufallszahlen für jede Kombination.")
    if st.button("Optimieren"):
        step_s = max(1, round(lead_time_demand_sd(item) / 3))
        grid_s = range(max(0, s_book - 6 * step_s), s_book + 10 * step_s, step_s)
        grid_q = sorted({max(1, round(Q * f)) for f in (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)})
        table = optimize_sq(Scenario.draw(item, days=365 * 10, seed=2), grid_s, grid_q)
        best = table.iloc[0]
        st.success(f"Günstigste Kombination: s = {best['s']:.0f}, Q = {best['Q']:.0f}, "
                   f"{best['kosten']:,.0f} EUR pro Jahr bei Servicegrad {best['csl']:.1%}")
        st.line_chart(table[table["Q"] == best["Q"]].sort_values("s").set_index("s")[["kosten_lager", "kosten_fehlmenge", "kosten"]])
        st.dataframe(table.head(15).round(3), hide_index=True)

with tab_abc:
    classified = classify(sample_assortment())
    st.write("Beispielsortiment mit 60 Artikeln. ABC nach Verbrauchswert, XYZ nach Schwankung der Wochennachfrage.")
    st.dataframe(matrix(classified))
    st.dataframe(classified[["artikel", "klasse", "verbrauchswert", "variationskoeffizient", "ziel_servicegrad"]].round(3),
                 hide_index=True)
