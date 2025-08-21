import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

st.set_page_config(page_title="Ennismore Model", layout="wide")

# ---------- Logo ----------
LOGO_URL = "https://cdn.prod.website-files.com/66ec88f6d7b63833eb28d6a7/66ec8de11054852c315965b0_BAY%20STREET%20HOSPITALITY-03-p-800.png"
st.sidebar.image(LOGO_URL, width=150)
st.image(LOGO_URL, width=200)

# ---------- Helpers ----------
def xnpv(rate, cashflows):
    """Return the NPV of a series of cashflows at irregular intervals.
    cashflows: list of (date, amount) tuples"""
    if rate <= -1.0:
        return np.inf
    t0 = cashflows[0][0]
    return sum(cf / ((1 + rate) ** ((t - t0).days / 365.0)) for t, cf in cashflows)

def xirr(cashflows):
    """Compute the XIRR for irregular cashflows via bisection (robust)."""
    cfs = sorted(cashflows, key=lambda x: x[0])
    low, high = -0.9999, 10.0
    for _ in range(200):
        mid = (low + high) / 2.0
        npv_mid = xnpv(mid, cfs)
        npv_low = xnpv(low, cfs)
        if npv_mid == 0:
            return mid
        if np.sign(npv_mid) == np.sign(npv_low):
            low = mid
        else:
            high = mid
        if abs(high - low) < 1e-9:
            break
    return (low + high) / 2.0

def fmt_money(x, curr="€"):
    try:
        return f"{curr}{x:,.0f}"
    except Exception:
        return ""

# ---------- Default Inputs (aligned with IM-style IRR) ----------
DEFAULTS = {
    "stake_pct": 0.138,                  # 13.8%
    "purchase_price": 405_000_000,       # €405m
    "entry_date": date(2025, 8, 31),     # capital call
    "exit_date": date(2029, 12, 31),     # conservative ~4.3y hold to align with 17–19% IRR
    "exit_market_cap_mid": 7_250_000_000,# headline market cap midpoint
    "dividends": 0,
    "fwd_multiple": 18.0,
    "fwd_ebitda_2024": 350_000_000,
    "net_debt_current": 700_000_000,     # for EV→Equity bridge
    "current_ev": 6_300_000_000,         # reference EV for discount display
}

EBITDA_TABLE = pd.DataFrame({
    "Year": [2022, 2023, 2024, 2025, 2026, 2027],
    "EBITDA_EUR_m": [78, 118, 171, round(171*1.22, 1), round(171*1.22**2, 1), round(171*1.22**3, 1)],
    "Note": [
        "Actual (IM p.10)",
        "Actual (IM p.10)",
        "Actual (IM p.10)",
        "Projected @ 22% CAGR",
        "Projected @ 22% CAGR",
        "Projected @ 22% CAGR",
    ],
})

# ---------- Sidebar Inputs ----------
st.sidebar.header("Assumptions")
stake_pct = st.sidebar.number_input("Stake Percentage", min_value=0.0, max_value=1.0, value=DEFAULTS["stake_pct"], step=0.001, format="%.3f")
purchase_price = st.sidebar.number_input("Purchase Price (€)", min_value=0.0, value=float(DEFAULTS["purchase_price"]), step=1_000_000.0, format="%.0f")
entry_date = st.sidebar.date_input("Entry Date", value=DEFAULTS["entry_date"])
exit_date = st.sidebar.date_input("Exit / IPO Date (Base)", value=DEFAULTS["exit_date"])
exit_market_cap_mid = st.sidebar.number_input("Exit Market Cap (Base Midpoint, €)", min_value=0.0, value=float(DEFAULTS["exit_market_cap_mid"]), step=50_000_000.0, format="%.0f")
dividends = st.sidebar.number_input("Dividends During Hold (€)", min_value=0.0, value=float(DEFAULTS["dividends"]), step=1_000_000.0, format="%.0f")

st.sidebar.markdown("---")
st.sidebar.subheader("Multiple Context")
fwd_multiple = st.sidebar.number_input("Forward EV/EBITDA Multiple", min_value=0.0, value=DEFAULTS["fwd_multiple"], step=0.5, format="%.1f")
fwd_ebitda_2024 = st.sidebar.number_input("2024E EBITDA (EV context, €)", min_value=0.0, value=float(DEFAULTS["fwd_ebitda_2024"]), step=10_000_000.0, format="%.0f")
net_debt_current = st.sidebar.number_input("Current Net Debt (EV → Equity, €)", min_value=0.0, value=float(DEFAULTS["net_debt_current"]), step=50_000_000.0, format="%.0f")

# ---------- Main Layout ----------
st.title("Ennismore Model")
st.caption(
    "Prefilled with financials discussed in the IM (Feb/Mar 2025). "
    "This model assumes a full exit in a single tranche (no phased sell-down). "
    "Use equity vs. market cap toggles to align with IM‑style IRR assumptions."
)

# --- Top KPIs grouped: Stake | Purchase & Discounts | Current EV
current_ev = DEFAULTS["current_ev"]
implied_ev_stake_value   = stake_pct * current_ev
implied_equity_value_ref = max(0.0, current_ev - net_debt_current) * stake_pct

disc_vs_ev     = (1 - (purchase_price / implied_ev_stake_value)) if implied_ev_stake_value   > 0 else np.nan
disc_vs_equity = (1 - (purchase_price / implied_equity_value_ref)) if implied_equity_value_ref > 0 else np.nan

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Stake", f"{stake_pct*100:.1f}%")

with c2:
    st.metric("Purchase Price", fmt_money(purchase_price))
    st.metric("Discount vs EV (€6.3B)", f"{disc_vs_ev*100:.1f}%" if not np.isnan(disc_vs_ev) else "N/A")
    st.metric("Discount vs Equity (EV – Net Debt)", f"{disc_vs_equity*100:.1f}%" if not np.isnan(disc_vs_equity) else "N/A")

with c3:
    st.metric("Current EV (Ref)", fmt_money(current_ev))

st.caption(
    "Note: EV-based discount uses €6.3B EV. Equity-based discount uses EV − current net debt (default €0.7B). "
    "IRR in the IM is typically anchored off equity value."
)

# EV / Equity context (forward)
ev_context = fwd_multiple * fwd_ebitda_2024
equity_context = ev_context - net_debt_current
st.markdown("### Valuation Context")
v1, v2, v3 = st.columns(3)
with v1:
    st.metric("EV (Forward)", fmt_money(ev_context))
with v2:
    st.metric("Net Debt (Current)", fmt_money(net_debt_current))
with v3:
    st.metric("Equity Value (Forward)", fmt_money(equity_context))

# Choose basis for exit proceeds (default ON to align with IM)
st.markdown("**Exit Proceeds Basis**")
use_equity_basis = st.toggle(
    "Compute exit proceeds from Equity Value (EV − Net Debt) instead of headline Market Cap",
    value=True
)
st.caption("Turning this ON generally aligns IRR with the IM’s equity-based methodology.")

# EBITDA table (editable)
st.markdown("### EBITDA Path (Editable)")
ebitda_edit = st.data_editor(EBITDA_TABLE, num_rows="dynamic", key="ebitda_table")

# Cash flow schedule (base)
st.markdown("### Cash Flows (Base Case)")
if use_equity_basis:
    base_equity_value = max(0.0, exit_market_cap_mid - net_debt_current)
    exit_proceeds = base_equity_value * stake_pct
else:
    exit_proceeds = exit_market_cap_mid * stake_pct

if dividends > 0:
    interim_date = date(entry_date.year + 1, entry_date.month, min(entry_date.day, 28))
    cashflows = [(entry_date, -purchase_price), (interim_date, dividends), (exit_date, exit_proceeds)]
else:
    cashflows = [(entry_date, -purchase_price), (exit_date, exit_proceeds)]

cf_df = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d, _ in cashflows],
    "Cash Flow (€)": [cf for _, cf in cashflows],
    "Note": ["Entry", "Dividend" if dividends > 0 else "Exit", "Exit"] if dividends > 0 else ["Entry", "Exit"],
})
st.dataframe(cf_df, use_container_width=True)

# MOIC & IRR (base)
moic = exit_proceeds / purchase_price if purchase_price > 0 else np.nan
irr_base = xirr(cashflows) if len(cashflows) >= 2 else np.nan

k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Exit Proceeds (Equity to Seller)", fmt_money(exit_proceeds))
with k2:
    st.metric("MOIC (Base)", f"{moic:.2f}x")
with k3:
    st.metric("IRR (Base, XIRR)", f"{irr_base*100:.1f}%")
    st.caption(
        "Displayed IRR conservatively assumes a ~4‑year hold to align with the IM; "
        "Bay Street anticipates an actual hold period of **9–18 months** subject to market conditions."
    )

# Sensitivities
st.markdown("### Sensitivities")
s1, s2 = st.columns(2)
with s1:
    exit_mult = st.slider("Exit Market Cap Sensitivity (€B)", min_value=6.0, max_value=9.0, value=7.25, step=0.25)
    if use_equity_basis:
        proceeds_sens = max(0.0, exit_mult*1_000_000_000 - net_debt_current) * stake_pct
    else:
        proceeds_sens = exit_mult * 1_000_000_000 * stake_pct
    moic_sens = proceeds_sens / purchase_price if purchase_price else np.nan
    irr_sens = xirr([(entry_date, -purchase_price), (exit_date, proceeds_sens)])
    st.write(f"**Proceeds:** {fmt_money(proceeds_sens)}")
    st.write(f"**MOIC:** {moic_sens:.2f}x")
    st.write(f"**IRR:** {irr_sens*100:.1f}%")
with s2:
    hold_years = st.slider("Hold Period (years)", min_value=1.0, max_value=5.0, value=((exit_date - entry_date).days/365.0), step=0.25)
    adj_exit_date = entry_date + timedelta(days=int(hold_years*365))
    irr_hold = xirr([(entry_date, -purchase_price), (adj_exit_date, exit_proceeds)])
    st.write(f"**Adj. Exit Date:** {adj_exit_date}")
    st.write(f"**IRR (hold adj.):** {irr_hold*100:.1f}%")

# ---------- Scenario Analysis ----------
st.markdown("## Scenario Analysis (IRR Paths & Probability‑Weighted IRR — assumes full exit, no phased sell‑down)")
st.caption("Edit exit year, market cap (in € billions), and probability. Includes delayed exits and a conservative €6.0B case.")

default_scenarios = pd.DataFrame({
    "Scenario": [
        "Base (2029)",
        "Bull (2029)",
        "Bear (2029, €6.0B)",
        "Base (2028)",
        "Bull (2028)",
        "Bear (2028, €6.0B)",
        "Base (2027)",
        "Bull (2027)",
        "Bear (2027, €6.0B)"
    ],
    "Exit_Year": [2029, 2029, 2029, 2028, 2028, 2028, 2027, 2027, 2027],
    "Exit_Market_Cap_Bn": [7.25, 8.50, 6.00, 7.25, 8.50, 6.00, 7.25, 8.50, 6.00],
    "Probability": [0.35, 0.10, 0.10, 0.15, 0.06, 0.06, 0.12, 0.03, 0.03],
})

scen_edit = st.data_editor(
    default_scenarios,
    num_rows="dynamic",
    use_container_width=True,
    key="scenario_table"
)

# Compute IRR per scenario
irr_list = []
for _, row in scen_edit.iterrows():
    try:
        scen_year = int(row["Exit_Year"])
        scen_cap_eur = float(row["Exit_Market_Cap_Bn"]) * 1_000_000_000
    except Exception:
        irr_list.append(np.nan)
        continue

    if use_equity_basis:
        scen_equity_value = max(0.0, scen_cap_eur - net_debt_current)
        scen_proceeds = scen_equity_value * stake_pct
    else:
        scen_proceeds = scen_cap_eur * stake_pct

    scen_exit_date = date(scen_year, 12, 31)
    scen_cfs = [(entry_date, -purchase_price), (scen_exit_date, scen_proceeds)]
    try:
        irr_val = xirr(scen_cfs)
    except Exception:
        irr_val = np.nan
    irr_list.append(irr_val)

scen_edit["IRR_%"] = [None if np.isnan(x) else round(x*100, 1) for x in irr_list]

# Probability-weighted IRR (normalize if not summing to 1)
prob_sum = scen_edit["Probability"].sum() if "Probability" in scen_edit else 0.0
weighted_irr = np.nan
if prob_sum > 0 and len(irr_list) == len(scen_edit):
    weights = [p / prob_sum for p in scen_edit["Probability"]]
    weighted_irr = 0.0
    for w, irr_v in zip(weights, irr_list):
        if np.isnan(irr_v):
            continue
        weighted_irr += w * irr_v

st.dataframe(scen_edit, use_container_width=True)
st.markdown("---")
if not np.isnan(weighted_irr):
    st.subheader(f"Probability‑Weighted IRR: {weighted_irr*100:.1f}%")
else:
    st.subheader("Probability‑Weighted IRR: N/A")

# ---------- Exports ----------
st.markdown("### Export")
inputs_df = pd.DataFrame({
    "Item": ["Stake Percentage", "Purchase Price (€)", "Entry Date", "Exit Date (Base)", "Exit Market Cap (Base Mid €)", "Dividends (€)", "Forward EV/EBITDA", "2024E EBITDA (€)", "Current Net Debt (€)"],
    "Value": [stake_pct, purchase_price, entry_date, exit_date, exit_market_cap_mid, dividends, fwd_multiple, fwd_ebitda_2024, net_debt_current],
})
export = {
    "inputs": inputs_df.to_csv(index=False).encode("utf-8"),
    "ebitda": EBITDA_TABLE.to_csv(index=False).encode("utf-8"),
    "cashflows": pd.DataFrame(cf_df).to_csv(index=False).encode("utf-8"),
    "scenarios": scen_edit.to_csv(index=False).encode("utf-8"),
}
cxa, cxb, cxc, cxd = st.columns(4)
with cxa: st.download_button("Download Inputs CSV", data=export["inputs"], file_name="ennismore_inputs.csv")
with cxb: st.download_button("Download EBITDA CSV", data=export["ebitda"], file_name="ennismore_ebitda.csv")
with cxc: st.download_button("Download Cash Flows CSV", data=export["cashflows"], file_name="ennismore_cash_flows.csv")
with cxd: st.download_button("Download Scenarios CSV", data=export["scenarios"], file_name="ennismore_scenarios.csv")

st.caption(
    "Notes: EBITDA 2022–2024 from IM (p.10). 22% CAGR guidance 2024→2027. "
    "Exit results assume a full exit (no phased sell‑down). Use the equity‑basis toggle to align with IM‑style IRR."
)
