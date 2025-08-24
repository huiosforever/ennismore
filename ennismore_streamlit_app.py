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
    if rate <= -1.0:
        return np.inf
    t0 = cashflows[0][0]
    return sum(cf / ((1 + rate) ** ((t - t0).days / 365.0)) for t, cf in cashflows)

def xirr(cashflows):
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

# ---------- Defaults ----------
DEFAULTS = {
    "stake_pct": 0.138,                    # 13.8%
    "purchase_price": 405_000_000,         # €405m
    "entry_date": date(2025, 8, 31),
    "exit_date": date(2029, 12, 31),       # conservative ~4y to align with IM IRR
    "exit_market_cap_mid": 7_250_000_000,
    "dividends": 0,
    "fwd_multiple": 18.0,                  # forward EV/EBITDA multiple
    "fwd_ebitda_2024": 350_000_000,        # 2024E EBITDA (for 18x)
    "current_multiple": 15.0,              # current EV/EBITDA multiple
    "current_ebitda": 350_000_000,         # editable; defaults to 2024E for convenience
    "net_debt_current": 700_000_000,       # EV → Equity bridge
    "current_ev_reference": 6_300_000_000, # for discount display context
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
stake_pct = st.sidebar.number_input("Stake Percentage", 0.0, 1.0, DEFAULTS["stake_pct"], step=0.001, format="%.3f")
purchase_price = st.sidebar.number_input("Purchase Price (€)", 0.0, value=float(DEFAULTS["purchase_price"]), step=1_000_000.0, format="%.0f")
entry_date = st.sidebar.date_input("Entry Date", value=DEFAULTS["entry_date"])
exit_date = st.sidebar.date_input("Exit / IPO Date (Base)", value=DEFAULTS["exit_date"])
exit_market_cap_mid = st.sidebar.number_input("Exit Market Cap (Base Midpoint, €)", 0.0, value=float(DEFAULTS["exit_market_cap_mid"]), step=50_000_000.0, format="%.0f")
dividends = st.sidebar.number_input("Dividends During Hold (€)", 0.0, value=float(DEFAULTS["dividends"]), step=1_000_000.0, format="%.0f")

st.sidebar.markdown("---")
st.sidebar.subheader("Multiple Context")
fwd_multiple = st.sidebar.number_input("Forward EV/EBITDA Multiple (for 2024E)", 0.0, value=DEFAULTS["fwd_multiple"], step=0.5, format="%.1f")
fwd_ebitda_2024 = st.sidebar.number_input("2024E EBITDA (for 18× EV, €)", 0.0, value=float(DEFAULTS["fwd_ebitda_2024"]), step=10_000_000.0, format="%.0f")
current_multiple = st.sidebar.number_input("Current EV/EBITDA Multiple", 0.0, value=DEFAULTS["current_multiple"], step=0.5, format="%.1f")
current_ebitda = st.sidebar.number_input("Current EBITDA (for 15× EV, €)", 0.0, value=float(DEFAULTS["current_ebitda"]), step=10_000_000.0, format="%.0f")
net_debt_current = st.sidebar.number_input("Current Net Debt (EV → Equity, €)", 0.0, value=float(DEFAULTS["net_debt_current"]), step=50_000_000.0, format="%.0f")

# ---------- Main ----------
st.title("Ennismore Model")
st.caption(
    "Prefilled with financials discussed in the IM (Feb/Mar 2025). "
    "Assumes a full exit in a single tranche (no phased sell‑down). "
    "Use the equity vs. market cap toggle to align with IM‑style IRR."
)

# Top KPIs
current_ev_ref = DEFAULTS["current_ev_reference"]
implied_ev_stake_value = stake_pct * current_ev_ref
implied_equity_value_ref = max(0.0, current_ev_ref - net_debt_current) * stake_pct
disc_vs_ev = (1 - (purchase_price / implied_ev_stake_value)) if implied_ev_stake_value > 0 else np.nan
disc_vs_equity = (1 - (purchase_price / implied_equity_value_ref)) if implied_equity_value_ref > 0 else np.nan

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Stake", f"{stake_pct*100:.1f}%")
with c2:
    st.metric("Purchase Price", fmt_money(purchase_price))
    st.metric("Discount vs EV (ref €6.3B)", f"{disc_vs_ev*100:.1f}%" if not np.isnan(disc_vs_ev) else "N/A")
    st.metric("Discount vs Equity (EV − Net Debt)", f"{disc_vs_equity*100:.1f}%" if not np.isnan(disc_vs_equity) else "N/A")
with c3:
    st.metric("EV Reference (for discount)", fmt_money(current_ev_ref))

# ---------- Valuation Contexts ----------
st.markdown("### Valuation Contexts (18× Forward vs 15× Current)")

# Forward 18x (2024E)
ev_forward = fwd_multiple * fwd_ebitda_2024
equity_forward = ev_forward - net_debt_current
implied_price_forward = equity_forward * stake_pct
disc_forward_vs_ticket = (1 - purchase_price / implied_price_forward) if implied_price_forward > 0 else np.nan

# Current 15x (separately editable EBITDA)
ev_current = current_multiple * current_ebitda
equity_current = ev_current - net_debt_current
implied_price_current = equity_current * stake_pct
disc_current_vs_ticket = (1 - purchase_price / implied_price_current) if implied_price_current > 0 else np.nan

vc1, vc2, vc3, vc4 = st.columns(4)
with vc1:
    st.metric("EV (18× × 2024E EBITDA)", fmt_money(ev_forward))
with vc2:
    st.metric("Equity (18× less Net Debt)", fmt_money(equity_forward))
with vc3:
    st.metric("Implied Price for 13.8% (18×)", fmt_money(implied_price_forward))
with vc4:
    st.metric("Discount vs 18× Equity Price", f"{disc_forward_vs_ticket*100:.1f}%" if not np.isnan(disc_forward_vs_ticket) else "N/A")

vc5, vc6, vc7, vc8 = st.columns(4)
with vc5:
    st.metric("EV (15× × Current EBITDA)", fmt_money(ev_current))
with vc6:
    st.metric("Equity (15× less Net Debt)", fmt_money(equity_current))
with vc7:
    st.metric("Implied Price for 13.8% (15×)", fmt_money(implied_price_current))
with vc8:
    st.metric("Discount vs 15× Equity Price", f"{disc_current_vs_ticket*100:.1f}%" if not np.isnan(disc_current_vs_ticket) else "N/A")

# Exit proceeds basis toggle
st.markdown("**Exit Proceeds Basis**")
use_equity_basis = st.toggle("Compute exit proceeds from Equity Value (EV − Net Debt) instead of headline Market Cap", value=True)
st.caption("Turning this ON generally aligns IRR with the IM’s equity‑based methodology.")

# EBITDA table (editable)
st.markdown("### EBITDA Path (Editable)")
ebitda_edit = st.data_editor(EBITDA_TABLE, num_rows="dynamic", key="ebitda_table")

# ---------- Base Case Cash Flows (Table A) ----------
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

st.markdown("## Base Case Cash Flows (Table A — Inputs → IRR)")
cf_df = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d, _ in cashflows],
    "Cash Flow (€)": [cf for _, cf in cashflows],
    "Note": ["Entry", "Dividend", "Exit"] if dividends > 0 else ["Entry", "Exit"],
})
st.dataframe(cf_df, use_container_width=True)

# Base MOIC/IRR
moic = exit_proceeds / purchase_price if purchase_price > 0 else np.nan
irr_base = xirr(cashflows) if len(cashflows) >= 2 else np.nan
m1, m2, m3 = st.columns(3)
with m1: st.metric("Exit Proceeds", fmt_money(exit_proceeds))
with m2: st.metric("MOIC (Base)", f"{moic:.2f}x")
with m3: st.metric("IRR (Base, XIRR)", f"{irr_base*100:.1f}%")
st.caption("IRR shown assumes a conservative ~4‑year hold to align with IM guidance; Bay Street anticipates an actual hold of ~9–18 months, market‑window dependent.")

# ---------- Sensitivities ----------
st.markdown("### Sensitivities")
s1, s2 = st.columns(2)
with s1:
    exit_mult = st.slider("Exit Market Cap Sensitivity (€B)", 6.0, 9.0, 7.25, 0.25)
    proceeds_sens = (max(0.0, exit_mult * 1_000_000_000 - net_debt_current) if use_equity_basis else exit_mult * 1_000_000_000) * stake_pct
    moic_sens = proceeds_sens / purchase_price if purchase_price else np.nan
    irr_sens = xirr([(entry_date, -purchase_price), (exit_date, proceeds_sens)])
    st.write(f"**Proceeds:** {fmt_money(proceeds_sens)}")
    st.write(f"**MOIC:** {moic_sens:.2f}x")
    st.write(f"**IRR:** {irr_sens*100:.1f}%")
with s2:
    hold_years = st.slider("Hold Period (years)", 1.0, 5.0, (exit_date - entry_date).days/365.0, 0.25)
    adj_exit_date = entry_date + timedelta(days=int(hold_years*365))
    irr_hold = xirr([(entry_date, -purchase_price), (adj_exit_date, exit_proceeds)])
    st.write(f"**Adj. Exit Date:** {adj_exit_date}")
    st.write(f"**IRR (hold adj.):** {irr_hold*100:.1f}%")

# ---------- Scenario Analysis ----------
st.markdown("---")
st.markdown("## Scenario Inputs (Table B — Editable)")
st.caption("Edit exit year, market cap (in € billions), and probability. Includes delayed exits and a conservative €6.0B case. Assumes full exit, no phased sell‑down.")

default_scenarios = pd.DataFrame({
    "Scenario": [
        "Base (2029)", "Bull (2029)", "Bear (2029, €6.0B)",
        "Base (2028)", "Bull (2028)", "Bear (2028, €6.0B)",
        "Base (2027)", "Bull (2027)", "Bear (2027, €6.0B)"
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

# ----- Scenario Results (labeled) -----
st.markdown("## Scenario Results (Table C — Computed)")
st.caption("Computed IRRs by scenario using your inputs above; also used for the probability‑weighted IRR shown below.")

irr_vals = []
for _, row in scen_edit.iterrows():
    try:
        scen_year = int(row["Exit_Year"])
        scen_cap_eur = float(row["Exit_Market_Cap_Bn"]) * 1_000_000_000
    except Exception:
        irr_vals.append(np.nan)
        continue
    scen_proceeds = (max(0.0, scen_cap_eur - net_debt_current) * stake_pct) if use_equity_basis else (scen_cap_eur * stake_pct)
    scen_exit_date = date(scen_year, 12, 31)
    scen_cfs = [(entry_date, -purchase_price), (scen_exit_date, scen_proceeds)]
    try:
        irr_vals.append(xirr(scen_cfs))
    except Exception:
        irr_vals.append(np.nan)

scen_results = scen_edit.copy()
scen_results["IRR_%"] = [None if np.isnan(x) else round(x*100, 1) for x in irr_vals]
st.dataframe(scen_results, use_container_width=True)

# Probability-weighted IRR
prob_sum = scen_edit["Probability"].sum() if "Probability" in scen_edit else 0.0
weighted_irr = np.nan
if prob_sum > 0 and len(irr_vals) == len(scen_edit):
    weights = [p / prob_sum for p in scen_edit["Probability"]]
    weighted_irr = sum(w * irr for w, irr in zip(weights, irr_vals) if not np.isnan(irr))

st.markdown("---")
st.subheader(f"Probability‑Weighted IRR: {weighted_irr*100:.1f}%" if not np.isnan(weighted_irr) else "Probability‑Weighted IRR: N/A")

# ---------- Exports ----------
st.markdown("### Export")
inputs_df = pd.DataFrame({
    "Item": ["Stake Percentage", "Purchase Price (€)", "Entry Date", "Exit Date (Base)",
             "Exit Market Cap (Base Mid €)", "Dividends (€)", "Forward EV/EBITDA",
             "2024E EBITDA (€)", "Current EV/EBITDA", "Current EBITDA (€)", "Current Net Debt (€)"],
    "Value": [stake_pct, purchase_price, entry_date, exit_date, exit_market_cap_mid, dividends,
              fwd_multiple, fwd_ebitda_2024, current_multiple, current_ebitda, net_debt_current],
})
export = {
    "inputs": inputs_df.to_csv(index=False).encode("utf-8"),
    "ebitda": ebitda_edit.to_csv(index=False).encode("utf-8"),
    "cashflows": cf_df.to_csv(index=False).encode("utf-8"),
    "scenarios_inputs": scen_edit.to_csv(index=False).encode("utf-8"),
    "scenarios_results": scen_results.to_csv(index=False).encode("utf-8"),
}
cxa, cxb, cxc, cxd, cxe = st.columns(5)
with cxa: st.download_button("Download Inputs CSV", data=export["inputs"], file_name="ennismore_inputs.csv")
with cxb: st.download_button("Download EBITDA CSV", data=export["ebitda"], file_name="ennismore_ebitda.csv")
with cxc: st.download_button("Download Cash Flows CSV", data=export["cashflows"], file_name="ennismore_cash_flows.csv")
with cxd: st.download_button("Download Scenario Inputs CSV", data=export["scenarios_inputs"], file_name="ennismore_scenarios_inputs.csv")
with cxe: st.download_button("Download Scenario Results CSV", data=export["scenarios_results"], file_name="ennismore_scenarios_results.csv")

st.caption(
    "Notes: EBITDA 2022–2024 from IM (p.10). 22% CAGR guidance 2024→2027. "
    "Exit results assume a full exit (no phased sell‑down). Use the equity‑basis toggle to align with IM‑style IRR."
)
