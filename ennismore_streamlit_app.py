import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

st.set_page_config(page_title="Ennismore Deal Model", layout="wide")

# ---------- Logo ----------
LOGO_URL = "https://cdn.prod.website-files.com/66ec88f6d7b63833eb28d6a7/66ec8de11054852c315965b0_BAY%20STREET%20HOSPITALITY-03-p-800.png"
st.sidebar.image(LOGO_URL, width=150)
st.image(LOGO_URL, width=200)

# ---------- Helpers ----------
def xnpv(rate, cashflows):
    '''Return the NPV of a series of cashflows at irregular intervals.
    cashflows: list of (date, amount) tuples'''
    if rate <= -1.0:
        return np.inf
    t0 = cashflows[0][0]
    return sum(cf / ((1 + rate) ** ((t - t0).days / 365.0)) for t, cf in cashflows)

def xirr(cashflows, guess=0.2):
    '''Compute the XIRR for irregular cashflows using a bisection search.'''
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
        if abs(high - low) < 1e-7:
            break
    return (low + high) / 2.0

def fmt_money(x, curr="€"):
    try:
        return f"{curr}{x:,.0f}"
    except Exception:
        return ""

# ---------- Default Inputs (prefilled from approved docs) ----------
DEFAULTS = {
    "stake_pct": 0.136,
    "purchase_price": 425_000_000,
    "entry_date": date(2025, 8, 31),
    "exit_date": date(2027, 12, 31),
    "exit_market_cap_mid": 7_250_000_000,
    "exit_net_debt": 0,
    "dividends": 0,
    "fwd_multiple": 18.0,
    "fwd_ebitda_2024": 350_000_000,
    "net_debt_current": 700_000_000,
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
exit_date = st.sidebar.date_input("Exit / IPO Date", value=DEFAULTS["exit_date"])
exit_market_cap_mid = st.sidebar.number_input("Exit Market Cap (Midpoint, €)", min_value=0.0, value=float(DEFAULTS["exit_market_cap_mid"]), step=50_000_000.0, format="%.0f")
dividends = st.sidebar.number_input("Dividends/Distributions During Hold (€)", min_value=0.0, value=float(DEFAULTS["dividends"]), step=1_000_000.0, format="%.0f")

st.sidebar.markdown("---")
st.sidebar.subheader("Multiple Context")
fwd_multiple = st.sidebar.number_input("Forward EV/EBITDA Multiple", min_value=0.0, value=DEFAULTS["fwd_multiple"], step=0.5, format="%.1f")
fwd_ebitda_2024 = st.sidebar.number_input("2024E EBITDA (for EV context, €)", min_value=0.0, value=float(DEFAULTS["fwd_ebitda_2024"]), step=10_000_000.0, format="%.0f")
net_debt_current = st.sidebar.number_input("Current Net Debt (EV → Equity, €)", min_value=0.0, value=float(DEFAULTS["net_debt_current"]), step=50_000_000.0, format="%.0f")

# ---------- Main Layout ----------
st.title("Ennismore Model")
st.caption("Prefilled with financials covered in the IM (Feb/Mar 2025). Adjust inputs in the sidebar to explore scenarios.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Stake", f"{stake_pct*100:.1f}%")
with col2:
    st.metric("Purchase Price", fmt_money(purchase_price))
with col3:
    st.metric("Exit Market Cap (Mid)", fmt_money(exit_market_cap_mid))

# EV / Equity context
ev_context = fwd_multiple * fwd_ebitda_2024
equity_context = ev_context - net_debt_current
st.markdown("### Valuation Context")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("EV (Forward)", fmt_money(ev_context))
with c2:
    st.metric("Net Debt (Current)", fmt_money(net_debt_current))
with c3:
    st.metric("Equity Value (Forward)", fmt_money(equity_context))

# EBITDA table (editable)
st.markdown("### EBITDA Path (Editable)")
ebitda_edit = st.data_editor(EBITDA_TABLE, num_rows="dynamic", key="ebitda_table")

# Cash flow schedule
st.markdown("### Cash Flows")
exit_proceeds = exit_market_cap_mid * stake_pct
if dividends > 0:
    interim_date = date(entry_date.year + 1, entry_date.month, min(entry_date.day, 28))
    cashflows = [
        (entry_date, -purchase_price),
        (interim_date, dividends),
        (exit_date, exit_proceeds),
    ]
else:
    cashflows = [
        (entry_date, -purchase_price),
        (exit_date, exit_proceeds),
    ]

cf_df = pd.DataFrame({
    "Date": [d.strftime("%Y-%m-%d") for d, _ in cashflows],
    "Cash Flow (€)": [cf for _, cf in cashflows],
    "Note": ["Entry", "Dividend" if dividends > 0 else "Exit", "Exit"] if dividends > 0 else ["Entry", "Exit"],
})

st.dataframe(cf_df, use_container_width=True)

# MOIC & IRR
moic = exit_proceeds / purchase_price if purchase_price > 0 else np.nan
irr = xirr(cashflows) if len(cashflows) >= 2 else np.nan

colm1, colm2, colm3 = st.columns(3)
with colm1:
    st.metric("Exit Proceeds (Equity)", fmt_money(exit_proceeds))
with colm2:
    st.metric("MOIC", f"{moic:.2f}x")
with colm3:
    st.metric("IRR (XIRR)", f"{irr*100:.1f}%")

# Sensitivities
st.markdown("### Sensitivities")
s1, s2 = st.columns(2)
with s1:
    exit_mult = st.slider("Exit Market Cap Sensitivity (€B)", min_value=6.0, max_value=9.0, value=7.25, step=0.25)
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

# Downloadable exports
st.markdown("### Export")
inputs_df = pd.DataFrame({
    "Item": ["Stake Percentage", "Purchase Price (€)", "Entry Date", "Exit Date", "Exit Market Cap (Mid €)", "Dividends (€)", "Forward EV/EBITDA", "2024E EBITDA (€)", "Current Net Debt (€)"],
    "Value": [stake_pct, purchase_price, entry_date, exit_date, exit_market_cap_mid, dividends, fwd_multiple, fwd_ebitda_2024, net_debt_current],
})
export = {
    "inputs": inputs_df.to_csv(index=False).encode("utf-8"),
    "ebitda": ebitda_edit.to_csv(index=False).encode("utf-8"),
    "cashflows": cf_df.to_csv(index=False).encode("utf-8"),
}
st.download_button("Download Inputs CSV", data=export["inputs"], file_name="ennismore_inputs.csv")
st.download_button("Download EBITDA CSV", data=export["ebitda"], file_name="ennismore_ebitda.csv")
st.download_button("Download Cash Flows CSV", data=export["cashflows"], file_name="ennismore_cash_flows.csv")

st.caption("Notes: EBITDA 2022–2024 from IM (p.10). 22% CAGR guidance 2024→2027. Exit cap midpoint €7.25B. Adjust any values to match your underwriting.")
