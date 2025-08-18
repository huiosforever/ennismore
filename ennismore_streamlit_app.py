import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

st.set_page_config(page_title="Ennismore Deal Model", layout="wide")

# ---------- Logo ----------
LOGO_URL = "https://cdn.prod.website-files.com/66ec88f6d7b63833eb28d6a7/66ec8de11054852c315965b0_BAY%20STREET%20HOSPITALITY-03-p-800.png  # <-- Replace with Bay Street logo URL
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
st.sidebar.subheader("Multiple Context (optional)")
fwd_multiple = st.sidebar.number_input("Forward EV/EBITDA Multiple", min_value=0.0, value=DEFAULTS["fwd_multiple"], step=0.5, format="%.1f")
fwd_ebitda_2024 = st.sidebar.number_input("2024E EBITDA (for EV context, €)", min_value=0.0, value=float(DEFAULTS["fwd_ebitda_2024"]), step=10_000_000.0, format="%.0f")
net_debt_current = st.sidebar.number_input("Current Net Debt (EV → Equity, €)", min_value=0.0, value=float(DEFAULTS["net_debt_current"]), step=50_000_000.0, format="%.0f")

# ---------- Main Layout ----------
st.title("Ennismore Deal Model — IRR / MOIC Explorer")
st.caption("Prefilled with figures discussed in the IM (Feb/Mar 2025). Adjust inputs in the sidebar to explore scenarios.")

# ... rest of your code unchanged ...
