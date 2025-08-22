# app.py
# CAPNOW Syndication Deal Calculator (v2.7 — stable inline popovers)
# -----------------------------------------------------------------
# - Payback = Funding * Rate
# - Origination Fee = % of Funding (Day 1 -> Capnow)
# - ACH Program Fee = fixed $ (Day 1 -> Capnow)
# - Broker Commission = % of Funding (Day 1 paid by investors, proportional to split)
# - Tracker Fees (CAFS) skim from each investor's daily share of collections
# - Business-day or calendar-day schedules
# - NEW: tiny “＊” popover above each investor metric (no session_state, no errors)

from datetime import date, timedelta
from typing import List, Dict
import math

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Utils ----------
def dollars(x: float) -> float:
    return float(np.round((x if x is not None else 0.0) + 1e-12, 2))

def fmt_money(x: float) -> str:
    return f"${x:,.2f}"

def business_dates(start: date, n: int) -> List[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out

def calendar_dates(start: date, n: int) -> List[date]:
    return [start + timedelta(days=i) for i in range(n)]

def validate_syndicators(rows: List[Dict]) -> str:
    if not rows:
        return "Add at least one syndicator."
    total = sum(float(r.get("Percent_of_Deal", 0) or 0) for r in rows)
    if not math.isclose(total, 100.0, abs_tol=0.001):
        return f"Syndicator % must sum to 100 (currently {total:.2f}%)."
    for r in rows:
        if not str(r.get("Name", "")).strip():
            return "Each syndicator must have a Name."
        tf = float(r.get("Tracker_Fee_%", 0) or 0)
        if tf < 0 or tf > 100:
            return f"Tracker fee % for {r.get('Name','(name)')} must be 0–100."
    return ""

# Small helper: render a tiny popover above a metric (falls back to expander if needed)
def metric_with_popover(key_prefix: str, label: str, value_str: str, help_md: str):
    star_col, _ = st.columns([0.12, 0.88])  # star sits a bit above/left of the metric
    if hasattr(star_col, "popover"):
        with star_col.popover("＊"):
            st.markdown(help_md)
    else:
        with star_col.expander("＊"):
            st.markdown(help_md)
    st.metric(label, value_str)

# ---------- Core calc ----------
def compute_schedule(
    start_dt: date,
    funding: float,
    rate: float,
    term_days: int,
    use_business_days: bool,
    orig_fee_pct_of_funding: float,
    ach_fee_capnow: float,
    broker_comm_pct_of_funding: float,
    syndicators_table: pd.DataFrame,
):
    payback = dollars(funding * rate)
    daily_payment = dollars(payback / term_days)
    dates = business_dates(start_dt, term_days) if use_business_days else calendar_dates(start_dt, term_days)

    # Normalize syndicators
    syns = []
    for _, r in syndicators_table.iterrows():
        syns.append({
            "name": str(r["Name"]).strip(),
            "pct": float(r["Percent_of_Deal"]) / 100.0,
            "tracker": float(r["Tracker_Fee_%"]) / 100.0,
        })

    # Day-1 fees
    orig_fee_capnow = dollars(funding * (orig_fee_pct_of_funding / 100.0))
    broker_comm_total = dollars(funding * (broker_comm_pct_of_funding / 100.0))
    broker_split = {s["name"]: dollars(broker_comm_total * s["pct"]) for s in syns}

    # Per-investor base
    invested_principal = {s["name"]: dollars(funding * s["pct"]) for s in syns}
    per_inv_daily_gross = {s["name"]: dollars(daily_payment * s["pct"]) for s in syns}

    # Build schedule & accumulate tracker totals (per investor)
    rows = []
    investor_collections_net = {s["name"]: 0.0 for s in syns}
    investor_tracker_totals = {s["name"]: 0.0 for s in syns}
    capnow_tracker_total = 0.0

    for i, d in enumerate(dates, start=1):
        row = {"Day": i, "Date": d.isoformat(), "Gross_Collection": daily_payment}
        capnow_today = 0.0

        for s in syns:
            name = s["name"]
            gross = per_inv_daily_gross[name]
            tracker_fee = dollars(gross * s["tracker"])        # rounded per day
            net_to_investor = dollars(gross - tracker_fee)

            row[f"{name}_Gross"] = gross
            row[f"{name}_TrackerFee"] = tracker_fee
            row[f"{name}_NetToInvestor"] = net_to_investor

            investor_collections_net[name] = dollars(investor_collections_net[name] + net_to_investor)
            investor_tracker_totals[name] = dollars(investor_tracker_totals[name] + tracker_fee)
            capnow_today += tracker_fee

        row["Capnow_Tracker_Fees"] = dollars(capnow_today)
        capnow_tracker_total = dollars(capnow_tracker_total + capnow_today)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Capnow & client (merchant) totals
    capnow_upfront = dollars(orig_fee_capnow + ach_fee_capnow)
    capnow_total = dollars(capnow_upfront + capnow_tracker_total)
    merchant_nets = dollars(funding - orig_fee_capnow - ach_fee_capnow)

    # Investor economics (Day-1 cash out = principal + broker share)
    investor_day1_cash = {}
    investor_profit = {}
    investor_roi = {}
    for s in syns:
        name = s["name"]
        day1_out = dollars(invested_principal[name] + broker_split[name])
        investor_day1_cash[name] = day1_out
        profit = dollars(investor_collections_net[name] - day1_out)
        investor_profit[name] = profit
        investor_roi[name] = round(100.0 * (profit / day1_out), 2) if day1_out > 0 else 0.0

    # Summaries
    summaries = {
        "Inputs": {
            "Start Date": dates[0].isoformat() if dates else "",
            "Funding Amount": dollars(funding),
            "Rate": rate,
            "Payback": payback,
            "Daily Payment": daily_payment,
            "Term Days": term_days,
            "Business Days": use_business_days,
        },
        "Fees": {
            "Origination % of Funding (Day 1)": dollars(orig_fee_pct_of_funding),
            "Origination $ (Day 1 → Capnow)": dollars(orig_fee_capnow),
            "ACH Program $ (Day 1 → Capnow)": dollars(ach_fee_capnow),
            "Broker % of Funding (Day 1 by Investors)": dollars(broker_comm_pct_of_funding),
            "Broker $ Total (on Funding)": dollars(broker_comm_total),
        },
        "Capnow": {
            "Upfront (Day 1)": capnow_upfront,
            "Tracker Fees (over term)": capnow_tracker_total,
            "Total Revenue": capnow_total,
        },
        "Client": {
            "Merchant Nets": merchant_nets
        },
        "Investors": {
            name: {
                "Deal %": round(100 * s["pct"], 2),
                "Invested Principal": invested_principal[name],
                "Broker Share (Day 1)": broker_split[name],                 # Paid by investor (to broker)
                "Tracker Fees (Total → Capnow)": investor_tracker_totals[name],
                "Total Fees Paid": dollars(broker_split[name] + investor_tracker_totals[name]),
                "Total Day-1 Cash Out": investor_day1_cash[name],
                "Collections Net (after tracker)": dollars(investor_collections_net[name]),
                "Profit": investor_profit[name],
                "ROI_%": investor_roi[name],
            }
            for s in syns
            for name in [s["name"]]
        },
    }

    return df, summaries

# ---------- UI ----------
st.set_page_config(page_title="CAPNOW Syndication Calculator", page_icon="💸", layout="wide")
st.title("💸 CAPNOW Syndication Deal Calculator")

tab_deal, tab_syn, tab_results = st.tabs(["🧾 Deal", "👥 Syndicators", "📊 Results"])

# Deal tab
with tab_deal:
    st.subheader("Deal Terms")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start_date = st.date_input("Start Date", value=date(2025, 8, 21))
        term_days = st.number_input("Term (days)", min_value=1, value=40, step=1)
    with c2:
        funding = st.number_input("Funding Amount ($)", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
