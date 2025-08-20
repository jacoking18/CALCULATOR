# app.py
# CAPNOW Syndication Deal Calculator (Tabbed UI)
# -----------------------------------------------------------------------------
# Core rules implemented:
# - Broker Commission = % of *Funding Amount* (not payback)
# - Broker Commission is paid on Day 1 by investors in proportion to their % split
#   (investors bring extra cash for this; not deducted from funding)
# - Origination + ACH Program Fee go to Capnow on Day 1
# - Tracker fees are skimmed DAILY from each investor's share of collections
# - Collections can be over business days (Mon–Fri) or calendar days
#
# Tabs:
#   1) Deal — enter deal terms and fees
#   2) Syndicators — set investor splits and tracker fees
#   3) Results — Summary metrics + Daily Cashflow Schedule + CSV download
#
# Notes:
# - Payback = override if provided else Funding * Rate
# - Profit/ROI use total investor cash-out on Day 1 = (Invested principal) + (Broker Commission share)

from datetime import date, timedelta
from typing import List, Dict
import math

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Utilities
# -----------------------------
def dollars(x: float) -> float:
    return float(np.round((x if x is not None else 0.0) + 1e-12, 2))


def make_business_dates(start: date, n_days: int) -> List[date]:
    out = []
    d = start
    while len(out) < n_days:
        if d.weekday() < 5:  # Mon-Fri
            out.append(d)
        d += timedelta(days=1)
    return out


def make_calendar_dates(start: date, n_days: int) -> List[date]:
    return [start + timedelta(days=i) for i in range(n_days)]


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


# -----------------------------
# Core calculation
# -----------------------------
def compute_schedule(
    start_dt: date,
    funding_amount: float,
    rate: float,
    payback_override: float,
    term_days: int,
    business_days: bool,
    orig_fee_capnow: float,
    ach_fee_capnow: float,
    broker_commission_pct_of_funding: float,
    syndicators_table: pd.DataFrame,
):
    # Payback & daily
    payback = payback_override if payback_override > 0 else dollars(funding_amount * rate)
    daily_payment = dollars(payback / term_days)

    # Dates
    dates = make_business_dates(start_dt, term_days) if business_days else make_calendar_dates(start_dt, term_days)

    # Normalized syndicators
    syns = []
    for _, r in syndicators_table.iterrows():
        syns.append(
            {
                "name": str(r["Name"]).strip(),
                "pct": float(r["Percent_of_Deal"]) / 100.0,
                "tracker": float(r["Tracker_Fee_%"]) / 100.0,
            }
        )

    # Precompute day-1 broker commission (on funding)
    broker_comm_total = dollars(funding_amount * (broker_commission_pct_of_funding / 100.0))
    broker_split = {s["name"]: dollars(broker_comm_total * s["pct"]) for s in syns}

    # Per-investor amounts
    invested_principal = {s["name"]: dollars(funding_amount * s["pct"]) for s in syns}
    per_inv_daily_gross = {s["name"]: dollars(daily_payment * s["pct"]) for s in syns}

    # Accumulators
    investor_collections_net = {s["name"]: 0.0 for s in syns}
    capnow_tracker_total = 0.0

    # Daily schedule rows
    rows = []
    for i, d in enumerate(dates, start=1):
        row = {
            "Day": i,
            "Date": d.isoformat(),
            "Gross_Collection": daily_payment,
        }
        capnow_tracker_today = 0.0

        for s in syns:
            name = s["name"]
            gross = per_inv_daily_gross[name]
            tracker_fee = dollars(gross * s["tracker"])
            net_to_investor = dollars(gross - tracker_fee)

            row[f"{name}_Gross"] = gross
            row[f"{name}_TrackerFee"] = tracker_fee
            row[f"{name}_NetToInvestor"] = net_to_investor

            investor_collections_net[name] = dollars(investor_collections_net[name] + net_to_investor)
            capnow_tracker_today += tracker_fee

        row["Capnow_Tracker_Fees"] = dollars(capnow_tracker_today)
        capnow_tracker_total = dollars(capnow_tracker_total + capnow_tracker_today)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Capnow totals
    capnow_upfront = dollars(orig_fee_capnow + ach_fee_capnow)
    capnow_total = dollars(capnow_upfront + capnow_tracker_total)

    # Investor economics: Day‑1 cash-out = principal + broker share
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

    summaries = {
        "Inputs": {
            "Start Date": dates[0].isoformat() if dates else "",
            "Funding Amount": dollars(funding_amount),
            "Rate": rate,
            "Payback": dollars(payback),
            "Daily Payment": daily_payment,
            "Term Days": term_days,
            "Business Days": business_days,
        },
        "Fees": {
            "Origination → Capnow (Day 1)": dollars(orig_fee_capnow),
            "ACH Program → Capnow (Day 1)": dollars(ach_fee_capnow),
            "Broker Commission % of Funding (Day 1 by Investors)": dollars(broker_commission_pct_of_funding),
            "Broker Commission Total (on Funding)": broker_comm_total,
        },
        "Capnow": {
            "Upfront (Day 1)": capnow_upfront,
            "Tracker Fees (over term)": capnow_tracker_total,
            "Total Revenue": capnow_total,
        },
        "Investors": {
            name: {
                "Deal %": round(100 * s["pct"], 2),
                "Invested Principal": invested_principal[name],
                "Broker Commission Share (Day 1)": broker_split[name],
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


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="CAPNOW Syndication Calculator", page_icon="💸", layout="wide")
st.title("💸 CAPNOW Syndication Deal Calculator")

tab_deal, tab_syn, tab_results = st.tabs(["🧾 Deal", "👥 Syndicators", "📊 Results"])

# ---- Deal tab
with tab_deal:
    st.subheader("Deal Terms")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start_date = st.date_input("Start Date", value=date(2025, 8, 21))
        term_days = st.number_input("Term (days)", min_value=1, value=40, step=1)
    with c2:
        funding_amount = st.number_input("Funding Amount ($)", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
        use_business_days = st.toggle("Use Business Days (Mon–Fri)", value=True)
    with c3:
        rate = st.number_input("Rate (×)", min_value=1.0, value=1.48, step=0.01, format="%.2f")
        payback_override = st.number_input("Payback Override ($) [optional]", min_value=0.0, value=7400.0, step=100.0, format="%.2f")
    with c4:
        orig_fee = st.number_input("Origination Fee → Capnow (Day 1)", min_value=0.0, value=500.0, step=50.0, format="%.2f")
        ach_fee = st.number_input("ACH Program Fee → Capnow (Day 1)", min_value=0.0, value=395.0, step=25.0, format="%.2f")

    st.markdown("### Broker Commission")
    bc1, bc2 = st.columns(2)
    with bc1:
        broker_comm_pct_funding = st.number_input("Broker Commission % of Funding (Day 1 by Investors)", min_value=0.0, value=7.0, step=0.5, format="%.2f")
    with bc2:
        st.info("This commission is calculated on the **Funding Amount** and is paid on **Day 1** by investors in proportion to their deal % (extra cash they must bring).")

# ---- Syndicators tab
with tab_syn:
    st.subheader("Syndicator Splits & Tracker Fees")
    st.caption("The % of Deal must total 100%. Tracker fee is skimmed daily from each investor’s share.")
    default_rows = [
        {"Name": "Jacobo", "Percent_of_Deal": 50.0, "Tracker_Fee_%": 5.0},
        {"Name": "Albert", "Percent_of_Deal": 50.0, "Tracker_Fee_%": 4.0},
    ]
    syn_df = st.data_editor(
        pd.DataFrame(default_rows),
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Name": st.column_config.TextColumn(required=True),
            "Percent_of_Deal": st.column_config.NumberColumn("Percent of Deal", format="%.2f", min_value=0.0, max_value=100.0, step=0.1),
            "Tracker_Fee_%": st.column_config.NumberColumn("Tracker Fee %", format="%.2f", min_value=0.0, max_value=100.0, step=0.1),
        },
        key="syn_table",
    )

    err = validate_syndicators(syn_df.to_dict(orient="records"))
    if err:
        st.error(err)

# ---- Results tab
with tab_results:
    # Guard if syndicators invalid
    err = validate_syndicators(syn_df.to_dict(orient="records"))
    if err:
        st.warning("Fix Syndicators before viewing results.")
        st.stop()

    df, summaries = compute_schedule(
        start_dt=start_date,
        funding_amount=float(funding_amount),
        rate=float(rate),
        payback_override=float(payback_override),
        term_days=int(term_days),
        business_days=bool(use_business_days),
        orig_fee_capnow=float(orig_fee),
        ach_fee_capnow=float(ach_fee),
        broker_commission_pct_of_funding=float(broker_comm_pct_funding),
        syndicators_table=syn_df,
    )

    st.subheader("Summary")
    inp, fees, cap, inv = summaries["Inputs"], summaries["Fees"], summaries["Capnow"], summaries["Investors"]

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Funding Amount", f"${inp['Funding Amount']:,.2f}")
    with m2: st.metric("Payback", f"${inp['Payback']:,.2f}")
    with m3: st.metric("Daily Payment", f"${inp['Daily Payment']:,.2f}")
    with m4: st.metric("Term Days", f"{inp['Term Days']} {'(Biz)' if inp['Business Days'] else '(Cal)'}")

    f1, f2, f3, f4 = st.columns(4)
    with f1: st.metric("Origination → Capnow (Day 1)", f"${fees['Origination → Capnow (Day 1)']:,.2f}")
    with f2: st.metric("ACH → Capnow (Day 1)", f"${fees['ACH Program → Capnow (Day 1)']:,.2f}")
    with f3: st.metric("Broker % of Funding", f"{fees['Broker Commission % of Funding (Day 1 by Investors)']:.2f}%")
    with f4: st.metric("Broker Commission Total", f"${fees['Broker Commission Total (on Funding)']:,.2f}")

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Capnow Upfront (Day 1)", f"${cap['Upfront (Day 1)']:,.2f}")
    with c2: st.metric("Capnow Tracker Fees", f"${cap['Tracker Fees (over term)']:,.2f}")
    with c3: st.metric("Capnow Total Revenue", f"${cap['Total Revenue']:,.2f}")

    st.markdown("#### Investor Results")
    cols = st.columns(len(inv))
    for i, (name, v) in enumerate(inv.items()):
        with cols[i]:
            st.metric(f"{name} — Deal %", f"{v['Deal %']:.2f}%")
            st.metric(f"{name} — Invested Principal", f"${v['Invested Principal']:,.2f}")
            st.metric(f"{name} — Broker (Day 1)", f"${v['Broker Commission Share (Day 1)']:,.2f}")
            st.metric(f"{name} — Day‑1 Cash Out", f"${v['Total Day-1 Cash Out']:,.2f}")
            st.metric(f"{name} — Collections Net", f"${v['Collections Net (after tracker)']:,.2f}")
            st.metric(f"{name} — Profit", f"${v['Profit']:,.2f}")
            st.metric(f"{name} — ROI", f"{v['ROI_%']:.2f}%")

    st.markdown("### Daily Cashflow Schedule")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Daily Schedule (CSV)", data=csv, file_name="syndication_cashflow_schedule.csv", mime="text/csv")

    with st.expander("Raw JSON: Inputs / Fees / Capnow / Investors"):
        st.json({"Inputs": inp, "Fees": fees, "Capnow": cap, "Investors": inv})
