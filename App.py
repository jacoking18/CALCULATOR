# app.py
# CAPNOW Syndication Deal Calculator (Streamlit)
# ------------------------------------------------
# Features:
# - Enter deal terms (start date, funding, rate or payback, term in days, fees)
# - Define any number of syndicators with % and tracker fee %
# - Choose business-day collections (Mon–Fri) or calendar days
# - Choose how broker commission is paid: upfront vs. withheld from collections
# - Full daily cashflow schedule with per-investor net, Capnow tracker fees, and upfront fees
# - Summary metrics (per-investor totals, profit, ROI; Capnow totals)
# - CSV download of the daily schedule
#
# Notes:
# - “Tracker fee” (%) is skimmed daily from each investor’s share of that day’s collection.
# - Origination fee + ACH program fee are paid to Capnow on Day 1.
# - Broker commission default is % of Payback (editable); payment mode is configurable.

import math
from datetime import date, datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Utilities
# -----------------------------
def business_days(start: date, n_days: int) -> List[date]:
    """Generate n_days business dates (Mon–Fri) starting from start (inclusive)."""
    days = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:  # 0..4 => Mon..Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def calendar_days(start: date, n_days: int) -> List[date]:
    """Generate n_days calendar dates starting from start (inclusive)."""
    return [start + timedelta(days=i) for i in range(n_days)]


def dollars(x: float) -> float:
    """Round to cents consistently."""
    return float(np.round(x + 1e-12, 2))


def validate_syndicators(syn_rows: List[Dict]) -> str:
    if len(syn_rows) == 0:
        return "Please add at least one syndicator."
    total_pct = sum(float(r.get("Percent_of_Deal", 0) or 0) for r in syn_rows)
    if not math.isclose(total_pct, 100.0, abs_tol=0.001):
        return f"Syndicator % must sum to 100. Currently: {total_pct:.2f}%."
    for r in syn_rows:
        name = str(r.get("Name", "")).strip()
        if not name:
            return "Each syndicator must have a Name."
        tf = float(r.get("Tracker_Fee_%", 0) or 0)
        if tf < 0 or tf > 100:
            return f"Tracker fee % for {name} must be between 0 and 100."
    return ""


# -----------------------------
# Core Calculation
# -----------------------------
def compute_schedule(
    start_dt: date,
    funding_amount: float,
    payback: float,
    term_days: int,
    use_business_days: bool,
    upfront_orig_fee: float,
    upfront_ach_fee: float,
    broker_commission_total: float,
    broker_commission_mode: str,  # "Upfront (split by %)" or "Withheld from collections"
    syndicators: List[Dict[str, float]],
):
    """
    Returns:
      df (DataFrame): daily schedule
      summaries (dict): capnow + investor totals/roi
    """
    # Daily payment (flat)
    daily_payment = payback / term_days

    # Build date list
    day_list = business_days(start_dt, term_days) if use_business_days else calendar_days(start_dt, term_days)

    # Convert syndicators to typed structure
    syns = []
    for r in syndicators:
        name = str(r["Name"]).strip()
        pct = float(r["Percent_of_Deal"]) / 100.0
        tracker = float(r["Tracker_Fee_%"]) / 100.0
        syns.append({"name": name, "pct": pct, "tracker": tracker})

    # Per‑investor daily gross (before tracker fee)
    per_inv_daily_gross = {s["name"]: daily_payment * s["pct"] for s in syns}

    # Initialize tracking
    records = []
    capnow_tracker_total = 0.0
    investor_collections_net = {s["name"]: 0.0 for s in syns}  # net after tracker fee and (if withholding) broker share
    investor_initials = {s["name"]: funding_amount * s["pct"] for s in syns}

    # Broker commission split per investor (proportional to % of deal)
    broker_split = {s["name"]: broker_commission_total * s["pct"] for s in syns}
    broker_remaining = broker_split.copy()  # for withholding mode

    # Build daily schedule
    for i, d in enumerate(day_list, start=1):
        row = {
            "Day": i,
            "Date": d.isoformat(),
            "Gross_Collection": dollars(daily_payment),
        }

        capnow_tracker_today = 0.0

        # Investor-level allocations
        for s in syns:
            name = s["name"]
            gross_share = per_inv_daily_gross[name]
            tracker_fee = gross_share * s["tracker"]
            after_tracker = gross_share - tracker_fee

            # If withholding mode, deduct broker commission from collections until covered
            withheld = 0.0
            if broker_commission_mode == "Withheld from collections":
                if broker_remaining[name] > 0:
                    withheld = min(after_tracker, broker_remaining[name])
                    broker_remaining[name] = dollars(broker_remaining[name] - withheld)

            net_to_investor = after_tracker - withheld

            row[f"{name}_Gross"] = dollars(gross_share)
            row[f"{name}_TrackerFee"] = dollars(tracker_fee)
            row[f"{name}_WithheldForBroker"] = dollars(withheld)
            row[f"{name}_NetToInvestor"] = dollars(net_to_investor)

            capnow_tracker_today += tracker_fee
            investor_collections_net[name] = dollars(investor_collections_net[name] + net_to_investor)

        row["Capnow_Tracker_Fees"] = dollars(capnow_tracker_today)
        capnow_tracker_total = dollars(capnow_tracker_total + capnow_tracker_today)
        records.append(row)

    df = pd.DataFrame(records)

    # Capnow upfront revenue
    capnow_upfront = dollars(upfront_orig_fee + upfront_ach_fee)
    capnow_total = dollars(capnow_upfront + capnow_tracker_total)

    # Investors: if commission was upfront, subtract their share now from net collections
    investor_final_net = {}
    for s in syns:
        name = s["name"]
        if broker_commission_mode == "Upfront (split by %)":
            final_net = investor_collections_net[name] - broker_split[name]
        else:
            final_net = investor_collections_net[name]
        investor_final_net[name] = dollars(final_net)

    # Profit & ROI
    investor_profit = {}
    investor_roi = {}
    for s in syns:
        name = s["name"]
        invested = investor_initials[name]
        profit = investor_final_net[name] - invested
        investor_profit[name] = dollars(profit)
        investor_roi[name] = round(100.0 * profit / invested, 2) if invested > 0 else 0.0

    # Summaries
    summaries = {
        "Inputs": {
            "Start Date": day_list[0].isoformat(),
            "Funding Amount": funding_amount,
            "Payback": payback,
            "Term Days": term_days,
            "Daily Payment": dollars(daily_payment),
            "Business Days": use_business_days,
            "Origination Fee (Day 1 → Capnow)": dollars(upfront_orig_fee),
            "ACH Program Fee (Day 1 → Capnow)": dollars(upfront_ach_fee),
            "Broker Commission (Total)": dollars(broker_commission_total),
            "Broker Commission Mode": broker_commission_mode,
        },
        "Capnow": {
            "Upfront Fees (Day 1)": dollars(capnow_upfront),
            "Tracker Fees (over term)": dollars(capnow_tracker_total),
            "Total Capnow Revenue": dollars(capnow_total),
        },
        "Investors": {
            name: {
                "Invested": dollars(investor_initials[name]),
                "Collections Net (after tracker & broker per mode)": dollars(investor_final_net[name]),
                "Profit": dollars(investor_profit[name]),
                "ROI_%": investor_roi[name],
            }
            for name in investor_final_net
        },
    }

    return df, summaries


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CAPNOW Syndication Calculator", page_icon="💸", layout="wide")
st.title("💸 CAPNOW Syndication Deal Calculator")

with st.sidebar:
    st.header("Deal Terms")

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start Date", value=date(2025, 8, 21))
        funding_amount = st.number_input("Funding Amount ($)", min_value=0.0, value=5000.0, step=100.0)
        term_days = st.number_input("Term (days)", min_value=1, value=40, step=1)
        use_business_days = st.toggle("Use Business Days (Mon–Fri)", value=True)
    with col_b:
        rate = st.number_input("Rate (e.g., 1.48)", min_value=1.0, value=1.48, step=0.01)
        payback_override = st.number_input("Payback Override ($) [optional]", min_value=0.0, value=7400.0, step=100.0,
                                           help="If non-zero, this overrides Rate×Funding for payback.")
        payback = payback_override if payback_override > 0 else dollars(funding_amount * rate)

    st.markdown("---")
    st.subheader("Fees")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        orig_fee = st.number_input("Origination Fee → Capnow (Day 1)", min_value=0.0, value=500.0, step=50.0)
    with col_f2:
        ach_fee = st.number_input("ACH Program Fee → Capnow (Day 1)", min_value=0.0, value=395.0, step=25.0)
    with col_f3:
        broker_comm_pct = st.number_input("Broker Commission % of Payback", min_value=0.0, value=7.0, step=0.5)
    broker_comm_override = st.number_input("Broker Commission Override ($) [optional]", min_value=0.0, value=0.0, step=50.0)
    if broker_comm_override > 0:
        broker_comm_total = broker_comm_override
    else:
        broker_comm_total = dollars(payback * (broker_comm_pct / 100.0))

    st.markdown("---")
    st.subheader("Broker Commission Handling")
    broker_mode = st.radio(
        "How should broker commission be paid?",
        options=["Upfront (split by %)", "Withheld from collections"],
        index=0,
        help=(
            "Upfront: each investor pays their proportional share immediately.\n"
            "Withheld: investor collections are reduced each day until their share is fully covered."
        ),
    )

st.subheader("Syndicators")
st.caption("Enter each syndicator’s % of the deal and their tracker-fee %. The % of Deal must total 100%.")
default_rows = [
    {"Name": "Jacobo", "Percent_of_Deal": 50.0, "Tracker_Fee_%": 5.0},
    {"Name": "Albert", "Percent_of_Deal": 50.0, "Tracker_Fee_%": 4.0},
]
syn_df = st.data_editor(
    pd.DataFrame(default_rows),
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Name": st.column_config.TextColumn(required=True),
        "Percent_of_Deal": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=100.0, step=0.1),
        "Tracker_Fee_%": st.column_config.NumberColumn(format="%.2f", min_value=0.0, max_value=100.0, step=0.1),
    },
    key="syn_table",
)

err = validate_syndicators(syn_df.to_dict(orient="records"))
if err:
    st.error(err)
    st.stop()

# Compute
df, summaries = compute_schedule(
    start_dt=start_date,
    funding_amount=funding_amount,
    payback=payback,
    term_days=term_days,
    use_business_days=use_business_days,
    upfront_orig_fee=orig_fee,
    upfront_ach_fee=ach_fee,
    broker_commission_total=broker_comm_total,
    broker_commission_mode=broker_mode,
    syndicators=syn_df.to_dict(orient="records"),
)

# -----------------------------
# Summaries UI
# -----------------------------
st.markdown("### Summary")

cap = summaries["Capnow"]
inp = summaries["Inputs"]
inv = summaries["Investors"]

cap_col1, cap_col2, cap_col3, cap_col4 = st.columns(4)
with cap_col1:
    st.metric("Funding Amount", f"${inp['Funding Amount']:,.2f}")
with cap_col2:
    st.metric("Payback", f"${inp['Payback']:,.2f}")
with cap_col3:
    st.metric("Daily Payment", f"${inp['Daily Payment']:,.2f}")
with cap_col4:
    st.metric("Term Days", f"{inp['Term Days']} {'(Biz)' if inp['Business Days'] else '(Cal)'}")

cap2_col1, cap2_col2, cap2_col3 = st.columns(3)
with cap2_col1:
    st.metric("Capnow Upfront (Day 1)", f"${cap['Upfront Fees (Day 1)']:,.2f}")
with cap2_col2:
    st.metric("Capnow Tracker Fees", f"${cap['Tracker Fees (over term)']:,.2f}")
with cap2_col3:
    st.metric("Capnow Total Revenue", f"${cap['Total Capnow Revenue']:,.2f}")

st.markdown("#### Investor Results")
inv_cols = st.columns(len(inv))
for i, (name, v) in enumerate(inv.items()):
    with inv_cols[i]:
        st.metric(f"{name} — Invested", f"${v['Invested']:,.2f}")
        st.metric(f"{name} — Collections Net", f"${v['Collections Net (after tracker & broker per mode)']:,.2f}")
        st.metric(f"{name} — Profit", f"${v['Profit']:,.2f}")
        st.metric(f"{name} — ROI", f"{v['ROI_%']:.2f}%")

# -----------------------------
# Daily Schedule Table
# -----------------------------
st.markdown("### Daily Cashflow Schedule")
st.dataframe(df, use_container_width=True)

# -----------------------------
# CSV Download
# -----------------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Schedule as CSV",
    data=csv,
    file_name="syndication_cashflow_schedule.csv",
    mime="text/csv",
)

# -----------------------------
# Debug / Inputs Echo (optional)
# -----------------------------
with st.expander("Deal Inputs (for reference)"):
    st.json(inp)
with st.expander("Capnow Totals"):
    st.json(cap)
with st.expander("Investor Totals"):
    st.json(inv)
