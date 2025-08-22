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
        use_biz = st.toggle("Use Business Days (Mon–Fri)", value=True)
    with c3:
        rate = st.number_input("Rate (×)", min_value=1.0, value=1.48, step=0.01, format="%.2f")
        st.caption("Computed Payback")
        st.info(fmt_money(funding * rate))
    with c4:
        orig_pct = st.number_input("Origination Fee (% of Funding)", min_value=0.0, value=10.0, step=0.5, format="%.2f")
        orig_val = dollars(funding * (orig_pct / 100.0))
        st.caption("Origination $ (auto)")
        st.info(fmt_money(orig_val))

    c5, c6 = st.columns(2)
    with c5:
        ach_fee = st.number_input("ACH Program Fee ($) → Capnow (Day 1)", min_value=0.0, value=395.0, step=25.0, format="%.2f")
    with c6:
        broker_pct = st.number_input(
            "Broker Commission (% of Funding, Day 1 by Investors)",
            min_value=0.0, value=7.0, step=0.5, format="%.2f"
        )
        broker_val = dollars(funding * (broker_pct / 100.0))
        st.caption("Broker Commission $ (auto)")
        st.info(fmt_money(broker_val))
        st.caption("Charged to investors Day-1, proportional to their deal %.")

# Syndicators tab
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

# Results tab
with tab_results:
    err = validate_syndicators(syn_df.to_dict(orient="records"))
    if err:
        st.warning("Fix Syndicators before viewing results.")
        st.stop()

    df, summaries = compute_schedule(
        start_dt=start_date,
        funding=float(funding),
        rate=float(rate),
        term_days=int(term_days),
        use_business_days=bool(use_biz),
        orig_fee_pct_of_funding=float(orig_pct),
        ach_fee_capnow=float(ach_fee),
        broker_comm_pct_of_funding=float(broker_pct),
        syndicators_table=syn_df,
    )

    st.subheader("Summary")
    inp, fees, cap, client, inv = summaries["Inputs"], summaries["Fees"], summaries["Capnow"], summaries["Client"], summaries["Investors"]

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1: st.metric("Funding Amount", fmt_money(inp['Funding Amount']))
    with m2: st.metric("Payback (Funding × Rate)", fmt_money(inp['Payback']))
    with m3: st.metric("Daily Payment", fmt_money(inp['Daily Payment']))
    with m4: st.metric("Term Days", f"{inp['Term Days']} {'(Biz)' if inp['Business Days'] else '(Cal)'}")
    with m5: st.metric("Merchant Nets", fmt_money(client['Merchant Nets']))

    f1, f2, f3, f4 = st.columns(4)
    with f1: st.metric("Origination % of Funding", f"{fees['Origination % of Funding (Day 1)']:.2f}%")
    with f2: st.metric("Origination $ (Day 1 → Capnow)", fmt_money(fees['Origination $ (Day 1 → Capnow)']))
    with f3: st.metric("ACH $ (Day 1 → Capnow)", fmt_money(fees['ACH Program $ (Day 1 → Capnow)']))
    with f4: st.metric("Broker % of Funding (Day 1 by Investors)", f"{fees['Broker % of Funding (Day 1 by Investors)']:.2f}%")

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Capnow Upfront (Day 1)", fmt_money(cap['Upfront (Day 1)']))
    with c2: st.metric("Capnow Tracker Fees", fmt_money(cap['Tracker Fees (over term)']))
    with c3: st.metric("Capnow Total Revenue", fmt_money(cap['Total Revenue']))

    st.markdown("#### Investor Results")
    cols = st.columns(len(inv))
    for i, (name, v) in enumerate(inv.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            metric_with_popover(f"{name}_dealpct", "% ON THE DEAL", f"{v['Deal %']:.2f}%",
                                "Investor’s share of the deal capital.")
            metric_with_popover(f"{name}_invested", "Invested Total $", fmt_money(v['Invested Principal']),
                                "Investor’s principal funded into the deal.")
            metric_with_popover(f"{name}_broker", "Commission Paid to Broker", fmt_money(v['Broker Share (Day 1)']),
                                "Due **on Day-1** from the investor: **Funding × Broker% × Investor%**. This goes to the broker.")
            metric_with_popover(f"{name}_cafs", "CAFS on Deal", fmt_money(v['Tracker Fees (Total → Capnow)']),
                                "**Collected on each payment.** Sum of **Daily Payment × Investor% × Tracker%** over the term. Paid to Capnow.")
            metric_with_popover(f"{name}_totalfees", "TOTAL FEES", fmt_money(v['Total Fees Paid']),
                                "**TOTAL FEES = Commission Paid to Broker + CAFS on Deal**.")
            metric_with_popover(f"{name}_day1", "Investment On Day 1", fmt_money(v['Total Day-1 Cash Out']),
                                "**Investment On Day 1 = Invested Principal + Commission Paid to Broker**.")
            metric_with_popover(f"{name}_net", "Net on Investment", fmt_money(v['Collections Net (after tracker)']),
                                "Total collected by the investor **after tracker fees** across all days.")
            metric_with_popover(f"{name}_profit", "Profit on Investment", fmt_money(v['Profit']),
                                "**Profit = Net on Investment − Investment On Day 1**.")
            metric_with_popover(f"{name}_roi", "ROI on Investment", f"{v['ROI_%']:.2f}%",
                                "**ROI = Profit ÷ Investment On Day 1**.")

    # Per-Investor Fees table
    st.markdown("### Per-Investor Fees")
    fees_rows = []
    for name, v in inv.items():
        fees_rows.append({
            "Investor": name,
            "Commission Paid to Broker": v["Broker Share (Day 1)"],
            "CAFS on Deal": v["Tracker Fees (Total → Capnow)"],
            "TOTAL FEES": v["Total Fees Paid"],
        })
    fees_df = pd.DataFrame(fees_rows)
    for col in ["Commission Paid to Broker", "CAFS on Deal", "TOTAL FEES"]:
        fees_df[col] = fees_df[col].map(lambda x: dollars(x))
    st.dataframe(fees_df, use_container_width=True)

    st.markdown("### Daily Cashflow Schedule")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Daily Schedule (CSV)", data=csv, file_name="syndication_cashflow_schedule.csv", mime="text/csv")

    with st.expander("Raw JSON: Inputs / Fees / Capnow / Client / Investors"):
        st.json({"Inputs": inp, "Fees": fees, "Capnow": cap, "Client": client, "Investors": inv})
