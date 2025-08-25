
import streamlit as st

# --- Guarded third-party imports ---
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    st.error("Missing dependencies. Please: pip install -r requirements.txt")
    st.stop()

import inspect, os, sys, math, time as _time
from io import BytesIO

# ------------------ Defaults ------------------
DEFAULT_STAFF = [
    {"name":"Jade",              "min_week_hours":35},
    {"name":"Ulisses",           "min_week_hours":25},
    {"name":"Angela",            "min_week_hours":20},
    {"name":"Jesus",             "min_week_hours":25},
    {"name":"Carla",             "min_week_hours":25},
    {"name":"Macarena_Sevilla",  "min_week_hours":25},
    {"name":"Rafael",            "min_week_hours":25},
    {"name":"Aitana",            "min_week_hours":25},
    {"name":"Diana",             "min_week_hours":25},
]
def _round_half(x): return math.floor(x*2+0.5)/2.0
for r in DEFAULT_STAFF:
    min_h = r["min_week_hours"]
    r["max_week_hours"] = _round_half(min(40.0, min_h*1.3))
    r["contract_hours"] = min_h
    r["weekend_only"] = False

# --- Import optimizer ---
opt_mod = None
try:
    import optimizer as _opt_mod
    opt_mod = _opt_mod
except Exception:
    here = os.path.dirname(__file__)
    if here and here not in sys.path:
        sys.path.insert(0, here)
    import optimizer as _opt_mod
    opt_mod = _opt_mod

def build_shift_set_fallback(T, min_len=4, max_len=8):
    S = []
    for s in T:
        for e in T:
            L = e - s + 1
            if min_len <= L <= max_len:
                S.append((s,e))
    return S

def adapt_to_user_optimizer(demand_df, staff_df, max_dev):
    if opt_mod is None or not hasattr(opt_mod, "build_and_solve_shift_model"):
        st.error("optimizer.py must define build_and_solve_shift_model(...)")
        st.stop()

    W = list(staff_df["name"])
    D = list(range(1, 8))
    T = list(range(1, 16))

    if hasattr(opt_mod, "build_shift_set"):
        try:
            S = opt_mod.build_shift_set(T, 4, 8)
        except Exception:
            S = build_shift_set_fallback(T, 4, 8)
    else:
        S = build_shift_set_fallback(T, 4, 8)

    MinHw = {row["name"]: float(row["min_week_hours"]) for _, row in staff_df.iterrows()}
    MaxHw = {row["name"]: float(row["max_week_hours"]) for _, row in staff_df.iterrows()}

    Demand = {d: [float(x) for x in demand_df.iloc[d-1, :].tolist()] for d in D}

    fn = getattr(opt_mod, "build_and_solve_shift_model")
    try:
        res = fn(W, D, T, S, MinHw, MaxHw, Demand, Max_Deviation=max_dev)
    except TypeError:
        res = fn(W, D, T, S, MinHw, MaxHw, Demand, Max_Deviation=max_dev, time_limit=None)

    # Normalize outputs
    out = {"status": res.get("status","OK"),
           "objective": res.get("objective", float("nan")),
           "elapsed_time": res.get("elapsed_time", float("nan"))}

    schedule = res.get("schedule", [])
    rows = []
    for d in D:
        for t in T:
            staffed = sum(1 for (w_, d_, t_) in schedule if d_ == d and t_ == t)
            demand_val = float(demand_df.iloc[d-1, t-1])
            rows.append({"day": d, "slot": t, "demand": demand_val, "staffed": staffed})
    out["coverage_df"] = pd.DataFrame(rows)
    return out

# ------------------ UI ------------------
st.set_page_config(page_title="Shift Scheduler — WEEKLY CHART ONLY (Minimal)", layout="wide")
st.title("Shift Scheduler — WEEKLY CHART ONLY (Minimal)")
st.caption(f"USING FILE: {__file__}")
st.caption(f"LAST MODIFIED: {_time.strftime('%Y-%m-%d %H:%M:%S', _time.localtime(os.path.getmtime(__file__)))}")
st.success("BUILD: WEEKLY_ONLY_MINIMAL")
st.warning("本版本严格只显示一张“周度汇总折线图”。不显示任何每日图表、柱状图、或逐日表格。")

SLOT_LABELS = ["10-11","11-12","12-13","13-14","14-15","15-16","16-17","17-18","18-19","19-20","20-21","21-22","22-23","23-00","00-01"]

with st.sidebar:
    st.header("Configuration")
    max_dev = st.number_input("Max deviation per slot (people)", min_value=0.0, value=2.5, step=0.5)

    st.markdown("---")
    st.subheader("Demand CSV")
    st.caption("Upload a 7x15 CSV (no header): rows=Mon..Sun, cols=15 hourly slots (10-11,...,00-01).")
    demand_file = st.file_uploader("Upload sales_demand_template.csv", type=["csv"])

    st.markdown("---")
    st.subheader("Staff table (编辑后仅用于求解，不会显示逐日表格)")
    uploaded_staff = st.file_uploader("Upload staff CSV (optional)", type=["csv"], key="staff_csv")
    if "staff_df" not in st.session_state:
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)
    if uploaded_staff is not None:
        st.session_state["staff_df"] = pd.read_csv(uploaded_staff)
    if st.button("Load default staff (10)"):
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)

    staff_df = st.data_editor(
        st.session_state["staff_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("name"),
            "min_week_hours": st.column_config.NumberColumn("min week hours"),
            "max_week_hours": st.column_config.NumberColumn("max week hours"),
            "contract_hours": st.column_config.NumberColumn("contract hours"),
            "weekend_only": st.column_config.CheckboxColumn("weekend only"),
        },
        hide_index=True
    )
    st.session_state["staff_df"] = staff_df
    st.caption(f"Current staff count: **{len(staff_df)}**")

# Load demand (or demo)
if demand_file is not None:
    demand = pd.read_csv(demand_file, header=None)
else:
    st.info("Using a demo 7x15 demand matrix (no upload).")
    demand = pd.DataFrame(np.array([
        [0.00,0.89,1.08,1.15,2.51,3.11,2.16,4.06,1.64,1.45,1.31,2.68,2.73,2.14,0.86],
        [0.37,1.08,0.90,0.59,2.64,3.40,3.26,3.97,0.86,1.51,1.63,1.77,2.53,2.58,0.07],
        [0.12,0.80,1.67,2.64,2.43,2.64,2.87,2.25,2.61,1.62,1.60,0.88,1.90,2.25,0.72],
        [0.63,1.00,1.67,2.46,1.56,1.91,2.58,2.04,2.63,2.11,1.04,1.34,2.31,2.12,0.61],
        [0.31,0.74,1.39,1.88,2.77,1.75,4.15,3.55,1.85,2.22,1.57,1.34,3.27,3.07,0.76],
        [0.66,0.48,0.64,1.05,1.85,3.61,4.63,3.06,1.99,2.04,1.77,1.82,2.87,3.40,0.88],
        [0.26,0.52,1.46,2.39,1.43,3.18,3.79,3.23,2.91,1.41,2.06,2.28,2.18,2.03,0.86],
    ]))

# Validate shape
if demand.shape != (7,15):
    st.error(f"Demand CSV must be 7 rows x 15 columns. Current shape: {demand.shape}")
    st.stop()

# --------- Main action ---------
st.markdown("### Run Optimizer")
if st.button("Solve now", type="primary"):
    with st.spinner("Solving..."):
        res = adapt_to_user_optimizer(demand, st.session_state["staff_df"], max_dev)

    st.success(f"Status: {res.get('status','N/A')}, Objective (total deviation): {res.get('objective', float('nan')):.4f}")

    # ------- WEEKLY ONLY CHART -------
    cov = res["coverage_df"]
    weekly_df = (cov.groupby("slot").agg({"demand":"sum", "staffed":"sum"}).reset_index().sort_values("slot"))
    x_labels = SLOT_LABELS if len(weekly_df) == len(SLOT_LABELS) else [str(s) for s in weekly_df["slot"]]

    st.markdown("### Weekly Demand vs Staffing (aggregate over 7 days)")
    st.line_chart(pd.DataFrame({"Demand": weekly_df["demand"].to_numpy(),
                                "Staffed": weekly_df["staffed"].to_numpy()}, index=x_labels))

    deviation = weekly_df["staffed"] - weekly_df["demand"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total under", f"{max(0, (weekly_df['demand']-weekly_df['staffed']).sum()):.1f}")
    c2.metric("Total over", f"{max(0, (weekly_df['staffed']-weekly_df['demand']).sum()):.1f}")
    c3.metric("Max |deviation|", f"{float(abs(deviation).max()):.1f}")

    # Do NOT render any daily tables or per-worker grids.
