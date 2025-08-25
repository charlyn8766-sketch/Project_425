
import streamlit as st

# Imports
try:
    import pandas as pd
    import numpy as np
except Exception:
    st.error("Please install dependencies: pip install -r requirements.txt")
    st.stop()

from optimizer import solve_schedule

st.set_page_config(page_title="Shift Scheduler — Weekly Chart Only", layout="wide")
st.title("Shift Scheduler — Weekly Chart Only")
st.caption("This build renders only the weekly aggregate line chart.")

# Constants for 13 slots (12:00–24:00)
DAYS = list(range(1,8))
SLOTS = list(range(1,14))
SLOT_LABELS = ["12-13","13-14","14-15","15-16","16-17","17-18","18-19","19-20","20-21","21-22","22-23","23-00","00-01"]

# Default staff (kept minimal; users can upload their own CSV with columns: name,min_week_hours,max_week_hours)
DEFAULT_STAFF = [
    {"name":"Cristina_Mata",    "min_week_hours":25.0, "max_week_hours":32.5},
    {"name":"Gabriela_Velasco", "min_week_hours":20.0, "max_week_hours":26.0},
    {"name":"Javi",             "min_week_hours":27.0, "max_week_hours":27.5},
    {"name":"Lorena",           "min_week_hours":25.0, "max_week_hours":32.5},
    {"name":"Aurora",           "min_week_hours":25.0, "max_week_hours":32.5},
    {"name":"Clara_Nogales",    "min_week_hours":25.0, "max_week_hours":32.5},
]

with st.sidebar:
    st.header("Config")
    max_dev = st.number_input("Max deviation per slot", min_value=0.0, value=2.5, step=0.1)
    require_min_staff = st.checkbox("Require at least 1 staff per slot", value=True)

    st.subheader("Staff CSV (optional)")
    st.caption("Columns: name,min_week_hours,max_week_hours")
    staff_upload = st.file_uploader("Upload staff.csv", type=["csv"], key="staff")
    if "staff_df" not in st.session_state:
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)
    if staff_upload is not None:
        st.session_state["staff_df"] = pd.read_csv(staff_upload)
    staff_df = st.session_state["staff_df"]

    st.subheader("Demand CSV (7x13, no header)")
    st.caption("Rows: Mon..Sun; Cols: 13 hourly slots (12..24). If omitted, demo will be used.")
    demand_file = st.file_uploader("Upload demand CSV", type=["csv"], key="demand")

# Demo demand (7x13) if none uploaded
DEMO = np.array([
    [0.23,0.25,0.70,1.39,0.80,1.16,1.27,0.28,0.89,1.18,0.91,0.08,0.26],
    [0.76,0.77,0.56,0.79,0.45,0.29,0.45,1.43,1.42,0.05,0.73,0.75,0.99],
    [0.31,0.38,0.29,0.53,0.49,1.63,0.53,0.35,0.01,0.26,0.29,0.88,0.35],
    [0.86,0.65,0.37,1.71,1.33,2.14,0.69,0.17,0.29,0.72,0.80,1.59,0.72],
    [0.30,0.75,1.05,1.28,0.56,1.09,0.55,0.80,1.20,0.43,0.97,0.50,0.74],
    [2.23,1.25,0.21,0.23,0.59,1.10,1.74,1.60,1.09,0.92,1.17,0.24,1.13],
    [0.37,0.59,1.10,1.74,1.60,1.09,0.92,1.17,0.24,1.13,1.89,0.80,0.00],
])
if demand_file is not None:
    demand_df = pd.read_csv(demand_file, header=None)
else:
    demand_df = pd.DataFrame(DEMO)

if demand_df.shape != (7,13):
    st.error(f"Demand CSV must be 7 rows x 13 columns. Got {demand_df.shape}.")
    st.stop()

# Convert staff table to dicts
def to_hours_dict(df, col):
    out = {}
    for _, r in df.iterrows():
        out[str(r["name"])] = float(r[col])
    return out

MinHw = to_hours_dict(staff_df, "min_week_hours")
MaxHw = to_hours_dict(staff_df, "max_week_hours")
W = [str(x) for x in staff_df["name"].tolist()]

# --- Solve & Plot ONLY the weekly line chart ---
if st.button("Solve", type="primary"):
    with st.spinner("Solving..."):
        Demand = {d: list(map(float, demand_df.iloc[d-1].tolist())) for d in DAYS}
        status, obj, schedule, metrics = solve_schedule(
            W=W, D=DAYS, T=SLOTS,
            MinHw=MinHw, MaxHw=MaxHw,
            Demand=Demand,
            Max_Deviation=max_dev,
            require_min_staff=require_min_staff,
        )

    # Build a coverage dataframe to aggregate weekly totals per slot
    rows = []
    for (d,t),(u,o,staffed,dem) in metrics.items():
        rows.append({"day": d, "slot": t, "staffed": staffed, "demand": dem})
    cov_df = pd.DataFrame(rows)

    # Weekly aggregate by slot
    weekly = (cov_df.groupby("slot")[["demand","staffed"]].sum()
                    .reindex(range(1,14))
                    .reset_index())

    # Draw the ONLY chart
    st.markdown("### Weekly Demand vs Staffing (aggregate over 7 days)")
    line_df = pd.DataFrame({"Demand": weekly["demand"], "Staffed": weekly["staffed"]},
                           index=SLOT_LABELS)
    st.line_chart(line_df)
