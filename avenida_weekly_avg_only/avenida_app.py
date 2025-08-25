
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import math

from optimizer import build_and_solve_shift_model

st.set_page_config(page_title="Avenida Scheduler (13 slots) — Weekly Chart Only", layout="wide")
st.title("Avenida Scheduler (13 slots) — Weekly Chart Only")
st.success("BUILD: WEEKLY_ONLY (13 slots) — Weekly line = AVERAGE")

SLOT_LABELS = ["12-13","13-14","14-15","15-16","16-17","17-18","18-19","19-20","20-21","21-22","22-23","23-00","00-01"]
DAY_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
D = list(range(1,8))
T = list(range(1,14))  # 13 slots
S = [(s,e) for s in T for e in T if 4 <= (e - s + 1) <= 8]

# Default staff (editable)
DEFAULT_STAFF = [
    {"name":"Cristina_Mata",    "min_week_hours":25.0},
    {"name":"Gabriela_Velasco", "min_week_hours":20.0},
    {"name":"Javi",             "min_week_hours":27.0},
    {"name":"Lorena",           "min_week_hours":25.0},
    {"name":"Aurora",           "min_week_hours":25.0},
    {"name":"Clara_Nogales",    "min_week_hours":25.0},
]
def _round_half(x): return math.floor(x*2+0.5)/2.0
for r in DEFAULT_STAFF:
    r["max_week_hours"] = _round_half(min(40.0, r["min_week_hours"]*1.3))
    r["contract_hours"] = r["min_week_hours"]
    r["weekend_only"] = False

with st.sidebar:
    st.header("Configuration")
    max_dev = st.number_input("Max deviation per slot (people)", min_value=0.0, value=2.5, step=0.5)

    st.markdown("---")
    st.subheader("Demand CSV (7×13, no header)")
    st.caption("Rows: Mon..Sun; Cols: 13 hourly slots (12..24).")
    demand_file = st.file_uploader("Upload demand.csv", type=["csv"])

    st.markdown("---")
    st.subheader("Staff table")
    st.caption("Edit staff below. You can add or delete rows. Download/Upload to reuse.")
    uploaded_staff = st.file_uploader("Upload staff CSV (optional)", type=["csv"], key="staff_csv")
    if "staff_df" not in st.session_state:
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)
    if uploaded_staff is not None:
        st.session_state["staff_df"] = pd.read_csv(uploaded_staff)
    if st.button("Load default staff"):
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)

    staff_df = st.data_editor(
        st.session_state["staff_df"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("name", help="Unique worker name"),
            "min_week_hours": st.column_config.NumberColumn("min work hour", help="Min weekly hours"),
            "max_week_hours": st.column_config.NumberColumn("max work hour", help="Max weekly hours"),
            "contract_hours": st.column_config.NumberColumn("contract hours", help="Reference contract hours"),
            "weekend_only": st.column_config.CheckboxColumn("weekend only", help="If True, only Fri/Sat/Sun"),
        },
        hide_index=True
    )
    st.session_state["staff_df"] = staff_df
    st.caption(f"Current staff count: **{len(staff_df)}**")

    staff_csv = staff_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download staff CSV", staff_csv, file_name="staff.csv", mime="text/csv")

# Load demand (or demo)
if demand_file is not None:
    demand = pd.read_csv(demand_file, header=None)
else:
    st.info("Using a demo 7×13 demand matrix (no upload).")
    demand = pd.DataFrame(np.array([
        [0.23,0.25,0.70,1.39,0.80,1.16,1.27,0.28,0.89,1.18,0.91,0.08,0.26],
        [0.76,0.77,0.56,0.79,0.45,0.29,0.45,1.43,1.42,0.05,0.73,0.75,0.99],
        [0.31,0.38,0.29,0.53,0.49,1.63,0.53,0.35,0.01,0.26,0.29,0.88,0.35],
        [0.86,0.65,0.37,1.71,1.33,2.14,0.69,0.17,0.29,0.72,0.80,1.59,0.72],
        [0.30,0.75,1.05,1.28,0.56,1.09,0.55,0.80,1.20,0.43,0.97,0.50,0.74],
        [2.23,1.25,0.21,0.23,0.59,1.10,1.74,1.60,1.09,0.92,1.17,0.24,1.13],
        [0.37,0.59,1.10,1.74,1.60,1.09,0.92,1.17,0.24,1.13,1.89,0.80,0.00],
    ]))

if demand.shape != (7,13):
    st.error(f"Demand CSV must be 7 rows × 13 columns. Current shape: {demand.shape}")
    st.stop()

# ---------- Solve ----------
if st.button("Solve now", type="primary"):
    # Prepare inputs
    W = list(st.session_state["staff_df"]["name"])
    MinHw = {row["name"]: float(row["min_week_hours"]) for _, row in st.session_state["staff_df"].iterrows()}
    MaxHw = {row["name"]: float(row["max_week_hours"]) for _, row in st.session_state["staff_df"].iterrows()}
    Demand = {d: [float(x) for x in demand.iloc[d-1, :].tolist()] for d in D}

    with st.spinner("Solving..."):
        res = build_and_solve_shift_model(W, D, T, S, MinHw, MaxHw, Demand, Max_Deviation=max_dev)

    st.success(f"Status: {res.get('status','N/A')}, Objective (total deviation): {res.get('objective', float('nan')):.4f}")

    # Hours per worker
    schedule = res.get("schedule", [])
    hours_rows = []
    for w in W:
        total_hours = sum(1 for (w_,_,_) in schedule if w_ == w)
        hours_rows.append({"name": w, "total_hours": total_hours, "min_week_hours": MinHw[w], "max_week_hours": MaxHw[w]})
    st.write("Weekly hours per worker")
    st.dataframe(pd.DataFrame(hours_rows), use_container_width=True)

    # Coverage by day-slot table
    cov_rows = []
    for d in D:
        for t in T:
            staffed = sum(1 for (w_,d_,t_) in schedule if d_==d and t_==t)
            dem = float(demand.iloc[d-1, t-1])
            under = max(0.0, dem - staffed)
            over  = max(0.0, staffed - dem)
            cov_rows.append({"day": d, "slot": t, "demand": dem, "staffed": staffed, "under": under, "over": over})
    coverage_df = pd.DataFrame(cov_rows)
    st.write("Coverage by day-slot (demand / staffed / under / over)")
    st.dataframe(coverage_df, use_container_width=True)

    # ---- Weekly line chart ONLY (AVERAGE over 7 days) ----
    st.markdown("### Weekly Demand vs Staffing (average per day)")
    weekly_avg = (coverage_df.groupby("slot")[["demand","staffed"]]
                              .mean()
                              .reindex(range(1,14))
                              .reset_index())

    line_df = pd.DataFrame({
        "Avg Demand": weekly_avg["demand"].to_numpy(),
        "Avg Staffed": weekly_avg["staffed"].to_numpy()
    }, index=SLOT_LABELS)
    st.line_chart(line_df, use_container_width=True)
    st.caption("Averages = mean across Mon–Sun for each slot.")

    # Per-worker 7×13 schedule (blue highlight)
    st.markdown("### Per-worker Schedule (7×13, color = scheduled)")
    def style_schedule(df):
        return df.style.apply(lambda s: ['background-color: #E6F4FF' if v==1 else '' for v in s], axis=1)

    worker_tables = {}
    for w in W:
        mat = np.zeros((7,13), dtype=int)
        for (w_, d, t) in schedule:
            if w_ == w:
                mat[d-1, t-1] = 1
        df = pd.DataFrame(mat, columns=SLOT_LABELS, index=DAY_LABELS)
        worker_tables[w] = df

    for w in W:
        with st.expander(w):
            st.dataframe(style_schedule(worker_tables[w]), use_container_width=True)

    # Download per-worker Excel
    output = BytesIO()
    try:
        import xlsxwriter
        engine = "xlsxwriter"
    except Exception:
        engine = None
    with pd.ExcelWriter(output, engine=engine) as writer:
        for w, df in worker_tables.items():
            sheet_name = w[:31] if w else "Worker"
            df.to_excel(writer, sheet_name=sheet_name)
    st.download_button(
        "Download per-worker schedule (Excel)",
        data=output.getvalue(),
        file_name="avenida_per_worker_schedule.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
