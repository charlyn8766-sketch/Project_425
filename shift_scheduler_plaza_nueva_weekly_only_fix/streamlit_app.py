
import streamlit as st

# Guarded imports
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    st.error("Please install pandas and numpy: pip install -r requirements.txt")
    st.stop()

from optimizer import solve_schedule

st.set_page_config(page_title="Plaza Nueva Shift Scheduler (12-24)", layout="wide")
st.title("Plaza Nueva Shift Scheduler (12:00–24:00)")

# Sets
D = list(range(1,8))
T = list(range(1,14))  # 13 slots
SLOT_LABELS = ["12-13","13-14","14-15","15-16","16-17","17-18",
               "18-19","19-20","20-21","21-22","22-23","23-24","00-01"]
DAY_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# Default staff
DEFAULT_STAFF = [
    {"name":"Irene",               "min_week_hours":35.0, "max_week_hours":40.0},
    {"name":"Leslie_Ann",          "min_week_hours":25.0, "max_week_hours":32.5},
    {"name":"Leonardo",            "min_week_hours":25.0, "max_week_hours":32.5},
    {"name":"Gabriela_Martinez",   "min_week_hours":30.0, "max_week_hours":39.0},
    {"name":"Eulogio",             "min_week_hours":25.0, "max_week_hours":32.5},
    {"name":"Antonio_S_Garcia",    "min_week_hours":20.0, "max_week_hours":26.0},
]

with st.sidebar:
    st.header("Configuration")
    max_dev = st.number_input("Max deviation per slot", min_value=0.0, value=2.5, step=0.1)
    ensure_min_staff = st.checkbox("Require at least 1 staff per slot", value=True)

    st.subheader("Staff table")
    if "staff_df" not in st.session_state:
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)
    staff_upload = st.file_uploader("Upload staff CSV (optional)", type=["csv"])
    if staff_upload is not None:
        st.session_state["staff_df"] = pd.read_csv(staff_upload)
    if st.button("Load default staff"):
        st.session_state["staff_df"] = pd.DataFrame(DEFAULT_STAFF)
    staff_df = st.data_editor(
        st.session_state["staff_df"],
        num_rows="dynamic",
        use_container_width=True
    )

    st.subheader("Demand CSV (7x13, no header)")
    st.caption("Rows: Mon..Sun; Cols: 13 hourly slots (12..24). If omitted, built-in default is used.")
    demand_file = st.file_uploader("Upload sales_demand_template.csv", type=["csv"])

# Default demand (7x13)
default_demand = {
    1: [0.12, 0.27, 0.36, 0.49, 1.45, 0.35, 1.23, 1.54, 1.30, 1.84, 1.01, 0.43, 0.05],
    2: [0.97, 0.43, 0.72, 0.60, 0.50, 0.55, 0.65, 1.11, 1.47, 0.91, 0.43, 0.00, 0.00],
    3: [0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.19, 0.61, 0.54, 0.50, 0.06, 0.74, 1.39],
    4: [1.00, 1.31, 1.16, 0.48, 0.61, 0.18, 1.45, 0.71, 1.35, 0.98, 0.68, 0.60, 0.48],
    5: [1.11, 0.10, 0.36, 0.13, 0.29, 2.31, 0.11, 1.49, 1.03, 0.65, 1.16, 1.34, 0.96],
    6: [1.08, 0.26, 0.07, 0.24, 0.80, 0.63, 0.83, 0.27, 0.59, 0.41, 0.44, 1.12, 1.84],
    7: [0.74, 0.43, 0.48, 0.35, 0.64, 0.70, 0.74, 0.96, 1.05, 0.88, 0.63, 0.71, 0.79]
}

if demand_file is not None:
    demand_df = pd.read_csv(demand_file, header=None)
    if demand_df.shape != (7,13):
        st.error(f"Demand CSV must be 7 rows x 13 columns. Got {demand_df.shape}.")
        st.stop()
else:
    demand_df = pd.DataFrame([default_demand[d] for d in D])

def to_hours_dict(df, col):
    return {str(r["name"]): float(r[col]) for _, r in df.iterrows()}

MinHw = to_hours_dict(staff_df, "min_week_hours")
MaxHw = to_hours_dict(staff_df, "max_week_hours")
W = [str(w) for w in staff_df["name"].tolist()]

# --------- Weekly chart only (defensive numeric casting) ---------
def render_weekly_demand_staffing_chart(coverage_df, SLOT_LABELS):
    st.markdown("### Weekly Demand vs Staffing (aggregate over 7 days)")
    if coverage_df is None or coverage_df.empty:
        st.warning("No coverage data to plot.")
        return

    weekly_df = (coverage_df.groupby("slot")
                            .agg({"demand":"sum", "staffed":"sum"})
                            .reset_index()
                            .sort_values("slot"))

    # Force numeric dtypes
    weekly_df["demand"] = pd.to_numeric(weekly_df["demand"], errors="coerce").fillna(0.0)
    weekly_df["staffed"] = pd.to_numeric(weekly_df["staffed"], errors="coerce").fillna(0.0)

    # Ensure we have 13 points (pad/truncate if needed)
    if len(weekly_df) != len(SLOT_LABELS):
        # reindex by slot number 1..13
        full = pd.DataFrame({"slot": list(range(1, len(SLOT_LABELS)+1))})
        weekly_df = full.merge(weekly_df, on="slot", how="left").fillna(0.0)

    x_labels = SLOT_LABELS
    line_df = pd.DataFrame({
        "Demand": weekly_df["demand"].astype(float).values,
        "Staffed": weekly_df["staffed"].astype(float).values
    }, index=x_labels)

    if line_df.isna().all().all() or (line_df.sum().sum() == 0):
        st.warning("The weekly totals are all zeros — nothing to plot.")
    else:
        st.line_chart(line_df, use_container_width=True)

    deviation = weekly_df["staffed"] - weekly_df["demand"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total under", f"{float(np.maximum(weekly_df['demand']-weekly_df['staffed'],0).sum()):.1f}")
    c2.metric("Total over", f"{float(np.maximum(weekly_df['staffed']-weekly_df['demand'],0).sum()):.1f}")
    c3.metric("Max |deviation|", f"{float(abs(deviation).max()):.1f}")

if st.button("Solve", type="primary"):
    with st.spinner("Solving..."):
        Demand = {d: list(map(float, demand_df.iloc[d-1].tolist())) for d in D}
        status, obj, schedule, metrics = solve_schedule(
            W=W, D=D, T=T,
            MinHw=MinHw, MaxHw=MaxHw,
            Demand=Demand,
            Max_Deviation=max_dev,
            require_min_staff=ensure_min_staff,
        )

    st.success(f"Status: {status}; Total deviation: {obj:.4f}")

    # Schedule table
    sched_df = pd.DataFrame(schedule, columns=["worker","day","slot"]).sort_values(["day","slot","worker"])
    st.dataframe(sched_df, use_container_width=True)

    # Coverage table
    rows = []
    for (d,t),(u,o,staffed,dem) in metrics.items():
        rows.append({"day":d,"slot":t,"staffed":staffed,"demand":dem,"under":u,"over":o})
    cov_df = pd.DataFrame(rows).sort_values(["day","slot"])
    st.dataframe(cov_df, use_container_width=True)

    # ---- Only weekly chart (defensive) ----
    render_weekly_demand_staffing_chart(cov_df, SLOT_LABELS)

    # Downloads
    st.download_button("Download schedule CSV", sched_df.to_csv(index=False).encode("utf-8"),
                       file_name="plaza_nueva_schedule.csv", mime="text/csv")
    st.download_button("Download coverage CSV", cov_df.to_csv(index=False).encode("utf-8"),
                       file_name="plaza_nueva_coverage.csv", mime="text/csv")

    # Per-worker 7x13 tables
    st.markdown("### Per-worker Schedule (7×13, color = scheduled)")
    workers = [str(w) for w in staff_df["name"].tolist()]
    def style_schedule(df):
        return df.style.apply(lambda s: ['background-color: #E6F4FF' if v==1 else '' for v in s], axis=1)
    worker_tables = {}
    if not sched_df.empty:
        for w in workers:
            mat = np.zeros((7,13), dtype=int)
            sub = sched_df[sched_df["worker"] == w]
            for _, r in sub.iterrows():
                d = int(r["day"]) - 1
                t = int(r["slot"]) - 1
                if 0 <= d < 7 and 0 <= t < 13:
                    mat[d, t] = 1
            idx = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            dfw = pd.DataFrame(mat, columns=SLOT_LABELS, index=idx)
            worker_tables[w] = dfw
    for w in workers:
        if w in worker_tables:
            with st.expander(w):
                st.dataframe(style_schedule(worker_tables[w]), use_container_width=True)

# Template download (7x13, no header)
templ = pd.DataFrame([default_demand[d] for d in D])
st.download_button("Download demand template (7x13 CSV, no header)",
                   templ.to_csv(index=False, header=False).encode("utf-8"),
                   file_name="sales_demand_template.csv",
                   mime="text/csv")
