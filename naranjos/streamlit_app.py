import streamlit as st

# Guarded imports
try:
    import pandas as pd
    import numpy as np
except Exception as e:
    st.error("Please install pandas and numpy: pip install -r requirements.txt")
    st.stop()

from optimizer import solve_schedule

st.set_page_config(page_title="Naranjos Shift Scheduler (12-24)", layout="wide")
st.title("Naranjos Shift Scheduler (12:00–24:00)")

# Sets
D = list(range(1,8))
T = list(range(1,14))  # 13 slots
SLOT_LABELS = ["12-13","13-14","14-15","15-16","16-17","17-18","18-19","19-20","20-21","21-22","22-23","23-24","00-01"]
DAY_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

# Default staff
DEFAULT_STAFF = [
    {"name":"Sara",        "min_week_hours":35.0, "max_week_hours":40.0},
    {"name":"Paula",       "min_week_hours":15.0, "max_week_hours":19.5},
    {"name":"Claudia",     "min_week_hours":20.0, "max_week_hours":26.0},
    {"name":"Vanessa_V",   "min_week_hours":20.0, "max_week_hours":26.0},
    {"name":"Valentyna",   "min_week_hours":25.0, "max_week_hours":32.5},
    {"name":"Paz",         "min_week_hours":25.0, "max_week_hours":32.5},
    {"name":"Kevin",       "min_week_hours":30.0, "max_week_hours":39.0},
    {"name":"Ariadna",     "min_week_hours":25.0, "max_week_hours":32.5},
    {"name":"Aroa",        "min_week_hours":25.0, "max_week_hours":32.5},
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

# Default demand (7x13) with interpolated Day 7 slots 5..13
default_demand = {
    1: [0.88, 1.00, 2.61, 2.49, 2.45, 2.01, 2.32, 1.00, 1.22, 1.71, 1.29, 1.99, 0.57],
    2: [2.07, 2.30, 2.48, 2.27, 2.01, 0.37, 0.82, 1.33, 1.72, 2.13, 0.24, 0.75, 1.59],
    3: [1.87, 1.15, 0.53, 1.32, 0.96, 0.72, 1.18, 1.96, 0.09, 0.67, 2.10, 1.90, 2.50],
    4: [2.51, 0.68, 1.30, 1.65, 2.14, 1.73, 0.22, 0.81, 1.31, 1.81, 2.17, 1.46, 1.58],
    5: [1.00, 0.80, 1.50, 2.75, 1.86, 0.29, 0.43, 1.87, 1.19, 1.89, 2.27, 3.22, 1.72],
    6: [1.37, 2.66, 1.97, 1.29, 0.16, 0.33, 0.45, 1.42, 1.68, 1.42, 2.24, 2.23, 2.67],
    7: [1.48, 2.03, 2.06, 0.37, 1.60, 0.91, 0.90, 1.40, 1.20, 1.61, 1.72, 1.92, 1.77]
}

if demand_file is not None:
    demand_df = pd.read_csv(demand_file, header=None)
    if demand_df.shape != (7,13):
        st.error(f"Demand CSV must be 7 rows x 13 columns. Got {demand_df.shape}.")
        st.stop()
else:
    demand_df = pd.DataFrame([default_demand[d] for d in D])

# Build Min/Max dicts
def to_hours_dict(df, col):
    out = {str(r["name"]): float(r[col]) for _, r in df.iterrows()}
    return out

MinHw = to_hours_dict(staff_df, "min_week_hours")
MaxHw = to_hours_dict(staff_df, "max_week_hours")
W = [str(w) for w in staff_df["name"].tolist()]


# --------- Visualization helper (UPDATED) ---------
def render_weekly_demand_staffing_chart(coverage_df, SLOT_LABELS, DAY_LABELS):
    st.markdown("### Weekly Demand vs Staffing (aggregate over 7 days)")

    # 按 slot 聚合一周所有天
    weekly_df = (coverage_df.groupby("slot")
                            .agg({"demand":"sum", "staffed":"sum"})
                            .reset_index()
                            .sort_values("slot"))
    
    # 用 slot label 显示横轴
    x_labels = SLOT_LABELS if len(weekly_df) == len(SLOT_LABELS) else [str(s) for s in weekly_df["slot"]]

    # 画折线图：demand vs staffed
    line_df = pd.DataFrame({
        "Demand": weekly_df["demand"].to_numpy(),
        "Staffed": weekly_df["staffed"].to_numpy(),
    }, index=x_labels)
    st.line_chart(line_df)

    # 显示 deviation summary
    deviation = weekly_df["staffed"] - weekly_df["demand"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total under", f"{max(0, (weekly_df['demand']-weekly_df['staffed']).sum()):.1f}")
    c2.metric("Total over", f"{max(0, (weekly_df['staffed']-weekly_df['demand']).sum()):.1f}")
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

    # Charts
    render_weekly_demand_staffing_chart(res["coverage_df"], SLOT_LABELS, DAY_LABELS)

    # Downloads
    st.download_button("Download schedule CSV", sched_df.to_csv(index=False).encode("utf-8"),
                       file_name="naranjos_schedule.csv", mime="text/csv")
    st.download_button("Download coverage CSV", cov_df.to_csv(index=False).encode("utf-8"),
                       file_name="naranjos_coverage.csv", mime="text/csv")

    # Per-worker 7x13 tables
    st.markdown("### Per-worker Schedule (7×13, color = scheduled)")
    workers = [str(w) for w in staff_df["name"].tolist()]
    def style_schedule(df):
        return df.style.apply(lambda s: ['background-color: #E6F4FF' if v==1 else '' for v in s], axis=1)
    worker_tables = {}
    if not sched_df.empty:
        import numpy as np
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
