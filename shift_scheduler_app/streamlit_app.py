
    import streamlit as st
    import pandas as pd
    import numpy as np
    from io import BytesIO
    import sys, os, importlib

    # --- Robust import of optimizer ---
    try:
        from optimizer import solve_schedule, build_shift_set
    except Exception as e1:
        # Try adding the current script's directory to sys.path and retry
        _here = os.path.dirname(__file__)
        if _here and _here not in sys.path:
            sys.path.insert(0, _here)
        try:
            from optimizer import solve_schedule, build_shift_set
        except Exception as e2:
            st.error("Failed to import `optimizer.py`. Please ensure it is in the same folder as this file.")
            st.code(f\"\"\"
Original error 1: {type(e1).__name__}: {e1}
Retry error     : {type(e2).__name__}: {e2}
Working dir     : {os.getcwd()}
__file__        : {__file__}
sys.path[0..3]  : {sys.path[:4]}
Folder contents : {os.listdir(os.path.dirname(__file__)) if os.path.dirname(__file__) else 'N/A'}
            \"\"\")
            st.stop()

    st.set_page_config(page_title="Shift Scheduler", layout="wide")

    st.title("Shift Scheduler (Streamlit + PuLP)")

    SLOT_LABELS = ["10-11","11-12","12-13","13-14","14-15","15-16","16-17","17-18","18-19","19-20","20-21","21-22","22-23","23-00","00-01"]
    DAY_LABELS = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

    with st.sidebar:
        st.header("Configuration")
        max_dev = st.number_input("Max deviation per slot (people)", min_value=0.0, value=2.5, step=0.5)
        time_limit = st.number_input("Solver time limit (seconds)", min_value=10, value=180, step=10)
        seed = st.number_input("Random seed (for reproducibility)", min_value=0, value=42, step=1)

        st.markdown("---")
        st.subheader("Demand CSV")
        st.caption("Upload a 7x15 CSV (no header): rows=Mon..Sun, cols=15 hourly slots (10-11,...,00-01).")
        demand_file = st.file_uploader("Upload sales_demand_template.csv", type=["csv"])

        st.markdown("---")
        st.subheader("Staff table")
        st.caption("Edit staff below. You can add or delete rows. Download/Upload to reuse.")
        uploaded_staff = st.file_uploader("Upload staff CSV (optional)", type=["csv"], key="staff_csv")
        if uploaded_staff is not None:
            staff_df_init = pd.read_csv(uploaded_staff)
        else:
            staff_df_init = pd.DataFrame([
                {"name":"Ana","min_week_hours":30,"max_week_hours":39,"contract_hours":30,"weekend_only":False},
                {"name":"Vanessa","min_week_hours":25,"max_week_hours":33,"contract_hours":25,"weekend_only":False},
                {"name":"Ines","min_week_hours":30,"max_week_hours":39,"contract_hours":30,"weekend_only":False},
                {"name":"Yuliia","min_week_hours":20,"max_week_hours":26,"contract_hours":20,"weekend_only":False},
            ])
        staff_df = st.data_editor(
            staff_df_init,
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

        staff_csv = staff_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download staff CSV", staff_csv, file_name="staff.csv", mime="text/csv")

    # Load demand
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

    # Build shift set (start<=end, length in [4,8])
    T = list(range(1,16))
    try:
        S = build_shift_set(T, min_len=4, max_len=8)
    except Exception:
        S = []
        for s in T:
            for e in T:
                L = e - s + 1
                if 4 <= L <= 8:
                    S.append((s,e))

    st.markdown("### Run Optimizer")
    if st.button("Solve now", type="primary"):
        with st.spinner("Solving..."):
            try:
                res = solve_schedule(demand, staff_df, S=S, max_deviation=max_dev, time_limit=time_limit, seed=seed)
            except TypeError:
                res = solve_schedule(demand, staff_df, S=S, max_deviation=max_dev, time_limit=time_limit)

        st.success(f"Status: {res.get('status','N/A')}, Objective (total deviation): {res.get('objective', float('nan')):.4f}")
        if 'hours_df' in res:
            st.write("Weekly hours per worker")
            st.dataframe(res['hours_df'], use_container_width=True)
        if 'coverage_df' in res:
            st.write("Coverage by day-slot (demand / staffed / under / over)")
            st.dataframe(res['coverage_df'], use_container_width=True)

        # Per-worker 7x15 with color highlight
        st.markdown("### Per-worker Schedule (7×15, color = scheduled)")
        assignments_df = res.get("assignments_df", pd.DataFrame(columns=["name","day","start_slot","end_slot"]))
        workers = list(staff_df['name'])

        def style_schedule(df):
            # green background for scheduled slots
            return df.style.apply(lambda s: ['background-color: #C6F6D5' if v==1 else '' for v in s], axis=1)

        worker_tables = {}
        for w in workers:
            mat = np.zeros((7,15), dtype=int)
            subset = assignments_df[assignments_df['name'] == w]
            for _, row in subset.iterrows():
                d = int(row['day']) - 1
                s = int(row['start_slot']) - 1
                e = int(row['end_slot']) - 1
                s = max(0, min(14, s))
                e = max(0, min(14, e))
                mat[d, s:e+1] = 1
            df = pd.DataFrame(mat, columns=SLOT_LABELS, index=DAY_LABELS)
            worker_tables[w] = df

        for w in workers:
            with st.expander(f"{w}"):
                st.dataframe(style_schedule(worker_tables[w]), use_container_width=True)

        # Download per-worker Excel
        if worker_tables:
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                for w, df in worker_tables.items():
                    sheet_name = w[:31] if w else "Worker"
                    df.to_excel(writer, sheet_name=sheet_name)
            st.download_button(
                "Download per-worker schedule (Excel)",
                data=output.getvalue(),
                file_name="per_worker_schedule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
