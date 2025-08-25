import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from optimizer import solve_schedule, default_staff

st.set_page_config(page_title="Naranjos Scheduler (Weekly Only)", layout="wide")

st.title("Naranjos — Weekly-Only Scheduler")
st.caption("Only one **weekly** line chart over 13 time slots (aggregated across 7 days). No daily charts or bar charts.")

# --- Sidebar: inputs ---
store_name = st.sidebar.text_input("Store name", "Naranjos")

st.sidebar.subheader("Upload (optional)")
demand_file = st.sidebar.file_uploader("Demand CSV (7 rows x 13 columns: slot1..slot13)", type=["csv"])
staff_file  = st.sidebar.file_uploader("Staff CSV (name,min_hours,max_hours)", type=["csv"])

if demand_file is not None:
    demand = pd.read_csv(demand_file)
    # Basic validation
    if demand.shape != (7,13):
        st.sidebar.error("Demand CSV must be 7 rows x 13 columns.")
        st.stop()
    demand.columns = [f"slot{i}" for i in range(1,14)]
else:
    # default demo demand
    demand = pd.DataFrame(
        [[0.2,0.3,0.4,0.5,0.9,1.1,1.3,1.5,1.2,1.0,0.8,0.5,0.3]]*7,
        columns=[f"slot{i}" for i in range(1,14)]
    )

if staff_file is not None:
    staff_df = pd.read_csv(staff_file)
    if not set(["name","min_hours","max_hours"]).issubset(staff_df.columns):
        st.sidebar.error("Staff CSV must contain columns: name,min_hours,max_hours")
        st.stop()
else:
    staff_df = default_staff()

st.sidebar.download_button("Download demand template", data=demand.to_csv(index=False).encode("utf-8"),
                           file_name="sales_demand_template.csv", mime="text/csv")

# --- Solve ---
with st.spinner("Solving schedule..."):
    result = solve_schedule(demand, staff_df)

st.success(f"Status: {result['status']}  |  Objective (total deviation): {result['objective']:.2f}")

# --- Weekly chart (ONLY) ---
st.subheader("Weekly Line — Demand vs Assigned Staff (Aggregated over 7 days)")

# Aggregate across days for each of the 13 slots
weekly_demand = result["demand"].sum(axis=0).values
weekly_coverage = result["coverage"].sum(axis=0).values

fig, ax = plt.subplots()
ax.plot(range(1,14), weekly_demand, marker="o", label="Demand (sum over week)")
ax.plot(range(1,14), weekly_coverage, marker="s", label="Assigned Staff (sum over week)")
ax.set_xlabel("Time Slots (13)")
ax.set_ylabel("Sum over Mon–Sun")
ax.set_xticks(range(1,14))
ax.set_xticklabels(result["time_labels"], rotation=45, ha="right")
ax.legend()
st.pyplot(fig)

# --- Deviation summary ---
total_under = result["under"].to_numpy().sum()
total_over  = result["over"].to_numpy().sum()
st.write(f"**Total Understaffing:** {total_under:.2f}  |  **Total Overstaffing:** {total_over:.2f}")

# --- Schedule tables (styled light blue) ---
st.subheader("Weekly Schedule Tables (Mon–Sun)")
tabs = st.tabs(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
for i, day in enumerate(range(1,8)):
    with tabs[i]:
        table = result["day_tables"][day].copy()
        # Light-blue highlight
        styled = table.style.set_properties(**{"background-color": "#e6f2ff"})
        st.dataframe(styled, use_container_width=True)

# --- Downloads ---
st.subheader("Downloads")
# 1) Coverage & deviations
cov_csv = result["coverage"]
und_csv = result["under"]
ove_csv = result["over"]
st.download_button("Download coverage (CSV)",
                   data=cov_csv.to_csv(index=False).encode("utf-8"),
                   file_name=f"{store_name}_coverage_week.csv",
                   mime="text/csv")
st.download_button("Download under (CSV)",
                   data=und_csv.to_csv(index=False).encode("utf-8"),
                   file_name=f"{store_name}_under_week.csv",
                   mime="text/csv")
st.download_button("Download over (CSV)",
                   data=ove_csv.to_csv(index=False).encode("utf-8"),
                   file_name=f"{store_name}_over_week.csv",
                   mime="text/csv")

# 2) Per-day schedule CSVs
for day in range(1,8):
    day_df = result["day_tables"][day]
    st.download_button(f"Download day {day} schedule (CSV)",
                       data=day_df.to_csv().encode("utf-8"),
                       file_name=f"{store_name}_day{day}_schedule.csv",
                       mime="text/csv")
