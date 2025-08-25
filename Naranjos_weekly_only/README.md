# Naranjos — Weekly-Only Streamlit Scheduler

This app shows ONLY the **weekly** line chart (13 time slots aggregated across 7 days) and keeps all the other results (schedule tables, deviation summary, and downloads). No daily line charts or bar charts are included.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Inputs
- **Demand CSV (optional):** 7 rows (Mon–Sun), 13 numeric columns (slot1..slot13) representing demand per hour-slot.  
- **Staff CSV (optional):** Columns: `name,min_hours,max_hours`. If not provided, a default staff list is used.

If no files are uploaded, the app uses built‑in demo data.

## Output
- One **weekly** line chart (x=13 time slots; values aggregated over 7 days).
- Light‑blue highlighted schedule tables.
- Download buttons for schedule CSVs.
