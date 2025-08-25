# Shift Scheduler — Weekly Chart Only (13 slots)

This package renders **only** a single weekly aggregate line chart (no daily charts/tables, no per-worker grids).
It uses a 13-slot model (12:00–24:00).

## Input formats
- Demand CSV: 7×13 (no header). Rows = Mon..Sun, Cols = 12-13, ..., 23-00, 00-01
- Staff CSV (optional): `name,min_week_hours,max_week_hours`

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
