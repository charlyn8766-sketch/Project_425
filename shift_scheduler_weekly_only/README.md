# Shift Scheduler — Weekly Chart Only

This package renders only a single weekly aggregate line chart (no daily charts, no tables).
It expects a 7×13 demand CSV (no header) and a staff CSV with columns
`name,min_week_hours,max_week_hours` (optional — a small demo staff is preloaded).

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
