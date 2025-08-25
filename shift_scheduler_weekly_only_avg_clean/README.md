
# Plaza Nueva Shift Scheduler (12:00–24:00)

Same constraints as Avenida/Naranjos:
- 13 slots/day (12:00–24:00)
- 12h rest only for last slot t=13 vs next day t=1
- Max 2 closing shifts at t=13

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
- `streamlit_app.py` — Streamlit UI (expects 7×13 CSV without header).
- `optimizer.py` — MILP in PuLP/CBC with the 13-slot rules above.
- `requirements.txt`
- `sales_demand_template.csv` — default 7×13 demand.
- `README.md`
