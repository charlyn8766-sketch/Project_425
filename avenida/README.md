
# Avenida Shift Scheduler (12:00–24:00)

This package matches your original Streamlit UI/flow, but adapted to **13 slots (12:00–24:00)** and with two constraint changes:
1) 12h rest: only applies to the last slot **t=13** vs next day's **t=1**.
2) Max 2 closing shifts: counted on **t=13**.

## Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files
- `streamlit_app.py` — Streamlit UI (expects 7×13 CSV without header).
- `optimizer.py` — MILP in PuLP/CBC with the 13-slot rules.
- `requirements.txt`
- `sales_demand_template.csv` — default 7×13 demand.
- `README.md`
