import itertools
import pandas as pd
import numpy as np
import pulp

# Time slots 1..13 (12:00–25:00). Shift length measured in "slot hours" = e - s.
T = list(range(1, 14))           # 13 slots
D = list(range(1, 8))            # 7 days

def default_staff():
    # Demo staff list (you can upload a CSV to replace this).
    # min/max hours are weekly.
    names = ["Ana","Vanessa_M","Ines","Yuliia","Giulia","Tomas","Ana_Bernabe","Raul_Calero","Jose_Antonio","Haely"]
    min_hw = [30,25,30,20,25,30,25,25,20,25]
    max_hw = [39,32.5,37.5,26,32.5,37.5,32.5,32.5,26,32.5]
    return pd.DataFrame({"name":names, "min_hours":min_hw, "max_hours":max_hw})

def timeslot_labels():
    # Map 1..13 -> "12:00–13:00", ..., "24:00–25:00"
    start = 12
    labels = []
    for i in range(13):
        s = start + i
        e = s + 1
        labels.append(f"{int(s):02d}:00–{int(e):02d}:00")
    return labels

def solve_schedule(demand_7x13: pd.DataFrame, staff_df: pd.DataFrame, weight_under=1.0, weight_over=1.0):
    """
    demand_7x13: 7 rows (Mon..Sun), 13 columns (slot1..slot13) numeric
    staff_df: columns [name, min_hours, max_hours]
    returns: dict with schedule, coverage, under/over, objective
    """
    # sets
    W = list(staff_df["name"])
    T = list(range(1,14))
    D = list(range(1,8))
    S = [(s,e) for s in T for e in T if 4 <= (e - s) <= 8]  # 4..8 slot-hours

    # parameters
    MinHw = {w: float(staff_df.loc[staff_df["name"]==w, "min_hours"].iloc[0]) for w in W}
    MaxHw = {w: float(staff_df.loc[staff_df["name"]==w, "max_hours"].iloc[0]) for w in W}
    Demand = {(d,t): float(demand_7x13.iloc[d-1, t-1]) for d in D for t in T}

    # model
    m = pulp.LpProblem("WeeklyScheduling_MinTotalDeviation", pulp.LpMinimize)

    # variables
    x = pulp.LpVariable.dicts("x", (W, D, S), lowBound=0, upBound=1, cat=pulp.LpBinary)  # assign shift (s,e) to worker w on day d
    b = pulp.LpVariable.dicts("b", (W, D, T), lowBound=0, upBound=1, cat=pulp.LpBinary)  # active at time t
    under = pulp.LpVariable.dicts("under", (D, T), lowBound=0, cat=pulp.LpContinuous)
    over  = pulp.LpVariable.dicts("over",  (D, T), lowBound=0, cat=pulp.LpContinuous)

    # objective: min sum (weighted L1 deviation)
    m += pulp.lpSum([weight_under * under[d][t] + weight_over * over[d][t] for d in D for t in T])

    # coverage equations: sum_w b[w,d,t] + over - under = Demand[d,t]
    for d in D:
        for t in T:
            m += pulp.lpSum([b[w][d][t] for w in W]) + over[d][t] - under[d][t] == Demand[(d,t)]

    # link b and x: b[w,d,t] = 1 if any shift covering t is chosen
    for w in W:
        for d in D:
            for t in T:
                covering = [x[w][d][(s,e)] for (s,e) in S if s <= t < e]
                if covering:
                    m += b[w][d][t] <= pulp.lpSum(covering)
                    for var in covering:
                        m += b[w][d][t] >= var
                else:
                    m += b[w][d][t] == 0

    # one shift per worker per day
    for w in W:
        for d in D:
            m += pulp.lpSum([x[w][d][se] for se in S]) <= 1

    # weekly hours for each worker: sum (e-s) * x in [min,max]
    for w in W:
        total_hours = pulp.lpSum([(e - s) * x[w][d][(s,e)] for d in D for (s,e) in S])
        m += total_hours >= MinHw[w]
        m += total_hours <= MaxHw[w]

    # Solve with CBC (default in PuLP). No time limit set.
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    # Build outputs
    status = pulp.LpStatus[m.status]
    obj = pulp.value(m.objective)

    # coverage & deviation frames
    cov = pd.DataFrame(0.0, index=range(1,8), columns=range(1,14))
    und = pd.DataFrame(0.0, index=range(1,8), columns=range(1,14))
    ove = pd.DataFrame(0.0, index=range(1,8), columns=range(1,14))
    for d in D:
        for t in T:
            cov.loc[d,t] = sum(pulp.value(b[w][d][t]) for w in W)
            und.loc[d,t] = pulp.value(under[d][t])
            ove.loc[d,t] = pulp.value(over[d][t])

    # schedule (per day dataframe with who works each time slot)
    day_tables = {}
    for d in D:
        table = pd.DataFrame("", index=range(1,14), columns=W)
        for w in W:
            for t in T:
                val = pulp.value(b[w][d][t])
                if val is not None and val > 0.5:
                    table.loc[t, w] = "✓"
        day_tables[d] = table

    return {
        "status": status,
        "objective": obj,
        "coverage": cov,
        "under": und,
        "over": ove,
        "day_tables": day_tables,
        "time_labels": timeslot_labels(),
        "demand": demand_7x13
    }
