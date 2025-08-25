
import time
import pulp

def solve_schedule(
    W,
    D=range(1,8),
    T=range(1,14),  # 13 slots (1..13)
    MinHw=None,
    MaxHw=None,
    Demand=None,
    Max_Deviation=2.5,
    require_min_staff=True,
):
    W = list(W); D = list(D); T = list(T)
    if MinHw is None or MaxHw is None:
        raise ValueError("MinHw and MaxHw must be provided.")
    if Demand is None:
        Demand = {d: [1.0 for _ in T] for d in D}
    else:
        for d in D:
            if d not in Demand:
                raise ValueError(f"Demand missing day={d}")
            if len(Demand[d]) != len(T):
                raise ValueError(f"Demand[day={d}] length should be {len(T)}, got {len(Demand[d])}")

    # Shift set 4..8 inclusive
    S = [(s,e) for s in T for e in T if s<=e and 4 <= (e - s + 1) <= 8]

    model = pulp.LpProblem("Shift_Scheduling_Naranjos", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (W, D, T), cat="Binary")
    y = pulp.LpVariable.dicts("y", (W, D), cat="Binary")
    z = pulp.LpVariable.dicts("z", (W, range(1,7)), cat="Binary")
    b = pulp.LpVariable.dicts("b", (W, D, S), cat="Binary")
    under = pulp.LpVariable.dicts("under", (D, T), lowBound=0)
    over  = pulp.LpVariable.dicts("over",  (D, T), lowBound=0)

    # Objective
    model += pulp.lpSum(under[d][t] + over[d][t] for d in D for t in T)

    # Link & daily min/max
    for w in W:
        for d in D:
            model += pulp.lpSum(b[w][d][se] for se in S) <= 1
            for t in T:
                model += x[w][d][t] == pulp.lpSum(b[w][d][se] for se in S if se[0] <= t <= se[1])
            model += y[w][d] == pulp.lpSum(b[w][d][se] for se in S)
            model += pulp.lpSum(x[w][d][t] for t in T) >= 4 * y[w][d]
            model += pulp.lpSum(x[w][d][t] for t in T) <= 8 * y[w][d]

    # Weekly hours
    for w in W:
        model += pulp.lpSum(x[w][d][t] for d in D for t in T) >= MinHw[w]
        model += pulp.lpSum(x[w][d][t] for d in D for t in T) <= MaxHw[w]

    # Two consecutive rest days
    for w in W:
        model += pulp.lpSum(z[w][d] for d in range(1,7)) == 1
        for d in range(1,7):
            model += z[w][d] <= 1 - y[w][d]
            model += z[w][d] <= 1 - y[w][d+1]

    # 12h rest only between t=13 and next day t=1
    if 13 in T and 1 in T:
        for w in W:
            for d in range(1,7):
                model += x[w][d][13] + x[w][d+1][1] <= 1

    # Max 2 closing (t=13)
    if 13 in T:
        for w in W:
            model += pulp.lpSum(x[w][d][13] for d in D) <= 2

    # Demand + bounds
    for d in D:
        for idx, t in enumerate(T):
            model += pulp.lpSum(x[w][d][t] for w in W) + under[d][t] - over[d][t] == Demand[d][idx]
            model += under[d][t] + over[d][t] <= Max_Deviation
            if require_min_staff:
                model += pulp.lpSum(x[w][d][t] for w in W) >= 1

    start = time.time()
    model.solve(pulp.PULP_CBC_CMD(msg=True))
    end = time.time()

    status = pulp.LpStatus[model.status]
    objective = pulp.value(model.objective)

    schedule = [(w,d,t) for w in W for d in D for t in T if pulp.value(x[w][d][t]) > 0.5]

    metrics = {}
    for d in D:
        for idx, t in enumerate(T):
            staffed = sum(1 for w in W if pulp.value(x[w][d][t]) > 0.5)
            metrics[(d,t)] = (pulp.value(under[d][t]), pulp.value(over[d][t]), staffed, Demand[d][idx])

    return status, objective, schedule, metrics
