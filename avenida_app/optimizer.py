
import pulp, time

def build_and_solve_shift_model(W, D, T, S, MinHw, MaxHw, Demand, Max_Deviation=2.5, time_limit=None):
    """
    13-slot optimizer (slots 1..13 correspond to 12-13,...,00-01).
    Constraints:
      - One shift (4–8h) per worker per day (via b variables)
      - Weekly hours in [MinHw, MaxHw]
      - Exactly one pair of consecutive rest days (z)
      - 12h rest: if close (t=13) then cannot open next day (t=1)
      - Max 2 closing shifts per worker (t=13)
      - Coverage with under/over and per-slot cap Max_Deviation
      - Optional time limit
    Returns dict with status, objective, elapsed_time, schedule list[(w,d,t)]
    """
    model = pulp.LpProblem("Shift_Scheduling_13slot", pulp.LpMinimize)

    # Variables
    x = pulp.LpVariable.dicts("x", (W, D, T), cat="Binary")
    y = pulp.LpVariable.dicts("y", (W, D), cat="Binary")
    z = pulp.LpVariable.dicts("z", (W, range(1,7)), cat="Binary")  # rest-day pair indicator between d and d+1
    b = pulp.LpVariable.dicts("b", (W, D, S), cat="Binary")
    under = pulp.LpVariable.dicts("under", (D, T), lowBound=0)
    over  = pulp.LpVariable.dicts("over",  (D, T), lowBound=0)

    # Objective: minimize total deviation
    model += pulp.lpSum(under[d][t] + over[d][t] for d in D for t in T)

    # Shift structure + daily bounds
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

    # Exactly one pair of consecutive rest days
    for w in W:
        model += pulp.lpSum(z[w][d] for d in range(1,7)) == 1
        for d in range(1,7):
            model += z[w][d] <= 1 - y[w][d]
            model += z[w][d] <= 1 - y[w][d+1]

    # 12h rest: close (t=13) then cannot open next day (t=1)
    if 13 in T and 1 in T:
        for w in W:
            for d in range(1,7):
                model += x[w][d][13] + x[w][d+1][1] <= 1

    # Max 2 closing shifts per week
    if 13 in T:
        for w in W:
            model += pulp.lpSum(x[w][d][13] for d in D) <= 2

    # Coverage
    for d in D:
        for idx, t in enumerate(T):
            model += pulp.lpSum(x[w][d][t] for w in W) + under[d][t] - over[d][t] == Demand[d][idx]
            model += under[d][t] + over[d][t] <= Max_Deviation
            model += pulp.lpSum(x[w][d][t] for w in W) >= 1

    # Solve
    start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=True)
    if time_limit is not None:
        solver.timeLimit = time_limit
    model.solve(solver)
    end = time.time()

    schedule = [(w,d,t) for w in W for d in D for t in T if pulp.value(x[w][d][t]) > 0.5]

    return {
        "status": pulp.LpStatus[model.status],
        "objective": pulp.value(model.objective),
        "elapsed_time": end - start,
        "schedule": schedule
    }
