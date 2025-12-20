import time
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from gurobipy import Model, GRB, quicksum

def load_data(fname="Group_40.xlsx"):               # Load data
    wb = load_workbook(fname, data_only=True)
    df_f = pd.DataFrame(wb["Flights"].values)
    df_i = pd.DataFrame(wb["Itineraries"].values)
    df_r = pd.DataFrame(wb["Recapture"].values)
    for df in (df_f, df_i, df_r):
        df.columns = df.iloc[0]
        df.drop(0, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df_f, df_i, df_r

# Parameters
df_flights, df_itin, df_rec = load_data("Group_40.xlsx")

L = list(range(len(df_flights)))      # all flights
P = list(range(len(df_itin)))  # all itineraries
P0 = len(P)                           # index fictivious spill‐itinerary
P_full = P + [P0]                     # all itineraries including fictivious
Pp = {p: set() for p in P_full}       # all possible itinary moves, build after b_pr is made

# Parameters
CAP = {l: float(df_flights.loc[l, "Capacity"] *10) for l in L }            # Capacity on a flight
fare = {p: float(df_itin.loc[p, "Price [EUR]"]) for p in P }    # Fare for flight(s) from itinary p
D = {p: float(df_itin.loc[p, "Demand"])      for p in P }       # Demand for itinary p

fare[P0] = 0.0                      # Fare of fictivious itinary
D[P0] = sum(D[p] for p in P)        # Demand for fictivious itinary is realy big so the demand constraint is not a problem

b_pr = {}                           # recapture‐rates
for _, row in df_rec.iterrows():
    p = int(row["From Itinerary"])
    r = int(row["To Itinerary"])
    b_pr[(p, r)] = float(row["Recapture Rate"])

for p in P:                  
    b_pr[(p, p)] = 1.0        # For whised itinary recapture rate is 1
    b_pr[(p, P0)] = 1.0       # spill naar P0

for (p, r), rate in b_pr.items():    # Building Pp
    if p!=P0 and (r!=p or r!=P0):
        Pp[p].add(r)
for p in P:
    Pp[p].add(p)
    Pp[p].add(P0)
Pp[P0] = set()                       # spill no outgoing recapture

flight_code_to_id = {df_flights.loc[l,"Flight No."] : l for l in L}
delta_lp = {}
for p, row in df_itin.iterrows():
    for col in ["Flight 1", "Flight 2"]:
        code = row[col]
        if pd.notna(code):
            l = flight_code_to_id[code]
            delta_lp[(l, p)] = 1             # If flight code l in itinerary p, delta = 1

Q = {l: quicksum(delta_lp.get((l,p),0) * D[p] for p in P) for l in L}     # Unconstrained demand on flight l


# Build the RMP
def init_RMP():
    m = Model("RMP")
    m.Params.OutputFlag = 0

    # Decision variable, here only fictisious
    t = m.addVars([(p,P0) for p in P], lb=0, vtype=GRB.INTEGER, name="t") 

    # Objective, minimize reveneu
    m.setObjective(quicksum((fare[p] - b_pr[(p,P0)] * fare[P0]) * t[p,P0] for p in P), GRB.MINIMIZE)

    # Capacity constraint, because only fictisous constraint is smaller
    for l in L:
        m.addConstr(quicksum(delta_lp.get((l,p),0) * t[p,P0] for p in P) >= Q[l] - CAP[l], name=f"cap_{l}")

    # Demand constraint
    for p in P:
        m.addConstr(t[p,P0] <= D[p], name=f"demand_{p}")

    m.update()
    return m, t


# Slack variable berekenen
def reduced_cost(p, r, pi, sigma):
    sum_pi_p = sum(pi[i] for i in L if delta_lp.get((i,p),0)==1)    # Delta_lp * pi_l
    sum_pi_r = sum(pi[i] for i in L if delta_lp.get((i,r),0)==1)    # Delta__lr * pi_l
    br = b_pr.get((p,r), 0)
    return ((fare[p] - br * fare[r]) - sum_pi_p + br * sum_pi_r - sigma[p])


## Column generation
def column_generation(max_iters=100, tol=1e-6):
    import time
    start_time = time.time()

    RMP, tvars = init_RMP()
    init_cols  = len(tvars)
    RMP_relaxed = RMP.relax()
    RMP_relaxed.Params.OutputFlag = 0

    # dual variables
    pi = {}
    sigma = {}
    for it in range(1, max_iters + 1):
        RMP_relaxed.optimize()
        if RMP_relaxed.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError("LP-relax infeasible")

        pi    = {l: RMP_relaxed.getConstrByName(f"cap_{l}").Pi    for l in L}
        sigma = {p: RMP_relaxed.getConstrByName(f"demand_{p}").Pi for p in P}

        # When to add column, slack < 0
        new_cols = []
        for p in P:
            for r in P_full:
                if (p, r) not in tvars and (p, r) in b_pr:
                    rc = reduced_cost(p, r, pi, sigma)
                    if rc < -tol:
                        new_cols.append((p, r))

        if not new_cols:
            break

        # D) add new column
        for p, r in new_cols:
            mip_var = RMP.addVar(lb=0, vtype=GRB.INTEGER, obj=(fare[p] - b_pr[(p, r)] * fare[r]), name=f"t[{p},{r}]") # Extend objective
            lp_var  = RMP_relaxed.addVar(lb=0, vtype=GRB.CONTINUOUS, obj=(fare[p] - b_pr[(p, r)] * fare[r]), name=f"t[{p},{r}]")
            for mdl, var in ((RMP, mip_var), (RMP_relaxed, lp_var)):
                for l in L:
                    a_out = delta_lp.get((l, p), 0)
                    a_in  = delta_lp.get((l, r), 0) * b_pr.get((r, p), 0.0)
                    if a_out:
                        mdl.chgCoeff(mdl.getConstrByName(f"cap_{l}"), var,  a_out)      # Extend capacity constraint
                    if a_in:
                        mdl.chgCoeff(mdl.getConstrByName(f"cap_{l}"), var, -a_in)       # Extend capacity constraint
                mdl.chgCoeff(mdl.getConstrByName(f"demand_{p}"), var, 1.0)              # Extend demand constraint

            tvars[(p, r)] = mip_var

        RMP.update()
        RMP_relaxed.update()

    # E) finale integer solve
    RMP.optimize()
    if RMP.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError("Final MIP heeft geen oplossing")

    # bereken eindwaarden
    final_cols = len(tvars)
    run_time   = time.time() - start_time

    return RMP, tvars, pi, sigma, init_cols, final_cols, run_time


# Print results
if __name__ == "__main__":
    rmp, t_vars, pi, sigma, init_cols, final_cols, run_time = column_generation()

    print(f"Initial number of columns : {init_cols}")
    print(f"Final   number of columns : {final_cols}")
    print(f"Total runtime (sec)       : {run_time:.2f}")

    # 1) Objective value
    print(f"Objective value           : {rmp.objVal:.2f}")

    # 2) Total spilled passengers (r == P0)
    total_spill = sum(var.X for (p, r), var in t_vars.items() if r == P0)
    print(f"Total spilled passengers  : {int(round(total_spill))}\n")

    # 3) First 5 dual π_i
    print("First 5 dual π_i")
    for i in range(5):
        print(f" π[{i}] = {pi[i]:.2f}")

    # 4) First 5 dual σ_p
    print("\nFirst 5 dual σ_p:")
    for p in range(5):
        print(f" σ[{p}] = {sigma[p]:.2f}")

    # 5) Recapture flows
    recaps = [
        (p, r, var.X)
        for (p, r), var in t_vars.items()
        if r != P0 and var.X > 1e-6
    ]
    print("\nAll recapture flows:")
    for p, r, x in recaps[:50]:
        print(f" t[{p}→{r}] = {int(round(x))}")