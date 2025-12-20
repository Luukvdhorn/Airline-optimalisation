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
CAP = {l: float(df_flights.loc[l, "Capacity"]) for l in L }            # Capacity on a flight
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
    t = m.addVars([(p,P0) for p in P], lb=0, vtype=GRB.CONTINUOUS, name="t") 

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


# Column generation
def column_generation(max_iters=100, tol=1e-6):
    start_time = time.time()
    RMP, tvars = init_RMP()
    print(f"Start CG: columns in RMP = {len(tvars)}")

    for it in range(1, max_iters+1):
        RMP.optimize()
        if RMP.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError("RMP is infeasible of error")

        # dual variable
        pi    = {l: RMP.getConstrByName(f"cap_{l}").Pi    for l in L}
        sigma = {p: RMP.getConstrByName(f"demand_{p}").Pi for p in P}

        # When to add a column
        new_cols = []
        for p in P:
            for r in P_full:
                if (p,r) not in tvars and (p,r) in b_pr:
                    rc = reduced_cost(p, r, pi, sigma)
                    if rc < -tol:
                        new_cols.append((p,r))

        if not new_cols:
            print(f"Converged after {it-1} iterations.")
            break

        # add new columns
        for p,r in new_cols:
            var = RMP.addVar(lb=0, vtype=GRB.CONTINUOUS, obj=(fare[p] - b_pr.get((p,r),0) * fare[r]), name=f"t[{p},{r}]") # expansion objective
            for l in L:
                a_out = delta_lp.get((l,p),0)
                a_in  = delta_lp.get((l,r),0) * b_pr.get((r,p),0)
                if a_out:
                    RMP.chgCoeff(RMP.getConstrByName(f"cap_{l}"), var, a_out)       # expansion capicity constraint
                if a_in:
                    RMP.chgCoeff(RMP.getConstrByName(f"cap_{l}"), var, -a_in)       # ecpansion capicity constraint
            RMP.chgCoeff(RMP.getConstrByName(f"demand_{p}"), var, 1.0)              # expansion demand constraint
            tvars[(p,r)] = var

        RMP.update()

    RMP.optimize()
    if RMP.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError("Final RMP has no solution")

    obj_val = RMP.objVal
    cols_final = len(tvars)
    runtime = time.time() - start_time
    print(f"End CG: columns in RMP = {cols_final}")
    print(f"Total CG iterations = {it-1}, runtime = {runtime:.2f}s")

    return RMP, tvars, obj_val


# Getting the results
if __name__ == "__main__":
    rmp, t_vars, obj_val = column_generation()

    print("\nFinal RMP")
    print(f"Objective = {obj_val:.2f}")

    # First 5 recaptures
    printed = 0
    print("\nFirst 5 recaptures:")
    for (p,r), var in t_vars.items():
        if r != P0 and var.X > 1e-6 and printed < 5:
            print(f"  t[{p} → {r}] = {var.X:.2f}")
            printed += 1

    # total spill
    total_spill = sum(t_vars[p,P0].X for p in P)
    print(f"\nTotal spill = {total_spill:.2f}")