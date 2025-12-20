import pandas as pd
import numpy as np
import time
from openpyxl import load_workbook
from gurobipy import Model, GRB, quicksum

def load_data(fname="Group_40.xlsx"):
    wb = load_workbook(fname, data_only=True)
    df_flights     = pd.DataFrame(wb["Flights"].values)
    df_itineraries = pd.DataFrame(wb["Itineraries"].values)
    df_recapture   = pd.DataFrame(wb["Recapture"].values)

    df_flights.columns     = df_flights.iloc[0];     df_flights     = df_flights.iloc[1:].reset_index(drop=True)
    df_itineraries.columns = df_itineraries.iloc[0]; df_itineraries = df_itineraries.iloc[1:].reset_index(drop=True)
    df_recapture.columns   = df_recapture.iloc[0];   df_recapture   = df_recapture.iloc[1:].reset_index(drop=True)

    return df_flights, df_itineraries, df_recapture

def build_model(df_flights, df_itineraries, df_recapture):
    # Sets
    L = list(range(len(df_flights)))      # all flights
    P = list(range(len(df_itineraries)))  # all itineraries
    P0 = len(P)                           # index fictivious spill‐itinerary
    P_full = P + [P0]                     # all itineraries including fictivious
    Pp = {p: set() for p in P_full}       # all possible itinary moves, build after b_pr is made

    # Parameters
    CAP = {l: float(df_flights.loc[l, "Capacity"]) for l in L }            # Capacity on a flight
    fare = {p: float(df_itineraries.loc[p, "Price [EUR]"]) for p in P }    # Fare for flight(s) from itinary p
    D = {p: float(df_itineraries.loc[p, "Demand"])      for p in P }       # Demand for itinary p

    fare[P0] = 0.0                      # Fare of fictivious itinary
    D[P0] = sum(D[p] for p in P)        # Demand for fictivious itinary is realy big so the demand constraint is not a problem

    b_pr = {}                           # recapture‐rates
    for _, row in df_recapture.iterrows():
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
    for p, row in df_itineraries.iterrows():
        for col in ["Flight 1", "Flight 2"]:
            code = row[col]
            if pd.notna(code):
                l = flight_code_to_id[code]
                delta_lp[(l, p)] = 1             # If flight code l in itinerary p, delta = 1

    Q = {l: quicksum(delta_lp.get((l,p),0) * D[p] for p in P) for l in L}     # Unconstrained demand on flight l

    model = Model("Keypath_Mixflow")    # Gurobi model
    model.setParam("TimeLimit", 300)    # Timelimit if optimal solution is not found then

    # Decision variable
    t = model.addVars([(p, r) for p in P_full for r in Pp[p]], lb=0, vtype=GRB.INTEGER, name="t") # People moved from prefeured itinary p to r

    # Objective: mimize lost reveneu
    obj = quicksum((fare[p] - b_pr[(p,r)] * fare[r]) * t[p,r] for p in P for r in Pp[p])
    model.setObjective(obj, GRB.MINIMIZE)

    # Capacity constraint
    for l in L:
        outflow  = quicksum( delta_lp.get((l, p), 0) * quicksum(t[p, r] for r in Pp[p]) for p in P_full )
        inflow   = quicksum( delta_lp.get((l, p), 0) * quicksum(b_pr[(r, p)] * t[r, p] for r in P_full if p in Pp[r]) for p in P_full )
        model.addConstr(outflow - inflow >= Q[l] - CAP[l], name=f"cap_{l}")

    # Demand constraint
    for p in P:
        model.addConstr(quicksum(t[p,r] for r in Pp[p]) <= D[p], name=f"demand_{p}")

    return model, t, P, P_full, Pp, b_pr, P0        # Needed for result calculation


def solve_and_report(model, t, P, Pp, b_pr, P0):
    model.optimize()
    model.write("keypath_mixflow.lp")

    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        print("No solution (status =", model.status, ")")
        if model.status == GRB.INFEASIBLE:
            model.computeIIS(); model.write("infeasible.ilp")
        return

    print("SOLUTION")
    print("Objective =", model.objVal)

    print("\nRecaptures")           # Values of t
    for p in P:
        for r in Pp[p]:
            if r == P0:
                continue
            val = t[p, r].X
            if val > 1e-6:
                print(f"t[{p} → {r}] = {val:.2f} pax")


    total_spill = sum(t[p, P0].X for p in P if P0 in Pp[p])     # Passengers in fictivious itinary are spilled
    print(f"\nTotal amount of spill: {total_spill:.2f} pax")

if __name__ == "__main__":
    df_flights, df_itineraries, df_recapture = load_data("Group_40.xlsx")                       # Our data
    model, t, P, P_full, Pp, b_pr, p0 = build_model(df_flights, df_itineraries, df_recapture)   # Build de model

    t1 = time.perf_counter()
    solve_and_report(model, t, P, Pp, b_pr, p0)                                                 # Optimize
    t2 = time.perf_counter()
    print(f"Time to optimize and report {t2 - t1:.4f} s")                                       # Print time to optimize and report results