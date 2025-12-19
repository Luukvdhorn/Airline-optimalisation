import time
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from gurobipy import Model, GRB, quicksum

# ---- 1) Data inlezen ----
def load_data(fname="Group_2.xlsx"):
    wb = load_workbook(fname, data_only=True)
    df_f = pd.DataFrame(wb["Flights"].values)
    df_i = pd.DataFrame(wb["Itineraries"].values)
    df_r = pd.DataFrame(wb["Recapture"].values)
    for df in (df_f, df_i, df_r):
        df.columns = df.iloc[0]
        df.drop(0, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df_f, df_i, df_r

# ---- 2) Sets & parameters ----
df_flights, df_itin, df_rec = load_data("Group_2.xlsx")

L      = list(range(len(df_flights)))       # flights
P      = list(range(len(df_itin)))          # echte itineraries
P0     = len(P)                             # spill‐itinerary index
P_full = P + [P0]

# Capacities
CAP = {l: float(df_flights.loc[l, "Capacity"]) for l in L}

# Fares & demands
fare = {p: float(df_itin.loc[p, "Price [EUR]"]) for p in P}
D    = {p: float(df_itin.loc[p, "Demand"])      for p in P}
fare[P0], D[P0] = 0.0, sum(D.values())

# Recapture‐rates
b_pr = {}
for _, row in df_rec.iterrows():
    p = int(row["From Itinerary"])
    r = int(row["To Itinerary"])
    b_pr[(p,r)] = float(row["Recapture Rate"])

for p in P:
    b_pr[(p,p)]  = 1.0
    b_pr[(p,P0)] = 1.0

# δ_lp: flight l in itinerary p?
flight2id = {df_flights.loc[l,"Flight No."]: l for l in L}
delta_lp = {}
for p, row in df_itin.iterrows():
    for col in ("Flight 1","Flight 2"):
        fcode = row[col]
        if pd.notna(fcode):
            l = flight2id[fcode]
            delta_lp[(l,p)] = 1

# Q[l] = ∑ δ_lp * D[p]
Q = {l: sum(delta_lp.get((l,p),0) * D[p] for p in P) for l in L}


# ---- 3) RMP initialisatie ----
def init_RMP():
    m = Model("RMP")
    m.Params.OutputFlag = 0

    # alleen spill‐kolommen
    t = m.addVars(
        [(p,P0) for p in P],
        lb=0, vtype=GRB.CONTINUOUS, name="t"
    )

    # objective
    m.setObjective(
        quicksum((fare[p] - b_pr[(p,P0)] * fare[P0]) * t[p,P0] for p in P),
        GRB.MINIMIZE
    )

    # capacity constraints
    for l in L:
        m.addConstr(
            quicksum(delta_lp.get((l,p),0) * t[p,P0] for p in P)
            >= Q[l] - CAP[l],
            name=f"cap_{l}"
        )

    # demand constraints
    for p in P:
        m.addConstr(
            t[p,P0] <= D[p],
            name=f"demand_{p}"
        )

    m.update()
    return m, t


# ---- 4) Reduced cost berekening ----
def reduced_cost(p, r, pi, sigma):
    sum_pi_p = sum(pi[i] for i in L if delta_lp.get((i,p),0)==1)
    sum_pi_r = sum(pi[i] for i in L if delta_lp.get((i,r),0)==1)
    br = b_pr.get((p,r), 0)
    return (
        (fare[p] - br * fare[r])
        - sum_pi_p + br * sum_pi_r
        - sigma[p]
    )


# ---- 5) Column generation loop ----
def column_generation(max_iters=100, tol=1e-6):
    start_time = time.time()
    RMP, tvars = init_RMP()
    print(f"Start CG: columns in RMP = {len(tvars)}")

    for it in range(1, max_iters+1):
        RMP.optimize()
        if RMP.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError("RMP is infeasible of error")

        # duale waarden
        pi    = {l: RMP.getConstrByName(f"cap_{l}").Pi    for l in L}
        sigma = {p: RMP.getConstrByName(f"demand_{p}").Pi for p in P}

        # pricing
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
            var = RMP.addVar(
                lb=0, vtype=GRB.CONTINUOUS,
                obj=(fare[p] - b_pr.get((p,r),0) * fare[r]),
                name=f"t[{p},{r}]"
            )
            for l in L:
                a_out = delta_lp.get((l,p),0)
                a_in  = delta_lp.get((l,r),0) * b_pr.get((r,p),0)
                if a_out:
                    RMP.chgCoeff(RMP.getConstrByName(f"cap_{l}"), var, a_out)
                if a_in:
                    RMP.chgCoeff(RMP.getConstrByName(f"cap_{l}"), var, -a_in)
            RMP.chgCoeff(RMP.getConstrByName(f"demand_{p}"), var, 1.0)
            tvars[(p,r)] = var

        RMP.update()

    # final solve & objective uitlezen
    RMP.optimize()
    if RMP.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        raise RuntimeError("Eind‐RMP gaf geen oplossing")

    obj_val    = RMP.objVal
    cols_final = len(tvars)
    runtime    = time.time() - start_time
    print(f"End CG: columns in RMP = {cols_final}")
    print(f"Total CG iterations = {it-1}, runtime = {runtime:.2f}s")

    return RMP, tvars, obj_val


# ---- 6) Main ----
if __name__ == "__main__":
    rmp, t_vars, obj_val = column_generation()

    print("\n=== Eind‐oplossing RMP ===")
    print(f"Objective LP = {obj_val:.2f}")

    # voorbeeld 5 echte recaptures
    printed = 0
    print("\nEerste 20 recaptures (r≠spill):")
    for (p,r), var in t_vars.items():
        if r != P0 and var.X > 1e-6 and printed < 20:
            print(f"  t[{p} → {r}] = {var.X:.2f}")
            printed += 1

    # totaal spill
    total_spill = sum(t_vars[p,P0].X for p in P)
    print(f"\nTotaal spill‐pax = {total_spill:.2f}")