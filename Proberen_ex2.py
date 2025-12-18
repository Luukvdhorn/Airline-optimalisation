from gurobipy import *
import pandas as pd
from openpyxl import load_workbook
import time

# =====================================================
# DATA
# =====================================================

wb = load_workbook("Group_40.xlsx", data_only=True)

df_flights = pd.DataFrame(wb["Flights"].values)
df_flights.columns = df_flights.iloc[0]
df_flights = df_flights.iloc[1:].reset_index(drop=True)

df_itin = pd.DataFrame(wb["Itineraries"].values)
df_itin.columns = df_itin.iloc[0]
df_itin = df_itin.iloc[1:].reset_index(drop=True)

df_rec = pd.DataFrame(wb["Recapture"].values)
df_rec.columns = df_rec.iloc[0]
df_rec = df_rec.iloc[1:].reset_index(drop=True)

# =====================================================
# SETS
# =====================================================

L = range(len(df_flights))
P_real = range(len(df_itin))
p_fict = len(df_itin)
P = list(P_real) + [p_fict]

# =====================================================
# PARAMETERS
# =====================================================

fare = {p: float(df_itin.loc[p, "Price [EUR]"]) for p in P_real}
fare[p_fict] = 0.0

D = {p: float(df_itin.loc[p, "Demand"]) for p in P_real}
D[p_fict] = sum(D[p] for p in P_real)

CAP = {i: float(df_flights.loc[i, "Capacity"]) for i in L}

# delta_{pi}
flight_id = {df_flights.loc[i, "Flight No."]: i for i in L}
delta = {}

for p, row in df_itin.iterrows():
    for col in ["Flight 1", "Flight 2"]:
        if pd.notna(row[col]):
            delta[(p, flight_id[row[col]])] = 1

for i in L:
    delta[(p_fict, i)] = 0

# Q_i
Q = {
    i: sum(delta.get((p, i), 0) * D[p] for p in P_real)
    for i in L
}

# recapture rates b_p^r
b = {}
for _, row in df_rec.iterrows():
    p = int(row["From Itinerary"])
    r = int(row["To Itinerary"])
    b[(p, r)] = float(row["Recapture Rate"])

# fictitious + self
for p in P_real:
    b[(p, p)] = 1.0
    b[(p, p_fict)] = 1.0

# =====================================================
# BUILD INITIAL RMP
# =====================================================

def build_initial_rmp():
    model = Model("RMP")
    model.setParam("OutputFlag", 0)

    t = {}

    # only fictitious columns initially
    for p in P_real:
        t[(p, p_fict)] = model.addVar(lb=0, name=f"t_{p}_{p_fict}")

    # objective
    model.setObjective(quicksum((fare[p] - b[(p, p_fict)] * fare[p_fict]) * t[(p, p_fict)] for p in P_real), GRB.MINIMIZE)

    # capacity constraints
    cap = {}
    for i in L:
        cap[i] = model.addConstr(
            quicksum(delta.get((p, i), 0) * t[(p, p_fict)]
                     for p in P_real)
            >= Q[i] - CAP[i],
            name=f"cap_{i}"
        )

    # demand constraints
    dem = {}
    for p in P_real:
        dem[p] = model.addConstr(
            t[(p, p_fict)] <= D[p],
            name=f"demand_{p}"
        )

    model.update()
    return model, t, cap, dem

# =====================================================
# PRICING PROBLEM
# =====================================================

def pricing(pi, sigma, existing_cols):
    candidates = []

    for (p, r), bpr in b.items():
        if r == p or (p, r) in existing_cols:
            continue

        rc = (fare[p] - sum(pi[i] * delta.get((p, i), 0) for i in L) - bpr * (fare[r] - sum(pi[i] * delta.get((r, i), 0) for i in L)) - sigma[p])


        if rc < -1e-6:
            candidates.append((rc, p, r))

    candidates.sort()
    return candidates[:10]

# =====================================================
# COLUMN GENERATION LOOP
# =====================================================

import time

def column_generation(max_iter=20):
    model, t, cap, dem = build_initial_rmp()
    existing_cols = set(t.keys())
    total_added_cols = 0

    start_time = time.time()  # ⏱️ start totale timer

    for it in range(max_iter):
        iter_start = time.time()
        model.optimize()
        iter_time = time.time() - iter_start

        pi = {i: cap[i].Pi for i in L}
        sigma = {p: dem[p].Pi for p in P_real}

        print(f"\n================ ITERATION {it} ================")
        print(f"Objective: {model.objVal:.2f}")
        print(f"Iteration solve time: {iter_time:.4f} sec")

        print("\nπ (pi) (Flight duals):")
        for i, v in pi.items():
            if abs(v) > 1e-6:
                print(f"  Flight {i}: {v:.2f}")

        print("\nσ (sigma) (Itinerary duals):")
        for p, v in sigma.items():
            if abs(v) > 1e-6:
                print(f"  Itin {p}: {v:.2f}")

        new_cols = pricing(pi, sigma, existing_cols)

        if not new_cols:
            print("\nNo negative reduced costs → OPTIMAL")
            break

        print(f"\nAdded columns (top {len(new_cols)}):")
        for rc, p, r in new_cols:
            print(f"  t[{p},{r}]  reduced cost = {rc:.2f}")

            var = model.addVar(lb=0, name=f"t_{p}_{r}")
            t[(p, r)] = var
            existing_cols.add((p, r))
            total_added_cols += 1

            # objective opbouwen zoals in de eerste versie
            model.setObjective(
                model.getObjective()
                + (fare[p] - b[(p, r)] * fare[r]) * var
            )

            # capacity constraints
            for i in L:
                model.chgCoeff(
                    cap[i],
                    var,
                    delta.get((p, i), 0) - delta.get((r, i), 0) * b[(p, r)]
                )

            # demand constraint
            model.chgCoeff(dem[p], var, 1)

        model.update()

    # =========================
    # FINAL SUMMARY
    # =========================
    total_time = time.time() - start_time  # ✅ nu gedefinieerd
    print("\n================ FINAL SUMMARY ================")
    print(f"Total optimization time: {total_time:.2f} seconds")
    print(f"Initial columns (fictitious): {len(P_real)}")
    print(f"Added columns: {total_added_cols}")
    print(f"Total columns in final RMP: {len(existing_cols)}")
    print(f"Final objective value: {model.objVal:.2f}")



# =====================================================
# RUN
# =====================================================

column_generation()
