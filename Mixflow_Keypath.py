import pandas as pd
import numpy as np
from openpyxl import load_workbook
from gurobipy import Model, GRB, quicksum

def load_data(fname="Group_40.xlsx"):
    wb = load_workbook(fname, data_only=True)
    df_flights     = pd.DataFrame(wb["Flights"].values)
    df_itineraries = pd.DataFrame(wb["Itineraries"].values)
    df_recapture   = pd.DataFrame(wb["Recapture"].values)

    # Hoofding rijen
    df_flights.columns     = df_flights.iloc[0];     df_flights     = df_flights.iloc[1:].reset_index(drop=True)
    df_itineraries.columns = df_itineraries.iloc[0]; df_itineraries = df_itineraries.iloc[1:].reset_index(drop=True)
    df_recapture.columns   = df_recapture.iloc[0];   df_recapture   = df_recapture.iloc[1:].reset_index(drop=True)

    return df_flights, df_itineraries, df_recapture

def build_model(df_flights, df_itineraries, df_recapture):
    # Sets
    L = list(range(len(df_flights)))      # alle flights
    P = list(range(len(df_itineraries)))  # alle originele itineraries
    P0 = len(P)                           # index van de fictieve spill‐itinerary
    P_full = P + [P0]

    # Parameters
    # capacity per flight
    CAP = { l: float(df_flights.loc[l, "Capacity"]) for l in L }

    # fare[p] en Demand D[p]
    fare = { p: float(df_itineraries.loc[p, "Price [EUR]"]) for p in P }
    D    = { p: float(df_itineraries.loc[p, "Demand"])      for p in P }

    # dummy: spill‐itinerary krijgt fare=0, grote vraag D
    fare[P0] = 0.0
    D[P0]    = sum(D[p] for p in P)  # of een ander groot getal

    # recapture‐rates b_pr
    b_pr = {}
    for _, row in df_recapture.iterrows():
        p = int(row["From Itinerary"])
        r = int(row["To Itinerary"])
        b_pr[(p,r)] = float(row["Recapture Rate"])

    # voeg self‐loops en spill‐optie p->P0 toe
    for p in P:
        b_pr[(p,p)] = 1.0        # self‐loop
        b_pr[(p,P0)] = 1.0       # spill naar P0

    # recapture‐sets Pp[p]
    Pp = { p: set() for p in P_full }
    # init met alle (p->r) uit Excel
    for (p,r), rate in b_pr.items():
        if p!=P0 and (r!=p or r!=P0):
            Pp[p].add(r)
    # self‐loops & spill
    for p in P:
        Pp[p].add(p)
        Pp[p].add(P0)
    Pp[P0] = set()  # spill heeft geen uitgaande recapture

    # δ_lp = 1 als flight l in itinerary p zit
    # maak flight_code->index map
    flight_code_to_id = { df_flights.loc[l,"Flight No."] : l for l in L }
    delta_lp = {}
    for p, row in df_itineraries.iterrows():
        for col in ["Flight 1", "Flight 2"]:
            code = row[col]
            if pd.notna(code):
                l = flight_code_to_id[code]
                delta_lp[(l,p)] = 1

    # Q[l] = oorspronkelijke (unconstrained) vraag op vlucht l
    Q = { l: quicksum(delta_lp.get((l,p),0) * D[p] for p in P) for l in L }

    # --- Build Gurobi model ---
    model = Model("Keypath_Mixflow")
    model.setParam("TimeLimit", 300)

    # Decision Variables: t[p,r] ≥ 0 continuous (of integer)
    t = model.addVars(
        [(p,r) for p in P_full for r in Pp[p]],
        lb=0, vtype=GRB.INTEGER, name="t"
    )

    # Objective: min ∑ₚ∈P ∑ᵣ∈Pp[p] (fare[p] − b_pr[p,r]*fare[r]) * t[p,r]
    obj = quicksum(
        (fare[p] - b_pr[(p,r)] * fare[r]) * t[p,r]
        for p in P
        for r in Pp[p]
    )
    model.setObjective(obj, GRB.MINIMIZE)

    # Capacity constraints voor elke l in L:
    # ∑ₚ δ_lp ( ∑ᵣ t[p,r] - ∑_{r: p∈Pp[r]} b_pr[r,p] t[r,p] ) ≥ Q[l] - CAP[l]
    for l in L:
        outflow  = quicksum( delta_lp.get((l,p),0) * quicksum(t[p,r] for r in Pp[p]) for p in P_full )
        inflow   = quicksum( delta_lp.get((l,p),0) * quicksum(b_pr[(r,p)] * t[r,p]
                       for r in P_full if p in Pp[r] )
                       for p in P_full )
        model.addConstr(outflow - inflow >= Q[l] - CAP[l], name=f"cap_{l}")

    # Demand constraints ∑ᵣ t[p,r] ≤ D[p] voor alle originele p (niet voor P0)
    for p in P:
        model.addConstr(quicksum(t[p,r] for r in Pp[p]) <= D[p], name=f"demand_{p}")

    return model, t, P, P_full, Pp, b_pr, P0


def solve_and_report(model, t, P, Pp, b_pr, P0):
    model.optimize()
    model.write("keypath_mixflow.lp")

    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        print("Geen oplossing (status =", model.status, ")")
        if model.status == GRB.INFEASIBLE:
            model.computeIIS(); model.write("infeasible.ilp")
        return

    print("=== Oplossing gevonden ===")
    print("Objective =", model.objVal)

    # 1) Print alleen recapture flows r ≠ P0
    print("\n-- Recaptures (p → r, r ≠ spill) --")
    for p in P:
        for r in Pp[p]:
            if r == P0:
                continue
            val = t[p, r].X
            if val > 1e-6:
                print(f"t[{p} → {r}] = {val:.2f} pax, x = {b_pr[(p,r)]*val:.2f}")

    # 2) Bereken en print totaal spill
    total_spill = sum(t[p, P0].X for p in P if P0 in Pp[p])
    print(f"\nTotaal spill‐passagiers (naar itinerary {P0}): {total_spill:.2f} pax")

# In je main‐blok
if __name__ == "__main__":
    # 1) Data inlezen
    df_flights, df_itineraries, df_recapture = load_data("Group_40.xlsx")

    # 2) Model bouwen
    model, t, P, P_full, Pp, b_pr, p0 = build_model(df_flights, df_itineraries, df_recapture)

    # 3) Oplossen en rapporteren
    solve_and_report(model, t, P, Pp, b_pr, p0)