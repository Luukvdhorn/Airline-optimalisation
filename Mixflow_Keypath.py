from gurobipy import * 
from openpyxl import * 
import openpyxl
from time import *
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

import pandas as pd
import numpy as np
from openpyxl import load_workbook


# DATA INLADEN
wb_data = load_workbook("Group_40.xlsx", data_only=True)

flights_sheet = wb_data["Flights"]
itineraries_sheet = wb_data["Itineraries"]
recapture_sheet = wb_data["Recapture"]

# Zet sheets om naar DataFrames
df_flights = pd.DataFrame(flights_sheet.values)
df_flights.columns = df_flights.iloc[0]
df_flights = df_flights.iloc[1:].reset_index(drop=True)

df_itineraries = pd.DataFrame(itineraries_sheet.values)
df_itineraries.columns = df_itineraries.iloc[0]
df_itineraries = df_itineraries.iloc[1:].reset_index(drop=True)

df_recapture = pd.DataFrame(recapture_sheet.values)
df_recapture.columns = df_recapture.iloc[0]
df_recapture = df_recapture.iloc[1:].reset_index(drop=True)


# SETS
L = range(len(df_flights))        # flights
P = range(len(df_itineraries))    # itineraries

# Pp: mogelijke recapture-itineraries per p
Pp = {p: set() for p in P}
for _, row in df_recapture.iterrows():
    p = int(row["From Itinerary"])
    r = int(row["To Itinerary"])
    Pp[p].add(r)


# PARAMETERS

# fare_p
fare = np.zeros(len(P))
for p in P:
    fare[p] = float(df_itineraries.loc[p, "Price [EUR]"])

# Demand_p
D = np.zeros(len(P))
for p in P:
    D[p] = float(df_itineraries.loc[p, "Demand"])

# CAP_l
CAP = np.zeros(len(L))
for l in L:
    CAP[l] = float(df_flights.loc[l, "Capacity"])


# b_p^r  (recapture rates)
b_pr = {}
for _, row in df_recapture.iterrows():
    p = int(row["From Itinerary"])
    r = int(row["To Itinerary"])
    b_pr[(p, r)] = float(row["Recapture Rate"])


# Maak mapping van flight code naar index l
flight_code_to_id = {df_flights.loc[l, 'Flight No.']: l for l in L}

# delta_lp = 1 als flight l in itinerary p zit
delta_lp = {}

for p, row in df_itineraries.iterrows():

    # Flight 1 (altijd aanwezig)
    code1 = row['Flight 1']
    if pd.notna(code1):
        l = flight_code_to_id[code1]
        delta_lp[(l, p)] = 1

    # Flight 2 (optioneel)
    code2 = row['Flight 2']
    if pd.notna(code2):
        l = flight_code_to_id[code2]
        delta_lp[(l, p)] = 1

#---------------------------
#Parameters keypath 

Q = {}
for l in L:
    Q[l] = quicksum(delta_lp.get((l, p), 0) * D[p] for p in P)

    
# CONTROLE

from gurobipy import Model, GRB, quicksum

def main():
    model = Model("Keypath_Formulation")
    model.setParam("TimeLimit", 300)

    # DECISION VARIABLES Keypath
    t = model.addVars(P, P, lb=0, vtype=GRB.INTEGER, name="t")

    # OBJECTIVE FUNCTION
    model.setObjective(quicksum((fare[p] - b_pr.get((p, r), 0) * fare[r]) * t[p, r] for p in P for r in Pp),
        GRB.MINIMIZE)
    

    # CAPACITY CONSTRAINTS
    for l in L:
        model.addConstr(quicksum(delta_lp.get((l, p), 0) * t[p, r] for p in P for r in P)
            - quicksum(delta_lp.get((l, p), 0) * b_pr.get((p, r), 0) * t[r, p] for p in P for r in Pp)
            >= Q[l] - CAP[l], name=f"cap_{l}")

    # DEMAND CONSTRAINTS
    for p in P:
        model.addConstr(quicksum(t[p, r] for r in P) <= D[p], name=f"demand_{p}")


    for p in P:
        for r in P:
            model.addConstr(t[p, r] >= 0, name=f"nonneg_{p}_{r}") # met lb = 0 ook al gevangen
    # RESULTS
    model.optimize()
    model.write("keypath_formulation.lp")

# Bereken x[p,r] uit t[p,r]
    x = {}

    for p in P:
        for r in Pp:
            b = b_pr.get((p, r), 0)
            x[(p, r)] = b * t[p, r].x
            print(x[(p, r)])

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        print("\nAfgeleide mix-flow x[p,r]:")
        for (p, r), val in x.items():
            if val > 1e-6:
                print(f"x[{p},{r}] = {val:.2f}")

        print(f"Objective value: {model.objVal}\n")
        for r in P:
            for p in P:
                if t[r, p].x > 1e-6:
                    print(f"t[{p},{r}] = {t[p,r].x}") 
        
    else:
        print("No feasible solution found.")

main()

