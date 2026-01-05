import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import * 
import openpyxl

wb = load_workbook("Assignment 2\DemandGroup40.xlsx", data_only=True)
ws = wb.active

icao_row = 5    
lat_row = 6    
lon_row = 7    
start_col = 3
runway_row = 8
runways = []
hub_index = 2

RE = 6371.0                                     # Radius Earth
f = 1.42                                        # Fuel cost constant 

airports = []
latitudes = []
longitudes = []


col = start_col

while True:
    icao = ws.cell(row=icao_row, column=col).value
    lat = ws.cell(row=lat_row, column=col).value
    lon = ws.cell(row=lon_row, column=col).value
    runway = ws.cell(row=runway_row, column=col).value
    if icao is None:
        break
    airports.append(icao)
    latitudes.append(float(lat))
    longitudes.append(float(lon))
    runways.append(float(runway))
    col += 1

#DISTANCE FORMULA

def distance(phi_i, lam_i, phi_j, lam_j):
    phi_i, phi_j = np.radians(phi_i), np.radians(phi_j)
    lam_i, lam_j = np.radians(lam_i), np.radians(lam_j)
    return 2 * RE * np.arcsin(
        np.sqrt(np.sin((phi_i - phi_j)/2)**2 +np.cos(phi_i)*np.cos(phi_j)*np.sin((lam_i - lam_j)/2)**2))


n = len(airports)
dij = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dij[i, j] = distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

#DEMAND IMPORT
demand_start_row = icao_row + 7
demand_start_col = start_col 

n = len(airports)
D = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        cell = ws.cell(row=demand_start_row + i, column=demand_start_col + j).value
        try:
            D[i, j] = float(cell) if cell is not None else 0
        except:
            D[i, j] = 0



print(D)


#FLEET IMPORT
wb2 = openpyxl.load_workbook("Assignment 2/FleetType.xlsx", data_only=True)
sheet2 = wb2.active

aircraft_names = [cell.value for cell in sheet2[1][1:]if cell.value is not None]
n_ac = len(aircraft_names)
data = {}

for row in sheet2.iter_rows(min_row=2, values_only=True):
    if row[0] is None:
        continue  
    parameter_name = row[0]
    values = row[1:1 + n_ac]  
    data[parameter_name] = values

df_aircraft = pd.DataFrame(data, index=aircraft_names)


#HOUR COEFFICIENT
wb = openpyxl.load_workbook("Assignment 2/HourCoefficients.xlsx", data_only=True)
ws = wb.active

start_row = 3        #AMSTERDAM
start_col = 4        #Hour 0 
n = len(airports)    #LENGTE RIJEN
T = 24               #AANTAL UREN (D t/m AA)

H = np.zeros((n, T))

for i in range(n):
    for t in range(T):
        cell = ws.cell(row=start_row + i, column=start_col + t).value
        try:
            H[i, t] = float(cell) if cell is not None else 0
        except:
            H[i, t] = 0

import numpy as np
import pandas as pd

# veronderstel dat je dit al hebt:
# airports = [...]       # lijst van ICAO-codes
# D        = np.ndarray  # shape (n,n), daily demand per route
# H        = np.ndarray  # shape (n,24), hour coefficients per airport

n, T = D.shape[0], H.shape[1]
hub_idx = airports.index('EHAM')   # index van Amsterdam

# 1) hub_arr[i,t] = D[i,hub] * H[i,t]
hub_arr = D[:, hub_idx][:, None] * H

# 2) hub_dep[j,t] = D[hub,j] * H[hub,t]
hub_dep = np.outer(D[hub_idx, :], H[hub_idx, :])

# 3) geen self-loops
hub_arr[hub_idx, :] = 0
hub_dep[hub_idx, :] = 0

# 4) optioneel: zet in DataFrame voor mooi overzicht
hours = [f'Uur_{t}' for t in range(T)]
df_arr = pd.DataFrame(hub_arr, index=airports, columns=hours)
df_dep = pd.DataFrame(hub_dep, index=airports, columns=hours)

print("=== Demand for flights that arive in EHAM, given hour is departure time in orgin ===")
print(df_dep)
print("\n=== Demand for flights that depart from EHAM ===")
print(df_dep)


# DATA IMPORT
N = range(len(airports))                    # Set of airports; i, j in N
K = range(len(df_aircraft))                 # Set of aircrafts; k in K
ac = len(df_aircraft)                       # Total amount of aircrafts

d = np.zeros((n, n))                        # Distance between i and j
for i in N:
    for j in N:
        d[i, j] = dij[i, j]


y = np.zeros((n, n))
for i in N:                                 #YIELD
    for j in N:
        if d[i, j] > 0:
            y[i, j] = (5.9 * d[i, j]**(-0.76) + 0.043)
        else:
            y[i, j] = 0 

v = np.zeros(ac)                            #SPEED OF AIRPLANES (k)
for k in K:
    v[k] = df_aircraft['Speed [km/h]'][k]
    
s = np.zeros(ac)                            #SEATS
for k in K:
    s[k] = df_aircraft['Seats'][k]

ra = np.zeros(ac)                           #RANGE OF AIRCRAFT
for k in K:
    ra[k] = df_aircraft['Maximum Range [km]'][k]

RAC = np.zeros(ac)                          #RUNWAY REQUIREMENT DEPARTURE
for k in K:
    RAC[k] = df_aircraft['Runway Required [m]'][k]

RAP = np.zeros(n)                           #RUNWAY REQUIREMENT ARRIVAL
for j in N:
    RAP[j] = runways[j]

TAT = np.zeros(ac)                                                      #TURN AROUND TIMES IN MIN
for k in K:
    TAT[k] = df_aircraft['Average TAT [min]'][k]

Fleet = np.zeros(ac)                                                      #Fleettypes
for k in K:
    Fleet[k] = df_aircraft['Fleet'][k]

#COSTS 

cl = np.zeros(ac)                           #COSTS WEEKLY LEASE
for k in K:
    cl[k] = df_aircraft['Lease Cost [€/day]'][k]
    

C = np.zeros(ac)                            #COST FIXED OPERATING
for k in K:
    C[k] = df_aircraft['Fixed Operating Cost (Per Fligth Leg)  [€]'][k]
    

CT = np.zeros(ac)                           #TIME COST PARAMETER
for k in K:
    CT[k] = df_aircraft['Cost per Hour'][k]
    


#Nog checken
C_Tij = np.zeros((n, n, ac))                #TIME COST PER LEG
for k in K:
    C_Tij[ :, :,k] = CT[k] * (d/ v[k])

    

CF = np.zeros(ac)                           #FUEL COST PARAMETER
for k in K:
    CF[k] = df_aircraft['Fuel Cost Parameter'][k]

C_Fij = np.zeros((n, n, ac))                #FUEL COST PER LEG
for k in K:
    C_Fij[:, :, k] = ((CF[k] * 1.42) / 1.5) * d
    


Ck_ij = np.zeros((n, n, ac))                #TOTAL OPERARION COST PER LEG (WITH DISCOUNT OF 30%)
for k in K:
    for i in N:
        for j in N:
            Ck_ij[i, j, k] = (C[k] + C_Tij[i, j, k] + C_Fij[i, j, k]) * 0.7     #(Nu heb ik die 0,7 er nog staan kan weg later)

a = {}                                  #BIG M CONSTRAIN FOR RUNWAYS
for i in N:
    for j in N:
        for k in K:
            if d[i,j] <= ra[k]:
                a[i,j,k] = 10000
            else:
                a[i,j,k] = 0

g = np.ones(n)                      #HUB IS AMSTERDAM 
g[hub_index] = 0 

LF = 0.80               # van 0.75 naar 0.80


 #VANAF HIER CODE DYNAMIC PROGRAMMEREN 

def possible_actions(state, aircraft):
   
   def is_valid_destination(destination):
        # Check runway requirement and range requirement
        return (RAC[k] < RAP[destination] and ra[k] >= distance[state, destination])
    
    # Only flights to and from the hub are considered. Also consider ground arc
    if state == hub_index:
        return [airport for airport in airports if is_valid_destination(airport)]
    else:
        possible_destinations = [state, hub_index]
        return [airport for airport in possible_destinations if is_valid_destination(airport)]

def get_flight_time(depart_from, arive_at, aircraft):
    """
    Compute flight time in minutes.
    Besides the TAT, include: 15 minutes extra for takeoff, 15 minutes extra for landing
    """
    distance = distance_matrix.loc[Airports[depart_from], Airports[arive_at]]
    speed = sp[aircraft]
    return 15 + (distance / speed)*60 + TAT[aircraft] + 15




def dynamic_programming(aircraft, D):




            