import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import * 
import openpyxl

wb = load_workbook("DemandGroup40.xlsx", data_only=True)
ws = wb.active

icao_row = 5    
lat_row = 6    
lon_row = 7    
start_col = 3
runway_row = 8
slots_row  = 9
runways = []
slots = []
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
    slot = ws.cell(row=slots_row, column=col).value
    if icao is None:
        break
    airports.append(icao)
    latitudes.append(float(lat))
    longitudes.append(float(lon))
    runways.append(float(runway))
    if slot == '-':
        slots.append(float('inf'))
    else:
        slots.append(float(slot))
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

print (df_aircraft)


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


cl = np.zeros(ac)                           #COSTS WEEKLY LEASE
for k in K:
    cl[k] = df_aircraft['Lease Cost [€/day]'][k]
    

C = np.zeros(ac)                            #COST FIXED OPERATINGH
for k in K:
    C[k] = df_aircraft['Fixed Operating Cost (Per Fligth Leg)  [€]'][k]
    

CT = np.zeros(ac)                           #TIME COST PARAMETER
for k in K:
    CT[k] = df_aircraft['Cost per Hour'][k]
    print(CT[k])


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
    


Ck_ij = np.zeros((n, n, ac))                #tOTAL OPERARION COST PER LEG (WITH DISCOUNT OF 30%)
for k in K:
    for i in N:
        for j in N:
            Ck_ij[i, j, k] = (C[k] + C_Tij[i, j, k] + C_Fij[i, j, k]) * 0.7

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

LF = 0.80 



#NIEUWE NOG TOEVOEGEN: FLEET
#SLOTS (STAAT NU IN DEMAND)
            