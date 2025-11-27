from gurobipy import * 
from openpyxl import * 
from time import *
import numpy as np
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm

wb = load_workbook("pop.xlsx", data_only=True)
ws = wb["General"]

population_2021 = []
gdp_2021 = []

population_2024 = []
gdp_2024 = []

row = 4  
while True:
    city = ws.cell(row=row, column=1).value          # kolom A
    pop2021 = ws.cell(row=row, column=2).value       # kolom B

    if city is None:
        break

    population_2021.append((city, pop2021))
    row += 1

row=4

while True:
    city = ws.cell(row=row, column=1).value          # kolom A
    pop2024 = ws.cell(row=row, column=3).value       # kolom C

    if city is None:
        break

    population_2024.append((city, pop2024))
    row += 1

row = 4  

while True:
    country = ws.cell(row=row, column=5).value       # kolom E
    gdp2021 = ws.cell(row=row, column=6).value       # kolom F

    if country is None:
        break  
    gdp_2021.append((country, gdp2021))
    row += 1

row = 4

while True:
    country = ws.cell(row=row, column=5).value       # kolom E
    gdp2024 = ws.cell(row=row, column=7).value       # kolom G

    if country is None:
        break  
    gdp_2024.append((country, gdp2024))
    row += 1


wb_codes = load_workbook("Airport_names.xlsx", data_only=True)
ws_codes = wb_codes.active

city_to_icao = {}
country_to_icao = {}

row = 1
while True:
    city = ws_codes.cell(row=row, column=1).value
    country = ws_codes.cell(row=row, column=2).value
    icao = ws_codes.cell(row=row, column=3).value

    if city is None:
        break

    city_to_icao[city] = icao
    country_to_icao[country] = icao

    row += 1

population_2021_dict = {}
population_2024_dict = {}
gdp_2021_dict = {}
gdp_2024_dict = {}

row = 4
while True:
    city = ws.cell(row=row, column=1).value        # kolom A
    pop2021 = ws.cell(row=row, column=2).value     # kolom B
    pop2024 = ws.cell(row=row, column=3).value     # kolom C
    gdp2021 = ws.cell(row=row, column=6).value     # kolom F
    gdp2024 = ws.cell(row=row, column=7).value     # kolom G

    if city is None:
        break

    if city in city_to_icao:
        icao = city_to_icao[city]
        population_2021_dict[icao] = pop2021
        population_2024_dict[icao] = pop2024
        gdp_2021_dict[icao] = gdp2021
        gdp_2024_dict[icao] = gdp2024

    row += 1

# print("Population:")
# print(population_2021_dict)

# print("GDP 2021:")
# print(gdp_2021_dict)

# print("Population 2024:")
# print(population_2024_dict)

# print("GDP 2024:")
# print(gdp_2024_dict)


wb = load_workbook("DemandGroup40.xlsx", data_only=True)
ws = wb.active 


icao_row = 5    
lat_row = 6     
lon_row = 7    
start_col = 3   

airports = []
latitudes = []
longitudes = []

col = start_col
while True:
    icao = ws.cell(row=icao_row, column=col).value
    lat = ws.cell(row=lat_row, column=col).value
    lon = ws.cell(row=lon_row, column=col).value
    if icao is None:
        break
    airports.append(icao)
    latitudes.append(float(lat))
    longitudes.append(float(lon))
    col += 1


RE = 6371.0 

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


# print("Afstandsmatrix (km):")
# print("\t" + "\t".join(airports))
# for i, row in enumerate(dij):
#     print(airports[i], "\t" + "\t".join(f"{val:.2f}" for val in row))


demand_start_row = icao_row + 8 
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

print("Demandmatrix 2021 (week):")
print("\t" + "\t".join(airports))
for i in range(n):
    print(airports[i], "\t" + "\t".join(f"{D[i,j]:.0f}" for j in range(n)))

# Determning B1, B2, B3 and k
f = 1.42   # fuel cost constant from assignment

n = len(airports)
rows = []

for i in range(n):        
    for j in range(n):
        if i == j:
            continue                    # If orgin and destionation is the same, go on

        Dij = D[i, j]
        if Dij <= 0:
            continue                    # It there is no demand, go on

        i_icao = airports[i]            #Collectiong ICAO code
        j_icao = airports[j]

        row = {
            "D": Dij,
            "pop": population_2021_dict[i_icao] * population_2021_dict[j_icao],
            "gdp": gdp_2021_dict[i_icao] * gdp_2021_dict[j_icao],
            "fd": f * dij[i, j]
        }
        rows.append(row)

df = pd.DataFrame(rows)
# print(df)

# --- LOGS ---
df['lnD']    = np.log(df['D'])                  #logarithm to lineariz the model, dat doe je door ln van alles te nemen heeft google mij verteld
df['ln_pop'] = np.log(df['pop'])
df['ln_gdp'] = np.log(df['gdp'])
df['ln_fd']  = np.log(df['fd'])

# --- OLS REGRESSIE ---
X = df[['ln_pop', 'ln_gdp', 'ln_fd']]           # Independent variables
X = sm.add_constant(X)                          # This creates ln(k)
y = df['lnD']                                   # Dependent variable

model = sm.OLS(y, X).fit()                      # Ordinary least squares
# print(model.summary())

# --- PARAMETERS ---
a = model.params['const']            # ln(k)
b1 = model.params['ln_pop']          # Creating b1
b2 = model.params['ln_gdp']

beta3 = model.params['ln_fd']        # = -b3 (afstand in noemer)
b3 = -beta3                          # Because in function it is in the denominator

k = np.exp(a)                       # Making k again after it is a ln()

# print("\n--- GRAVITY MODEL PARAMETERS ---")
# print(f"k  = {k}")
# print(f"b1 = {b1:.4f}")
# print(f"b2 = {b2:.4f}")
# print(f"b3 = {b3:.4f}")

# Maak een lege matrix voor voorspelde demand
D_pred = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            D_pred[i, j] = 0  # geen vraag van een stad naar zichzelf
            continue
        i_icao = airports[i]
        j_icao = airports[j]

        pop_product = population_2021_dict[i_icao] * population_2021_dict[j_icao]
        gdp_product = gdp_2021_dict[i_icao] * gdp_2021_dict[j_icao]
        fd_ij = dij[i, j]

        D_pred[i, j] = k * (pop_product**b1) * (gdp_product**b2) * ((f * fd_ij)**(beta3))

print("Voorspelde demand (gravity model):")
print("\t" + "\t".join(airports))
for i in range(n):
    print(airports[i], "\t" + "\t".join(f"{D_pred[i,j]:.0f}" for j in range(n)))


# Verschilmatrix
D_diff = D_pred - D  # absoluut verschil
D_pct  = np.divide(D_diff, D, out=np.zeros_like(D_diff), where=D!=0) * 100  # percentage verschil

print("Absoluut verschil (voorspeld - werkelijke vraag):")
print("\t" + "\t".join(airports))
for i in range(n):
    print(airports[i], "\t" + "\t".join(f"{D_diff[i,j]:.0f}" for j in range(n)))

