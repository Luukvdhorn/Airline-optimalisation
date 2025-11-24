from gurobipy import * 
from openpyxl import * 
from time import *
import numpy as np


# Laad het Excel-bestand
wb = load_workbook("pop.xlsx", data_only=True)

# Selecteer de sheet
ws = wb["General"]

population_2021 = []
gdp_2021 = []

population_2024 = []
gdp_2024 = []

# --------------------------------------------------------------------
# Stap 1 — Population data (kolommen A–C)
# --------------------------------------------------------------------
row = 4   # data begint op rij 3 (rij 1 leeg, rij 2 kolomnamen)

while True:
    city = ws.cell(row=row, column=1).value          # kolom A
    pop2021 = ws.cell(row=row, column=2).value       # kolom B

    if city is None:
        break  # einde van het population-blok

    population_2021.append((city, pop2021))
    row += 1

row=4

while True:
    city = ws.cell(row=row, column=1).value          # kolom A
    pop2024 = ws.cell(row=row, column=3).value       # kolom C

    if city is None:
        break  # einde van het population-blok

    population_2024.append((city, pop2024))
    row += 1

# --------------------------------------------------------------------
# Stap 2 — GDP data (kolommen E–G)
# --------------------------------------------------------------------
row = 4   # ook hier start de data op rij 3

while True:
    country = ws.cell(row=row, column=5).value       # kolom E
    gdp2021 = ws.cell(row=row, column=6).value       # kolom F

    if country is None:
        break  # einde van GDP-blok

    gdp_2021.append((country, gdp2021))
    row += 1

row = 4

while True:
    country = ws.cell(row=row, column=5).value       # kolom E
    gdp2024 = ws.cell(row=row, column=7).value       # kolom G

    if country is None:
        break  # einde van GDP-blok

    gdp_2024.append((country, gdp2024))
    row += 1

# Print resultaten
print("Population per city (2021):")
for city, pop in population_2021:
    print(f"{city}: {pop}")

print("Population per city (2024):")
for city, pop in population_2024:
    print(f"{city}: {pop}")


print("\nGDP per country (2021):")
for country, gdp in gdp_2021:
    print(f"{country}: {gdp}")

print("\nGDP per country (2024):")
for country, gdp in gdp_2024:
    print(f"{country}: {gdp}")


wb = load_workbook("DemandGroup40.xlsx", data_only=True)
ws = wb.active 

# ---- ICAO codes, latitudes en longitudes ophalen ----
icao_row = 5    # ICAO Code labels in kolom B
lat_row = 6     # Latitude (deg) labels in kolom B
lon_row = 7     # Longitude (deg) labels in kolom B
start_col = 3   # Waarden beginnen vanaf kolom C

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

# ---- Haversine afstandsfunctie ----
RE = 6371.0  # km

def distance(phi_i, lam_i, phi_j, lam_j):
    phi_i, phi_j = np.radians(phi_i), np.radians(phi_j)
    lam_i, lam_j = np.radians(lam_i), np.radians(lam_j)
    return 2 * RE * np.arcsin(
        np.sqrt(
            np.sin((phi_i - phi_j)/2)**2 +
            np.cos(phi_i)*np.cos(phi_j)*np.sin((lam_i - lam_j)/2)**2
        )
    )

# ---- d_ij matrix opbouwen ----
n = len(airports)
dij = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dij[i, j] = distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

# ---- Afdrukken ----
print("Afstandsmatrix (km):")
print("\t" + "\t".join(airports))
for i, row in enumerate(dij):
    print(airports[i], "\t" + "\t".join(f"{val:.2f}" for val in row))

