from gurobipy import * 
from openpyxl import * 
from time import *


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

