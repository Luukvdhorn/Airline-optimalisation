from openpyxl import *
from time import *
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import math
import matplotlib.pyplot as plt
import pandas as pd
import os as os

wb = load_workbook("pop.xlsx")

sheet = wb.active  
rows = tuple(sheet.iter_rows(values_only=True))

# Zoek header rij (de rij waar "City" en "Country" staan)
for i, r in enumerate(rows):
    if r[0] == "City":
        header_index = i
        break

# Extract population section
pop_rows = []
for r in rows[header_index+1:]:
    if r[0] is None:
        break
    pop_rows.append(r[0:3])   # city, 2021, 2024

df_pop = pd.DataFrame(pop_rows, columns=["City", "Pop2021", "Pop2024"])

# Extract GDP section
gdp_rows = []
for r in rows[header_index+1:]:
    if r[4] is None:
        break
    gdp_rows.append(r[4:7])   # country, 2021, 2024

df_gdp = pd.DataFrame(gdp_rows, columns=["Country", "GDP2021", "GDP2024"])

print(df_pop)
print(df_gdp)
