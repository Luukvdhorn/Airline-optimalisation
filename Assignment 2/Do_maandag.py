import numpy as np
import pandas as pd
import openpyxl
import math

# --- Data Import ---
wb = openpyxl.load_workbook("Assignment 2/DemandGroup40.xlsx", data_only=True)
ws = wb.active

icao_row = 5    
lat_row = 6    
lon_row = 7    
start_col = 3
runway_row = 8
runways = []
hub_index = 2

airports = []
latitudes = []
longitudes = []

col = start_col
while True:
    icao = ws.cell(row=icao_row, column=col).value
    if icao is None:
        break
    lat = ws.cell(row=lat_row, column=col).value
    lon = ws.cell(row=lon_row, column=col).value
    runway = ws.cell(row=runway_row, column=col).value

    airports.append(icao)
    latitudes.append(float(lat))
    longitudes.append(float(lon))
    runways.append(float(runway))
    col += 1

n = len(airports)
RE = 6371.0

def distance(phi_i, lam_i, phi_j, lam_j):
    phi_i, phi_j = np.radians(phi_i), np.radians(phi_j)
    lam_i, lam_j = np.radians(lam_i), np.radians(lam_j)
    return 2 * RE * np.arcsin(
        np.sqrt(np.sin((phi_i - phi_j)/2)**2 + np.cos(phi_i)*np.cos(phi_j)*np.sin((lam_i - lam_j)/2)**2)
    )

d = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        d[i, j] = distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

# Demand Import
demand_start_row = icao_row + 7
demand_start_col = start_col

D = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        val = ws.cell(row=demand_start_row + i, column=demand_start_col + j).value
        try:
            D[i, j] = float(val) if val is not None else 0
        except:
            D[i,j] = 0

# Hour coefficients import
wb_h = openpyxl.load_workbook("Assignment 2/HourCoefficients.xlsx", data_only=True)
ws_h = wb_h.active

T = 24  # 24 hours
H = np.zeros((n, T))
start_row = 3
start_col = 4
for i in range(n):
    for t in range(T):
        val = ws_h.cell(row=start_row + i, column=start_col + t).value
        try:
            H[i, t] = float(val) if val is not None else 0
        except:
            H[i, t] = 0

print("Sum HourCoefficients per airport (time band spread):")
for i in range(n):
    print(f"{airports[i]}: {np.sum(H[i]):.3f}")

# Calculate demand per hour by spreading weekly demand using hour coefficients
dem_hour = np.zeros((n, n, T))   # demand per origin-dest-hour
for i in range(n):
    for j in range(n):
        for t in range(T):
            dem_hour[i, j, t] = D[i, j] * H[i, t]

# Fleet data import
wb2 = openpyxl.load_workbook("Assignment 2/FleetType.xlsx", data_only=True)
sheet2 = wb2.active

aircraft_names = [cell.value for cell in sheet2[1][1:] if cell.value is not None]
n_ac = len(aircraft_names)
data = {}
for row in sheet2.iter_rows(min_row=2, values_only=True):
    pname = row[0]
    if pname is None:
        continue
    values = row[1:1 + n_ac]
    data[pname] = values

df_aircraft = pd.DataFrame(data, index=aircraft_names)

# Extract parameters
s = df_aircraft['Seats'].values
v = df_aircraft['Speed [km/h]'].values
ra = df_aircraft['Maximum Range [km]'].values
RAC = df_aircraft['Runway Required [m]'].values
TAT = df_aircraft['Average TAT [min]'].values
Fleet = df_aircraft['Fleet'].values.astype(int)
cl = df_aircraft['Lease Cost [€/day]'].values
C_fix = df_aircraft['Fixed Operating Cost (Per Fligth Leg)  [€]'].values
CT = df_aircraft['Cost per Hour'].values
CF = df_aircraft['Fuel Cost Parameter'].values

runway_arrival = np.array(runways)

hub_index = 2
step_minutes = 6
total_steps = 24 * 10  # 6 min steps in 24h

def timestep_to_hour(timestep):
    total_minutes = timestep * step_minutes
    return total_minutes // 60

def action_possible(stage, ac_type):
    possible = []
    if stage == hub_index:
        candidates = range(n)
    else:
        candidates = [stage, hub_index]
    for dest in candidates:
        if RAC[ac_type] < runway_arrival[dest] and ra[ac_type] >= d[stage, dest]:
            possible.append(dest)
    return possible

def block_time(origin, dest, ac_type):
    if origin == dest:
        return 0
    return 15 + (d[origin,dest] / v[ac_type]) * 60 + TAT[ac_type] + 15

def operating_cost(origin, dest, ac_type):
    if origin == dest:
        return 0
    C_fixed = C_fix[ac_type]
    C_time = CT[ac_type] * (d[origin,dest] / v[ac_type])
    C_fuel = (CF[ac_type] * 1.42 / 1.5) * d[origin,dest]
    return C_fixed + C_time + C_fuel

def revenue_func(origin, dest, flow):
    if origin == dest:
        return 0
    y = 5.9 * d[origin,dest]**(-0.76) + 0.043
    return y * d[origin,dest] * flow

def capture_demand(origin, dest, timestep, ac_type, dem_hour):
    hour, _ = divmod(timestep * step_minutes, 60)
    capacity = 0.8 * s[ac_type]
    remaining_capacity = capacity
    flow = 0
    for h in [hour, hour-1, hour-2]:
        if h < 0:
            continue
        available = dem_hour[origin, dest, h]
        if available <= 0:
            continue
        taken = min(available, remaining_capacity)
        dem_hour[origin, dest, h] -= taken
        flow += taken
        remaining_capacity -= taken
        if remaining_capacity <= 0:
            break
    return flow, dem_hour

def dynamic_programming(ac_type, dem_hour):
    profit_matrix = np.full((n, total_steps), -1e9)
    action_matrix = np.full((n, total_steps), -1)
    profit_matrix[:, -1] = 0

    total_flow_dp = 0

    for t in range(total_steps-2, -1, -1):
        current_hour = timestep_to_hour(t)
        for loc in range(n):
            best_profit = -1e9
            best_action = -1
            for dest in action_possible(loc, ac_type):
                if dest == loc:
                    profit = profit_matrix[loc, t+1]
                    flow = 0
                else:
                    blockt = block_time(loc, dest, ac_type)
                    arrival_time = t + math.ceil(blockt/step_minutes)
                    future_profit = profit_matrix[dest, arrival_time] if arrival_time < total_steps else 0
                    expected_demand = 0
                    for hh in [current_hour, current_hour-1, current_hour-2]:
                        if hh >= 0:
                            expected_demand += dem_hour[loc,dest,hh]
                    capacity = 0.8 * s[ac_type]
                    flow = min(capacity, expected_demand)
                    revenue = revenue_func(loc, dest, flow)
                    cost = operating_cost(loc, dest, ac_type)
                    profit = revenue - cost + future_profit
                if profit > best_profit:
                    best_profit = profit
                    best_action = dest
                    total_flow_dp += flow
            profit_matrix[loc, t] = best_profit
            action_matrix[loc, t] = best_action

    #print(f"[DP] Aircraft {ac_type} total estimated flow: {total_flow_dp:.1f}")
    return action_matrix, profit_matrix

def schedule(ac_type, action_matrix, profit_matrix, dem_hour):
    route = [(0, hub_index)]
    current_pos = hub_index
    t = 0
    total_block_time = 0
    flows = []

    while t < total_steps:
        next_pos = int(action_matrix[current_pos, t])
        if next_pos < 0 or next_pos >= n:
            break
        if next_pos == current_pos:
            t += 1
            continue
        bt = block_time(current_pos, next_pos, ac_type)
        t_next = t + math.ceil(bt/step_minutes)
        if t_next >= total_steps:
            break
        total_block_time += bt
        flow, dem_hour = capture_demand(current_pos, next_pos, t, ac_type, dem_hour)
        flows.append(flow)
        route.append((t_next, next_pos))
        #print(f"[SCHEDULE] AC {ac_type}: flown {flow:.1f} pax from {airports[current_pos]} to {airports[next_pos]} at t={t}")
        current_pos = next_pos
        t = t_next

    profit = profit_matrix[hub_index, 0] - cl[ac_type]
    min_block = 6*60
    if total_block_time < min_block:
        profit = -1e9

    # print(f"[SCHEDULE] AC {ac_type} total flown pax: {sum(flows):.1f}")
    return route, profit, total_block_time, flows, dem_hour

# Main loop
demand = dem_hour.copy()
available_ac = Fleet.copy()
total_passengers_transported = 0
solution_dict = {}
iteration = 0

# Teller hoeveel vliegtuigen per type zijn ingezet
used_ac_count = np.zeros(len(Fleet), dtype=int)

while any(available_ac > 0):
    profits = np.full(n_ac, -1e9)
    ut_time = np.zeros(n_ac)
    routes = {}

    for k in range(n_ac):
        if available_ac[k] > 0:
            action_mat, profit_mat = dynamic_programming(k, demand)
            r, p, t_block, flown, demand_new = schedule(k, action_mat, profit_mat, demand)
            routes[k] = (r, p, t_block, flown, demand_new)
            profits[k] = p
            ut_time[k] = t_block

    # Zet winst op -1e9 voor vliegtuigen met te lage block time
    for k in range(n_ac):
        if ut_time[k] < 6*60:
            profits[k] = -1e9

    print("Iteration: {}" .format(iteration), profits, available_ac)

    if np.all(profits < 0):
        print("Geen rendabele vluchten meer, stoppen.")
        break

    k_best = np.argmax(profits)
    r, p, t_block, flown, d_new = routes[k_best]
    demand = d_new 
    available_ac[k_best] -= 1
    used_ac_count[k_best] += 1
    total_passengers_transported += sum(flown)
    solution_dict[f"Route {iteration}"] = {"Aircraft type": k_best, "Profit": p,
                                          "Utilization": t_block,
                                          "Route": r}

    print(f"[MAIN LOOP] Iteration {iteration}: Added plane type {k_best} with profit {p:.1f}, block time {t_block:.1f} min")
    print(f"Remaining demand sum: {np.sum(demand):.1f}")
    print(f"Total passengers transported so far: {total_passengers_transported:.1f}")

    iteration +=1

# Print na planning hoeveel vliegtuigen zijn ingezet per type
print("\n===== Gebruik van vliegtuigen per type =====")
for idx, ac_name in enumerate(df_aircraft.index):
    print(f"{ac_name}: {used_ac_count[idx]} gebruikt van {Fleet[idx]} beschikbaar")