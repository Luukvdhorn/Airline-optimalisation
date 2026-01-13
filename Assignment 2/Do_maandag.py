import numpy as np
import pandas as pd
import openpyxl
import math
import matplotlib.pyplot as plt


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

step_minutes = 6
total_steps = 24 * 10  # 6 min steps in 24h

def timestep_to_hour(timestep):
    total_minutes = timestep * step_minutes
    hour = total_minutes // 60
    return hour

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
    bt = 15 + (d[origin,dest] / v[ac_type]) * 60 + TAT[ac_type] + 15
    return bt

def operating_cost(origin, dest, ac_type):
    if origin == dest:
        return 0
    C_fixed = C_fix[ac_type]
    C_time = CT[ac_type] * (d[origin,dest] / v[ac_type])
    C_fuel = (CF[ac_type] * 1.42 / 1.5) * d[origin,dest]
    tot_cost = C_fixed + C_time + C_fuel
    return tot_cost

def revenue_func(origin, dest, flow):
    if origin == dest:
        return 0
    y = 5.9 * d[origin,dest]**(-0.76) + 0.043
    rev = y * d[origin,dest] * flow
    return rev


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
        taken_int = math.floor(taken)  # Afromen naar beneden, zodat alleen hele passagiers worden meegenomen
        if taken_int == 0:
            continue  # als afgerond 0, overslaan
        dem_hour[origin, dest, h] -= taken_int
        flow += taken_int
        remaining_capacity -= taken_int
        if remaining_capacity <= 0:
            break
    return flow, dem_hour

def dynamic_programming(ac_type, dem_hour):
    profit_matrix = np.full((n, total_steps), -1e9)
    action_matrix = np.full((n, total_steps), -1)

    # Laatste tijdstip alleen winst op de hub
    profit_matrix[:, -1] = -1e7
    profit_matrix[hub_index, -1] = 0

    for t in range(total_steps-2, -1, -1):
        current_hour = timestep_to_hour(t)
        for loc in range(n):
            best_profit = -1e6
            best_action = -1

            if t >= total_steps - 1:
                if loc == hub_index:
                    candidate_dests = [hub_index]
                else:
                    candidate_dests = [hub_index]
            else:
                candidate_dests = action_possible(loc, ac_type)

            for dest in candidate_dests:
                if dest == loc:
                    profit = profit_matrix[loc, t+1]
                    flow = 0
                else:
                    blockt = block_time(loc, dest, ac_type)
                    arrival_time = t + math.ceil(blockt/step_minutes)
                    if arrival_time >= total_steps:
                        #print(f"[DP] Vlucht van {loc} naar {dest} arriveert buiten planningstijd (arr: {arrival_time}), winst wordt -1e9 gezet")
                        future_profit = -1e9
                    else:
                        future_profit = profit_matrix[dest, arrival_time]

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

            profit_matrix[loc, t] = best_profit
            action_matrix[loc, t] = best_action

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
        
        revenue_leg = revenue_func(current_pos, next_pos, flow)
        cost_leg = operating_cost(current_pos, next_pos, ac_type)
        profit_leg = revenue_leg - cost_leg

        print(f"[SCHEDULE] AC {ac_type}: Vlucht {airports[current_pos]}->{airports[next_pos]} flow={flow:.1f} "
            f"revenue={revenue_leg:.2f} cost={cost_leg:.2f} profit_leg={profit_leg:.2f}")
        
        flows.append(flow)
        route.append((t_next, next_pos))
        current_pos = next_pos
        t = t_next

    min_block = 6*60
    if total_block_time < min_block:
        print(f"[SCHEDULE] Aircraft type {ac_type} totale blocktime {total_block_time:.1f} min < minimale blocktijd {min_block} min, winst wordt -1e9 gezet")
        profit = -1e9
    else:
        profit = profit_matrix[hub_index, 0] - cl[ac_type]

    return route, profit, total_block_time, flows, dem_hour

# Main loop
demand = dem_hour.copy()
available_ac = Fleet.copy()
total_passengers_transported = 0
solution_dict = {}
iteration = 0

used_ac_count = np.zeros(len(Fleet), dtype=int)
total_profit = 0  # Voor cumulatieve winst

while any(available_ac > 0):
    profits = np.full(n_ac, -1e8)
    ut_time = np.zeros(n_ac)
    routes = {}

    for k in range(n_ac):
        if available_ac[k] > 0:
            action_mat, profit_mat = dynamic_programming(k, demand)
            r, p, t_block, flown, d_new = schedule(k, action_mat, profit_mat, demand)
            routes[k] = (r, p, t_block, flown, d_new)
            profits[k] = p
            ut_time[k] = t_block

    # for k in range(n_ac):
    #     if ut_time[k] < 6*60:
    #         print(f"[MAIN LOOP] Aircraft type {k} blocktime {ut_time[k]:.1f} min < 360 min, winst wordt -1e9 gezet")
    #         profits[k] = -1e5

    print(f"Iteration: {iteration} - Profits: {profits}, Available AC: {available_ac}")

    if np.all(profits < 0):
        print("Geen rendabele vluchten meer, stoppen.")
        break

    k_best = np.argmax(profits)
    r, p, t_block, flown, d_new = routes[k_best]
    demand = d_new 
    available_ac[k_best] -= 1
    used_ac_count[k_best] += 1
    total_profit += p
    total_passengers_transported += sum(flown)
    solution_dict[f"Route {iteration}"] = {"Aircraft type": k_best, "Profit": p,
                                        "Utilization": t_block,
                                        "Route": r,
                                        "Flows": flown}

    print(f"[MAIN LOOP] Iteration {iteration}: Added plane type {k_best} with profit {p:.1f}, block time {t_block:.1f} min")
    print(f"Remaining demand sum: {np.sum(demand):.1f}")
    print(f"Total passengers transported so far: {total_passengers_transported:.1f}")

    iteration += 1

print(f"\nTotale winst over alle vluchten: {total_profit:.2f} euro")
print("\n===== Gebruik van vliegtuigen per type =====")
for idx, ac_name in enumerate(df_aircraft.index):
    print(f"{ac_name}: {used_ac_count[idx]} gebruikt van {Fleet[idx]} beschikbaar")

origin_idx = airports.index("EHAM")
dest_idx = airports.index("EGLL")

print("\nResterende vraag per uur van EHAM naar EGLL (na planning):")
for t in range(T):
    remaining = demand[origin_idx, dest_idx, t]
    print(f"Uur {t:02d}: {remaining:.1f} passagiers")

print("\nResterende vraag per uur van EGLL naar EHAM (na planning):")
for t in range(T):
    remaining2 = demand[dest_idx, origin_idx, t]
    print(f"Uur {t:02d}: {remaining2:.1f} passagiers")

# De rest van je analyses en visualisaties blijf je gebruiken zoals in je oorspronkelijke code.

# import pandas as pd
# import numpy as np

# # Aannemende dat `D` is originele vraag (2D matrix n x n),
# # en `demand` is huidige vraag (3D matrix n x n x tijdstappen)

# # Som van resterende vraag per route over alle uren:
# resterend_per_route = np.sum(demand, axis=2)

# # Totaal vervoerd per route
# vervoerd_per_route = D*2.5 - resterend_per_route


# # Maak het overzichtelijk als DataFrame met luchthavencodes
# df_orig = pd.DataFrame(D, index=airports, columns=airports)
# df_restant = pd.DataFrame(resterend_per_route, index=airports, columns=airports)
# df_vervoerd = pd.DataFrame(vervoerd_per_route, index=airports, columns=airports)

# print("\nOriginele vraag per route (passagiers/week):")
# print(df_orig.round(1))

# print("\nResterende vraag per route na planning:")
# print(df_restant.round(1))


# tot_vervoerd = df_vervoerd.to_numpy().sum() 
# print(f'Totaal vervoerd passagiers: {tot_vervoerd:.1f}')

def timestep_to_label(ts, timestep_duration=6):
    minutes = ts * timestep_duration
    h = minutes // 60
    m = minutes % 60
    return f"{int(h):02d}:{int(m):02d}"

def print_all_routes(solution_dict, airports, timestep_to_label):
    print("\n=== Overzicht van alle geplande vluchten per vliegtuig ===")
    for route_name, sched in solution_dict.items():
        route = sched["Route"]           # lijst van (tijd, luchthaven_idx)
        ac_type = sched["Aircraft type"]
        flown = sched.get("Flows", None) # je moet flows opslaan om deze te kunnen tonen

        print(f"\n{route_name} - Aircraft type {ac_type}:")
        for i in range(len(route)-1):
            dep_time, origin = route[i]
            arr_time, dest = route[i+1]
            passengers = 0
            if flown is not None and i < len(flown):
                passengers = flown[i]
            print(f"  Vertrek {timestep_to_label(dep_time)} van {airports[origin]} naar {airports[dest]} met {passengers:.1f} passagiers")

print_all_routes(solution_dict, airports, timestep_to_label)

# Indexen ophalen van luchthavens EGLL en LFPG
origin_idx = airports.index("EHAM")
dest_idx = airports.index("EGLL")

flight_count = 0
total_passengers = 0

for sched in solution_dict.values():
    route = sched["Route"]
    flows = sched.get("Flows", None)  # flows moet je in solution_dict opslaan
    for i in range(len(route)-1):
        dep_airport = route[i][1]
        arr_airport = route[i+1][1]
        if dep_airport == origin_idx and arr_airport == dest_idx:
            flight_count += 1
            if flows is not None and i < len(flows):
                total_passengers += flows[i]

print(f"Aantal vluchten EGLL -> LFPG: {flight_count}")
print(f"Totaal vervoerde passagiers EHAM -> EGLL: {total_passengers:.1f}")

ASK = 0.0
RPK = 0.0

TOTAL_REVENUE = 0.0
TOTAL_COST = 0.0
TOTAL_LEASE = 0.0

for sched in solution_dict.values():
    route = sched["Route"]
    flows = sched["Flows"]
    ac_type = sched["Aircraft type"]
    seats = s[ac_type]

    TOTAL_LEASE += cl[ac_type]   

    for i, passengers in enumerate(flows):
        if passengers <= 0:
            continue

        _, origin = route[i]
        _, dest   = route[i+1]

        distance_leg = d[origin, dest]

        ASK += seats * distance_leg
        RPK += passengers * distance_leg

        y = 5.9 * distance_leg**(-0.76) + 0.043
        revenue_leg = y * distance_leg * passengers
        TOTAL_REVENUE += revenue_leg

        cost_leg = operating_cost(origin, dest, ac_type)
        TOTAL_COST += cost_leg

TOTAL_COST += TOTAL_LEASE
CASK = TOTAL_COST / ASK 
RASK = TOTAL_REVENUE / ASK 
YIELD = TOTAL_REVENUE / RPK
ANLF = RPK / ASK 
BELF = CASK / YIELD 

print(f"ASK   : {ASK:,.0f} seat-km")
print(f"RPK   : {RPK:,.0f} pax-km")
print(f"CASK  : {CASK:.4f} €/ASK")
print(f"RASK  : {RASK:.4f} €/ASK")
print(f"YIELD : {YIELD:.4f} €/RPK")
print(f"ANLF  : {ANLF:.3f}")
print(f"BELF  : {BELF:.3f}")
print(f"Totaal winst (Profit uit solution_dict): {sum(sched['Profit'] for sched in solution_dict.values()):.2f} €")
print(f"Total revenue: {TOTAL_REVENUE:.2f}")
print(f"Total costs: {TOTAL_COST:.2f}")
print(f"Totale profit {TOTAL_REVENUE-TOTAL_COST:.2f}")

#MAKING A GRAPH VIA CHATGPT
def add_ground_arcs(route):
    full_route = []
    for i in range(len(route) - 1):
        t0, a0 = route[i]
        t1, a1 = route[i + 1]

        full_route.append((t0, a0))
        for t in range(t0 + 1, t1):
            full_route.append((t, a0))
    full_route.append(route[-1])
    return full_route

def timestep_to_label(ts, timestep_duration=6):
    minutes = ts * timestep_duration
    h = minutes // 60
    m = minutes % 60
    return f"{int(h):02d}:{int(m):02d}"

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_yticks(range(len(airports)))
ax.set_yticklabels(airports)
ax.set_ylim(-1, len(airports))

timestep_duration = 6
total_timesteps = 24*60 // timestep_duration

minor_ticks = range(0, total_timesteps + 1, 2*60 // timestep_duration)
ax.set_xlim(0, total_timesteps)
ax.set_xticks([])
ax.set_xticks(minor_ticks, minor=True)
ax.set_xticklabels([timestep_to_label(t) for t in minor_ticks], minor=True)
ax.tick_params(axis='x', which='minor', labelsize=8, length=5)

colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

for idx, (route_name, sched) in enumerate(solution_dict.items()):
    route = sched["Route"]
    ac_type = sched["Aircraft type"]
    expanded_route = add_ground_arcs(route)

    times = [step[0] for step in expanded_route]
    airport_ids = [step[1] for step in expanded_route]
    ac_count_idx = idx 
    color = colors[ac_count_idx % len(colors)]
    ax.plot(times, airport_ids, color=color, label=f"{route_name} - AC type {ac_type}")

ax.set_xlabel("Time")
ax.set_ylabel("Airports")
ax.grid()
ax.legend()
plt.title("Aircraft Timetable")
plt.tight_layout()
plt.show()