
#from ..vrp import VRP
import sys
sys.path.append("..")

import numpy as np

from vrp import VRP

n = 5  # number of customers
k = 4  # number of vehicles

Q = 15


############## DFJ ############### 
vrp_dfj = VRP()

vrp_dfj.setup_random_data(number_of_customers=n, number_of_vehicles=k, vehicle_capacity=Q, demand_lower=1, demand_higher=10,seed=420)
vrp_dfj.gap_goal = 0.1
vrp_dfj.subtour_type = 'DFJ'

vrp_dfj.setup()
vrp_dfj.optimize()


print(vrp_dfj.model._A)

active_arcs = []

for i,j in vrp_dfj.A:
    for k in vrp_dfj.K:
        if round(vrp_dfj.x[i,j,k].x) == 1:
            active_arcs.append([i,j,k])
active_arcs = np.vstack(active_arcs)

print(active_arcs)

print("Subtours")
print(vrp_dfj.subtour(vrp_dfj.K, active_arcs))

vrp_dfj.visualize()




# if input("continue?").lower() not in ['y', 'yes']:
#     sys.exit()

# ############## MTZ ############### 
# vrp_mtz = VRP()

# vrp_mtz.setup_random_data(number_of_customers=n, number_of_vehicles=k, demand_lower=1, demand_higher=10)
# vrp_mtz.gap_goal = 0.1

# vrp_mtz.subtour_type = 'MTZ'
# vrp_mtz.setup()
# vrp_mtz.optimize()
# vrp_mtz.visualize()


