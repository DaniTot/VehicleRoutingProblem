
#from ..vrp import VRP
import sys
sys.path.append("..")
from vrp import VRP

n = 15 # number of customers
k = 2  # number of vehicles


############## MTZ ############### 
vrp_mtz = VRP()

vrp_mtz.setup_random_data(number_of_customers=n, number_of_vehicles=k, demand_lower=1, demand_higher=10)
vrp_mtz.gap_goal = 0.1

vrp_mtz.subtour_type = 'MTZ'
vrp_mtz.setup()
vrp_mtz.optimize()
vrp_mtz.visualize()

if input("continue?").lower() not in ['y', 'yes']:
    sys.exit()

############## DFJ ############### 
vrp_dfj = VRP()

vrp_dfj.setup_random_data(number_of_customers=n, number_of_vehicles=k, demand_lower=1, demand_higher=10)
vrp_dfj.gap_goal = 0.1
vrp_dfj.subtour_type = 'DFJ'
vrp_dfj.setup()
vrp_dfj.optimize()
# vrp_dfj.visualize()