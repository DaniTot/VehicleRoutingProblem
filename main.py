from vrp import VRP

vrp = VRP()

random = True
formulation = ["capacitated", "timed"][0]
subtour_type = ['MTZ', 'DFJ'][1]

n = 5  # number of customers
k = 1  # number of vehicles

Q = 32

plot_all_nodes = True
if formulation is "capacitated":
    if random:
        vrp.subtour_type = subtour_type
        vrp.setup_random_data(number_of_customers=n,
                              number_of_vehicles=k,
                              vehicle_capacity=Q,
                              x_range=20, y_range=20,
                              demand_lower=1, demand_higher=10,
                              seed=420)
    else:
        vrp.setup_preset_data(file_name="validation_data_A/A-n32-k5.vrp",
                              number_of_vehicles=5)
    vrp.CVRP_setup()
elif formulation is "timed":
    vrp.subtour_type = 'TW'
    vrp.time_window = 10
    vrp.opening_time = 0
    vrp.closing_time = 30
    vrp.processing_time = 1
    if random:
        vrp.setup_random_data(number_of_customers=5,
                              number_of_vehicles=2,
                              vehicle_capacity=10,
                              x_range=10, y_range=10,
                              demand_lower=1, demand_higher=5,
                              seed=420)
    else:
        vrp.setup_preset_data(file_name="validation_data_A/A-n32-k5.vrp",
                              number_of_vehicles=5)
    vrp.CVRPTW_setup()

if plot_all_nodes == True:
    vrp.visualize(plot_sol='n')

vrp.optimize()

vrp.visualize()
