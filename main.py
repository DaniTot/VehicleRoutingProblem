from vrp import VRP

vrp = VRP()

random = True

n = 4  # number of customers
k = 1  # number of vehicles

Q = 32

plot_all_nodes = True

if random:
    vrp.setup_random_data(number_of_customers=n,
                          number_of_vehicles=k,
                          vehicle_capacity=Q,
                          x_range=10, y_range=5,
                          demand_lower=1, demand_higher=10,
                          seed=420)
else:
    vrp.setup_preset_data(file_name="validation_data_A/A-n32-k5.vrp",
                          number_of_vehicles=5)


if plot_all_nodes == True:
    vrp.visualize(plot_sol='n')

vrp.gap_goal = 0.1
vrp.subtour_type = 'DFJ'
vrp.setup()
vrp.optimize()
vrp.visualize()