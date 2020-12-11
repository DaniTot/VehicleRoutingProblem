from vrp import VRP

vrp = VRP()

random = True
formulation = ["capacitated", "timed"][0]


# vrp.gap_goal = 0.01


if formulation is "capacitated":
    if random:
        vrp.subtour_type = ['MTZ', 'DFJ'][1]
        vrp.setup_random_data(number_of_customers=5,
                              number_of_vehicles=2,
                              vehicle_capacity=10,
                              x_range=10, y_range=10,
                              demand_lower=1, demand_higher=5,
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

print("All set")
vrp.optimize()
vrp.visualize()
