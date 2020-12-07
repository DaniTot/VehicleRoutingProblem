from vrp import VRP

vrp = VRP()

random = True

if random:
    vrp.setup_random_data(number_of_customers=15,
                          number_of_vehicles=2,
                          vehicle_capacity=10,
                          x_range=10, y_range=10,
                          demand_lower=1, demand_higher=10,
                          seed=420)
else:
    vrp.setup_preset_data(file_name="validation_data_A/A-n32-k5.vrp",
                          number_of_vehicles=5)


vrp.gap_goal = 0.1
vrp.subtour_type = 'MTZ'

vrp.setup()

vrp.optimize()

vrp.visualize()
