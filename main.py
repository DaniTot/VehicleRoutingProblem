from vrp import VRP

vrp = VRP()

random = False

if random:
    vrp.setup_random_data(number_of_customers=4,
                          number_of_vehicles=1,
                          vehicle_capacity=5,
                          x_range=10, y_range=10,
                          demand_lower=1, demand_higher=2,
                          seed=420)
else:
    vrp.setup_preset_data(file_name="validation_data_A/A-n38-k5.vrp",
                          number_of_vehicles=5)

vrp.optimize()

vrp.visualize()
