from vrp import VRP

vrp = VRP(4, 1)

if True:
    vrp.setup_random_data()
else:
    vrp.setup_preset_data()

vrp.optimize()

vrp.visualize()
