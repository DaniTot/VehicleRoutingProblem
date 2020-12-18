from vrp import VRP

vrp = VRP()

random = True
formulation = ["capacitated", "timed"][1]
subtour_type = ['DFJ','MTZ'][0]

n = 15  # number of customers
k = 5  # number of vehicles

Q = 50

goal_gap = 0.0

plot_all_nodes = False

vrp.gap_goal = goal_gap
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
    vrp.time_window = 20
    vrp.opening_time = 0
    vrp.closing_time = 40
    vrp.processing_time = 1
    if random:
        vrp.setup_random_data(number_of_customers=n,
                              number_of_vehicles=k,
                              vehicle_capacity=Q,
                              x_range=20, y_range=20,
                              demand_lower=1, demand_higher=10,
                              seed=420)
    else:
        vrp.setup_preset_data(file_name="validation_data_A/A-n32-k5.vrp",
                              number_of_vehicles=5, subtour_type='TW')
    vrp.CVRPTW_setup()

if plot_all_nodes == True:
    vrp.visualize(plot_sol='n')

# vrp.model.Params.Threads = 2
vrp.optimize()

vrp.visualize()


###  Sensitivity Analysis ###
#
# from matplotlib import pyplot as plt
# from mpl_toolkits.axes_grid1 import host_subplot
# import mpl_toolkits.axisartist as AA
#
# fig, ax1 = plt.subplots()
#
# par1 = ax1.twinx()
#
# color_r = 'tab:red'
# color_b = 'tab:blue'
#
# ax1.set_xlabel("Time window")
# ax1.set_ylabel("Objective value", color=color_r)
# par1.set_ylabel("Computation time", color=color_b)
#
# X = []
# Y1 = []
# Y2 = []
# for i in range(1, 8):
#     random = True
#     formulation = ["capacitated", "timed"][1]
#     subtour_type = ['DFJ','MTZ'][1]
#
#     n = 10  # number of customers
#     k = 5  # number of vehicles
#
#     Q = 10
#
#     goal_gap = 0.0
#
#     plot_all_nodes = False
#     vrp = VRP()
#     vrp.gap_goal = goal_gap
#     if formulation is "capacitated":
#         if random:
#             vrp.subtour_type = subtour_type
#             vrp.setup_random_data(number_of_customers=n,
#                                   number_of_vehicles=k,
#                                   vehicle_capacity=Q,
#                                   x_range=20, y_range=20,
#                                   demand_lower=1, demand_higher=10,
#                                   seed=420)
#         else:
#             vrp.setup_preset_data(file_name="validation_data_A/A-n32-k5.vrp",
#                                   number_of_vehicles=5)
#         vrp.CVRP_setup()
#     elif formulation is "timed":
#         vrp.subtour_type = 'TW'
#         vrp.time_window = 5*i
#         vrp.opening_time = 0
#         vrp.closing_time = 40
#         vrp.processing_time = 1
#         if random:
#             vrp.setup_random_data(number_of_customers=n,
#                                   number_of_vehicles=k,
#                                   vehicle_capacity=Q,
#                                   x_range=20, y_range=20,
#                                   demand_lower=1, demand_higher=10,
#                                   seed=420)
#         else:
#             vrp.setup_preset_data(file_name="validation_data_A/A-n32-k5.vrp",
#                                   number_of_vehicles=5, subtour_type='TW')
#         vrp.CVRPTW_setup()
#
#     if plot_all_nodes == True:
#         vrp.visualize(plot_sol='n')
#
#     # vrp.model.Params.Threads = 2
#     vrp.optimize()
#     # vrp.visualize()
#     print("Runtime: ", vrp.model.Runtime)
#     print("MIPGap: ", vrp.model.MIPGap)
#     print("Objective Value: ", vrp.model.objVal)
#     X.append(vrp.time_window)
#     Y1.append(vrp.model.objVal)
#     Y2.append(vrp.model.Runtime)
#
# p1, = ax1.plot(X, Y1, label="Objective value", color=color_r)
# p2, = par1.plot(X, Y2, label="Runtime", color=color_b)
# # host.legend()
#
# # host.set_xlim(0, 2)
# ax1.set_ylim(min(Y1)*0.9, max(Y1)*1.1)
# par1.set_ylim(min(Y2)*0.9, max(Y2)*1.1)
# ax1.tick_params(axis='y', labelcolor=color_r)
# par1.tick_params(axis='y', labelcolor=color_b)
#
# print(X)
# print(Y1)
# print(Y2)
#
# # plt.draw()
# plt.savefig("sensitivity_TW.png")
# plt.show()

