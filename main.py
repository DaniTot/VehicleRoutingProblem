from vrp import VRP
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

# vrp = VRP()
#
# random = True
# formulation = ["capacitated", "timed"][0]
# subtour_type = ['DFJ','MTZ'][0]
#
# n = 5  # number of customers
# k = 1  # number of vehicles
#
# Q = 32
#
# goal_gap = 0.1
#
# plot_all_nodes = True
#
# vrp.gap_goal = goal_gap
# if formulation is "capacitated":
#     if random:
#         vrp.subtour_type = subtour_type
#         vrp.setup_random_data(number_of_customers=n,
#                               number_of_vehicles=k,
#                               vehicle_capacity=Q,
#                               x_range=20, y_range=20,
#                               demand_lower=1, demand_higher=10,
#                               seed=420)
#     else:
#         vrp.setup_preset_data(file_name="validation_data_A/A-n32-k5.vrp",
#                               number_of_vehicles=5)
#     vrp.CVRP_setup()
# elif formulation is "timed":
#     vrp.subtour_type = 'TW'
#     vrp.time_window = 20
#     vrp.opening_time = 0
#     vrp.closing_time = 40
#     vrp.processing_time = 1
#     if random:
#         vrp.setup_random_data(number_of_customers=n,
#                               number_of_vehicles=k,
#                               vehicle_capacity=Q,
#                               x_range=20, y_range=20,
#                               demand_lower=1, demand_higher=10,
#                               seed=420)
#     else:
#         vrp.setup_preset_data(file_name="validation_data_A/A-n32-k5.vrp",
#                               number_of_vehicles=5, subtour_type='TW')
#     vrp.CVRPTW_setup()
#
# if plot_all_nodes == True:
#     vrp.visualize(plot_sol='n')
#
# # vrp.model.Params.Threads = 2
# vrp.optimize()
#
# vrp.visualize()

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()

host.set_xlim(0, 2)
host.set_ylim(0, 2)

host.set_xlabel("Vehicle capacity")
host.set_ylabel("Objective value")
par1.set_ylabel("Computation time")

X = []
Y1 = []
Y2 = []
for i in range(5):
    random = True
    formulation = ["capacitated", "timed"][0]
    subtour_type = ['DFJ','MTZ'][0]

    n = 10  # number of customers
    k = 5  # number of vehicles

    Q = 8*i

    goal_gap = 0.1

    plot_all_nodes = True
    vrp = VRP()
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
    print("Runtime: ", vrp.model.Runtime)
    print("MIPGap: ", vrp.model.MIPGap)
    print("Objective Value: ", vrp.model.objVal)
    X.append(Q)
    Y1.append(vrp.model.objVal)
    Y2.append(vrp.model.Runtime)

p1, = host.plot(X, Y1, label="Objective value")
p2, = par1.plot(X, Y2, label="Runtime")
host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())

plt.draw()
plt.show()
plt.savefig("sensitivity1.png")

