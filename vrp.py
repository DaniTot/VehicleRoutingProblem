import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from gurobipy import Model, GRB, quicksum


class VRP:
    def __init__(self):

        self.Q = None
        self.n = None
        self.x_range = None
        self.y_range = None
        self.number_of_customers = None
        self.demand_range = None

        self.K = None
        self.nodes = None
        self.N = None  # Customer Nodes
        self.V = None  # Depot and customer nodes
        self.A = None  # Arcs between depot and customer nodes
        self.c = None  # Cost associated with each arc
        self.q = None  # The weight that has to be delivered to each customer

        self.model = None
        self.x = None
        self.u = None

        self.subtour_type = None

    def setup_random_data(self,
                          number_of_customers,
                          number_of_vehicles,
                          vehicle_capacity=5,
                          x_range=10, y_range=10,
                          demand_lower=1, demand_higher=2,
                          subtour_type="DFJ",
                          seed=420):
        if seed is not None:
            np.random.seed(seed)

        self.subtour_type = subtour_type

        self.Q = vehicle_capacity
        self.n = number_of_customers + 1
        self.x_range = x_range
        self.y_range = y_range
        self.number_of_customers = number_of_customers
        self.demand_range = [demand_lower, demand_higher]
        self.K = list(range(number_of_vehicles))

        self.create_dataset()

        self.create_arcs()

        self.setup()

    def setup_preset_data(self, file_name, number_of_vehicles):

        nb_customers, truck_capacity, n, v, demands, nodes = self.read_input_cvrp(file_name)

        self.read_input_cvrp(file_name)
        self.Q = truck_capacity
        self.n = nb_customers + 1
        self.number_of_customers = nb_customers
        self.K = list(range(number_of_vehicles))

        self.nodes = nodes
        self.N = n
        self.V = v
        self.q = demands

        self.create_arcs()

        self.setup()

    def visualize(self):
        plt.title('Capacitated Vehicle Routing Problem')
        color_list = plt.cm.Set3(np.linspace(0, 12, 12))

        plt.plot(np.array(self.nodes.iloc[self.V[0]])[0], np.array(self.nodes.iloc[self.V[0]])[1], marker='*',
                 markersize=20, label='Depot', c='r', )
        plt.scatter(np.array(self.nodes.iloc[self.N])[:, 0],
                    np.array(self.nodes.iloc[self.N])[:, 1], c='b', label='Customer')
        plt.annotate([self.V[0], self.V[-1]], (self.nodes.iloc[self.V[0]][0], self.nodes.iloc[self.V[0]][1]),
                     xytext=(20, -20), textcoords='offset points', ha='right', va='bottom')
        for idx in self.V[1:-1]:
            plt.annotate(idx, (self.nodes.iloc[idx][0], self.nodes.iloc[idx][1]),
                         xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        travel_paths = {}
        for k in self.K:
            travel_paths[k] = []
            for (i, j) in self.A:
                # Using the round function, in case truncation leads to something like 0.9999
                if round(self.x[i, j, k].x) == 1:
                    travel_paths[k].append((i, j))

                    plt.plot([self.nodes.iloc[i][0], self.nodes.iloc[j][0]],
                             [self.nodes.iloc[i][1], self.nodes.iloc[j][1]], c=color_list[k])
                    if (i, j) != (self.V[0], self.V[-1]) and (i, j) != (self.V[-1], self.V[0]):
                        plt.annotate(round(self.c[(i, j)], 2),
                                     ((self.nodes.iloc[i][0] + self.nodes.iloc[j][0]) / 2,
                                      (self.nodes.iloc[i][1] + self.nodes.iloc[j][1]) / 2),
                                     xytext=(20, 2), textcoords='offset points', ha='right', va='bottom')
        print(self.V)
        print(travel_paths)
        print(self.q)
        plt.legend(loc='lower right')
        plt.show()

    def create_dataset(self):
        # If data sheet does not exist, generate one
        self.generate_node_table()
        self.create_customers()
        print("Node data generated")

    def optimize(self):
        self.model.optimize()
        self.model.write("out.sol")
        return

    def setup(self):

        self.model = Model('CVRP')

        # Variables
        self.x = {}
        for i, j in self.A:
            for k in self.K:
                self.x[i, j, k] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"x[{i},{j},{k}]")

        self.u = {}
        for j in self.N:
            for k in self.K:
                self.u[j, k] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"u[{j},{k}]")

        # Make sure there are no duplicates of i,j,k combination in x
        assert len(self.x) == len(set(self.x.keys()))

        self.model.update()

        # Objective function
        # explanation for self.V[:-1] = [0, 1, ..., n]
        # explanation for self.V[1:] = [1, ..., n, n+1]
        self.model.setObjective(quicksum(self.x[i, j, k] * self.c[i, j] for i in self.V[:-1] for j in self.V[1:]
                                         for k in self.K if i != j),
                                sense=GRB.MINIMIZE)
        self.model.update()

        # Constraints
        # Each vehicle must leave the depot
        for k in self.K:
            self.model.addConstr(quicksum(self.x[self.V[0], j, k] for j in self.V[1:]) == 1, name=f"Start_{k}")
        # Each vehicle must return to depot
        for k in self.K:
            self.model.addConstr(quicksum(self.x[j, self.V[-1], k] for j in self.V[:-1]) == 1, name=f"Finish_{k}")
        # Each customer must be visited by one vehicle
        for i in self.N:
            self.model.addConstr(quicksum(self.x[j, i, k] for k in self.K for j in self.V[:-1] if j != i) == 1,
                                 name=f"Visit_{i}")
        # If a vehicle visits a customer, it must also leave that customer

        for i in self.N:
            for k in self.K:
                self.model.addConstr(quicksum(self.x[j, i, k] for j in self.V[:-1] if j != i) ==  # What arrives to i
                                     quicksum(self.x[i, j, k] for j in self.V[1:] if j != i),  # Must leave from i
                                     name=f"Leave_{i}_{k}")

        # Capacity constraint
        for k in self.K:
            self.model.addConstr(
                quicksum(self.q[j] * self.x[i, j, k] for i in self.V[:-1] for j in self.V[1:-1] if i != j) <= self.Q,
                name=f"Capacity_{k}")

        # # TODO: Sub-tour elimination via Miller-Tucker-Zemlin formulation
        # for k in self.K:
        #     for i, j in self.A:
        #         if i not in (self.V[0], self.V[-1]) and j not in (self.V[0], self.V[-1]):
        #             self.model.addConstr(self.u[j, k] - self.u[i, k] >= self.q[j] - self.Q * (1 - self.x[i, j, k]),
        #                                  name=f"Subtour_{i}_{j}_{k}")

        # Subtour elimination constraint (miller-tucker-zemlin)
        #
        # $$u_{j} - u_{i} \geq q_{j} - Q(1-x_{ijk}), i,j = \{1,....,n\}, i \neq j$$

        # Miller-Tucker-Zemlin formulation for subtour elimination

        if self.subtour_type == 'MTZ':
            for k in self.K:
                for i, j in self.A:
                    if i >= 1 and j >= 1:
                        if i != self.number_of_customers + 1 and j != self.number_of_customers + 1:
                            self.model.addConstr(self.u[j, k] - self.u[i, k] >= self.q[j] - self.Q * (1 - self.x[i, j, k]))

            # Capacity constraint
            for i in self.N:
                for k in self.K:
                    self.model.addConstr(self.u[i, k] >= self.q[i])
                    self.model.addConstr(self.u[i, k] <= self.Q)

        # Subtour elimination constraint (Dantzig-Fulkerson Johnson)
        #
        # $$\sum_{i\in S}\sum_{j \in S,j \neq i} x_{ij} \leq |S| - 1$$
        elif self.subtour_type == 'DFJ':
            for k in self.K:
                self.model.addConstr(quicksum(quicksum(self.q[j] * self.x[i, j, k] for j in self.N if j != i) for i in self.V if i < self.number_of_customers + 1) <= self.Q)

        self.model.update()
        self.model.write("model.lp")

        # TODO: Set optimization limit

        return

    def subtourelim(self, where):
        if where == GRB.callback.MIPSOL:

            active_arcs = []
            # TODO: look into self._vars, self.cbGetSolution
            for i, j in self.A:
                for k in self.K:
                    solutions = self.model.cbGetSolution(self.model._vars)
                    if solutions[i, j, k] > 0.5:
                        active_arcs.append([i, j, k])

            active_arcs = np.vstack(active_arcs)

            tours = self.subtour(active_arcs)

            for k in tours.keys():
                if len(tours[k]) > 1:
                    for tour in tours[k]:
                        S = np.unique(tour)
                        expr = quicksum(self.model._vars[i, j, k] for i in S for j in S if j != i)
                        self.model.cbLazy(expr <= len(S) - 1)

    def subtour(self, active_arcs):
        tours = {}

        for k in self.K:
            vehicle_tours = []
            vehicle_arcs = active_arcs[np.where(active_arcs[:, 2] == k)][:, 0:2]
            start_node, finish_node = vehicle_arcs[0]
            if finish_node == self.V[-1]:
                finish_node = self.V[0]

            tour = [start_node, finish_node]
            vehicle_arcs = np.delete(vehicle_arcs, [0], axis=0)

            subtour_done = 0

            while True:
                while True:
                    next_node = np.where(vehicle_arcs[:, 0] == finish_node)

                    if next_node[0].size == 0:
                        vehicle_tours.append(tour)
                        break
                    else:
                        start_node, finish_node = vehicle_arcs[next_node][0]
                        # if finish_node == V[-1]:
                        #     finish_node = V[0]
                        vehicle_arcs = np.delete(vehicle_arcs, next_node[0], axis=0)

                        tour.append(finish_node)

                if vehicle_arcs.size != 0:
                    start_node, finish_node = vehicle_arcs[0]
                    vehicle_arcs = np.delete(vehicle_arcs, [0], axis=0)

                    # if finish_node == V[-1]:
                    #     finish_node = V[0]

                    tour = [start_node, finish_node]
                else:
                    tours[k] = vehicle_tours
                    break

        return tours

    @staticmethod
    def calc_distance(p1, p2):
        dist = (((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
        return dist

    def generate_node_table(self, file_name="nodes.csv"):
        nodes_coord_x = np.random.rand(self.n) * self.x_range
        nodes_coord_y = np.random.rand(self.n) * self.y_range
        nodes_table = pd.DataFrame(np.transpose(np.array([nodes_coord_x, nodes_coord_y])),
                                   columns=['x_coord', 'y_coord'])
        nodes_table.to_csv(file_name)
        self.nodes = nodes_table
        return nodes_table

    def read_node_table(self, file_name="nodes.csv"):
        nodes_table = pd.read_csv(file_name)
        self.nodes = nodes_table
        return nodes_table

    def create_customers(self):
        random_selection = list(self.nodes.sample(n=self.number_of_customers + 1).index)
        # Select the depot node
        # Depot node is randomly selected
        start_depot_idx = random_selection.pop(np.random.randint(0, len(random_selection) - 1, size=1)[0])
        print("start_depot_idx", start_depot_idx)

        # Add a new end depot node to the end of nodes, which has the same coordinates as the start depot
        end_depot_coord = list(self.nodes.iloc[start_depot_idx])
        self.nodes = self.nodes.append(pd.DataFrame([end_depot_coord], columns=['x_coord', 'y_coord']),
                                       ignore_index=True)
        print(self.nodes)
        end_depot_idx = self.nodes.index[-1]
        print("end_depot_idx", end_depot_idx)

        self.N = random_selection  # The rest will be the customer nodes
        temp_list = self.N[:]
        temp_list.insert(0, start_depot_idx)
        temp_list.append(end_depot_idx)
        # self.V[0] is the start depot, and self.V[-1] is the end depot
        self.V = temp_list[:]

        # Assign the customer demand to each customer
        self.q = {}
        for i in self.N:
            self.q[i] = np.random.randint(self.demand_range[0], self.demand_range[1], size=1)[0]

        assert sum(self.q.values()) <= self.Q * len(self.K), f"The total customer demand {sum(self.q.values())} " \
                                                             f"exceeds the total vehicle capacity: " \
                                                             f"{self.Q * len(self.K)} = {self.Q} * {len(self.K)}."
        return

    def create_arcs(self):
        # Create arcs between customer nodes
        self.A = [c for c in itertools.product(self.N, self.N)]

        # Add depot nodes
        # Make sure there are only leaving arcs from the depot start node...
        for j in self.V[1:]:
            self.A.append((self.V[0], j))
        # ...and there are only leading nodes to the depot end node.
        for i in self.V[1:-1]:
            self.A.append((i, self.V[-1]))

        # Remove the elements where i=j
        for i, tup in enumerate(self.A):
            if tup[0] == tup[1]:
                self.A.pop(i)

        # Make sure there are no duplicate arcs
        duplicates = [(k, v) for k, v in Counter(self.A).items() if v > 1]
        assert len(duplicates) == 0, f"Duplicate arc found: {duplicates}"

        # The cost to travel an arc equals its length
        self.c = {}
        for i, j in self.A:
            x_i = self.nodes.get('x_coord')[i]
            y_i = self.nodes.get('y_coord')[i]
            x_j = self.nodes.get('x_coord')[j]
            y_j = self.nodes.get('y_coord')[j]
            self.c[(i, j)] = self.calc_distance((x_i, y_i), (x_j, y_j))
        return

    ##################################################################################################
    # The reading functions are taken and modified from the original
    # https://www.localsolver.com/docs/last/exampletour/vrp.html#
    ##################################################################################################
    @staticmethod
    def read_elem(filename):
        with open(filename) as f:
            return [str(elem) for elem in f.read().split()]

    # The input files follow the "Augerat" format.
    def read_input_cvrp(self, filename):
        import sys
        # file_it = iter(self.read_elem(sys.argv[1]))
        file_it = iter(self.read_elem(filename))

        nb_nodes = 0
        nb_customers = 0
        truck_capacity = 0
        while True:
            token = next(file_it)
            if token == "DIMENSION":
                next(file_it)  # Removes the ":"
                nb_nodes = int(next(file_it))
                nb_customers = nb_nodes - 1
            elif token == "CAPACITY":
                next(file_it)  # Removes the ":"
                truck_capacity = int(next(file_it))
            elif token == "EDGE_WEIGHT_TYPE":
                next(file_it)  # Removes the ":"
                token = next(file_it)
                if token != "EUC_2D":
                    print("Edge Weight Type " + token + " is not supported (only EUD_2D)")
                    sys.exit(1)
            elif token == "NODE_COORD_SECTION":
                break

        assert nb_customers != 0
        assert nb_nodes != 0
        assert truck_capacity != 0

        customers_x = [None] * nb_customers
        customers_y = [None] * nb_customers
        depot_x = 0
        depot_y = 0
        for n in range(nb_nodes):
            node_id = int(next(file_it))
            if node_id != n + 1:
                print("Unexpected index")
                sys.exit(1)
            if node_id == 1:
                depot_x = int(next(file_it))
                depot_y = int(next(file_it))
            else:
                # -2 because orginal customer indices are in 2..nbNodes
                customers_x[node_id - 2] = int(next(file_it))
                customers_y[node_id - 2] = int(next(file_it))

        nodes_coord_x = customers_x[:]
        nodes_coord_x.append(depot_x)
        nodes_coord_x.insert(0, depot_x)
        nodes_coord_y = customers_y[:]
        nodes_coord_y.append(depot_y)
        nodes_coord_y.insert(0, depot_y)
        nodes = pd.DataFrame(np.transpose(np.array([nodes_coord_x, nodes_coord_y])), columns=['x_coord', 'y_coord'])

        # Create customer index list N, and node index list V
        N = list(nodes.index[1:-1])
        assert len(N) == nb_customers
        V = list(nodes.index)
        assert len(V) == nb_nodes + 1

        token = next(file_it)
        if token != "DEMAND_SECTION":
            print("Expected token DEMAND_SECTION")
            sys.exit(1)

        demands = {}
        for n in N:
            demands[n] = None

        for n in range(nb_nodes):
            node_id = int(next(file_it))
            if node_id != n + 1:
                print("Unexpected index")
                sys.exit(1)
            if node_id == 1:
                if int(next(file_it)) != 0:
                    print("Demand for depot should be 0")
                    sys.exit(1)
            else:
                # First element in N is 1, but the first customer in the file is 2
                demands[node_id - 1] = int(next(file_it))

        token = next(file_it)
        if token != "DEPOT_SECTION":
            print("Expected token DEPOT_SECTION")
            sys.exit(1)

        warehouse_id = int(next(file_it))
        if warehouse_id != 1:
            print("Warehouse id is supposed to be 1")
            sys.exit(1)

        end_of_depot_section = int(next(file_it))
        if end_of_depot_section != -1:
            print("Expecting only one warehouse, more than one found")
            sys.exit(1)

        return nb_customers, truck_capacity, N, V, demands, nodes

    # ###########################################################
    # # TODO: use the code below for inspiration, don't steal it!
    # ###########################################################
    # def subtourelim(self, where):
    #     if where == GRB.callback.MIPSOL:
    #         # make a list of edges selected in the solution
    #         for k in self.K:
    #             selected = []
    #             visited = set()
    #             for i in self.N:
    #                 sol = self.model.cbGetSolution([self.x[i, j, k] for j in self.N])
    #                 new_selected = [(i, j) for j in self.N if sol[j] > 0.5]
    #                 selected += new_selected
    #
    #                 if new_selected:
    #                     visited.add(i)
    #
    #             # find the shortest cycle in the selected edge list
    #             print(str(k) + ' selected ' + str(len(selected)) + ' ' + repr(selected))
    #             print(str(k) + ' len visited ' + str(len(visited)) + ' ' + repr(visited))
    #             tour = self.subtour(selected, visited)
    #             print(str(k) + ' len tour ' + str(len(tour)) + ' ' + repr(tour))
    #
    #             if len(tour) < len(visited):
    #                 # add a subtour elimination constraint
    #                 expr = quicksum(self.x[i, j, k] for i, j in itertools.combinations(tour, 2))
    #                 self.model.cbLazy(expr <= len(tour) - 1)
    #
    # def subtour(self, edges, visited):
    #     unvisited = list(visited)
    #     cycle = range(len(visited) + 1)
    #     selected = {}
    #     for x, y in edges:
    #         selected[x] = []
    #     for x, y in edges:
    #         selected[x].append(y)
    #     print(selected)
    #     while unvisited:
    #         thiscycle = []
    #         neighbors = unvisited
    #         while neighbors:
    #             current = neighbors[0]
    #             thiscycle.append(current)
    #             unvisited.remove(current)
    #             print(thiscycle)
    #             neighbors = [j for j in selected[current] if j in unvisited]
    #         if len(cycle) > len(thiscycle):
    #             cycle = thiscycle
    #     return cycle
    #
    # def optimize_subtour(self):
    #     self.model.params.LazyConstraints = 1
    #     self.model.optimize(self.subtourelim)
    #     solution = self.model.getAttr('x', self.x)
    #     return solution
