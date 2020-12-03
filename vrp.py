import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from gurobipy import Model, GRB, quicksum


class VRP:
    def __init__(self, number_of_customers, number_of_vehicles,
                 vehicle_capacity=5, x_range=10, y_range=10, demand_lower=1, demand_higher=2,
                 seed=420):
        if seed is not None:
            np.random.seed(seed)

        self.Q = vehicle_capacity
        self.n = number_of_customers + 1
        self.x_range = x_range
        self.y_range = y_range
        self.number_of_customers = number_of_customers
        self.demand_range = [demand_lower, demand_higher]

        self.K = list(range(number_of_vehicles))
        self.nodes = None
        self.N = None  # Customer Nodes
        self.V = None  # Depot and customer nodes
        self.A = None  # Arcs between depot and customer nodes
        self.c = None  # Cost associated with each arc
        self.q = None  # The weight that has to be delivered to each customer

        self.model = None
        self.x = None
        self.u = None

    def setup_random_data(self):
        self.create_dataset()
        self.setup()

    def setup_preset_data(self):
        self.load_dataset()
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
        self.create_arcs()
        print("Node data generated")

    def load_dataset(self):
        # TODO
        raise NotImplementedError
        return

    def optimize(self):
        self.model.optimize()
        self.model.write("out.sol")
        return

    def setup(self):
        self.model = Model('CVRP')

        # Variables
        # self.x = self.model.addVars(self.A, vtype=GRB.BINARY)
        # self.u = self.model.addVars(self.N, vtype=GRB.CONTINUOUS)
        self.x = {}
        for i, j in self.A:
            for k in self.K:
                self.x[i, j, k] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"x[{i},{j},{k}]")
        # TODO: Not sure if this below is needed
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
                                     name=f"Leave_{i}{k}")

        # Capacity constraint
        for k in self.K:
            self.model.addConstr(
                quicksum(self.q[j] * self.x[i, j, k] for i in self.V[:-1] for j in self.V[1:-1] if i != j) <= self.Q,
                name=f"Capacity_{k}")

        # TODO: Sub-tour elimination via Miller-Tucker-Zemlin formulation
        for k in self.K:
            for i, j in self.A:
                if i not in (self.V[0], self.V[-1]) and j not in (self.V[0], self.V[-1]):
                    self.model.addConstr(self.u[j, k] - self.u[i, k] >= self.q[j] - self.Q * (1 - self.x[i, j, k]),
                                         name=f"Subtour_{i}{j}{k}")

        self.model.update()
        self.model.write("model.lp")

        return

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

        assert sum(self.q.values()) <= self.Q * len(self.K), f"The total customer demand {sum(self.q.values())} exceeds the total vehicle capacity: {self.Q * len(self.K)} = {self.Q} * {len(self.K)}."
        return

    def create_arcs(self):
        # Create arcs between customer nodes
        self.A = [c for c in itertools.product(self.N, self.N)]

        # Add depot nodes
        # Make sure there are only leaving arcs from the depot start node...
        for j in self.V[1:]:
            self.A.append((self.V[0], j))
        # ...and there are only leading nodes to the depot end node.
        for i in self.V[:-1]:
            self.A.append((i, self.V[-1]))

        # Remove the elements where i=j
        for i, tup in enumerate(self.A):
            if tup[0] == tup[1]:
                self.A.pop(i)

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
        file_it = iter(self.read_elem(sys.argv[1]))

        nb_nodes = 0
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

        # Compute distance matrix
        distance_matrix = compute_distance_matrix(customers_x, customers_y)
        distance_warehouses = compute_distance_warehouses(depot_x, depot_y, customers_x, customers_y)

        token = next(file_it)
        if token != "DEMAND_SECTION":
            print("Expected token DEMAND_SECTION")
            sys.exit(1)

        demands = [None] * nb_customers
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
                # -2 because orginal customer indices are in 2..nbNodes
                demands[node_id - 2] = int(next(file_it))

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

        return (nb_customers, truck_capacity, distance_matrix, distance_warehouses, demands)

    ###########################################################
    # TODO: use the code below for inspiration, don't steal it!
    ###########################################################
    def subtourelim(self, where):
        if where == GRB.callback.MIPSOL:
            # make a list of edges selected in the solution
            for k in self.K:
                selected = []
                visited = set()
                for i in self.N:
                    sol = self.model.cbGetSolution([self.x[i, j, k] for j in self.N])
                    new_selected = [(i, j) for j in self.N if sol[j] > 0.5]
                    selected += new_selected

                    if new_selected:
                        visited.add(i)

                # find the shortest cycle in the selected edge list
                print(str(k) + ' selected ' + str(len(selected)) + ' ' + repr(selected))
                print(str(k) + ' len visited ' + str(len(visited)) + ' ' + repr(visited))
                tour = self.subtour(selected, visited)
                print(str(k) + ' len tour ' + str(len(tour)) + ' ' + repr(tour))

                if len(tour) < len(visited):
                    # add a subtour elimination constraint
                    expr = quicksum(self.x[i, j, k] for i, j in itertools.combinations(tour, 2))
                    self.model.cbLazy(expr <= len(tour) - 1)

                # must contain the start point
                # start_point = self.delivers[k].pos
                # if not start_point in visited:
                #     model.cbLazy(degree[start_point, k] >= 1)

    # Given a list of edges, finds the shortest subtour

    # def subtour(edges):
    #     visited = [False] * n
    #     cycles = []
    #     lengths = []
    #     selected = [[] for i in range(n)]
    #     for x, y in edges:
    #         selected[x].append(y)
    #     while True:
    #         current = visited.index(False)
    #         thiscycle = [current]
    #         while True:
    #             visited[current] = True
    #             neighbors = [x for x in selected[current] if not visited[x]]
    #             if len(neighbors) == 0:
    #                 break
    #             current = neighbors[0]
    #             thiscycle.append(current)
    #         cycles.append(thiscycle)
    #         lengths.append(len(thiscycle))
    #         if sum(lengths) == n:
    #             break
    #     return cycles[lengths.index(min(lengths))]

    def subtour(self, edges, visited):
        unvisited = list(visited)
        cycle = range(len(visited) + 1)
        selected = {}
        for x, y in edges:
            selected[x] = []
        for x, y in edges:
            selected[x].append(y)
        print(selected)
        while unvisited:
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                print(thiscycle)
                neighbors = [j for j in selected[current] if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    def optimize_subtour(self):
        self.model.params.LazyConstraints = 1
        self.model.optimize(self.subtourelim)
        solution = self.model.getAttr('x', self.x)
        return solution

