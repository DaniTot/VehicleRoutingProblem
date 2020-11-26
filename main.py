import numpy as np
import pandas as pd
import itertools
from gurobipy import Model, GRB, quicksum


class VRP:
    def __init__(self, number_of_nodes, number_of_vehicles,
                 vehicle_capacity=30, x_range=10, y_range=10, number_of_customers=None, demand_lower=1, demand_higher=3,
                 seed=420):
        if seed is not None:
            np.random.seed(seed)

        self.Q = vehicle_capacity
        self.n = number_of_nodes
        self.x_range = x_range
        self.y_range = y_range
        if number_of_customers is None:
            self.number_of_customers = number_of_nodes - 1
        else:
            self.number_of_customers = number_of_customers
        self.demand_range = [demand_lower, demand_higher]
        
        self.K = list(range(number_of_vehicles+1))
        self.nodes = None
        self.N = None  # Customer Nodes
        self.V = None  # Depot and customer nodes
        self.A = None  # Arcs between depot and customer nodes
        self.c = None  # Cost associated with each arc
        self.q = None  # The weight that has to be delivered to each customer

        self.model = None
        self.x = None
        self.u = None

        self.create_dataset()
        print("setup")
        self.setup()
        print("done")
    
    def create_dataset(self):
        # If data sheet does not exist, generate one
        self.generate_node_table()
        self.create_customers()
        self.create_arcs()

    def load_dataset(self):
        # TODO
        return

    def setup(self):
        self.model = Model('CVRP')

        # Variables
        # self.x = self.model.addVars(self.A, vtype=GRB.BINARY)
        # self.u = self.model.addVars(self.N, vtype=GRB.CONTINUOUS)
        self.x = {}
        for i, j in self.A:
            for k in self.K:
                self.x[i, j, k] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"x[{i}, {j}, {k}]")
        # TODO: Not sure if this below is needed
        # self.u = {}
        # for j in self.N:
        #         self.u[j] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"u[{j}]")
        # Make sure there are no duplicates of i,j,k combination in x
        assert len(self.x) == len(set(self.x.keys()))
        assert len(self.u) == len(set(self.u.keys()))
        self.model.update()

        # Objective function
        # explanation for self.V[:-1] = [0, 1, ..., n]
        # explanation for self.V[1:] = [1, ..., n, n+1]
        self.model.setObjective(quicksum(self.x[i, j, k]*self.c[i, j] for i in self.V[:-1] for j in self.V[1:]
                                         for k in self.K),
                                sense=GRB.MINIMIZE)
        self.model.update()

        # Constraints
        # Each vehicle must leave the depot
        for k in self.K:
            self.model.addConstr(quicksum(self.x[0, j, k] for j in self.V[1:]) == 1, name=f"Start_{k}")
        # Each vehicle must return to depot
        for k in self.K:
            self.model.addConstr(quicksum(self.x[j, 0, k] for j in self.V[:-1]) == 1, name=f"Finish_{k}")
        # Each customer must be visited by one vehicle
        for i in self.N:
            self.model.addConstr(quicksum(self.x[i, j, k] for k in self.K for j in self.V[:-1] if i != j) == 1,
                                 name=f"Visit_{i}")
        # If a vehicle visits a customer, it must also leave that customer
        for i in self.N:
            for k in self.K:
                self.model.addConstr(quicksum(self.x[j, j, k] for j in self.V[:-1] if i != j) -
                                     quicksum(self.x[i, j, k] for j in self.V[1:] if i != j) == 0,
                                     name=f"Leave_{i}{k}")
        # Capacity constraint
        for k in self.K:
            self.model.addConstr(quicksum(self.q[j] * self.x[i, j, k]
                                          for i in self.V for j in self.N if i != j) <= self.Q,
                                 name=f"Capacity_{k}")

        # TODO: I'm not sure if we need these, these might have been completely replaced by the constraint above
        # for i, j in itertools.product(self.A, self.A):
        #     if i != 0 and j != 0:
        #         self.model.addConstr((self.x[i, j] == 1) >> (self.u[i] + self.q[i] == self.u[j]),
        #                              name=f"Constr3_{i}{j}")
        # for i in self.N:
        #     self.model.addConstr(self.q[i] <= self.u[i], name=f"Constr4_{i}")
        #     self.model.addConstr(self.u[i] <= self.Q, name=f"Constr5_{i}")

        # TODO: Sub-tour elimination

        self.model.update()

        return

    @staticmethod
    def calc_distance(p1, p2):
        # p1 = tuple(p1)
        # p2 = tuple(p2)
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
        start_depot_idx = random_selection.pop(int(np.random.random_integers(0, len(random_selection)-1, size=1)))

        # Add a new end depot node to the end of nodes, which has the same coordinates as the start depot
        end_depot_coord = list(self.nodes.iloc[start_depot_idx])
        self.nodes = self.nodes.append(pd.DataFrame([end_depot_coord], columns=['x_coord', 'y_coord']),
                                       ignore_index=True)
        end_depot_idx = self.nodes.index[-1]

        self.N = random_selection  # The rest will be the customer nodes
        temp_list = self.N[:]
        temp_list.insert(0, start_depot_idx)
        temp_list.append(end_depot_idx)
        # self.V[0] is the start depot, and self.V[-1] is the end depot
        self.V = temp_list[:]

        # Assign the customer demand to each customer
        self.q = {}
        for i in self.N:
            self.q[i] = np.random.random_integers(self.demand_range[0], self.demand_range[1], size=1)
        return

    def create_arcs(self):
        self.A = [c for c in itertools.product(self.V, self.V)]

        for i, tup in enumerate(self.A):
            # Remove the elements where i=j
            if tup[0] == tup[1]:
                self.A.pop(i)
            # TODO: is the following correct? is the following needed?
            # Remove the elements that lead from the start node to the end node
            elif (tup[0] == self.V[0] and tup[1] == self.V[-1]) or (tup[0] == self.V[-1] and tup[1] == self.V[0]):
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


vrp = VRP(20, 3)
