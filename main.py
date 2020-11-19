import numpy as np
import pandas as pd
import itertools
from gurobipy import Model, GRB, quicksum


class VRP:
    def __init__(self, number_of_nodes, number_of_vehicles,
                 vehicle_capacity=30, x_range=10, y_range=10, number_of_customers=None, demand_lower=1, demand_higher=3,
                 seed=420):
        if seed is not None:
            self.rnd = np.random.seed(seed)
        else:
            self.rnd = np.random

        self.Q = vehicle_capacity
        self.n = number_of_nodes
        self.x_range = x_range
        self.y_range = y_range
        if number_of_customers is None:
            self.number_of_customers = number_of_nodes
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
        self.x = self.model.addVars(self.A, vtype=GRB.BINARY)
        self.u = self.model.addVars(self.N, vtype=GRB.CONTINUOUS)
        self.model.update()

        # Objective function
        # TODO: Alter obj. func. for multiple vehicles
        self.model.setObjective(quicksum(self.x[i, j]*self.c[i, j] for i, j in self.A), sense=GRB.MINIMIZE)
        self.model.update()

        # Constraints
        for i in self.N:
            self.model.addConstr(quicksum(self.x[i, j] for j in self.V if i != j) == 1, name=f"Constr1_{i}")
        for j in self.N:
            self.model.addConstr(quicksum(self.x[i, j] for i in self.V if j != i) == 1, name=f"Constr2_{j}")
        for i, j in itertools.product(self.A, self.A):
            if i != 0 and j != 0:
                self.model.addConstr((self.x[i, j] == 1) >> (self.u[i] + self.q[i] == self.u[j]),
                                     name=f"Constr3_{i}{j}")
        for i in self.N:
            self.model.addConstr(self.q[i] <= self.u[i], name=f"Constr4_{i}")
            self.model.addConstr(self.u[i] <= self.Q, name=f"Constr5_{i}")

        # TODO: multiple vehicle constraint
        self.model.update()

        return

    @staticmethod
    def calc_distance(p1, p2):
        # p1 = tuple(p1)
        # p2 = tuple(p2)
        dist = (((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
        return dist

    def generate_node_table(self, file_name="nodes.csv"):
        nodes_coord_x = self.rnd.rand(self.n) * self.x_range
        nodes_coord_y = self.rnd.rand(self.n) * self.y_range
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
        self.V = random_selection.pop(self.rnd.random_integers(min(random_selection),
                                                               max(random_selection) + 1,
                                                               size=1))
        self.N = random_selection  # The rest will be the customer nodes
        self.V = self.V + self.N

        # Assign the customer demand to each customer
        self.q = {}
        for i in self.N:
            self.q[i] = self.rnd.random_integers(self.demand_range[0], self.demand_range[1], size=1)
        return

    def create_arcs(self):
        self.A = [c for c in itertools.product(self.V, self.V)]
        # Remove the elements where i=j
        for i, tup in enumerate(self.A):
            if tup[0] == tup[1]:
                self.A.pop(i)

        # The cost to travel an arc equals its length
        self.c = {}
        for i, j in self.A:
            x_i = self.nodes.at(i, 'x_coord')
            y_i = self.nodes.at(i, 'y_coord')
            x_j = self.nodes.at(i, 'x_coord')
            y_j = self.nodes.at(i, 'y_coord')
            self.c[(i, j)] = self.calc_distance((x_i, y_i), (x_j, y_j))
        return
