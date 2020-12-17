import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import itertools
import os


from collections import Counter
from matplotlib.markers import MarkerStyle
from gurobipy import Model, GRB, quicksum, LinExpr


import time



class VRP:
    """ VRP class function

    Methods
    -------
    example2(bla=blabla)
        lorem ipsun
    example3()
        lorem ipsun

    """
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

        self.subtour_type = 'DFJ'  # DFJ or MTZ
        self.gap_goal = 0.1

        self.random_data_n_model_p = "random_datasets"

    def setup_random_data(self,
                          number_of_customers,
                          number_of_vehicles,
                          vehicle_capacity=5,
                          x_range=10, y_range=10,
                          demand_lower=1, demand_higher=2,
                          seed=420):

        """
        Random dataset generation 

        Parameters
        ----------
        number_of_customers : int
            Total number of customers in problem
        number_of_vehicles : int
            Number of vehicles
        Vehicle capacity : int, optional
            Vehicle capacity
        x_range : int, optional
            Horizontal Scaling for the random dataset
        y_range : int, optional
            Vertical Scaling for the random dataset
        demand_lower : int, optional
            Lower bound of customer demand range.
        demand_higher : int, optional
            Lower bound of customer demand range.
        seed : int, optional
            Seed for RNG
        """

        if seed is not None:
            np.random.seed(seed)

        self.Q = vehicle_capacity
        self.n = number_of_customers
        self.x_range = x_range
        self.y_range = y_range
        self.number_of_customers = number_of_customers
        self.demand_range = [demand_lower, demand_higher]
        self.K = np.arange(1,number_of_vehicles+1)

        self.create_dataset()

        self.create_arcs()

    def setup_preset_data(self, file_name, number_of_vehicles, subtour_type = 'DFJ'):

        nb_customers, truck_capacity, n, v, demands, nodes = self.read_input_cvrp(file_name)

        self.read_input_cvrp(file_name)
        self.Q = truck_capacity
        self.n = nb_customers + 1

        self.number_of_customers = nb_customers
        self.K = np.arange(1,number_of_vehicles+1)

        self.nodes = nodes
        self.N = n
        self.V = v
        self.q = demands

        self.subtour_type = subtour_type

        self.create_arcs()

    def visualize(self,plot_sol='y'):

        cmap = plt.cm.get_cmap('hsv', len(self.K) + 1)
        fig, ax = plt.subplots(1, 1)  # a figure with a 1x1 grid of Axes

        
        if plot_sol in ['y', 'yes']:

            plt.title('Capacitated Vehicle Routing Problem')

            self.find_active_arcs()
            tours = self.subtour(self.K, self.active_arcs)

            for k in tours.keys():
                vehicle_color = cmap(k-1)[0:3]
                vehicle_arcs = tours[k]
                G = nx.DiGraph()
                for tour in vehicle_arcs:
                    idx = 0
                    for node in tour:
                        node_pos = (self.nodes.iloc[node][0],self.nodes.iloc[node][1])
                        G.add_node(node,pos=node_pos)

                        if idx < len(tour)-1:
                            node_i = tour[idx]
                            node_j = tour[idx+1]
                            edge_cost = round(self.c[(node_i,node_j)],2)
                            G.add_edge( node_i, node_j, weight=edge_cost )
                            idx += 1

                    node_pos = nx.get_node_attributes(G,'pos')
                    weights = nx.get_edge_attributes(G,'weight')

                    nx.draw(G,node_pos,ax=ax, node_size=400, node_color='w', edgecolors=vehicle_color, edge_color= vehicle_color )
                    nx.draw_networkx_edge_labels(G, node_pos, ax=ax, edge_labels=weights)      
                    
                    ax.scatter([],[], c=cmap(k-1), label='k='+str(k) )

                    for node in node_pos.keys():
                        pos = node_pos[node]   

                        if node == self.V[0]:
                            offset = -0.1
                            comma_on = ','
                        elif node == self.V[-1]:
                            offset = 0.1
                        else:
                            offset = 0
                            comma_on=''
                        ax.text(pos[0] + offset, pos[1], s=str(node)+comma_on, horizontalalignment='center',verticalalignment='center')


            # Recolor depot node to black
            G = nx.DiGraph()
            pos = (self.nodes.iloc[0][0],self.nodes.iloc[0][1])
            G.add_node(0,pos=pos)
            node_pos = nx.get_node_attributes(G,'pos')
            nx.draw_networkx_nodes(G,node_pos,ax=ax, node_size=400, node_color='w', edgecolors='k')

            # Add axes
            xmin = self.nodes.min()['x_coord'] - 1
            ymin = self.nodes.min()['y_coord'] - 1

            xmax = self.nodes.max()['x_coord'] + 1
            ymax = self.nodes.max()['y_coord'] + 1

            plt.axis('on') # turns on axis
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            plt.legend(loc='lower right')

            print("Subtours")
            print(tours)

            plt.show()     


        elif plot_sol == 'n':
            plt.title('Capacitated Vehicle Routing Problem')

            G = nx.Graph()

            for node in self.V:
                node_pos = (self.nodes.iloc[node][0],self.nodes.iloc[node][1])
                G.add_node(node,pos=node_pos)

            for edge in self.c.keys():
                if edge != (self.V[0],self.V[-1]) and edge[1] != self.V[-1]:
                    G.add_edge(edge[0],edge[1],weight=round(self.c[edge],2))

            node_pos = nx.get_node_attributes(G, 'pos')
            weights = nx.get_edge_attributes(G , 'weight')

            offset = 0
            nx.draw(G, node_pos, ax=ax, node_color='w', edgecolors='k')

            for node in node_pos.keys():
                pos = node_pos[node]   

                if node == self.V[0]:
                    offset = -0.1
                    comma_on = ','
                elif node == self.V[-1]:
                    offset = 0.1
                else:
                    offset = 0
                    comma_on=''
                ax.text(pos[0] + offset, pos[1], s=str(node)+comma_on, horizontalalignment='center',verticalalignment='center')

            nx.draw_networkx_edge_labels(G,node_pos,edge_labels=weights)


            # Add axes
            xmin = self.nodes.min()['x_coord'] - 1
            ymin = self.nodes.min()['y_coord'] - 1

            xmax = self.nodes.max()['x_coord'] + 1
            ymax = self.nodes.max()['y_coord'] + 1

            plt.axis('on') # turns on axis
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            plt.legend(loc='lower right')

            plt.show()


        return

    ################################           DATASET        ###################################### 
    ################################################################################################

    def create_dataset(self):
        # If data sheet does not exist, generate one
        self.generate_node_table()
        self.create_customers()
        print("Node data generated")

    @staticmethod
    def calc_distance(p1, p2):
        dist = np.linalg.norm( np.subtract(p1, p2) ) # Euclidian distance
        # dist = (((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)
        return dist

    def generate_node_table(self, file_name="nodes.csv"):
        nodes_coord_x = np.random.rand(self.n + 1) * self.x_range
        nodes_coord_y = np.random.rand(self.n + 1) * self.y_range
        nodes_table = pd.DataFrame(np.transpose(np.array([nodes_coord_x, nodes_coord_y])),
                                   columns=['x_coord', 'y_coord'])


        if file_name == "nodes.csv":
            file_name = "n{}k{}_nodes.csv".format( self.n,len(self.K) ) 
            
        nodes_table.to_csv( os.path.join(self.random_data_n_model_p,file_name) )
        self.nodes = nodes_table

        return nodes_table

    def read_node_table(self, file_name="nodes.csv"):
        if file_name == "nodes.csv":
            file_name = "n{}k{}_nodes.csv".format(self.n,self.k)

        nodes_table = pd.read_csv( os.path.join(self.random_data_n_model_p,file_name) )
        self.nodes = nodes_table
        return nodes_table

    def create_customers(self):
        random_selection = list(self.nodes.sample(n=self.n + 1).index)

        # Select the depot node
        # Depot node is randomly selected
        start_depot_idx = random_selection.pop(np.random.randint(0, len(random_selection) - 1, size=1)[0])
        # print("start_depot_idx", start_depot_idx)

        # Add a new end depot node to the end of nodes, which has the same coordinates as the start depot
        end_depot_coord = list(self.nodes.iloc[start_depot_idx])
        self.nodes = self.nodes.append(pd.DataFrame([end_depot_coord], columns=['x_coord', 'y_coord']),
                                       ignore_index=True)

        end_depot_idx = self.nodes.index[-1]
        # print("end_depot_idx", end_depot_idx)

        customer_nodes = self.nodes.to_numpy()[random_selection]
        depot_nodes    = self.nodes.to_numpy()[ [start_depot_idx,end_depot_idx] ] 

        self.nodes     = pd.DataFrame( np.vstack( (depot_nodes[0], customer_nodes, depot_nodes[1]) ), columns=['x_coord', 'y_coord'])

        self.N         = np.arange(1,self.n+1)
        self.V         = np.hstack((0,self.N,self.n+1)) 

        print(self.nodes)

        # Assign the customer demand to each customer
        self.q = {}
        for i in self.N:
            self.q[i] = np.random.randint(self.demand_range[0], self.demand_range[1], size=1)[0]
       
        assert sum(self.q.values()) <= self.Q * len(self.K), f"The total customer demand {sum(self.q.values())} " \
                                                             f"exceeds the total vehicle capacity: " \
                                                             f"{self.Q * len(self.K)} = {self.Q} * {len(self.K)}."

        return

    def create_arcs(self):
        """
        Create arcs between customer nodes
        :return:
        """
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

        # sort all the arcs
        A = np.array(self.A)

        sort_ = np.lexsort((A[:,1],A[:,0]),axis=0)

        self.A = A[sort_]

        # The cost to travel an arc equals its length
        self.c = {}
        for i, j in self.A:
            x_i = self.nodes.get('x_coord')[i]
            y_i = self.nodes.get('y_coord')[i]
            x_j = self.nodes.get('x_coord')[j]
            y_j = self.nodes.get('y_coord')[j]
            self.c[(i, j)] = self.calc_distance((x_i, y_i), (x_j, y_j))
        return



    ################################           READ CVRP        #################################### 
    ################################################################################################


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


    ################################      OPTIMIZATION  MODEL      ################################# 
    ################################################################################################
    def setup(self):

        self.general_setup()

        self.model.update()

        # Miller-Tucker-Zemlin formulation for subtour elimination
        # $$u_{j} - u_{i} \geq q_{j} - Q(1-x_{ijk}), i,j = \{1,....,n\}, i \neq j$$
        if self.subtour_type == 'MTZ':
            for k in self.K:
                for i, j in self.A:
                    if i >= 1 and j >= 1:
                        if i != self.number_of_customers + 1 and j != self.number_of_customers + 1:
                            self.model.addConstr(self.u[j, k] - self.u[i, k] >= self.q[j] - self.Q * (1 - self.x[i, j, k]))

            # Capacity constraint
            for k in self.K:
                for i in self.N:
                    self.model.addConstr(self.u[i, k] >= self.q[i])
                    self.model.addConstr(self.u[i, k] <= self.Q)

        # Subtour elimination constraint (Dantzig-Fulkerson Johnson)
        # $$\sum_{i\in S}\sum_{j \in S,j \neq i} x_{ij} \leq |S| - 1$$
        elif self.subtour_type == 'DFJ':
            for k in self.K:
                self.model.addConstr(
                    quicksum(
                        quicksum(
                            self.q[j] * self.x[i, j, k] for j in self.N if j != i
                            ) 
                        for i in self.V[:-1] 
                    ) <= self.Q )

        else:
            pass

        self.model.update()

        # optimization limit
        self.model.setParam("MIPGap", self.gap_goal)
        self.model.update()

        mdl_name = "n{}k{}.lp".format(self.n, len(self.K) )

        self.model.write( os.path.join("models",mdl_name) )

        return


    def general_setup(self):
        """
        Basic model elements shared by both CVRP and CVRPTW
        :return:
        """

        self.model = Model()

        # Variables
        self.x = {}
        for i, j in self.A:
            for k in self.K:
                self.x[i, j, k] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"x[{i},{j},{k}]")

        self.u = {}
        for j in self.N:
            for k in self.K:
                self.u[j, k] = self.model.addVar(vtype=GRB.CONTINUOUS, name=f"u[{j},{k}]")

        # Objective function
        # explanation for self.V[:-1] = [0, 1, ..., n]
        # explanation for self.V[1:] = [1, ..., n, n+1]
        self.model.setObjective(quicksum(self.x[i, j, k] * self.c[i, j] for i in self.V[:-1] for j in self.V[1:]
                                         for k in self.K if i != j),
                                sense=GRB.MINIMIZE)

        # Make sure there are no duplicates of i,j,k combination in x
        assert len(self.x) == len(set(self.x.keys()))

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

        # # Capacity constraint
        # for k in self.K:
        #     self.model.addConstr(
        #         quicksum(self.q[j] * self.x[i, j, k] for i in self.V[:-1] for j in self.V[1:-1] if i != j) <= self.Q,
        #         name=f"Capacity_{k}")

        return



    def add_model_vars_dfj(self):
            self.model._x = self.x # User made variables get passed along with underscore VarName
            self.model._A = self.A
            self.model._K = self.K
            self.model._V = self.V
            self.model._subtour = self.subtour

    def optimize(self):

        if self.subtour_type == 'MTZ':
            self.model.optimize()
        elif self.subtour_type == 'DFJ':
            self.add_model_vars_dfj()
            self.model.Params.lazyConstraints = 1
            self.model.optimize(self.subtourelim)
        elif self.subtour_type == '':
            if input("no subtour type chosen, continue?: ") in ['y', 'yes']:
                self.model.optimize()

        if input("Save optimized result?:").lower() in ['y', 'yes']:
            solution_name = "n{}k{}.sol".format(self.n,len(self.K))
            self.model.write( os.path.join("solutions",solution_name))

        return


    def find_active_arcs(self):

        active_arcs = []

        for i,j in self.A:
            for k in self.K:
                if round(self.x[i,j,k].x) == 1:
                    active_arcs.append([i,j,k])

        self.active_arcs = np.vstack(active_arcs)

    @staticmethod
    def subtourelim(mdl,where):
        if where == GRB.callback.MIPSOL:
            active_arcs = []
            solutions = mdl.cbGetSolution(mdl._x)

            for i, j in mdl._A:
                for k in mdl._K:
                    if round(solutions[i, j, k]) == 1:
                        active_arcs.append([i, j, k])

            active_arcs = np.vstack(active_arcs)

            tours = mdl._subtour(mdl._K, active_arcs)

            # add lazy constraints
            for k in tours.keys():
                if len(tours[k]) > 1:
                    for tour in tours[k]:
                        S = np.unique(tour)
                        expr = LinExpr()

                        for i in S:
                            if i != mdl._V[-1]:
                                for j in S:
                                    if j != i and j != mdl._V[0]:
                                            expr += mdl._x[i, j, k] 

                        # expr = quicksum(mdl._x[i, j, k] for i in S for j in S if j != i if i != mdl._V[-1] or j != mdl._V[0])
                        mdl.cbLazy(expr <= len(S) - 1)
                        
    @staticmethod
    def subtour(K, active_arcs):
        tours = {}

        for k in K:
            vehicle_tours = []
            vehicle_arcs = active_arcs[np.where(active_arcs[:, 2] == k)][:, 0:2]
            start_node, finish_node = vehicle_arcs[0]

            tour = [start_node, finish_node]
            vehicle_arcs = np.delete(vehicle_arcs, [0], axis=0)

            while True:
                while True:
                    next_node = np.where(vehicle_arcs[:, 0] == finish_node)

                    if next_node[0].size == 0:
                        vehicle_tours.append(tour)
                        break
                    else:
                        start_node, finish_node = vehicle_arcs[next_node][0]
                        vehicle_arcs = np.delete(vehicle_arcs, next_node[0], axis=0)

                        tour.append(finish_node)

                if vehicle_arcs.size != 0:
                    start_node, finish_node = vehicle_arcs[0]
                    vehicle_arcs = np.delete(vehicle_arcs, [0], axis=0)

                    tour = [start_node, finish_node]
                else:
                    tours[k] = vehicle_tours
                    break

        return tours