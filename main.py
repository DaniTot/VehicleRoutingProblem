import numpy as np
import pandas as pd


def generate_node_table(number_of_nodes, x_range, y_range, file_name="nodes.csv"):
    nodes_coord_x = np.random.rand(number_of_nodes) * x_range
    nodes_coord_y = np.random.rand(number_of_nodes) * y_range
    nodes_table = pd.DataFrame(np.transpose(np.array([nodes_coord_x, nodes_coord_y])), columns=['x_coord', 'y_coord'])
    nodes_table.to_csv(file_name)
    return nodes_table


def read_node_table(file_name="nodes.csv"):
    node_table = pd.read_csv(file_name)
    return node_table


def select_customers(node_table, number_of_cutomers):
    N = node_table.sample(n=number_of_cutomers)
    return N
