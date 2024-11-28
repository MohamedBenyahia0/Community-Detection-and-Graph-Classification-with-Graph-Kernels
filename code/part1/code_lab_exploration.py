"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
file_path='datasets//CA-HepTh.txt'
G = nx.read_edgelist(
    file_path,  
    delimiter='\t',           
    comments='#',            
    create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

############## Task 2
n=len(list(nx.connected_components(G)))
print("Number of connected components : ",n)
largest_cc = max(nx.connected_components(G), key=len)
largest_cc_graph=G.subgraph(largest_cc).copy()
print("Number of nodes of largest cc:", largest_cc_graph.number_of_nodes())
print("Number of edges of largest cc:", largest_cc_graph.number_of_edges())
fraction_nodes=largest_cc_graph.number_of_nodes()/G.number_of_nodes()
fraction_edges=largest_cc_graph.number_of_edges()/G.number_of_edges()
print("fraction of nodes : ",fraction_nodes)
print("fraction of edges : ",fraction_edges)
