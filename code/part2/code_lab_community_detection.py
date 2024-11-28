"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    A = nx.adjacency_matrix(G)
    
    degrees=np.array(A.sum(axis=1)).flatten()
    D_inv = diags(1 / degrees)
    I=eye(A.shape[0], format="csr")
    L=I-D_inv@A
    S, U = eigs(L, k=k,which='SM')
    
    kmeans = KMeans(n_clusters=k).fit(np.real(U))
    clustering={}
    for i,node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]
    ##################
    
    return clustering



############## Task 4

##################
file_path='.//datasets//CA-HepTh.txt'
G = nx.read_edgelist(
    file_path,  
    delimiter='\t',           
    comments='#',            
    create_using=nx.Graph())
largest_cc = max(nx.connected_components(G), key=len)
largest_cc_graph=G.subgraph(largest_cc).copy()
clustering=spectral_clustering(largest_cc_graph, 50)
print(clustering)
##################



############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    
    
    A = nx.to_numpy_array(G)
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}
    degrees = np.sum(A, axis=1)  # Degree of each node
    m = G.number_of_edges()  # Total number of edges
    
    Q = 0  # Initialize modularity
    communities = set(clustering.values())  # Unique communities
    
    for c in communities:
        # Nodes in community c
        nodes_in_c = [node_to_index[node] for node in G.nodes if clustering[node] == c]
        l_c = sum(A[i, j] for i in nodes_in_c for j in nodes_in_c) / 2  # Edges within c
        d_c = sum(degrees[i] for i in nodes_in_c)  # Degree sum for nodes in c
        
        Q += (l_c / m) - (d_c / (2 * m)) ** 2
    
    

    
    modularity=Q
    ##################

    
    return modularity



############## Task 6

##################
random_clustering={node:np.random.randint(0,50) for k , node in enumerate(largest_cc_graph.nodes())}
rd_cluster_mod=modularity(largest_cc_graph,random_clustering)
cluster_mod=modularity(largest_cc_graph,clustering)
print('Random clustering modularity: ',rd_cluster_mod)
print('Spectral Clustering : ',cluster_mod)
##################








