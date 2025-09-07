import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from networkx.algorithms.community import greedy_modularity_communities
import random
import igraph as ig
import leidenalg

def drawGraph(G, new_edges=None):
    plt.figure(figsize=(8, 6))
    colors = ["#275DAD" for _ in G.nodes()]
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_color=colors,
        edgecolors="#272727",
        node_size=80,
        linewidths=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    if new_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=new_edges,
            edge_color="red",
            alpha=0.4,
            width=2
        )

    plt.title(f"Graph Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def addClique(G, central_node):
    H = G.copy()
    neighbors = list(H.neighbors(central_node))
    new_edges = []

    # Connect every pair of neighbors
    for u, v in combinations(neighbors, 2):
        if not H.has_edge(u, v):
            H.add_edge(u, v)
            new_edges.append((u, v))

    return H, new_edges

def addCliqueDegreeNodes(G, n=1):
    G = G.copy()
    all_edges = []
    # Top n degree nodes
    top_n = sorted(G.degree, key=lambda x: x[1], reverse=True)[:n]
    for node, _ in top_n:
        G, new_edges = addClique(G, central_node=node)
        all_edges.extend(new_edges)
    return G, all_edges

def addCliqueBetweennessNodes(G, n=1):
    G = G.copy()
    
    # Top n betweenness nodes
    bet = nx.betweenness_centrality(G)
    top_nodes = sorted(bet, key=bet.get, reverse=True)[:n]
    all_edges = []

    for node in top_nodes:
        G, new_edges = addClique(G, central_node=node)
        all_edges.extend(new_edges)

    return G, all_edges

def reinforceBridges(G):
    G = G.copy()
    bridges = list(nx.bridges(G))
    for u, v in bridges:
        for n in (set(G.neighbors(u)) | set(G.neighbors(v))):
            if n != u and not G.has_edge(u, n):
                G.add_edge(u, n)
            if n != v and not G.has_edge(v, n):
                G.add_edge(v, n)
    return G

def addIntercommunityShortcuts(G, seed=None):
    rng = random.Random(seed)
    H = G.copy()
    new_edges = []

    G_ig = ig.Graph.from_networkx(G)
    partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
    
    # convert to NetworkX node labels
    communities = [[G_ig.vs[v]["_nx_name"] for v in community] for community in partition]

    # add one edge between each pair of communities
    for i in range(len(communities)):
        for j in range(i+1, len(communities)):
            comm_i = list(communities[i])
            comm_j = list(communities[j])
            rng.shuffle(comm_i)
            rng.shuffle(comm_j)
            
            for u in comm_i:
                for v in comm_j:
                    if not H.has_edge(u, v):
                        H.add_edge(u, v)
                        new_edges.append((u, v))
                        break
                else:
                    continue
                break

    return H, new_edges

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_file> <method:degree|betweenness|randomize|bridge|intercommunity>")
        sys.exit(1)

    graph = nx.read_gml(sys.argv[1])
    method = sys.argv[2] if len(sys.argv) > 2 else "degree"
    newGraph = None

    if method == "degree":
        newGraph, new_edges = addCliqueDegreeNodes(graph, n=3)
        drawGraph(newGraph, new_edges)
    elif method == "betweenness":
        newGraph, new_edges = addCliqueBetweennessNodes(graph, n=3)
        drawGraph(newGraph, new_edges)
    elif method == "randomize":
        newGraph = nx.gnm_random_graph(graph.number_of_nodes(), graph.number_of_edges())
    elif method == "bridge":
        newGraph = reinforceBridges(graph)
        drawGraph(newGraph)
    elif method == "intercommunity":
        newGraph, new_edges = addIntercommunityShortcuts(graph, seed=42)
        drawGraph(newGraph, new_edges)
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)

    if newGraph:
        nx.write_gml(newGraph, f"enhanced_{method}.gml")
        print(f"Enhanced graph saved as: enhanced_{method}.gml")
