#!/usr/bin/env python3
import networkx as nx
from itertools import combinations
import sys
import matplotlib.pyplot as plt
import numpy as np

def drawGraph(G, bridges=None):
    plt.figure(figsize=(16, 9)) 
    maxCC = max(nx.connected_components(G), key=len)
    Gcc = G.subgraph(maxCC)
    colors = ["#5BC0EB" if n in maxCC else "#275DAD" for n in G.nodes()]
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_color=colors,
        edgecolors="#272727",
        node_size=100,
        linewidths=0.8
    )

    if bridges:
        nx.draw_networkx_edges(G, pos, edgelist=bridges, edge_color="#FF5733", width=2)

    nx.draw_networkx_edges(G, pos, alpha=0.3)

    plt.title(f"Graph Visualization")
    plt.axis("off")
    plt.tight_layout() 
    plt.show()

def plotHist(measures, title="Histogram", xlabel="", ylabel="", logAxes=False):
    plt.figure(figsize=(8, 6))
    plt.hist(measures, bins=30, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if logAxes:
        plt.xscale("log")
        plt.yscale("log")
    plt.grid()
    plt.show()

def analysis(input_file):
    G = nx.read_gml(input_file)
    print("Sample edge:", list(G.edges(data=True))[0])
    print("\n---- Size ----")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    print("\n---- Characteristics ----")
    maxCC = max(nx.connected_components(G), key=len)
    Gcc = G.subgraph(maxCC)
    bridges = list(nx.bridges(G))
    local_bridges = list(nx.local_bridges(G, with_span=False))
    print("Bridges:", len(bridges))
    print("Local Bridges:", len(local_bridges))

    print("Average degree:", sum(dict(G.degree()).values()) / G.number_of_nodes())
    print("Average shortest path length (largest connected component):", nx.average_shortest_path_length(Gcc))
    print("Density:", nx.density(G))
    print("Connected components:", nx.number_connected_components(G))
    print("Diameter (largest connected component):", nx.diameter(Gcc))

    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G)

    top_deg = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    top_close = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
    top_betw = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

    print("\n--- Degree:")
    for node, value in top_deg:
        name = G.nodes[node].get("name", None)
        if name:
            print(f"{node} ({name}): {value:.4f}")
        else:
            print(f"{node}: {value:.4f}")

    print("\n--- Closeness:")
    for node, value in top_close:
        name = G.nodes[node].get("name", None)  # only persons have this
        if name:
            print(f"{node} ({name}): {value:.4f}")
        else:
            print(f"{node}: {value:.4f}")

    print("\n--- Betweenness:")
    for node, value in top_betw:
        name = G.nodes[node].get("name", None)
        if name:
            print(f"{node} ({name}): {value:.4f}")
        else:
            print(f"{node}: {value:.4f}")

    print("")
    triangles_per_node = list(nx.triangles(G).values())
    print(f"Total triangles: {sum(triangles_per_node) // 3}")
    print(f"Average triangles per node: {sum(triangles_per_node) / G.number_of_nodes():.4f}")
    print(f"Average clustering: {nx.average_clustering(G):.4f}")

    degrees = [G.degree(n) for n in G.nodes()]
    
    plotHist(degrees, title="Degree Distribution", xlabel="Degree", ylabel="Frequency")
    plotHist(closeness.values(), title="Closeness Centrality Distribution", xlabel="Closeness Centrality", ylabel="Frequency")
    plotHist(betweenness.values(), title="Betweenness Centrality Distribution", xlabel="Betweenness Centrality", ylabel="Frequency")
    plotHist(nx.clustering(G).values(), title="Clustering Coefficient Distribution", xlabel="Clustering Coefficient", ylabel="Frequency")

    drawGraph(G, bridges)
    pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    analysis(input_file)