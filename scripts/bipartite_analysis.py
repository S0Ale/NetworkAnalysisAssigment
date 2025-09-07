import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import itertools

def drawGraph(G):
    plt.figure(figsize=(16, 9)) 
    colors = ["#5BC0EB" if G.nodes[n]["bipartite"] == 0 else "#275DAD" for n in G.nodes()]

    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_color=colors,
        edgecolors="#272727",
        node_size=100,
        linewidths=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', label='Person',
                    markerfacecolor='#5BC0EB', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Crime',
                    markerfacecolor='#275DAD', markersize=10)
    ], loc="best")

    plt.title(f"Graph Visualization")
    plt.axis("off")
    plt.tight_layout() 
    plt.show()

def plotHist(measures, title="Histogram", xlabel="", ylabel=""):
    plt.figure(figsize=(8, 6))
    plt.hist(measures, bins=30, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def count4Cycles(G, nodes=None):
    if nodes is None:
        # use the bipartite attribute
        nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]

    count = 0
    for u, v in itertools.combinations(nodes, 2):
        common = set(G.neighbors(u)) & set(G.neighbors(v))
        k = len(common)
        if k >= 2:
            # choose 2 common neighbors
            count += k * (k - 1) // 2
    return count

G = nx.read_gml("moreno_graph.gml")

print("---- Size ----")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Sample edge:", list(G.edges(data=True))[0])

# Left and right nodes
leftNodes = {n for n, d, in G.nodes(data=True) if d["bipartite"] == 0}
rightNodes = set(G) - leftNodes

print("Left nodes (persons):", len(leftNodes))
print("Right nodes (crimes):", len(rightNodes))

print("\n---- Characteristics ----")
degLeft, _ = bipartite.degrees(G, leftNodes)
_, degRight = bipartite.degrees(G, rightNodes)

maxCC = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(maxCC)

print("Average degree (bipartite, left):", sum(d for _, d in degLeft) / len(leftNodes))
print("Average degree (bipartite, right):", sum(d for _, d in degRight) / len(rightNodes))
print("Average shortest path length (largest connected component):", nx.average_shortest_path_length(Gcc))
print("Density (bipartite, left):", bipartite.density(G, leftNodes))
print("Density (bipartite, right):", bipartite.density(G, rightNodes))

maxCC = max(nx.connected_components(G), key=len)
Gcc = G.subgraph(maxCC)
print("Diameter (largest connected component):", nx.diameter(Gcc))

closeness_bi = bipartite.closeness_centrality(G, leftNodes)
betweenness_bi = bipartite.betweenness_centrality(G, leftNodes)

top_close = sorted(closeness_bi.items(), key=lambda x: x[1], reverse=True)[:5]
top_betw = sorted(betweenness_bi.items(), key=lambda x: x[1], reverse=True)[:5]

print("\n--- Top bipartite closeness:")
for node, value in top_close:
    name = G.nodes[node].get("name", None)  # Only persons have this
    if name:
        print(f"{node} ({name}): {value:.4f}")
    else:
        print(f"{node}: {value:.4f}")

print("\n--- Top bipartite betweenness:")
for node, value in top_betw:
    name = G.nodes[node].get("name", None)
    if name:
        print(f"{node} ({name}): {value:.4f}")
    else:
        print(f"{node}: {value:.4f}")


print(f"\nCount of 4-cycles (bipartite): {count4Cycles(G):d}")
print(f"Average clustering: {bipartite.average_clustering(G):3.4f}")
print(f"Degree assortativity: {nx.degree_assortativity_coefficient(G):.4f}")
print(f"Assortativity (bipartite): {nx.attribute_assortativity_coefficient(G, 'bipartite'):.4f}")
print(f"Attribute assortativity (sex): {nx.attribute_assortativity_coefficient(G, 'sex'):.4f}")

print("Plotting degree distribution...")

person_degrees = [G.degree(n) for n in leftNodes]
crime_degrees = [G.degree(n) for n in rightNodes]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(person_degrees, bins=range(1, max(person_degrees)+2), edgecolor='black', align='left')
plt.title("Degree Distribution of Persons")
plt.xlabel("Number of Crimes per Person")
plt.ylabel("Number of Persons")

plt.subplot(1, 2, 2)
plt.hist(crime_degrees, bins=range(1, max(crime_degrees)+2), edgecolor='black', align='left', color='orange')
plt.title("Degree Distribution of Crimes")
plt.xlabel("Number of Persons per Crime")
plt.ylabel("Number of Crimes")

plt.tight_layout()
plt.show()

plotHist(list(betweenness_bi.values()), title="Betweenness Centrality (Bipartite)", xlabel="Betweenness", ylabel="Frequency")
plotHist(list(closeness_bi.values()), title="Closeness Centrality (Bipartite)", xlabel="Closeness", ylabel="Frequency")
plotHist(list(bipartite.clustering(G).values()), title="Clustering Coefficient (Bipartite)", xlabel="Clustering", ylabel="Frequency")

drawGraph(G)
