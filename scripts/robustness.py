import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def plotRobustnessCurves(G, strategies):
    plt.figure(figsize=(10,6))

    N = G.number_of_nodes()
    results = simulateAttack(G, strategies)
    num_steps = len(next(iter(results.values())))
    nodes_removed = list(range(num_steps))
    percent_removed = [100 * n / N for n in nodes_removed]
    remaining_fraction = [1 - n / N for n in nodes_removed]

    for strategy, vals in results.items():
        num_steps = len(next(iter(results.values())))
        plt.plot(percent_removed, vals, label=strategy, linewidth=1.5)

    # Add dotted line for remaining nodes
    plt.plot(percent_removed, remaining_fraction, 'k--', label='Remaining nodes', linewidth=1.5)
    plt.xlabel("Percentage of nodes removed")
    plt.ylabel("Giant component size (fraction of original nodes)")
    plt.title("Network Robustness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def giantComponentFraction(G, N):
    if G.number_of_nodes() == 0:
        return 0.0
    largest_cc = max(nx.connected_components(G), key=len)
    return len(largest_cc) / N

def simulateAttack(G, strategies):
    N = G.number_of_nodes()
    results = {}
    for strategy in strategies:
        G_tmp = G.copy()
        removed_nodes = set()
        fractions = [giantComponentFraction(G, N)] 

        while G_tmp.number_of_nodes() > 0:
            if strategy == "random":
                chosen = random.choice(list(G_tmp.nodes()))
            else:
                if strategy == "degree":
                    centrality = dict(G_tmp.degree())
                elif strategy == "pagerank":
                    centrality = nx.pagerank(G_tmp)
                elif strategy == "betweenness":
                    centrality = nx.betweenness_centrality(G_tmp)

                chosen = max(centrality, key=centrality.get)

            G_tmp.remove_node(chosen)
            removed_nodes.add(chosen)

            frac = giantComponentFraction(G_tmp, N)
            fractions.append(frac)

        results[strategy] = fractions
        print(f"{strategy}: completed ({len(fractions)} steps)")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    G = nx.read_gml(input_file)
    plotRobustnessCurves(G, strategies=["random", "degree", "betweenness", "pagerank"])