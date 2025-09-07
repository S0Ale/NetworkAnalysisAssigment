import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def criticalThreshold(G):
    degrees = [d for n, d in G.degree()]
    k_avg = sum(degrees) / len(degrees)
    k2_avg = sum(d**2 for d in degrees) / len(degrees)
    
    kappa = k2_avg / k_avg
    if kappa <= 1:
        return 1.0
    f_c = 1 - 1 / (kappa - 1)
    return f_c

def analyzeRobustness(G, name="Graph"):
    print(f"{name}: N={G.number_of_nodes()}, E={G.number_of_edges()}, 1st moment={np.mean([d for n, d in G.degree()]):.4f}, 2nd moment={np.mean([d**2 for n, d in G.degree()]):.4f}, fc={criticalThreshold(G):.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input_file>")
        sys.exit(1)

    graph = nx.read_gml(sys.argv[1])
    print("")
    analyzeRobustness(graph, name=sys.argv[1])