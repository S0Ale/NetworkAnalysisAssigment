import sys
import networkx as nx
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community import girvan_newman, modularity, partition_quality
import igraph as ig
import numpy as np
import leidenalg
import time
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def drawPartition(G, partition, title="Communities"):
    print(f"Drawing {len(partition)} communities (networkx)...")

    # assign a community ID to each node
    community_map = {}
    for cid, community in enumerate(partition):
        for node in community:
            community_map[node] = cid


    pos = nx.spring_layout(G, seed=42)
    node_colors = [community_map[node] for node in G.nodes()]

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        cmap=plt.colormaps.get_cmap("tab20").resampled(len(partition)),
        edgecolors="black",
        node_size=100,
        linewidths=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3)

    centrality = nx.degree_centrality(G)

    labels = {}
    for cid, community in enumerate(partition):
        if community:  # pick node with biggest degree centrality
            rep_node = max(community, key=lambda n: centrality[n])
            labels[rep_node] = G.nodes[rep_node].get("name", rep_node)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight="bold")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def partitionToLabels(G, partition):
    node_index = {node: i for i, node in enumerate(G.nodes())}
    labels = [-1] * len(G.nodes())
    for cid, community in enumerate(partition):
        for node in community:
            labels[node_index[node]] = cid
    return labels

def checkQuality(G, partition):
    mod = modularity(G, partition)
    cov, perf = partition_quality(G, partition)
    print(f"Modularity: {mod:.4f}")
    print(f"Coverage: {cov:.4f}")
    print(f"Performance: {perf:.4f}")

def leiden(G):
    start_time = time.time()
    partition = leidenalg.find_partition(ig.Graph.from_networkx(G), leidenalg.ModularityVertexPartition)
    end_time = time.time()

    # Convert igraph partition to networkx node labels
    node_list = list(G.nodes())
    nx_partition = [set(node_list[v] for v in community) for community in partition]

    print(f"Number of communities found: {len(nx_partition)}")
    checkQuality(G, nx_partition)
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    return nx_partition

def louvain(G):
    start_time = time.time()
    partition = louvain_communities(G)
    end_time = time.time()

    print(f"Number of communities found: {len(partition)}")
    checkQuality(G, partition)
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    return partition

def girvanNewman(G, k=-1):
    start_time = time.time()
    comp = girvan_newman(G)

    best_partition = None
    best_modularity = -1.0
    partitions = []

    if k >= 0:
        for i, communities in enumerate(comp):
            if i == k:
                partitions = [set(c) for c in communities]
                break
    else:
        for i, communities in enumerate(comp):
            partition = [set(c) for c in communities]
            q = modularity(G, partition)
            print(f"Step {i}: modularity = {q:.4f}")
            if q > best_modularity:
                best_modularity = q
                best_partition = partition
            partitions = best_partition  # keep the best one

    end_time = time.time()

    if k >= 0:
        print(f"Partition at level {k} found with {len(partitions)} communities")
    else:
        print(f"Best partition has modularity = {best_modularity:.4f} with {len(partitions)} communities")

    checkQuality(G, partitions)

    print(f"Execution time: {end_time - start_time:.4f} seconds")

    return partitions

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <graph_file.gml.gz> <algorithm:louvain|girvan|leiden> [k]")
        sys.exit(1)

    graph_file, algo = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else -1

    G = nx.read_gml(graph_file)

    partitions = None
    if algo == "louvain":
        partitions = louvain(G)
    elif algo == "girvan":
        partitions = girvanNewman(G, k)
    elif algo == "leiden":
        partitions = leiden(G)
    else:
        print("Unknown algorithm. Use 'louvain' or 'girvan' or 'leiden'.")
        sys.exit(1)

    drawPartition(G, partitions, title=f"Communities detected by {algo.capitalize()}")