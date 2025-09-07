import sys
import networkx as nx
from networkx.algorithms.community import louvain_communities
from networkx.algorithms.community import girvan_newman, modularity, partition_quality
import igraph as ig
import leidenalg
import time
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

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
    print("")
    print("--- Leiden Algorithm ---")
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
    print("")
    print("--- Louvain Algorithm ---")
    start_time = time.time()
    partition = louvain_communities(G)
    end_time = time.time()

    print(f"Number of communities found: {len(partition)}")
    checkQuality(G, partition)
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    return partition

def girvanNewman(G, k=-1):
    print("\n--- Girvan-Newman Algorithm ---")
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
    print("")

    if k >= 0:
        print(f"Partition at level {k} found with {len(partitions)} communities")
    else:
        print(f"Best partition has modularity = {best_modularity:.4f} with {len(partitions)} communities")

    checkQuality(G, partitions)

    print(f"Execution time: {end_time - start_time:.4f} seconds")

    return partitions

def mapAlgorithm(algo):
    if algo == "louvain":
        return louvain
    elif algo == "girvan":
        return girvanNewman
    elif algo == "leiden":
        return leiden
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <graph_file.gml> <first_algorithm:louvain|girvan|leiden> <second_algorithm:louvain|girvan|leiden> [k]")
        sys.exit(1)

    graph_file, algo1, algo2 = sys.argv[1], sys.argv[2], sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    G = nx.read_gml(graph_file)

    algo1_func = mapAlgorithm(algo1)
    algo2_func = mapAlgorithm(algo2)

    if algo1 == "girvan":
        partition1 = algo1_func(G, k)
    else:
        partition1 = algo1_func(G)

    if algo2 == "girvan":
        partition2 = algo2_func(G, k)
    else:
        partition2 = algo2_func(G)

    labels1 = partitionToLabels(G, partition1)
    labels2 = partitionToLabels(G, partition2)

    print("")
    print("ARI:", adjusted_rand_score(labels1, labels2))
    print("NMI:", normalized_mutual_info_score(labels1, labels2))