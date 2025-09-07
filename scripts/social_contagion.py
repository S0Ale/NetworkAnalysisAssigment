import sys
import random
import string
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def randomWord(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def makeTamperFunction():
    def tamper(msg):
        words = msg.split()
        idx = random.randrange(len(words))
        words[idx] = randomWord(len(words[idx]))
        return " ".join(words)
    return tamper


def drawGraph(G, step, states, malicious_nodes, initial_seeds, messages, newly_adopted=None, save_path=None):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)

    color_map = []
    for n in G.nodes():
        if states[n] == 0:
            base_color = "#275DAD"
        elif states[n] == 1:
            base_color = "#3EB489"
        elif states[n] == 2:
            base_color = "#FF4B3E"

        if newly_adopted and n in newly_adopted:
            if states[n] == 1:
                base_color = "#A8F5D1"
            elif states[n] == 2:
                base_color = "#FFD1C4"

        color_map.append(base_color)

    nx.draw_networkx_nodes(
        G, pos,
        node_color=color_map,
        node_size=50,
        edgecolors="black",
        linewidths=0.8
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=initial_seeds,
        node_color=[color_map[list(G.nodes()).index(n)] for n in initial_seeds],
        node_size=70,
        edgecolors="red",
        linewidths=1.5
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=malicious_nodes,
        node_color=[color_map[list(G.nodes()).index(n)] for n in malicious_nodes],
        node_size=70,
        edgecolors="purple",
        linewidths=1.5
    )

    nx.draw_networkx_edges(G, pos, alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Inactive',
               markerfacecolor='#275DAD', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Original',
               markerfacecolor='#3EB489', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Original (new)',
               markerfacecolor='#A8F5D1', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Tampered',
               markerfacecolor='#FF4B3E', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Tampered (new)',
               markerfacecolor='#FFD1C4', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Initial gossipers',
               markerfacecolor='white', markersize=10, markeredgecolor='red'),
        Line2D([0], [0], marker='o', color='w', label='Malicious nodes',
               markerfacecolor='white', markersize=10, markeredgecolor='purple')
    ]
    plt.legend(handles=legend_elements, loc="best")

    counts = {0: 0, 1: 0, 2: 0}  # count nodes per state
    for s in states.values():
        counts[s] += 1
    count_text = f"Inactive: {counts[0]}, Original: {counts[1]}, Tampered: {counts[2]}"
    plt.title(f"Step {step} | {count_text}")

    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/frame_{step:03d}.png")
        plt.close()
        print(f"Frame {step:03d} saved.")
    else:
        plt.show()

def plotSimilarityHeatmap(messages, G, max_labels=50):
    nodes = list(G.nodes())
    texts = [messages[n] if messages[n] is not None else "" for n in nodes]

    # Mask inactive nodes
    active_mask = [i for i, t in enumerate(texts) if t.strip() != ""]
    active_nodes = [nodes[i] for i in active_mask]
    active_texts = [texts[i] for i in active_mask]

    if not active_texts:
        print("No active messages found, skipping similarity heatmap.")

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(active_texts)

    sim_matrix = cosine_similarity(X)

    n_active = len(active_nodes)
    if n_active <= max_labels:
        # show full messages
        labels = active_texts
        rotation = 90
        fontsize = 7
        xticks = range(n_active)
        yticks = range(n_active)

    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix, cmap="viridis", interpolation="nearest")
    plt.colorbar(label="Cosine similarity")
    plt.title("Message similarity across active nodes")
    if n_active <= max_labels:
        plt.xticks(xticks, labels, rotation=rotation, fontsize=fontsize)
        plt.yticks(yticks, labels, fontsize=fontsize)
    else:
        plt.xlabel("Active Node index")
        plt.ylabel("Active Node index")
    plt.tight_layout()
    plt.show()

def printSampleMessages(messages, original_message, k=3):
    tampered = list({msg for msg in messages.values() if msg and msg != original_message})
    print("\n=== Sample Messages ===")
    print(f"Original: {original_message}")
    if tampered:
        print(f"\n{min(k, len(tampered))} Tampered samples:")
        for msg in random.sample(tampered, min(k, len(tampered))):
            print(f"- {msg}")
    else:
        print("\n(no tampered messages were generated)")

def simulateContagion(G, theta, states, malicious_nodes, initial_seeds, steps, messages, tamper_functions, save_path=None):
    if steps < 0:
        steps = int(1e6)

    for step in range(steps):
        new_states = states.copy()
        newly_adopted = []

        for node in G.nodes():
            if states[node] == 0:
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    continue

                adopted_original = sum(1 for n in neighbors if states[n] == 1)
                adopted_tampered = sum(1 for n in neighbors if states[n] == 2)
                total = len(neighbors)

                if adopted_original / total >= theta:
                    # pick a source neighbor (someone with the message)
                    source = random.choice([n for n in neighbors if states[n] == 1])
                    incoming_msg = messages[source]

                    if node in malicious_nodes:
                        new_states[node] = 2
                        messages[node] = tamper_functions[node](incoming_msg)
                    else:
                        new_states[node] = 1
                        messages[node] = incoming_msg
                    newly_adopted.append(node)

                elif adopted_tampered / total >= theta:
                    source = random.choice([n for n in neighbors if states[n] == 2])
                    incoming_msg = messages[source]

                    if node in malicious_nodes:
                        new_states[node] = 2
                        messages[node] = tamper_functions[node](incoming_msg)  # tamper even tampered
                    else:
                        new_states[node] = 2
                        messages[node] = incoming_msg
                    newly_adopted.append(node)

        if not newly_adopted:
            print(f"Stopped early at step {step}, no new adoptions.")
            break

        states = new_states
        drawGraph(G, step + 1, states, malicious_nodes, initial_seeds, messages, newly_adopted, save_path=save_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <graph_file.gml> [initial_nodes] [initial_type:random|high_degree] [theta] [malicious_percent] [save_path] [steps]")
        sys.exit(1)

    graph_file = sys.argv[1]
    initial_nodes = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 2
    initial_type = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else "random"
    theta = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else 0.3
    malicious_percent = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else 10
    save_path = sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] else None
    steps = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] else -1

    G = nx.read_gml(graph_file)
    originalMessage = "this is my secret message"

    # States: 0=inactive, 1=original, 2=tampered
    states = {n: 0 for n in G.nodes()}
    messages = {n: None for n in G.nodes()}

    # Initial gossipers
    if initial_type == "random":
        initial_gossipers = random.sample(list(G.nodes()), initial_nodes)
    elif initial_type == "high_degree":
        sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)
        initial_gossipers = [n for n, deg in sorted_nodes[:initial_nodes]]

    # Pick malicious nodes
    possible_malicious = list(set(G.nodes()) - set(initial_gossipers))
    num_malicious = max(1, len(G) * malicious_percent // 100)
    malicious_nodes = random.sample(possible_malicious, num_malicious)

    tamper_functions = {node: makeTamperFunction() for node in malicious_nodes}

    for n in initial_gossipers:
        states[n] = 1
        messages[n] = originalMessage

    drawGraph(G, 0, states, malicious_nodes, initial_gossipers, messages, save_path=save_path)
    simulateContagion(G, theta, states, malicious_nodes, initial_gossipers, steps, messages, tamper_functions, save_path=save_path)

    printSampleMessages(messages, originalMessage, k=10)
    plotSimilarityHeatmap(messages, G)
