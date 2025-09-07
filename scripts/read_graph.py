import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite

print("Reading graph...")
with open("out.moreno_crime_crime", "r") as f:
    edge_lines = f.readlines()[2:]

with open("rel.moreno_crime_crime.person.role", "r") as f:
    roles = [line.strip() for line in f.readlines()]

with open("ent.moreno_crime_crime.person.name", "r") as f:
    names = [line.strip() for line in f.readlines()]

with open("ent.moreno_crime_crime.person.sex", "r") as f:
    sexes = [line.strip() for line in f.readlines()]

print("- Graph read successfully")

print("Creating graph...")
G = nx.Graph()
for idx, line in enumerate(edge_lines):
    personId, crimeId = map(int, line.strip().split())
    if not G.has_node(f"p{personId}"):
        G.add_node(f"p{personId}", bipartite=0, name=names[personId-1], sex=sexes[personId-1])
    if not G.has_node(f"c{crimeId}"):
        G.add_node(f"c{crimeId}", bipartite=1)
    G.add_edge(f"p{personId}", f"c{crimeId}", role=roles[idx])

print("- Graph created successfully")

print("Saving graph...")
nx.write_gml(G, "./moreno_graph.gml")
print("- Graph saved.")

