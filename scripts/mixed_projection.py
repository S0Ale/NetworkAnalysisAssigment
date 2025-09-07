import networkx as nx
from itertools import combinations

G = nx.read_gml("moreno_graph.gml")
leftNodes = {n for n, d in G.nodes(data=True) if d["bipartite"] == 0}

P = nx.Graph()

for crime in (n for n, d in G.nodes(data=True) if d["bipartite"] == 1):
    people_in_crime = list(G.neighbors(crime))
    
    for i, u in enumerate(people_in_crime):
        for v in people_in_crime[i+1:]:
            left, right = (u, v) if u < v else (v, u)

            if not P.has_node(left):
                P.add_node(left, name=G.nodes[left].get("name"), sex=G.nodes[left].get("sex"))
            if not P.has_node(right):
                P.add_node(right, name=G.nodes[right].get("name"), sex=G.nodes[right].get("sex"))
            
            def normalize_role(role_str):
                roles = role_str.strip().split()
                if len(roles) > 1:
                    return "/".join(sorted(roles))
                return roles[0]
            
            leftRole = normalize_role(G[left][crime]["role"])
            rightRole = normalize_role(G[right][crime]["role"])
            
            if not P.has_edge(left, right):
                P.add_edge(left, right, leftRole=[], rightRole=[], crimes=[])
            
            P[left][right]["leftRole"].append(leftRole)
            P[left][right]["rightRole"].append(rightRole)
            P[left][right]["crimes"].append(crime)


print("- Projection with roles ready.")

print("Saving graph...")
nx.write_gml(P, "./mixed_projection.gml")
print("- Graph saved.")