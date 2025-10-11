import matplotlib.pyplot as plt
import networkx as nx

# cria grafo direcionado
G = nx.DiGraph()

# define nós e conexões
edges = [
    ('Create New Feature', 'Normalize City'),
    ('Normalize City', 'Encoding Categories'),
    ('Encoding Categories', 'End')
]
G.add_edges_from(edges)

# define layout
pos = nx.spring_layout(G, seed=42)

# cores e tamanhos personalizados
node_colors = ['#6EC1E4' if n != 'End' else '#F9A825' for n in G.nodes()]
node_sizes = [2200 if n != 'End' else 2600 for n in G.nodes()]

# desenha grafo
plt.figure(figsize=(8, 5))
nx.draw(
    G, pos,
    with_labels=True,
    node_color=node_colors,
    node_size=node_sizes,
    arrowsize=20,
    edgecolors='black',
    font_size=10,
    font_weight='bold'
)

plt.title('Feature Engineering Logic', fontsize=14, fontweight='bold', pad=20)
plt.axis('off')
plt.show()
