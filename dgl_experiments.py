import os
import dgl
import numpy as np
import torch
import networkx
import matplotlib.pyplot as plt

def visualize_graph(g, nodeLabel, edgeLabel, savefile):
    plt.figure()
    G = g.to_networkx(node_attrs=nodeLabel.split(), edge_attrs=edgeLabel.split())
    pos = networkx.spring_layout(G)
    
    # draw nodes
    networkx.draw(G, pos, edge_color="grey", node_size=500, with_labels=True) 
    node_data = networkx.get_node_attributes(G, nodeLabel)
    '''
    node_labels = {index:"N:"+ str(data) for index,data in enumerate(node_data)}  
    pos_higher = {}
    for k, v in pos.items():  
        if(v[1]>0):
            pos_higher[k] = (v[0]-0.04, v[1]+0.04)
        else:
            pos_higher[k] = (v[0]-0.04, v[1]-0.04)
    networkx.draw_networkx_labels(G, pos_higher, labels=node_labels,font_color="brown", font_size=12)
    '''
    # draw edges
    edge_labels = networkx.get_edge_attributes(G, edgeLabel)
    edge_labels= {(key[0],key[1]): str(edge_labels[key].item()) for key in edge_labels}
    networkx.draw_networkx_edges(G, pos, alpha=0.5)
    networkx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12) 
    
    # save
    plt.savefig(savefile)


# difine a new graph
# 7 nodes, 7 edges
g = dgl.graph((torch.tensor([0,1,1,2,4,0,2]), torch.tensor([1,3,5,0,3,6,4])), num_nodes=7)
g.ndata["node"] = torch.tensor([0,1,2,3,4,5,6])
g.edata["edge"] = torch.tensor([0,1,2,3,4,5,6])

# subgraph: node 0, 2, 3, 4
sg1 = g.subgraph([0, 2, 3, 4])

# subgraph: edge 1, 2, 3, 4, 6
sg2 = g.edge_subgraph([1, 2, 3, 4, 6])

# visualization
if not os.path.exists('./visual'):
    os.mkdir('./visual')
    
visualize_graph(g, "node", "edge", './visual/grade.png')
visualize_graph(sg1, "node", "edge", './visual/subgrade1.png')
visualize_graph(sg2, "node", "edge", './visual/subgrade2.png')


dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]

# degree
out_degrees = []
in_degrees = []
for i in range(graph.num_nodes()):
    out_degrees.append(graph.out_degrees(i))
    in_degrees.append(graph.in_degrees(i))

max_d = max(max(out_degrees), max(in_degrees))

plt.figure()
plt.hist(out_degrees, bins=max_d, facecolor='blue', edgecolor='black', alpha=1)
plt.xlabel('degrees')
plt.ylabel('number')
plt.savefig('./visual/out_degrees.png')

plt.figure()
plt.hist(in_degrees, bins=max_d, facecolor='blue', edgecolor='black', alpha=1)
plt.xlabel('degrees')
plt.ylabel('number')
plt.savefig('./visual/in_degrees.png')


