import pandas as pd
import networkx as nx
from stellargraph import StellarGraph as sg

def create_nx_graph(graph_DF,test_DF):
    nx_graphs = []
    nx_test = []
    for i in range(len(graph_DF)):
        nx_graphs.append(nx.Graph(graph_DF['A'][i]))
    

    for i in range(len(test_DF)):
        nx_test.append(nx.Graph(test_DF['A'][i]))
    
    for i in range(len(nx_graphs)):
        nx_graphs[i].graph['nz'] = graph_DF['nz'][i]
        nx_graphs[i].graph['np'] = graph_DF['np'][i]
        #nx_graphs[i].graph['performance'] = graph_DF['Labels'][i]
    
    for i in range(len(nx_test)):
        nx_test[i].graph['nz'] = graph_DF['nz'][i]
        nx_test[i].graph['np'] = graph_DF['np'][i]
        #nx_test[i].graph['performance'] = graph_DF['Labels'][i]
    
    for i in range(len(nx_graphs)):
        for j in range(nx_graphs[i].number_of_nodes()):
            nx_graphs[i].nodes[j]['feature'] = graph_DF['NL3'][i][j]

    for i in range(len(nx_test)):
        for j in range(nx_test[i].number_of_nodes()):
            nx_test[i].nodes[j]['feature'] = test_DF['NL3'][i][j]
        
    return nx_graphs, nx_test

def create_stellargraphs(nx_graphs,nx_test):
    sg_graphs = []
    for i in range(len(nx_graphs)):
        sg_graphs.append(sg.from_networkx(nx_graphs[i], node_features='feature'))

    sg_test = []
    for i in range(len(nx_test)):
        sg_test.append(sg.from_networkx(nx_test[i], node_features='feature'))
        
    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in sg_graphs],
        columns=["nodes", "edges"],
    )
    print(summary.describe().round(1))

    test_summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_nodes()) for g in sg_test],
        columns=['nodes', 'edges'],
    )
    print(test_summary.describe().round(1))
    
    return sg_graphs, sg_test

def create_graph_binary_labels(graph_DF):
    graph_labels = graph_DF['Labels']
    
    binary = []
    for i in graph_labels:
        if i < graph_labels.median():
            binary.append(1)
        else:
            binary.append(0)
    graph_DF['bin2'] = binary
    graph_labels = graph_DF['bin2']
    
    return graph_labels

def create_performance_labels(graph_DF):
    graph_labels = graph_DF['Labels']
    
    return graph_labels