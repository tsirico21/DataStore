import pandas as pd
from scipy.io import loadmat

def load_data(path):
    path = ('data4.mat')

    raw_data = loadmat(path, squeeze_me=True)
    graph_data = raw_data['Graphs']
    test_data = raw_data['test']



    graph_DF = pd.DataFrame()
    graph_DF['A'] = graph_data['A']
    graph_DF['Labels'] = graph_data['Labels']
    graph_DF['Binary_Lbls'] = graph_data['Binary']
    graph_DF['NL'] = graph_data['nl']
    graph_DF['NL2'] = graph_data['Ln2']
    graph_DF['nz'] = graph_data['nz']
    graph_DF['np'] = graph_data['np']


    test_DF = pd.DataFrame()
    test_DF['A'] = test_data['A']
    #test_DF['Labels'] = test_data['Labels']
    #test_DF['Binary_Lbls'] = test_data['Binary']
    test_DF['NL'] = test_data['nl']
    test_DF['NL2'] = test_data['Ln2']
    test_DF['nz'] = test_data['nz']
    test_DF['np'] = test_data['np']
    
    for i in range(len(graph_DF)):
        for j in range(len(graph_DF['NL2'][i])):
            graph_DF['NL2'][i][j] = ord(graph_DF['NL2'][i][j])
            
    for i in range(len(graph_DF)):
        graph_DF['NL'][i] = dict(enumerate(graph_DF['NL'][i]))
        graph_DF['NL2'][i] = dict(enumerate(graph_DF['NL2'][i]))
        
    for i in range(len(test_DF)):        
        test_DF['NL'][i] = dict(enumerate(test_DF['NL'][i]))
        test_DF['NL2'][i] = dict(enumerate(test_DF['NL2'][i]))
        
    for i in range(len(test_DF)):
        for j in range(len(test_DF['NL2'][i])):
            test_DF['NL2'][i][j] = ord(test_DF['NL2'][i][j])
    NL4 = []
    for nl1, nl2 in zip(graph_DF['NL'], graph_DF['NL2']):
        nl3 = {k: [nl1[k], nl2[k]] for k in nl1.keys()}
        NL4.append(nl3)
    
    NL4_test = []
    for nl1_test, nl2_test in zip(test_DF['NL'], test_DF['NL2']):
        nl3_test = {k:[nl1_test[k], nl2_test[k]] for k in nl1_test.keys()}
        NL4_test.append(nl3_test)

    graph_DF['NL3'] = NL4
    test_DF['NL3'] = NL4_test
    
    return graph_DF, test_DF