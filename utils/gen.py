import os
import networkx as nx
import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Any, List
import numpy as np
import random
import csv


    
def generate_graph(N, M, directed = False, weight_func: Callable = lambda:None, seed = None, modulation: str = 'normal'):
    '''
    generate and return a nx.Graph or nx.DiGraph

    Parameters
    ----------
    N: number of nodes

    M: number of edges

    directed: True for directed graph, False for undirected graph

    weight_func: random weight generator, a function with no input and one single number output as weight of an edge, default return None

    seed: random seed

    modulation: one string under following options, biasing graph generation to meet requirements such as more hops, 
            this parameter has less priority than N or M, which means the effects are not guaranteed.
        'normal': randomly choose M edges for N nodes graph.
        '(s)chain': the graph consists a chain that connect all nodes at maximum, 
            'schain' (shuffled chain) means the edges in the chain may not directed to the same way, and chain does not eventually cover all nodes.
        'tree': it will first generate a tree structure in the graph.
        '(s)circle': the graph consists a circle that connect all nodes at maximum, 
            'scircle' (shuffled circle) means the edges in the circle may not directed to the same way, and circle does not eventually cover all nodes.
        'iso': try to leave isolated nodes (degree = 0) in the graph.
        'noniso': try to generate multiple connected components instead of isolated nodes to make undirectivity.
    '''
    if M < 0 or (M > N*(N-1)//2 and not directed) or (M > N*(N-1) and directed):
            raise ValueError("M must be between N-1 and N*(N-1)/2 for a simple graph.")
    
    if seed is not None:
        random.seed(seed)

    def get_potential_edges(directed: bool, nodes):
        potential_edges = []
        if directed:
            for u in nodes:
                for v in nodes:
                    if u != v:
                        potential_edges.append((u, v))
        else:
            sorted_node = sorted(nodes)
            for i in range(len(sorted_node) - 1):
                for j in range(i + 1, len(sorted_node)):
                    potential_edges.append((sorted_node[i], sorted_node[j]))

        return potential_edges

    potential_edges = get_potential_edges(directed, list(range(N)))
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(range(N))

    rand_rate = (1 - random.random()**2)

    if(modulation == 'chain'):
        for n in range(min(M, N - 1)):
            G.add_edge(n, n + 1, weight=weight_func())
            potential_edges.remove((n, n + 1))
    elif(modulation == 'schain'):
        chain_nodes = random.sample(list(range(N)), int(min(M + 1, N) * rand_rate))
        for i in range(len(chain_nodes) - 1):
            n = chain_nodes[i]
            n1 = chain_nodes[i + 1]
            if directed and random.random() < 0.5:
                G.add_edge(max(n, n1), min(n, n1), weight=weight_func())
                potential_edges.remove((max(n, n1), min(n, n1)))
            else:
                G.add_edge(min(n ,n1), max(n, n1), weight=weight_func())
                potential_edges.remove((min(n ,n1), max(n, n1)))
    elif(modulation == 'tree'):
        for n in range(min(M, N - 1)):
            n1 = random.randint(n + 1, N - 1)
            G.add_edge(n, n1, weight=weight_func())
            potential_edges.remove((n, n1))
    elif(modulation == 'circle'):
        for n in range(min(M - 1, N - 1)):
            G.add_edge(n, n + 1, weight=weight_func())
            potential_edges.remove((n, n + 1))
        if min(M - 1, N - 1) > 0:
            if directed:
                G.add_edge(min(M - 1, N - 1), 0, weight=weight_func())
                potential_edges.remove((min(M - 1, N - 1), 0))
            elif min(M - 1, N - 1) > 1:
                G.add_edge(0, min(M - 1, N - 1), weight=weight_func())
                potential_edges.remove((0, min(M - 1, N - 1)))
    elif(modulation == 'scircle'):
        circle_nodes = random.sample(list(range(N)), int(min(M, N) * rand_rate))
        if len(circle_nodes) > 2:
            for i in range(len(circle_nodes) - 1):
                n = circle_nodes[i]
                n1 = circle_nodes[i + 1]
                if directed and random.random() < 0.5:
                    G.add_edge(max(n, n1), min(n, n1), weight=weight_func())
                    potential_edges.remove((max(n, n1), min(n, n1)))
                else:
                    G.add_edge(min(n, n1), max(n, n1), weight=weight_func())
                    potential_edges.remove((min(n, n1), max(n, n1)))
            if directed and random.random() < 0.5:
                G.add_edge(max(circle_nodes[-1], circle_nodes[0]), min(circle_nodes[-1], circle_nodes[0]), weight=weight_func())
                potential_edges.remove((max(circle_nodes[-1], circle_nodes[0]), min(circle_nodes[-1], circle_nodes[0])))
            else:
                G.add_edge(min(circle_nodes[-1], circle_nodes[0]), max(circle_nodes[-1], circle_nodes[0]), weight=weight_func())
                potential_edges.remove((min(circle_nodes[-1], circle_nodes[0]), max(circle_nodes[-1], circle_nodes[0])))
    elif(modulation == 'iso'):
        noniso_nodes = random.sample(range(N), int(N * rand_rate))
        max_M = len(noniso_nodes) * (len(noniso_nodes) - 1)
        if not directed:
            max_M //= 2
        sub_G = generate_graph(len(noniso_nodes), min(max_M, M), directed)
        for edge in sub_G.edges():
            n = noniso_nodes[edge[0]]
            n1 = noniso_nodes[edge[1]]
            if directed:
                potential_edges.remove((n, n1))
            else:
                potential_edges.remove((min(n, n1), max(n, n1)))
            G.add_edge(n, n1, weight=weight_func())
    elif(modulation == 'noniso'):
        nodes1 = []
        nodes2 = []
        for n in range(N):
            if random.random() < 0.5:
                nodes1.append(n)
            else:
                nodes2.append(n)
        max_M1 = len(nodes1) * (len(nodes1) - 1)
        max_M2 = len(nodes2) * (len(nodes2) - 1)
        if not directed:
            max_M1 //= 2
            max_M2 //= 2
        sub_G1 = generate_graph(len(nodes1), min(max_M1, M // 2), directed)
        sub_G2 = generate_graph(len(nodes2), min(max_M2, M // 2), directed)
        for edge1 in sub_G1.edges():
            n = nodes1[edge1[0]]
            n1 = nodes1[edge1[1]]
            if directed:
                potential_edges.remove((n, n1))
            else:
                potential_edges.remove((min(n, n1), max(n, n1)))
            G.add_edge(n, n1, weight=weight_func())
        for edge2 in sub_G2.edges():
            n = nodes2[edge2[0]]
            n1 = nodes2[edge2[1]]
            if directed:
                potential_edges.remove((n, n1))
            else:
                potential_edges.remove((min(n, n1), max(n, n1)))
            G.add_edge(n, n1, weight=weight_func())



    while G.number_of_edges() < M:
        chosen_edge = potential_edges.pop(random.randint(0, len(potential_edges) - 1))
        G.add_edge(chosen_edge[0], chosen_edge[1], weight=weight_func())

    return G


def gen_datapool(directed = False, N_range: list = list(range(5,16)), M_num: int = 5, weight_func: Callable = lambda:None, modulation_dict: dict = {'chain':5,'tree':5,'circle':5,'normal':10}, filt_isomorphic=False, randbare_max = 50, verbose = False):
    '''
    generate and return a graph datapool

    Parameters
    ----------
    directed: True for directed graph, False for undirected graph

    N_range: list of N (number of nodes) for graphs in the datapool

    M_num: number of M (number of edges) sampled for each N

    weight_func: random weight generator, a function with no input and one single number output as weight of an edge, default return None

    modulation_dict: dictionary of {modulation: number}, interfering graph generation to ensure diversity and sufficient high hop cases. 
            Sum of values is the number of graphs generated for each N M.
        modulation: one string under following options, biasing graph generation to meet requirements such as more hops, 
                this parameter has less priority than N or M, which means the effects are not guaranteed.
            'normal': randomly choose M edges for N nodes graph.
            '(s)chain': the graph consists a chain that connect all nodes at maximum, 
            'schain' means the edges in the chain may not directed to the same way.
            'tree': it will first generate a tree structure in the graph.
            '(s)circle': the graph consists a circle that connect all nodes at maximum, 
            'scircle' means the edges in the circle may not directed to the same way.
            'iso': try to leave isolated nodes (degree = 0) in the graph.
            'noniso': try to generate multiple connected components instead of isolated nodes to make undirectivity.

    filt_isomorphic: bool, True to ensure all graphs in the datapool are isomorphic, **very costy**

    randbare_max: when filt_isomorphics, quit after randbare_max failed trials

    verbose: bool
    '''
    graph_datapool = {}
    for N in N_range:
        graph_datapool[N] = {}
        if directed:
            max_M = int(N * (N - 1))
        else:
            max_M = int(N * (N - 1) // 2)
        for M in np.linspace(1, max_M - 1, min(M_num, (max_M - 1)), dtype=int):
            graph_datapool[N][M] = []
            for modulation in modulation_dict:
                for _ in range(modulation_dict[modulation]):
                    G = generate_graph(N, M, directed=directed, weight_func=weight_func, modulation=modulation)
                    if filt_isomorphic:
                        is_isomorphic = False
                        for G1 in graph_datapool[N][M]:
                            if(nx.is_isomorphic(G, G1)):
                                is_isomorphic = True
                                break
                        if(is_isomorphic):
                            continue
                    graph_datapool[N][M].append(G)

            num_graph = sum(modulation_dict.values())
            randbare = 0
            while len(graph_datapool[N][M]) < num_graph:
                G = generate_graph(N, M, directed=directed, weight_func=weight_func)
                if filt_isomorphic:
                    is_isomorphic = False
                    for G1 in graph_datapool[N][M]:
                        if(nx.is_isomorphic(G, G1)):
                            is_isomorphic = True
                            break
                    if(is_isomorphic):
                        if(randbare >= randbare_max):
                            warnings.warn(f'N: {N}, M: {M}, not enough graph generated.')
                            break
                        randbare += 1
                        continue
                    randbare = 0
                graph_datapool[N][M].append(G)

        if verbose:
            print(f'N: {N}, cnt: {sum([len(graph_datapool[N][M]) for M in graph_datapool[N]])}, finish generation')

    return graph_datapool



def build_dataset(datapool, num_samples, filters: List[Callable[[nx.Graph], bool]], balances: List[Callable[[nx.Graph], Any]], weight: list, build_question: Callable[[nx.Graph], Any], filt_isomorphic=False, filt_isomorphic_from:list=[]):
    '''
    build and return a dataset under certain requirements

    Parameters
    ----------
    datapool: a datapool from gen_datapool()

    num_samples: size of the dataset

    filters: list of functions to filter out not needed graphs, functions take graph as input, output any True to remove, output all False to remain

    balances: list of functions, each function takes graph as input and output a customized attribute, 
        the attribute will be averagely distributed as possible in the dataset

    weight: list of weights according to attribute 'balances'

    build_question: one customized function takes one graph as input and returns information about questioning using this graph, 
        eg. pair of nodes in connectivity or shortest path tasks.

    filt_isomorphic: ensure all graphs are mutually isomorphic in this dataset

    filt_isomorphic_from: a list of graphs, ensure all grahps in this dataset and the list given are all mutually isomorphic

    Returns
    ----------
    dataset: list of graphs
    question_list: list of data used in the question, equal size with 'dataset'
    '''
    dataset = []
    question_list = []

    graph_indexer = []
    for N in datapool:
        for M in datapool[N]:
            for iG, G in enumerate(datapool[N][M]):
                if True not in [filter(G) for filter in filters]:
                    index = {'attr': []}
                    for balance in balances:
                        index['attr'].append(balance(G))
                    index['N'] = N
                    index['M'] = M
                    index['i'] = iG
                    graph_indexer.append(index)

    balance_cnt = []
    for i in range(len(balances)):
        balance_cnt.append({})
    for index in graph_indexer:
        for i, idx in enumerate(index['attr']):
            balance_cnt[i][idx] = 0

    if len(graph_indexer) < num_samples:
        warnings.warn(f'pool size after filt: {len(graph_indexer)}, num_samples: {num_samples}, not enough graph.') 

    cnt = 0
    while cnt < num_samples and len(graph_indexer) > 0:
        rank = []
        for balance_cnt_dict in balance_cnt:
            rank.append(sorted(balance_cnt_dict, key=lambda x: (balance_cnt_dict.get(x), random.random())))

        graph_choose_lambda = lambda x: sum([weight[i] * rank[i].index(x['attr'][i]) / len(rank[i]) for i in range(len(balances))])
        graph_chosen = min(graph_indexer, key=graph_choose_lambda)
        G_chosen = datapool[graph_chosen['N']][graph_chosen['M']][graph_chosen['i']]
        question_chosen = build_question(G_chosen)
        
        if filt_isomorphic:
            gis_isomorphic = False
            for G1 in (dataset + filt_isomorphic_from):
                if nx.is_isomorphic(G1, G_chosen):
                    gis_isomorphic = True
                    break
            if gis_isomorphic:
                graph_indexer.remove(graph_chosen)
                continue

        dataset.append(G_chosen)
        question_list.append(question_chosen)
        graph_indexer.remove(graph_chosen)

        for i, balance_cnt_dict in enumerate(balance_cnt):
            balance_cnt_dict[graph_chosen['attr'][i]] += 1
        
        cnt += 1

    return dataset, question_list



def cnt_datapool(graph_datapool: dict, attribute: Callable[[nx.Graph], Any]):
    '''
    count each classification in the datapool.

    Parameters
    ----------
    graph_datapool: datapool
    attribute: function input graph and output classification

    Returns
    ----------
    stat_cnt: dictionary {attribute: count}
    '''
    stat_cnt = {}
    for N_index, N in enumerate(graph_datapool):
        for M_index, M in enumerate(graph_datapool[N]):
            for G in graph_datapool[N][M]:
                attr = attribute(G)
                if attr not in stat_cnt:
                    stat_cnt[attr] = 1
                else:
                    stat_cnt[attr] += 1

    stat_cnt = dict(sorted(stat_cnt.items()))
                
    return stat_cnt

def cnt_dataset(graph_dataset: list, question_list: list, attribute: Callable[[nx.Graph, Any], Any]):
    '''
    count each classification in the dataset.

    Parameters
    ----------
    graph_dataset: dataset, list of graphs
    question_list: list of question data
    attribute: function input (graph, question data) and output classification

    Returns
    ----------
    stat_cnt: dictionary {attribute: count}
    '''
    stat_cnt = {}
    for G, question_chosen in zip(graph_dataset, question_list):
        attr = attribute(G, question_chosen)
        if attr not in stat_cnt:
            stat_cnt[attr] = 1
        else:
            stat_cnt[attr] += 1

    stat_cnt = dict(sorted(stat_cnt.items()))
                
    return stat_cnt

def plot_distribution(X_cnt: dict, xlabel: str, save_path: str = None, xtick: list = None, xticklabel: list = None, verbose=True):
    '''plot X_cnt keys at x-axis and values at y-axis'''
    plt.plot(list(X_cnt.keys()), list(X_cnt.values()))
    if xtick is None:
        plt.xticks(list(X_cnt.keys()), xticklabel)
    else:
        plt.xticks(xtick, xticklabel)
    plt.xlabel(xlabel)
    plt.ylim(bottom=0)
    if save_path is not None:
        plt.savefig(save_path)
    if verbose:
        plt.show()

def save_datapool(datapool, path):
    '''save datapool to pickle file, create path if not exist'''
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(path,'wb') as f:
        pickle.dump(datapool,f)

def save_dataset(dataset, question_list, path):
    '''save dataset to pickle file, create path if not exist'''
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(path,'wb') as f:
        pickle.dump([dataset, question_list],f)

def load_data(path):
    '''load pickle file'''
    with open(path, 'rb') as f:
        return pickle.load(f)
    

def shuffle_dataset(path=None, dataset=None, question_list=None):
    '''shuffle dataset, input could be pickle path or actual parameters, save or return the result'''
    if path is not None:
        with open(path, 'rb') as f:
            dataset, question_list = pickle.load(f)

    data_shuffle = list(zip(dataset, question_list))
    random.shuffle(data_shuffle)
    dataset, question_list = zip(*data_shuffle)
    dataset, question_list = list(dataset), list(question_list)

    if path is not None:
        with open(path, 'wb') as f:
            pickle.dump([dataset, question_list], f)
    else:
        return dataset, question_list

def path_contains_negative_edge(G, path):
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G[u][v]['weight'] < 0:
            return True
    return False

def graph_contains_negative_edge(G):
    for u, v, data in G.edges(data=True):
        if data.get('weight', 1) < 0:
            return True
    return False

def make_class(datapool, save_path, plot_path, num_samples, target_hop, connectivity_type, weigth=[1,1], filt_isomorphic=True, filt_isomorphic_from_path='', extra_filter_funcs=[], npath = False, verbose=True):
    '''
    construct and save a dataset of connectivity and shortest path task given connectivity_type

    Parameters
    ----------
    datapool: a datapool from gen_datapool()

    save_path: file path where the dataset is going to be saved, create path if not exist

    plot_path: folder path to save plotted images about statistics of the dataset, create path if not exist

    num_samples: size of the dataset

    target_hop: number of hops desired, 0 if unconnected

    connectivity_type: string of following options
        'hop(k)': k-hop connected
        'singleton': singleton or isolated, indicating in the node pair, one or both nodes have 0 degree
        'isoc': two nodes belongs to different components and they are not isolated in an undirected graph
        'disoc': two nodes belongs to different components and they are not isolated in a directed graph
        'asymmetric': in the directed graph, two nodes are connected in the undirected way but unconnected in the directed way

    weigth: weight of balancing number of nodes and number of edges in the dataset

    filt_isomorphic: ensure all graphs are mutually isomorphic in this dataset

    filt_isomorphic_from_path: path of another dataset, ensure all grahps in both datasets are all mutually isomorphic

    extra_filter_funcs: other requirements could be fufilled here using custom filter functions

    npath: bool, True to force all path contains negative weight

    verbose: bool

    Returns
    ----------
    avg_N, avg_M, avg_density: average number of nodes, edges and density of graphs in the dataset
    '''
    if verbose: print(save_path)
    TARGET_HOP = target_hop
    filt_isomorphic_from = []
    if filt_isomorphic_from_path != '':
        filt_isomorphic_from = load_data(filt_isomorphic_from_path)[0]
    

    def filter_xhop(G: nx.Graph):
        target_hop = TARGET_HOP
        if G.is_directed():
            for u in range(len(G.nodes)):
                for v in range(len(G.nodes)):
                    if u == v:
                        continue
                    try:
                        spath = nx.shortest_path(G, u, v, method='bellman-ford')
                    except nx.exception.NetworkXNoPath:
                        spath = []

                    if len(spath) - 1 == target_hop and (not npath or (path_contains_negative_edge(G, spath))):
                        return False

            return True
        
        for u in range(len(G.nodes)):
            for v in range(u + 1, len(G.nodes)):
                try:
                    spath = len(nx.shortest_path(G, u, v)) - 1
                except nx.exception.NetworkXNoPath:
                    spath = 0

                if spath == target_hop:
                    return False
        return True

    def have_iso(G: nx.Graph):
        degree_list = [G.degree(node) for node in G.nodes]
        return 0 in degree_list

    def not_have_iso(G: nx.Graph):
        degree_list = [G.degree(node) for node in G.nodes]
        return 0 not in degree_list


    def balance_N(G: nx.Graph):
        return len(G.nodes)

    def balance_M(G: nx.Graph):
        return len(G.edges)

    def build_question_xhop(G: nx.Graph):
        target_hop = TARGET_HOP
        question_options = []
        if G.is_directed():
            for u in range(len(G.nodes)):
                for v in range(len(G.nodes)):
                    if u == v:
                        continue
                    try:
                        spath = nx.shortest_path(G, u, v)
                    except nx.exception.NetworkXNoPath:
                        spath = []

                    if len(spath) - 1 == target_hop:
                        question_options.append((u, v))
        else:
            for u in range(len(G.nodes)):
                for v in range(u + 1, len(G.nodes)):
                    try:
                        spath = len(nx.shortest_path(G, u, v)) - 1
                    except nx.exception.NetworkXNoPath:
                        spath = 0

                    if spath == target_hop and (not npath or (path_contains_negative_edge(G, spath))):
                        question_options.append((u, v))
        return random.choice(question_options)

    def build_question_iso(G: nx.Graph):
        iso_node = random.choice([node for node in G.nodes if G.degree(node) == 0])
        another_node = random.choice([node for node in G.nodes if node != iso_node])
        if random.random() < 0.5:
            return (iso_node, another_node)
        else:
            return (another_node, iso_node)
        
    def build_question_noniso(G: nx.Graph):
        noniso_list = [node for node in G.nodes if G.degree(node) != 0]
        question_options = []
        for u in noniso_list:
            for v in noniso_list:
                if u == v:
                    continue
                try:
                    spath = len(nx.shortest_path(G, u, v)) - 1
                except nx.exception.NetworkXNoPath:
                    question_options.append((u, v))
        return random.choice(question_options)
        
    def filter_question_noniso(G: nx.Graph):
        noniso_list = [node for node in G.nodes if G.degree(node) != 0]
        question_options = []
        for u in noniso_list:
            for v in noniso_list:
                if u == v:
                    continue
                try:
                    spath = len(nx.shortest_path(G, u, v)) - 1
                except nx.exception.NetworkXNoPath:
                    return False
        return True

    def filter_question_compsame(G: nx.Graph):
        noniso_list = [node for node in G.nodes if G.degree(node) != 0]
        undG = G.to_undirected()
        question_options = []
        for u in noniso_list:
            for v in noniso_list:
                if u == v:
                    continue
                try:
                    spath = len(nx.shortest_path(G, u, v)) - 1
                except nx.exception.NetworkXNoPath:
                    try:
                        spath_und = len(nx.shortest_path(undG, u, v)) - 1
                        return False
                    except nx.exception.NetworkXNoPath:
                        spath_und = 0
                        
        return True

    def build_question_compsame(G: nx.Graph):
        noniso_list = [node for node in G.nodes if G.degree(node) != 0]
        undG = G.to_undirected()
        question_options = []
        for u in noniso_list:
            for v in noniso_list:
                if u == v:
                    continue
                try:
                    spath = len(nx.shortest_path(G, u, v)) - 1
                except nx.exception.NetworkXNoPath:
                    try:
                        spath_und = len(nx.shortest_path(undG, u, v)) - 1
                        question_options.append((u, v))
                    except nx.exception.NetworkXNoPath:
                        spath_und = 0
                        
        return random.choice(question_options)

    def filter_question_compdiff(G: nx.Graph):
        noniso_list = [node for node in G.nodes if G.degree(node) != 0]
        undG = G.to_undirected()
        question_options = []
        for u in noniso_list:
            for v in noniso_list:
                if u == v:
                    continue
                try:
                    spath = len(nx.shortest_path(G, u, v)) - 1
                except nx.exception.NetworkXNoPath:
                    try:
                        spath_und = len(nx.shortest_path(undG, u, v)) - 1
                    except nx.exception.NetworkXNoPath:
                        return False
                        
        return True

    def build_question_compdiff(G: nx.Graph):
        noniso_list = [node for node in G.nodes if G.degree(node) != 0]
        undG = G.to_undirected()
        question_options = []
        for u in noniso_list:
            for v in noniso_list:
                if u == v:
                    continue
                try:
                    spath = len(nx.shortest_path(G, u, v)) - 1
                except nx.exception.NetworkXNoPath:
                    try:
                        spath_und = len(nx.shortest_path(undG, u, v)) - 1
                    except nx.exception.NetworkXNoPath:
                        question_options.append((u, v))
                        
        return random.choice(question_options)
    
    def filter_posweight(G: nx.Graph):
        return not graph_contains_negative_edge(G)

    filters = extra_filter_funcs
    if npath:
        filters.append(filter_posweight)
        
    if 'hop' in connectivity_type:
        filters += [filter_xhop]
        dataset, question_list = build_dataset(datapool=datapool, num_samples=num_samples, filters=filters, balances=[balance_N, balance_M], weight=weigth, build_question=build_question_xhop, filt_isomorphic=filt_isomorphic, filt_isomorphic_from=filt_isomorphic_from)
    elif connectivity_type == 'isoc':
        filters += [filter_xhop, filter_question_noniso]
        dataset, question_list = build_dataset(datapool=datapool, num_samples=num_samples, filters=filters, balances=[balance_N, balance_M], weight=weigth, build_question=build_question_noniso, filt_isomorphic=filt_isomorphic, filt_isomorphic_from=filt_isomorphic_from)
    elif connectivity_type == 'singleton':
        filters += [not_have_iso]
        dataset, question_list = build_dataset(datapool=datapool, num_samples=num_samples, filters=filters, balances=[balance_N, balance_M], weight=weigth, build_question=build_question_iso, filt_isomorphic=filt_isomorphic, filt_isomorphic_from=filt_isomorphic_from)
    elif connectivity_type == 'asymmetric':
        filters += [filter_question_compsame]
        dataset, question_list = build_dataset(datapool=datapool, num_samples=num_samples, filters=filters, balances=[balance_N, balance_M], weight=weigth, build_question=build_question_compsame, filt_isomorphic=filt_isomorphic, filt_isomorphic_from=filt_isomorphic_from)
    elif connectivity_type == 'disoc':
        filters += [filter_question_compdiff]
        dataset, question_list = build_dataset(datapool=datapool, num_samples=num_samples, filters=filters, balances=[balance_N, balance_M], weight=weigth, build_question=build_question_compdiff, filt_isomorphic=filt_isomorphic, filt_isomorphic_from=filt_isomorphic_from)

    dataset, question_list = shuffle_dataset(dataset=dataset, question_list=question_list)
    
    avg_N, avg_M, avg_density = stat_class(dataset, question_list, plot_path, verbose)

    save_dataset(dataset, question_list, save_path)
    return avg_N, avg_M, avg_density


def stat_class(dataset, question_list, plot_path, verbose=True):
    '''
    do statistic on dataset, save plot and return the average number of nodes, edges and density
    '''
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    
    def attribute_N(G: nx.Graph, question_chosen):
        return len(G.nodes)
    stat_cnt = cnt_dataset(dataset, question_list, attribute=attribute_N)
    plot_distribution(stat_cnt, xlabel='number of nodes', save_path=f'{plot_path}/N.png', verbose=verbose)
    avg_N = sum([k * stat_cnt[k] for k in stat_cnt]) / sum([stat_cnt[k] for k in stat_cnt])

    def attribute_M(G: nx.Graph, question_chosen):
        return len(G.edges)
    stat_cnt = cnt_dataset(dataset, question_list, attribute=attribute_M)
    plot_distribution(stat_cnt, xlabel='number of edges', xtick=list(range(0, 50, 5)), save_path=f'{plot_path}/M.png', verbose=verbose)
    avg_M = sum([k * stat_cnt[k] for k in stat_cnt]) / sum([stat_cnt[k] for k in stat_cnt])

    def attribute_density(G: nx.Graph, question_chosen):
        N = len(G.nodes)
        M = len(G.edges)
        if G.is_directed():
            density = M / (N * (N - 1))
        else:
            density = M / (N * (N - 1) // 2)
        return density
    stat_cnt = cnt_dataset(dataset, question_list, attribute=attribute_density)
    plot_distribution(stat_cnt, xlabel='density', xtick=np.linspace(0,1,11), save_path=f'{plot_path}/density.png', verbose=verbose)
    avg_density = sum([k * stat_cnt[k] for k in stat_cnt]) / sum([stat_cnt[k] for k in stat_cnt])

    if verbose:
        for G, pair in zip(dataset, question_list):
            print(G.nodes)
            print(G.edges)
            print(pair)
            print()

    with open(f'{plot_path}/avg.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['density', 'edge num', 'node num'])
        writer.writerow([avg_density, avg_M, avg_N])

    return avg_N, avg_M, avg_density


