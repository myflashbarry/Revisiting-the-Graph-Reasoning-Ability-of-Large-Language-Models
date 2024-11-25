import random

def _graph_describe_incident(graph,shuffle='',rename=False,start_node=0,dicts=None):
    if graph.is_directed():
        directed_str = 'a directed'
        connect_str = 'directed'
    else:
        directed_str = 'an undirected'
        connect_str = 'connected'
    strings=f'G describes {directed_str} graph among '
    nodes=graph.nodes
    if dicts==None:
        dicts={}
        for i in nodes:
            dicts[i]=i
    degrees=list(graph.degree())
    sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
    name_dicts={}
    for idx,i in enumerate(sorted_nodes):
        name_dicts[i[0]]=idx
    # print(name_dicts)
    # print(graph.nodes)
    nodes_list=list(graph.nodes)
    for idx,i in enumerate(range(len(nodes))):
        if idx==len(nodes)-1:
            strings+='and '+str(dicts[nodes_list[idx]+start_node])+'.\n'
        else:
            # print(nodes_list,idx)
            strings+=str(dicts[nodes_list[idx]+start_node])+', '
    strings+='In this graph:\n'
    edge_dicts={}
    for e in graph.edges(data=True):
        i0, i1 = 0, 1
        do_shuffle = (('shuffle' in shuffle) and random.random() < 0.5)
        if not graph.is_directed() and do_shuffle:
            i0, i1 = 1, 0
        if e[i0] not in edge_dicts:
            edge_dicts[e[i0]]=[]
        edge_dicts[e[i0]].append((e[i1], e[2]))
    edge_list=[]

    if 'sort' in shuffle:
        edge_dicts = dict(sorted(edge_dicts.items()))

        
    for key in edge_dicts.keys():
        edge_sentence=''
        edge_sentence+='Node '+str(dicts[key+start_node])+' is ' + connect_str + ' to nodes '

        for i in range(len(edge_dicts[key])):
            edge_sentence+=str(dicts[edge_dicts[key][i][0]+start_node])
            if 'weight' in edge_dicts[key][i][1] and edge_dicts[key][i][1]['weight'] is not None:
                edge_sentence += f" (weight: {edge_dicts[key][i][1]['weight']})"
            if i==len(edge_dicts[key])-1:
                edge_sentence+='.'
            else:
                edge_sentence+=', '
            
        edge_list.append(edge_sentence)
    
    if 'conswap' in shuffle or 'shuffle' in shuffle:
        random.shuffle(edge_list)

    for i in range(len(edge_list)):
        strings+=edge_list[i]

        strings+='\n'
    return strings


def _graph_describe_incident_v2(graph,shuffle='',use_degree=False,rename=False,start_node=0,dicts=None):
    assert not graph.is_directed()
    strings='G describes an undirected graph among '
    nodes=graph.nodes
    if dicts==None:
        dicts={}
        for i in nodes:
            dicts[i]=i
    degrees=list(graph.degree())
    sorted_nodes = sorted(degrees, key=lambda x: x[1], reverse=True)
    name_dicts={}
    for idx,i in enumerate(sorted_nodes):
        name_dicts[i[0]]=idx
    nodes_list=list(graph.nodes)
    if use_degree:
        for idx in range(len(sorted_nodes)-1,-1,-1):
            if rename:
                n=name_dicts[sorted_nodes[idx][0]]
            else:
                n=sorted_nodes[idx][0]
            if idx==len(nodes)-1:
                strings+='and '+str(dicts[n+start_node])+'.\n'
            else:
                strings+=str(dicts[n+start_node])+', '
    else:
        for idx,i in enumerate(range(len(nodes))):
            if idx==len(nodes)-1:
                strings+='and '+str(dicts[nodes_list[idx]+start_node])+'.\n'
            else:
                # print(nodes_list,idx)
                strings+=str(dicts[nodes_list[idx]+start_node])+', '
    strings+='In this graph:\n'
    edge_dicts={}
    for e in graph.edges(data=True):
        if e[0] not in edge_dicts:
            edge_dicts[e[0]]=[]
        if e[1] not in edge_dicts:
            edge_dicts[e[1]]=[]
        edge_dicts[e[0]].append((e[1], e[2]))
        edge_dicts[e[1]].append((e[0], e[2]))
    edge_list=[]
    edge_dicts = {key: edge_dicts[key] for key in sorted(edge_dicts)}



    for key in edge_dicts.keys():
        edge_sentence=''
        edge_sentence+='Node '+str(dicts[key+start_node])+' is connected to nodes '

        for i in range(len(edge_dicts[key])):
            edge_sentence+=str(dicts[edge_dicts[key][i][0]+start_node])
            if 'weight' in edge_dicts[key][i][1] and edge_dicts[key][i][1]['weight'] is not None:
                edge_sentence += f" (weight: {edge_dicts[key][i][1]['weight']})"
            if i==len(edge_dicts[key])-1:
                edge_sentence+='.'
            else:
                edge_sentence+=', '

        edge_list.append(edge_sentence)

    if shuffle:
        random.shuffle(edge_list)

    for i in range(len(edge_list)):
        strings+=edge_list[i]
        strings+='\n'
        # if i ==len(edge_list)-1:
        #     for _ in range(5):
        #         strings+=edge_list[i]+'\n'

    return strings

def _graph_discribe_adj_n(graph,shuffle=False,dicts=None):
    begins=f"G describes {'a directed' if graph.is_directed() else 'an undirected'} graph among node "
    nodes=graph.nodes
    # print(nodes)
    if dicts==None or len(dicts)==0:
        dicts={}
        for i in nodes:
            dicts[i]=i
    for idx,i in enumerate(range(len(nodes))):
        if idx==len(nodes)-1:
            begins+='and '+str(dicts[i]) +'.\n'# +'.The edges in G are: '
        else:
            begins+=str(dicts[i])+', '
    edges=graph.edges
    edge_list=[]
    for idx,e in enumerate(edges):
        if (not graph.is_directed()) and 'shuffle' in shuffle and random.random() < 0.5:
            edge_list.append((e[1], e[0]))
        else:
            edge_list.append(e)
    data_list=[]
    for idx,e in enumerate(graph.edges(data=True)):
        data_list.append(e[2])
    if 'shuffle' in shuffle:
        zipped = list(zip(edge_list, data_list))
        random.shuffle(zipped)
        edge_list, data_list = zip(*zipped)
        edge_list = list(edge_list)
        data_list = list(data_list)
    for e, data in zip(edge_list, data_list):
        weight_str = ''
        if 'weight' in data and data['weight'] is not None:
            weight_str = f" with weight {data['weight']}"
        begins+='Node '+str(dicts[e[0]])+f" is {'directed' if graph.is_directed() else 'connected'} to Node "+str(dicts[e[1]])+weight_str+'.\n'
    # begins+=str(edges)
    return begins


def graph_txt(G, method, seed=None, shuffle='', dicts=None):
    if seed is not None:
        random.seed(seed)

    if method=='node':
        if G.is_directed():
            return _graph_describe_incident(G,shuffle=shuffle,dicts=dicts)
        else:
            return _graph_describe_incident_v2(G,shuffle=shuffle,dicts=dicts)
    elif method == 'edge':
        return _graph_discribe_adj_n(G,shuffle=shuffle,dicts=dicts)

    