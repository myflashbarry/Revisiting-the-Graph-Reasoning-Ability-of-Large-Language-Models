import pickle
import string
import random
import networkx as nx
from utils.graph_nlp import graph_txt
import pickle

def degree_analysis(G):
    d=[]
    for g in G.degree:
        d.append(g[1])
    return sum(d)/len(d),max(d),min(d)

def random_string(length):
    # Generates a random string of a given length
    letters = string.ascii_uppercase  # This contains 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    result = ''.join(random.choice(letters) for _ in range(length))
    return result

def get_single_data(folder_name,dataset_name,describe_method,prompt_method,prompt_path,seed=None,shuffle=''):
    if seed is not None:
        random.seed(seed)

    dataset_path = f'dataset/{folder_name}/{dataset_name}.pkl'
    with open(dataset_path,'rb') as f:
        dataset,nodes_selected_list=pickle.load(f)

        input_text=[]
        ans=[]
        for idx,(G,nodes) in enumerate(zip(dataset,nodes_selected_list)):
            rename_dict = _gen_rename_dict(G, shuffle)
            selected_nodes = _shuffle_selected_nodes(G, nodes)
            if prompt_path:
                with open(prompt_path,'r') as f:
                    prompt_text=f.read()
            else:
                prompt_text = ''
            input_string, ans_bool = get_question_txt(G, selected_nodes, describe_method, prompt_method, shuffle, rename_dict, prompt_text)
            input_text.append(input_string)
            ans.append(ans_bool)
        return input_text,ans
    
def get_shot(folder_name,dataset_name,describe_method,prompt_method,seed=None,rename=''):
    '''
    generate and return shot prompt for fewshot and cot for entire dataset

    Parameters
    ----------
    folder_name: name of folder of dataset

    dataset_name: name of dataset used for generate shot, dataset_path = f'dataset/{folder_name}/{dataset_name}.pkl'

    describe_method: 'node', 'edge'

    prompt_method: zero-shot version of prompting method, 'zero' for fewshot prompt, 'zerocot' for 'cot' prompt, 
    
    seed: random seed

    rename: '', 'renamei', 'renamec', node representation rename. 
        '': ordered ID start from 0, 'renamei': random ID from random.randrange(0,100000), 'renamec': random characters of 5 uppercase letters

    Returns
    ----------
    input_text: list of strings, text of questions from every graph in the dataset

    ans: list of strings, shot answers matching every questions
    '''
    assert 'zero' in prompt_method
    if seed is not None:
        random.seed(seed)

    dataset_path = f'dataset/{folder_name}/{dataset_name}.pkl'
    with open(dataset_path,'rb') as f:
        dataset,nodes_selected_list=pickle.load(f)

        input_text=[]
        ans=[]
        for idx,(G,nodes) in enumerate(zip(dataset,nodes_selected_list)):
            rename_dict = _gen_rename_dict(G, rename)
            selected_nodes = _shuffle_selected_nodes(G, nodes)
            input_string, ans_bool = get_question_txt(G, selected_nodes, describe_method, prompt_method, rename, rename_dict)
            if prompt_method == 'zero':
                ans_string, _ = get_ans_txt(G, selected_nodes, rename_dict)
            elif prompt_method == 'zerocot':
                ans_string, _ = get_ans_path_txt(G, selected_nodes, rename_dict)
            input_text.append(input_string)
            ans.append(ans_string)
        return input_text,ans
    
def _random_string(length):
    # Generates a random string of a given length
    letters = string.ascii_uppercase  # This contains 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    result = ''.join(random.choice(letters) for _ in range(length))
    return result

def _gen_rename_dict(G, shuffle):
    dicts={}
    if 'renamei' in shuffle:
        for i in G.nodes:
            rand_value = random.randrange(0,100000)
            while rand_value in dicts.values():
                rand_value = random.randrange(0,100000)
            dicts[i] = rand_value
    elif 'renamec' in shuffle:
        for i in G.nodes:
            rand_value = _random_string(5)
            while rand_value in dicts.values():
                rand_value = _random_string(5)
            dicts[i] = rand_value
    else:
        for i in G.nodes:
            dicts[i] = i
    return dicts

def _shuffle_selected_nodes(G, nodes):
    if not G.is_directed() and random.random() < 0.5:
        selected_nodes = [nodes[1], nodes[0]]
    else:
        selected_nodes = nodes
    return selected_nodes

def get_question_txt(G, selected_nodes, describe_method, prompt_method, shuffle, dicts, prompt_text=''):
    '''
    prompt_method: 'zero', 'zerocot', 'cot', 'fewshot', 'zerov2', 'zeropath', 'cotpath'
    '''
    if 'selfloop' in shuffle:
        if G.degree(selected_nodes[0]) == 0:
            G.add_edge(selected_nodes[0], selected_nodes[0])
        if G.degree(selected_nodes[1]) == 0:
            G.add_edge(selected_nodes[1], selected_nodes[1])
    elif 'fullloop' in shuffle:
        for node in G.nodes:
            G.add_edge(node, node)
            
    string=graph_txt(G,describe_method,shuffle=shuffle,dicts=dicts)
    ans_bool = nx.has_path(G, selected_nodes[0], selected_nodes[1])

    
    if prompt_method=='zero':
        if G.is_directed():
            input_string="Given a directed graph: " + string+' Is there a directed path from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ ' Choose from the following answers ["Yes","No"] in the form of "The answer is Yes" or "The answer is No".'
        else:
            input_string="Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Is there a path between Node ' +str(dicts[selected_nodes[0]]) +' and '+ str(dicts[selected_nodes[1]])+ ' Choose from the following answers ["Yes","No"] in the form of "The answer is Yes" or "The answer is No".'
    elif prompt_method=='zerocot':
        if G.is_directed():
            input_string="Given a directed graph: " + string+' Is there a directed path from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ ' Choose from the following answers ["Yes","No"] in the form of "The answer is Yes" or "The answer is No". Let\'s think step by step.'
        else:
            input_string="Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Is there a path between Node ' +str(dicts[selected_nodes[0]])+' and '+ str(dicts[selected_nodes[1]])+ ' Choose from the following answers ["Yes","No"] in the form of "The answer is Yes" or "The answer is No". Let\'s think step by step.'
    elif prompt_method=='cot':
        if G.is_directed():
            input_string=prompt_text+ "\n\nQ: Given a directed graph: " + string+' Is there a directed path from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ ' Choose from the following answers ["Yes","No"] in the form of "The answer is Yes" or "The answer is No". Let\'s think step by step.\nA:'
        else:
            input_string=prompt_text+"\n\nQ: Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Is there a path between Node ' +str(dicts[selected_nodes[0]])+' and '+ str(dicts[selected_nodes[1]])+ ' Choose from the following answers ["Yes","No"] in the form of "The answer is Yes" or "The answer is No". Let\'s think step by step.\nA:'
    
    elif prompt_method=='fewshot':
        if G.is_directed():
            input_string=prompt_text+ "\n\nQ: Given a directed graph: " + string+' Is there a directed path from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ ' Choose from the following answers ["Yes","No"] in the form of "The answer is Yes" or "The answer is No".\nA:'
        else:
            input_string=prompt_text+"\n\nQ: Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Is there a path between Node ' +str(dicts[selected_nodes[0]])+' and '+ str(dicts[selected_nodes[1]])+ ' Choose from the following answers ["Yes","No"] in the form of "The answer is Yes" or "The answer is No".\nA:'
    elif prompt_method=='zerov2':
        if not G.is_directed():
            input_string="Determine if there is a path between two nodes in the graph. " + string+'\nQ: Is there a path between Node ' +str(dicts[selected_nodes[0]])+' and '+ str(dicts[selected_nodes[1]])+ '? Choose from the following answers ["Yes","No"] in the form of "The answer is Yes" or "The answer is No".\nA:'
    elif prompt_method=='zeropath':
        if G.is_directed():
            input_string="Given a directed graph: " + string+' Is there a directed path from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ ' If the path exists, give the path in the form of "Node #1 -> Node #2". Otherwise, give "No path"'
        else:
            input_string="Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Is there a path between Node ' +str(dicts[selected_nodes[0]]) +' and '+ str(dicts[selected_nodes[1]])+ ' If the path exists, give the path in the form of "Node #1 -> Node #2". Otherwise, give "No path"'
    elif prompt_method=='cotpath':
        if G.is_directed():
            input_string=prompt_text+"\n\nQ: Given a directed graph: " + string+' Is there a directed path from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ ' If the path exists, give the path in the form of "Node #1 -> Node #2". Otherwise, give "No path".\nA:'
        else:
            input_string=prompt_text+"\n\nQ: Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Is there a path between Node ' +str(dicts[selected_nodes[0]]) +' and '+ str(dicts[selected_nodes[1]])+ ' If the path exists, give the path in the form of "Node #1 -> Node #2". Otherwise, give "No path".\nA:'
    else:
        raise Exception(f'prompt_method: {prompt_method} not implemented')

    return input_string, ans_bool

def get_ans_txt(G, selected_nodes, dicts):
    ans_bool = nx.has_path(G, selected_nodes[0], selected_nodes[1])
    ans_string = 'The answer is Yes.' if ans_bool else 'The answer is No.'
    return ans_string, ans_bool

def get_ans_path_txt(G, selected_nodes, dicts):
    try:
        ans_path = nx.shortest_path(G, selected_nodes[0], selected_nodes[1])
        ans_string = 'Exist path: '
        for n in ans_path:
            ans_string += f'{dicts[n]} -> '
        ans_string = ans_string[:-4] + '.'
    except:
        reason_text = ''
        if G.degree(selected_nodes[0]) == 0:
            reason_text = f'Node {dicts[selected_nodes[0]]} is an isolated node. '
        elif G.degree(selected_nodes[1]) == 0:
            reason_text = f'Node {dicts[selected_nodes[1]]} is an isolated node. '
        elif not G.is_directed():
            for component in nx.connected_components(G):
                if selected_nodes[0] in component:
                    component1 = component
                if selected_nodes[1] in component:
                    component2 = component
            for node, component in [(selected_nodes[0], component1), (selected_nodes[1], component2)]:
                reason_text += f'Node {dicts[node]} is in the connected block consisted of '
                for i, n in enumerate(component):
                    if i != len(component) - 1:
                        reason_text += f'node {dicts[n]}, '
                    else:
                        reason_text = reason_text[:-2] + f' and node {dicts[n]}. '
        ans_string = reason_text + f'There is no path between node {dicts[selected_nodes[0]]} and node {dicts[selected_nodes[1]]}. No path'
        ans_path = []
    return ans_string, ans_path

