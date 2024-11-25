import pickle
import string
import random
import networkx as nx
import pickle
import numpy as np


from utils.graph_nlp import graph_txt
from task_sp.utils.eval import graph_has_weight



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
            input_string, ans_single = get_question_txt(G, selected_nodes, describe_method, prompt_method, shuffle, rename_dict, prompt_text)
            input_text.append(input_string)
            ans.append(ans_single)
        return input_text,ans
    
def get_shot(folder_name,dataset_name,describe_method,prompt_method,seed=None,rename=''):
    '''
    generate and return shot prompt for fewshot and cot for entire dataset

    Parameters
    ----------
    folder_name: name of folder of dataset

    dataset_name: name of dataset used for generate shot, dataset_path = f'dataset/{folder_name}/{dataset_name}.pkl'

    describe_method: 'node', 'edge'

    prompt_method: zero-shot version of prompting method, 'zero' for fewshot prompt, 'zerocot<algorithm>' for 'cot<algorithm>' prompt, 
        <algorithm> includes 'dijkstra', 'bf', 'fw'
    
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
            input_string, ans_single = get_question_txt(G, selected_nodes, describe_method, prompt_method, rename, rename_dict)
            if prompt_method == 'zero':
                ans_string, ans_single1 = get_ans_txt(G, selected_nodes, rename_dict)
            elif prompt_method == 'zerocotdijkstra':
                ans_string, ans_single1 = get_dijkstra_txt(G, selected_nodes, rename_dict)
            elif prompt_method == 'zerocotbf':
                ans_string, ans_single1 = get_bf_txt(G, selected_nodes, rename_dict)
            elif prompt_method == 'zerocotfw':
                ans_string, ans_single1 = get_fw_txt(G, selected_nodes, rename_dict)
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
    prompt_method: 'zero', 'zerocot', 'zerocotdijkstra', 'zerocotbf', 'zerocotfw', 'cotdijkstra', 'cotbf', 'cotfw', 'fewshot', 'fewshotcot', 'zerov2'
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
    try:
        weight_attr = 'weight' if graph_has_weight(G) else None
        ans_single = nx.shortest_path(G, selected_nodes[0], selected_nodes[1], weight_attr, method='bellman-ford')
    except:
        ans_single = []

    
    if prompt_method=='zero':
        if G.is_directed():
            input_string="Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path."'
        else:
            input_string="Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path."'
    elif prompt_method=='zerocot':
        if G.is_directed():
            input_string="Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step.'
        else:
            input_string="Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step.'
    elif prompt_method=='zerocotdijkstra':
        if G.is_directed():
            input_string="Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Dijkstra\'s Algorithm.'
        else:
            input_string="Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Dijkstra\'s Algorithm.'
        if not graph_has_weight(G):
            input_string = input_string[:-1] + ', all edges have weight 1.'
    elif prompt_method=='zerocotbf':
        if G.is_directed():
            input_string="Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Bellman-Ford Algorithm.'
        else:
            input_string="Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Bellman-Ford Algorithm.'
        if not graph_has_weight(G):
            input_string = input_string[:-1] + ', all edges have weight 1.'
    elif prompt_method=='zerocotfw':
        if G.is_directed():
            input_string="Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Floyd-Warshall Algorithm.'
        else:
            input_string="Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Floyd-Warshall Algorithm.'
        if not graph_has_weight(G):
            input_string = input_string[:-1] + ', all edges have weight 1.'
    elif prompt_method=='cotdijkstra':
        if G.is_directed():
            input_string=prompt_text+ "\n\nQ: Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Dijkstra\'s Algorithm.\nA:'
        else:
            input_string=prompt_text+"\n\nQ: Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Dijkstra\'s Algorithm.\nA:'
        if not graph_has_weight(G):
            input_string = input_string[:-4] + ', all edges have weight 1.\nA:'
    elif prompt_method=='cotbf':
        if G.is_directed():
            input_string=prompt_text+ "\n\nQ: Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Bellman-Ford Algorithm.\nA:'
        else:
            input_string=prompt_text+"\n\nQ: Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Bellman-Ford Algorithm.\nA:'
        if not graph_has_weight(G):
            input_string = input_string[:-4] + ', all edges have weight 1.\nA:'
    elif prompt_method=='cotfw':
        if G.is_directed():
            input_string=prompt_text+ "\n\nQ: Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Floyd-Warshall Algorithm.\nA:'
        else:
            input_string=prompt_text+"\n\nQ: Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step using Floyd-Warshall Algorithm.\nA:'
        if not graph_has_weight(G):
            input_string = input_string[:-4] + ', all edges have weight 1.\nA:'
    elif prompt_method=='fewshot':
        if G.is_directed():
            input_string=prompt_text+ "\n\nQ: Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.".\nA:'
        else:
            input_string=prompt_text+"\n\nQ: Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.".\nA:'
    elif prompt_method=='fewshotcot':
        if G.is_directed():
            input_string=prompt_text+ "\n\nQ: Given a directed graph: " + string+' Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step.\nA:'
        else:
            input_string=prompt_text+"\n\nQ: Determine if there is a path between two nodes in the graph. The graph is: " + string+' The question is: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.". Let\'s think step by step.\nA:'
    elif prompt_method=='zerov2':
        if not G.is_directed():
            input_string="Determine if there is a path between two nodes in the graph. " + string+'\nQ: Does a path exist from node ' +str(dicts[selected_nodes[0]]) +' to node '+ str(dicts[selected_nodes[1]])+ '? If so, provide the shortest path formatted as "Node #1 -> Node #2." If not, state "No path.".\nA:'
    else:
        raise Exception(f'prompt_method: {prompt_method} not implemented')

    return input_string, ans_single

def get_ans_txt(G, selected_nodes, dicts):
    try:
        weight_attr = 'weight' if graph_has_weight(G) else None
        ans_single = nx.shortest_path(G, selected_nodes[0], selected_nodes[1], weight_attr, method='bellman-ford')
    except:
        ans_single = []
    if len(ans_single) == 0:
        ans_string = 'No Path.'
    else:
        ans_string = ''
        for node in ans_single:
            ans_string += f'Node {dicts[node]} -> '
        ans_string = ans_string[:-4]
    return ans_string, ans_single

def _node_sequence_txt(nodes, dicts):
    assert len(nodes) > 0
    string = 'node ' if len(nodes) == 1 else 'nodes '
    for i, node in enumerate(nodes):
        if i == len(nodes) - 1:
            string = string[:-2] + f' and {dicts[node]}'
        else:
            string += f'{dicts[node]}, '
    return string

def get_dijkstra_txt(G, selected_nodes, dicts):
    ans_string = f"To determine if there is a path from node {dicts[selected_nodes[0]]} to node {dicts[selected_nodes[1]]} and find the shortest path if it exists, we can use Dijkstra's Algorithm. Let's apply the algorithm step by step:\n\nInitialization\n- Start with node {dicts[selected_nodes[0]]}.\n- Set the distance to node {dicts[selected_nodes[0]]} (the starting node) to 0 and to all other nodes to infinity.\n- Keep a priority queue to select the node with the smallest tentative distance that hasn't been permanently set yet.\n- Mark all nodes as unvisited."
    distance_dict = {}
    previous_dict = {}
    Q = []
    for node in G.nodes:
        distance_dict[node] = float('inf')
        previous_dict[node] = None
        Q.append(node)
    distance_dict[selected_nodes[0]] = 0
    previous_dict[selected_nodes[0]] = selected_nodes[0]

    ans_string += f"\n\nStep by Step Process"
    num_iter = 1
    while len(Q) > 0:
        u = min(Q, key=lambda x: distance_dict[x])
        if distance_dict[u] == float('inf'):
            break
        Q.remove(u)

        ans_string += f"\n{num_iter}. Select node {dicts[u]} ({'next ' if num_iter != 1 else ''}smallest distance in the priority queue). From node {dicts[u]},"
        num_iter += 1
        if len(list(G.neighbors(u))) <= 0:
            ans_string += f" we cannot reach any node."
            ans_string += '\n'
            continue
        ans_string += f" we can reach {_node_sequence_txt(list(G.neighbors(u)), dicts)}."
        to_ignore, to_visit = [], []
        for neighbor in G.neighbors(u):
            if neighbor in Q:
                to_visit.append(neighbor)
            else:
                to_ignore.append(neighbor)
        if len(to_ignore) > 0:
            ans_string += f" However, {_node_sequence_txt(to_ignore, dicts)} {'has' if len(to_ignore) == 1 else 'have'} already been selected."
        if len(to_visit) <= 0:
            ans_string += f" We have nothing to update."
            ans_string += '\n'
            continue
        ans_string += f" We update the distance{'' if len(to_visit) == 1 else 's'} to {_node_sequence_txt(to_visit, dicts)}."
        for neighbor in to_visit:
            u_neighbor_weight = 1
            if 'weight' in G.get_edge_data(u, neighbor) and G.get_edge_data(u, neighbor)['weight'] is not None:
                u_neighbor_weight = G.get_edge_data(u, neighbor)['weight']
            alternative_distance = distance_dict[u] + u_neighbor_weight
            ans_string += f"\n  - Distance to node {dicts[neighbor]} (from node {dicts[u]}) is {alternative_distance}"
            if alternative_distance < distance_dict[neighbor]:
                ans_string += f", which is better than the previous, update the priority queue."
                distance_dict[neighbor] = alternative_distance
                previous_dict[neighbor] = u
            else:
                ans_string += f", which is not better than the previous, and will not update the priority queue."

        ans_string += '\n'

    ans_string += f"\nConclusion"
    ans_single = []
    if previous_dict[selected_nodes[1]] is not None:
        ans_single = [selected_nodes[1]]
        node = selected_nodes[1]
        while node != selected_nodes[0]:
            node = previous_dict[node]
            ans_single = [node] + ans_single

        spath_string = ''
        for node in ans_single:
            spath_string += f'Node {dicts[node]} -> '
        spath_string = spath_string[:-4]
        ans_string += f"\nA path exists from node {dicts[selected_nodes[0]]} to node {dicts[selected_nodes[1]]}.\nAnswer: {spath_string}."
    else:
        ans_string += f"We have now considered all possible paths from node {dicts[selected_nodes[0]]} and updated the distances accordingly. Unfortunately, node {dicts[selected_nodes[1]]} was never reached in our exploration, indicating that there is no path from node {dicts[selected_nodes[0]]} to node {dicts[selected_nodes[1]]} in the graph as described.\nAnswer: No path."

    return ans_string, ans_single


def get_bf_txt(G, selected_nodes, dicts):
    ans_string = f"To determine if there is a path from node {dicts[selected_nodes[0]]} to node {dicts[selected_nodes[1]]} and find the shortest path if it exists, we can use the Bellman-Ford algorithm.\nThe Bellman-Ford algorithm works by iteratively relaxing the edges, updating the cost to reach each vertex from the source vertex if a shorter path is found."
    if not G.is_directed():
        ans_string += " Since the graph is undirected, we can consider each connection as bidirectional for the purpose of this explanation, but we'll follow the directed edges as described."
    ans_string += "\nGiven the graph description, let's apply the Bellman-Ford algorithm step by step:"

    ans_string += f"\n\n1. Initialize distances:\n- Start with initializing the distance to all nodes as infinity, except for the source node (node {dicts[selected_nodes[0]]} in this case), which is set to 0."
    distance_dict = {}
    previous_dict = {}
    for node in G.nodes:
        distance_dict[node] = float('inf')
        previous_dict[node] = None
    distance_dict[selected_nodes[0]] = 0
    previous_dict[selected_nodes[0]] = selected_nodes[0]

    ans_string += f"\n\n2. Relaxation:\n- Update distances based on the graph's edges. We'll do this for each edge, for V-1 iterations, where V is the number of vertices ({len(G.nodes)} in this case). However, we can stop early if there are no updates in a round."

    no_update = False
    for i in range(len(G.nodes) - 1):
        no_update = True
        ans_string += f"\n\nIteration {i+1}:"
        for edge in G.edges(data=True):
            edge_w = 1
            if 'weight' in edge[2] and edge[2]['weight'] is not None:
                edge_w = edge[2]['weight']

            if distance_dict[edge[0]] + edge_w < distance_dict[edge[1]]:
                distance_dict[edge[1]] = distance_dict[edge[0]] + edge_w
                previous_dict[edge[1]] = edge[0]
                ans_string += f"\n- Update distance to {dicts[edge[1]]} via {dicts[edge[0]]}: {distance_dict[edge[0]]} + {edge_w} = {distance_dict[edge[0]] + edge_w}"
                no_update = False
            if not G.is_directed():
                if distance_dict[edge[1]] + edge_w < distance_dict[edge[0]]:
                    distance_dict[edge[0]] = distance_dict[edge[1]] + edge_w
                    previous_dict[edge[0]] = edge[1]
                    ans_string += f"\n- Update distance to {dicts[edge[0]]} via {dicts[edge[1]]}: {distance_dict[edge[1]]} + {edge_w} = {distance_dict[edge[1]] + edge_w}"
                    no_update = False
        if no_update:
            ans_string += f"\n- No update in a round, stop early."
            break
    
    if not no_update:
        no_update = True
        ans_string += f"\n\n3. Check for Negative Weight Cycles:"
        for edge in G.edges(data=True):
            edge_w = 1
            if 'weight' in edge[2] and edge[2]['weight'] is not None:
                edge_w = edge[2]['weight']

            if distance_dict[edge[0]] + edge_w < distance_dict[edge[1]]:
                distance_dict[edge[1]] = distance_dict[edge[0]] + edge_w
                previous_dict[edge[1]] = edge[0]
                ans_string += f"\n- Update distance to {dicts[edge[1]]} via {dicts[edge[0]]}, Negative weight cycle detacted."
                no_update = False
                break
            if not G.is_directed():
                if distance_dict[edge[1]] + edge_w < distance_dict[edge[0]]:
                    distance_dict[edge[0]] = distance_dict[edge[1]] + edge_w
                    previous_dict[edge[0]] = edge[1]
                    ans_string += f"\n- Update distance to {dicts[edge[0]]} via {dicts[edge[1]]}, Negative weight cycle detacted."
                    no_update = False
                    break
        if no_update:
            ans_string += f"\n- No update in a round, no negative weight cycle."

    ans_single = []
    if not no_update:
        ans_string += f"\n\nBy examining the graph, we notice that there exists a negative weight cycle. Therefore, the answer is 'No path.'"
    elif previous_dict[selected_nodes[1]] is not None:
        ans_single = [selected_nodes[1]]
        node = selected_nodes[1]
        while node != selected_nodes[0]:
            node = previous_dict[node]
            ans_single = [node] + ans_single

        spath_string = ''
        for node in ans_single:
            spath_string += f'Node {dicts[node]} -> '
        spath_string = spath_string[:-4]
        ans_string += f"\n\nTherefore, there is a path from node {dicts[selected_nodes[0]]} to node {dicts[selected_nodes[1]]}, and the shortest path is {spath_string}."
    else:
        ans_string += f"\n\nBy examining the graph, we notice that the graph does not show any connection between node {dicts[selected_nodes[0]]} and node {dicts[selected_nodes[1]]}. Therefore, based on the graph's connections, the answer is 'No path.'"
    
    return ans_string, ans_single


def get_fw_txt(G:nx.Graph, selected_nodes, dicts):
    ans_string = f"To determine if a path exists from node {dicts[selected_nodes[0]]} to node {dicts[selected_nodes[1]]} and to find the shortest path if it exists, we can use the Floyd-Warshall algorithm. This algorithm computes the shortest paths between all pairs of nodes in a weighted graph. Here's a step-by-step explanation of how to apply the Floyd-Warshall algorithm to this graph:"
    if not G.is_directed():
        ans_string += "\nSince the graph is undirected, we can consider each connection as bidirectional for the purpose of this explanation, but we'll follow the directed edges as described."
    
    ans_string += f"\n\nStep 1: Initialize the Distance Matrix"
    ans_string += f"\nFirst, we create a distance matrix `dist` where `dist[i][j]` represents the shortest distance from node `i` to node `j`. Initially, `dist[i][j]` is set to infinity for all pairs of nodes except when `i = j`, in which case `dist[i][j] = 0`. If there is a direct edge from `i` to `j`, then `dist[i][j]` is the weight of that edge.\nGiven the graph, the initial `dist` matrix (only showing relevant initializations for brevity) would look something like this:"
    dist = np.full((G.number_of_nodes(), G.number_of_nodes()), np.inf)
    for i in range(G.number_of_nodes()):
        dist[i][i] = 0
    for e in G.edges(data=True):
        dist[e[0]][e[1]] = e[2]['weight'] if 'weight' in e[2] and e[2]['weight'] is not None else 1
        if not G.is_directed():
            dist[e[1]][e[0]] = e[2]['weight'] if 'weight' in e[2] and e[2]['weight'] is not None else 1
    for i in range(G.number_of_nodes()):
        ans_string += '\n'
        for j in range(G.number_of_nodes()):
            if np.isinf(dist[i][j]):
                continue
            ans_string += f"dist[{i}][{j}] = {dist[i][j]}, "
        ans_string = ans_string[:-2] + '.'
    ans_string += f"\nSince we need to reconstruct the shortest path, we create a matrix `prev` where `prev[i][j]` represents the penultimate vertex on the path from `i` to `j`. Initially. `prev[i][j]` is set to null for all pairs of nodes except `prev[i][i] = i`. If there is a direct edge from `i` to `j`, then `prev[i][j] = i`.\nGiven the graph, the initial `prev` matrix (only showing relevant initializations for brevity) would look something like this:"
    prev = np.full((G.number_of_nodes(), G.number_of_nodes()), None)
    for i in range(G.number_of_nodes()):
        prev[i][i] = i
    for e in G.edges(data=True):
        prev[e[0]][e[1]] = e[0]
        if not G.is_directed():
            prev[e[1]][e[0]] = e[1]
    for i in range(G.number_of_nodes()):
        ans_string += '\n'
        for j in range(G.number_of_nodes()):
            if prev[i][j] is None:
                continue
            ans_string += f"prev[{i}][{j}] = {prev[i][j]}, "
        ans_string = ans_string[:-2] + '.'

    ans_string += f"\n\nStep 2: Apply the Floyd-Warshall Algorithm"
    ans_string += f"\nThe core of the Floyd-Warshall algorithm involves updating the `dist` matrix by considering all pairs of nodes `(i, j)` and checking if a path from `i` to `j` through an intermediate node `k` is shorter than the current known path from `i` to `j`. The algorithm iterates through all nodes as possible intermediate nodes.\nFor each node `k`, and for each pair of nodes `(i, j)`, we update `dist[i][j]` as follows:"
    for k in range(G.number_of_nodes()):
        ans_string += f"\nFor k = {k}:"
        have_update = False
        for i in range(G.number_of_nodes()):
            for j in range(G.number_of_nodes()):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    prev[i][j] = prev[k][j]
                    ans_string += f" Update dist[{i}][{j}] = {dist[i][j]}, prev[{i}][{j}] = {prev[i][j]}."
                    have_update = True
        if not have_update:
            ans_string += f" No update."

    ans_string += f"\n\nStep 3: Check for Negative Weight Cycles"
    ans_string += f"\nCheck the diagonal of the distance matrix. If any `dist[i][i]` is negative, a negative cycle exists."
    have_negcycle = any([dist[i][i] < 0 for i in range(G.number_of_nodes())])
    if have_negcycle:
        ans_string += f"\nIn this case, a negative cycle exists."
        ans_single = []
    else:
        ans_string += f"\nIn this case, a negative cycle does not exist."

        ans_string += f"\n\nStep 4: Check for a Path from Node {dicts[selected_nodes[0]]} to Node {dicts[selected_nodes[1]]}"
        ans_string += f"\nAfter applying the Floyd-Warshall algorithm, we look at `dist[{dicts[selected_nodes[0]]}][{dicts[selected_nodes[1]]}]`. If it's not infinity, a path exists, its length is `dist[{dicts[selected_nodes[0]]}][{dicts[selected_nodes[1]]}]`, and the path could be reconstructed from `prev` matrix."
        if np.isinf(dist[selected_nodes[0]][selected_nodes[1]]):
            ans_string += f"\nIn this case, `dist[{dicts[selected_nodes[0]]}][{dicts[selected_nodes[1]]}]` is infinity, thus there is no path."
            ans_single = []
        else:
            ans_single = [selected_nodes[1]]
            v = selected_nodes[1]
            while v != selected_nodes[0]:
                v = prev[selected_nodes[0]][v]
                ans_single = [v] + ans_single
            spath_string = ''
            for node in ans_single:
                spath_string += f'Node {dicts[node]} -> '
            spath_string = spath_string[:-4]
            ans_string += f"\nIn this case, `dist[{dicts[selected_nodes[0]]}][{dicts[selected_nodes[1]]}]` is not infinity, thus the path exists, the path is {spath_string}."
    
    ans_string += f"\n\nConclusion"
    if have_negcycle:
        ans_string += f"\nBy examining the graph, we notice that there exists a negative weight cycle. Therefore, the answer is 'No path.'"
    elif len(ans_single) == 0:
        ans_string += f"\nBy examining the graph, we notice that the graph does not show any connection between node {dicts[selected_nodes[0]]} and node {dicts[selected_nodes[1]]}. Therefore, based on the graph's connections, the answer is 'No path.'"
    else:
        ans_string += f"\nBy examining the graph, there is a path from node {dicts[selected_nodes[0]]} to node {dicts[selected_nodes[1]]}, and the shortest path is {spath_string}."            
    return ans_string, ans_single
