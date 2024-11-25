import json
import pickle
import networkx as nx
import regex as re
import Levenshtein

discrib_method='inc'

dataset_name = 'dummy100_connect_rename'#'connect_1-1'#
pkl_name = 'dummy100_connect_rename'#'connect_1-1'#'dummy100_connect'
json_name = 'gpt3dummy100_connect_rename_inc_zero-qswapshuffle'#'gpt3connect_1-1_inc_zero-nodeshuffled'#

def load_pkl(path):
    with open(path,'rb') as f:
        return pickle.load(f)
    
def load_json(path):
    with open(path, 'r') as file:
        return json.load(file)
    
def graph_has_weight(G):
    assert any('weight' in G[u][v] and G[u][v]['weight'] is not None for u, v in G.edges()) == all('weight' in G[u][v] and G[u][v]['weight'] is not None for u, v in G.edges())
    return any('weight' in G[u][v] and G[u][v]['weight'] is not None for u, v in G.edges())

def check_complete_res(responses):
    return not (len(responses) <= 0 or len(responses[0][0][0]) <= 0 or 'error' in responses[0][0][0].lower())

def check_complete(json_path, verbose=True):
    incomplete_cnt = 0
    with open(json_path, 'r') as f:
        response_list = json.load(f)
    for res in response_list:
        if not check_complete_res(res['response']):
            if verbose:
                print(res['response'])
            incomplete_cnt += 1
    return incomplete_cnt, len(response_list)

    
def extract_sentence(description, node_type, json_path, text):
    sentence = description
    for split_key in ['->', '\\to', '\\rightarrow']:
        sentence_split = sentence.split('->')
        if len(sentence_split) > 1:
            break
    start_index = 0
    if len(sentence_split) > 1:
        for sentence_str in sentence_split[len(sentence_split)-2::-1]:
            if len(sentence_str) > 12:
                start_index = len(sentence_split) - sentence_split[::-1].index(sentence_str) - 1
                break
    sentence_split_path = sentence_split[start_index:]
    do_manual = False
    if len(sentence_split_path) < 2:
        if sentence == 'The answer is Yes.':
            sentence_extract = sentence
        else:
            do_manual = True
    elif len(sentence_split_path) > 100:
        sentence_extract = sentence
    else:
        if node_type=='num':
            pattern = r"\b\d+\b"
        elif node_type=='char':
            pattern = r'\b[A-Z]{5}\b'
        elif node_type=='random':
            pattern = r'\b\d+\b'
        head_match = list(re.finditer(pattern, sentence_split_path[0]))
        tail_match = re.search(pattern, sentence_split_path[-1])
        if len(head_match) < 1 or tail_match is None:
            do_manual = True
        else:
            head_match = head_match[-1]
            start_index = head_match.start() + sum([len(split_str) + 2 for split_str in sentence_split[:start_index]])
            end_index = tail_match.end() + sum([len(split_str) + 2 for split_str in sentence_split[:-1]])
            sentence_extract = sentence[start_index:end_index]

    if do_manual:
        # print(f'[{json_path}]\n[Q]:\n{text}\n[A]:\n{sentence}')
        # sentence_extract = input("[input sentence to extract, '' means 'no path']:")
        sentence_extract = 'n'

    if node_type=='num':
        pattern = r"\b\d+\b"
    elif node_type=='char':
        pattern = r'\b[A-Z]{5}\b'
    elif node_type=='random':
        pattern = r'\b\d+\b'
    extracted_ids = re.findall(pattern, sentence_extract)

    return extracted_ids, sentence_extract

def verify_specific_path(G, path, nodes):
    is_path = len(path) > 1 and all(G.has_edge(path[i], path[i+1]) for i in range(len(path) - 1))
    is_target = len(path) > 1 and ((path[0] == nodes[0] and path[-1] == nodes[1]) or (path[0] == nodes[1] and path[-1] == nodes[0] and not G.is_directed()))
    return is_path and is_target

def match_nopath(sentence):
    nopath_pattern = r'cannot \w*\s?find a path|[Nn]o \w*\s?[Pp]ath|[Pp]ath[^.:\n]+not \w*\s?(outlined|exist[s]?|revealed|shown)'
    return re.search(nopath_pattern, sentence) is not None

def judge_spath(res, G, nodepair, idx, json_path, verbose):
    if not check_complete_res(res['response']):
        raise Exception('incomplete')

    text=res['text']
    sentence=res['response'][0][0][0]
    ans_response=res['ans']
    try:
        weight_attr = 'weight' if graph_has_weight(G) else None
        ans = nx.shortest_path(G, nodepair[0], nodepair[1], weight_attr, method='bellman-ford')
    except:
        ans = []

    if 'zero' not in json_path:
        if 'Q:' not in text:
            print(f'[text]:\n{text}')
            text = input('[input text area]:')
        else:
            text = text.split('Q:')[-1]
    if verbose:
        print(f'[{idx}]')
        print(text)
        print(sentence)

    dictions={}
    if 'renamec' in json_path:
        descript_method = 'char'
        if 'inc' in json_path:
            s='graph among'
            e='In this graph:'
            text=text[text.index(s):text.index(e)]
            nodes_list = re.findall(r'\b[A-Z]+\b', text)
        elif 'adj' in json_path:
            s='graph among'
            e='The edges in G'
            if 'adjn' in json_path:
                e = '.\nNode '
            text=text[text.index(s):text.index(e)]
            nodes_list = re.findall(r'\b[A-Z]+\b', text)
        for idxn in range(len(nodes_list)):
            dictions[nodes_list[idxn]]=idxn
    elif 'renamei' in json_path:
        descript_method = 'random'
        if 'inc' in json_path:
            s='graph among'
            e='In this graph:'
            text=text[text.index(s):text.index(e)]
            nodes_list = re.findall(r'\b\d+\b', text)
            # print(nodes_list)
        elif 'adj' in json_path:
            s='graph among'
            e='The edges in G'
            if 'adjn' in json_path:
                e = '.\nNode '
            text=text[text.index(s):text.index(e)]
            nodes_list = re.findall(r'\b\d+\b', text)
        for idxn in range(len(nodes_list)):
            dictions[nodes_list[idxn]]=idxn
    else:
        descript_method = 'num'
        for idxn in range(len(G.nodes)):
            dictions[str(idxn)]=idxn

    if match_nopath(sentence[-100:]):
        if len(ans)==0:
            return 1, 1, -1
        else: 
            return 0, 0, -1
    else:
        if len(ans)==0:
            return 0, 0, -1
        else:
            extracted_ids, sentence_extract = extract_sentence(sentence, descript_method, json_path, text)
            if sentence_extract == '': # as predicted no path
                return 0, 0, -1

            new_ids=[]
            for ids in extracted_ids:
                if ids in dictions:
                    new_ids.append(int(dictions[ids]))
                else:
                    ids = min(dictions, key=lambda x: Levenshtein.distance(str(x), str(ids)))
                    new_ids.append(int(dictions[ids]))
            
            extracted_ids=new_ids
            # G.get_edge_data(u, neighbor)['weight']
            path_verify=verify_specific_path(G,extracted_ids, nodepair)
            if verbose:
                print(extracted_ids)
                print(path_verify)
            if path_verify:
                extracted_length = len(extracted_ids) - 1
                shortest_path_length = len(ans) - 1
                if graph_has_weight(G):
                    extracted_length = 0
                    for ipath in range(len(extracted_ids) - 1):
                        extracted_length += G.get_edge_data(extracted_ids[ipath], extracted_ids[ipath+1])['weight']
                    shortest_path_length = 0
                    for ipath in range(len(ans) - 1):
                        shortest_path_length += G.get_edge_data(ans[ipath], ans[ipath+1])['weight']
                assert extracted_length >= shortest_path_length
                if extracted_length == shortest_path_length:
                    return 1, 1, 1
                else:
                    return 0, 1, (shortest_path_length/extracted_length if extracted_length > 0 else 0)
            else:
                return 0, 0, -1


def get_acc_spath(json_path, pkl_path, verbose=True):
    '''
    evaluate accuracy, F-acc and pcr of shortest path task, has to be answered with path.

    Parameters
    ----------
    json_path: the path of json file to be evaluated
    pkl_path: the path of dataset for evaluate
    verbose: bool

    Returns
    accuracy: ACC, judgment of shortest path, to be right if and only if it is the valid shortest path or successfully identify that the path does not exist
    fidelity: F-acc, judge to be right as long as the path is valid or truly no path
        always equal with acc in singleton, isoc, asymmetric
    decrease: PCR, the shortest path length versus output path length, this is only be taken into account if a path exists and the path is valid
        always 0 in singleton, isoc, asymmetric
    ----------
    '''
    true_count=0
    with open(json_path,'r') as f:
        data=json.load(f)
    answer_list=[]
    pred_list=[]
    with open(pkl_path,'rb')as f:
        graph_list,node_list=pickle.load(f)
    # data=data[0]
    if verbose:
        print(json_path, len(graph_list))
    
    acc_dicts = []
    fidelity_score_dicts=[]
    decrease_score_dicts=[]
    for idx,(res,G,nodes) in enumerate(zip(data,graph_list,node_list)):
        acc, fed, pcr = judge_spath(res=res, G=G, nodepair=nodes, idx=idx, json_path=json_path, verbose=verbose)
        acc_dicts.append(acc)
        fidelity_score_dicts.append(fed)
        if pcr > 0:
            decrease_score_dicts.append(pcr)

    if len(acc_dicts) == 0:
        acc_dicts.append(0)
    if len(fidelity_score_dicts) == 0:
        fidelity_score_dicts.append(0)
    if len(decrease_score_dicts) == 0:
        decrease_score_dicts.append(0)
    accuracy = sum(acc_dicts)/len(acc_dicts)
    fidelity = sum(fidelity_score_dicts)/len(fidelity_score_dicts)# accuracy_score(length_dicts[key]['ans'], length_dicts[key]['pred'])
    decrease=sum(decrease_score_dicts)/len(decrease_score_dicts)
        
    return accuracy, fidelity, decrease


