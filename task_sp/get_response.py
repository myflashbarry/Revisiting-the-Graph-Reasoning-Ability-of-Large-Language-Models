from task_sp.graph2nlp import get_single_data
import json
from tqdm import tqdm
import os
from chat import chat_dict


def get_response(folder_name, model, dataset_name, comment='', describe_method='node', prompt_method='zero', rename='', prompt_path='', seed=None, verbose=True, cover=False ):
    '''
    get LLM response of shortest path task and record into json
    json format: [{'text': <question text as string>, 'ans': <nx.shortest_path() one sp example>, 'response':[[[<response as string>], 0]]}]
    json_name = f"{model}{'_'.join(dataset_name.split('_')[:-1])}_sp{comment}_{describe_method}_{prompt_method}{'' if rename == '' else '-'+rename}"
    json_path = f'dataset/{folder_name}/{json_name}.json'

    Parameters
    ----------
    folder_name: name of the folder where json will be saved
    
    model: name of LLM defined in chat.py chat_dict
    
    dataset_name: name of dataset, file name of pickle without extention
    
    comment: comment add in json file name
    
    describe_method: 'node' or 'edge', how to describe graph in text
    
    prompt_method: 'zero', 'fewshot', 'zerocot', 'zerocotdijkstra', 'zerocotbf', 'zerocotfw', 'cotdijkstra', 'cotbf', 'cotfw', 
        for 'fewshot' and 'cot<algorithm>' please make sure that shot prompts are generated
        
    rename: '', 'renamei', 'renamec', node representation rename. 
        '': ordered ID start from 0, 'renamei': random ID from random.randrange(0,100000), 'renamec': random characters of 5 uppercase letters
        
    prompt_path: path of shot prompt txt file, default '' means none
    
    seed: random seed
    
    verbose: bool
    
    cover: bool, True then cover previous result stored in the same path
    '''
    if len(dataset_name.split('_')) == 1:
        dataset_name += '_sp'
    json_name = f"{model}{'_'.join(dataset_name.split('_')[:-1])}_sp{comment}_{describe_method}_{prompt_method}{'' if rename == '' else '-'+rename}"
    json_path = f'dataset/{folder_name}/{json_name}.json'

    if os.path.exists(json_path) and not cover:
        with open(json_path, 'r') as f:
            responses_list = json.load(f)
    else:
        input_texts,ans=get_single_data(folder_name,dataset_name,describe_method,prompt_method,prompt_path,seed,rename)
        responses_list = []
        for text, one_ans in zip(input_texts, ans):
            responses_dict = {}
            responses_dict['text'] = text
            responses_dict['ans'] = one_ans
            responses_dict['response'] = []
            responses_list.append(responses_dict)

    print(f'[json path]: {json_path}, [num of questions]: {len(responses_list)}')
    if verbose:
        print('[please confirm, press \'q\' to quit]:')
        command = 'default'
        while command != '':
            command = input()
            if command == 'q':
                return
            if command.isdigit():
                print(responses_list[int(command)]['text'])
                print(responses_list[int(command)]['ans'])

    for idx,res in tqdm(enumerate(responses_list),total=len(responses_list), desc=json_path):
        responses_list[idx]['response'] = [[[chat_dict[model](res['text'])], 0]]

    with open(json_path,'w') as f:
        json.dump(responses_list,f)
    print(f'[saved]: {json_path}')
