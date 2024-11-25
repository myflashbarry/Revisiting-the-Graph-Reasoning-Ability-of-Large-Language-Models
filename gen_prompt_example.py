from task_sp.graph2nlp import get_shot
from task_sp.get_response import get_response
from task_sp.utils.eval import get_acc_spath
import random

prompt_strings = []
for dataset_name in ['hop1', 'hop2', 'hop3', 'hop4', 'hop5', 'singleton', 'disoc', 'asymmetric']:
    dataset_name += f'_connect_shot'
    question_texts, ans_strings = get_shot('directed_easy_posweight', dataset_name, 'node', 'zerocotdijkstra', rename='renamei')
    question_text, ans_string = random.choice(list(zip(question_texts, ans_strings)))
    prompt_string = f'Q: {question_text}\nA: {ans_string}\n'
    prompt_strings.append(prompt_string)

random.shuffle(prompt_strings)
prompt_strings = prompt_strings[:3]
prompt_text = ''
for prompt_string in prompt_strings:
    prompt_text += prompt_string + '\n'

prompt_path = 'prompt_test.txt'
with open(prompt_path, 'w') as f:
    f.write(prompt_text)

get_response(folder_name='directed_easy_posweight',
            model='gpt3', # model name defined in chat.py
            dataset_name=f'hop4_connect',
            comment='',
            describe_method='node',
            prompt_method='cotdijkstra',
            rename='',
            prompt_path='prompt_test.txt',
            verbose=False,
            cover=False)

json_path = f'dataset/directed_easy_posweight/gpt3hop4_sp_node_cotdijkstra.json'
pkl_path = f'dataset/directed_easy_posweight/hop4_connect.pkl'
acc, fed, pcr = get_acc_spath(json_path, pkl_path, verbose=False)
print(f"ACC: {acc}, F-acc: {fed}, PCR: {pcr}")