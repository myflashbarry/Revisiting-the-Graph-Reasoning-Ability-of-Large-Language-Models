from task_connect.get_response import get_response
from utils.gen import gen_datapool, make_class, save_datapool
from task_connect.utils.eval import get_acc_path

# generate datapool
modulation_dict = {'chain':1, 'schain':5, 'tree': 1, 'scircle': 1, 'iso':5, 'noniso':5, 'normal': 20}
datapool = gen_datapool(directed=False, N_range=list(range(5, 16)), M_num=50, modulation_dict=modulation_dict, filt_isomorphic=False, verbose=True)
save_datapool(datapool,'dataset/datapool/undirected_easy_unweight.pkl')

# generate datasets
connectivity_type_tuples = [(5, k, f'hop{k}') for k in range(1, 6)] + [(5, 0, 'singleton'), (20, 0, 'isoc')]
for num_samples, target_hop, connectivity_type in connectivity_type_tuples:
    make_class(datapool,
            save_path=f'dataset/undirected_easy_unweight/{connectivity_type}_connect.pkl',
            plot_path=f'dataset/undirected_easy_unweight/{connectivity_type}_plot',
            num_samples=num_samples,
            target_hop=target_hop,
            connectivity_type=connectivity_type,
            verbose=False)
    
# get response
for num_samples, target_hop, connectivity_type in connectivity_type_tuples:
    get_response(folder_name='undirected_easy_unweight',
            model='gpt3', # model name defined in chat.py
            dataset_name=f'{connectivity_type}_connect',
            comment='',
            describe_method='node',
            prompt_method='zeropath',
            rename='',
            verbose=False,
            cover=False)
    
# evaluation
for num_samples, target_hop, connectivity_type in connectivity_type_tuples:
    json_path = f'dataset/undirected_easy_unweight/gpt3{connectivity_type}_connect_node_zeropath.json'
    pkl_path = f'dataset/undirected_easy_unweight/{connectivity_type}_connect.pkl'
    acc, fed, pcr = get_acc_path(json_path, pkl_path, verbose=False)
    print(f"connectivity type: {connectivity_type}, ACC: {acc}, F-acc: {fed}, PCR: {pcr}")
