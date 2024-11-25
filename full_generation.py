from utils.gen import gen_datapool, make_class, save_datapool, load_data
import random
import multiprocessing as mp

simple = True # set to True to make a small size dataset with full structure, False to make a full size dataset

max_process = 64 # maximum number of process used in generation
semaphore = mp.Semaphore(max_process)

def mp_full_datapool(arg_tuple):
    with semaphore:
        modulation_dict = {'chain':1, 'schain':1, 'tree': 1, 'circle': 1, 'scircle': 1, 'iso':1, 'noniso':1, 'normal': 5} if simple else {'chain':1, 'schain':5, 'tree': 5, 'scircle': 5, 'iso':10, 'noniso':10, 'normal': 50}
        directed, size, weight_type = arg_tuple
        M_num = 10 if simple else 100
        if size == 'easy':
            N_range = list(range(5, 16))
        elif size == 'medium':
            N_range = list(range(16, 26))
        elif size == 'hard':
            N_range = list(range(26, 36))
        if weight_type == 'unweight':
            weight_func = lambda: None
        elif weight_type == 'posweight':
            weight_func = lambda: random.randint(1, 9)
        elif weight_type == 'negweight':
            weight_func = lambda: random.randint(-3, 9)
        datapool = gen_datapool(directed=directed, N_range=N_range, M_num=M_num, weight_func=weight_func, modulation_dict=modulation_dict, filt_isomorphic=False, verbose=True)
        save_datapool(datapool,f"dataset/datapool/{'directed' if directed else 'undirected'}_{size}_{weight_type}.pkl")


def gen_full_datapool():
    arg_list = []
    for directed in [False, True]:
        for size in ['easy', 'medium', 'hard']:
            for weight_type in ['unweight', 'posweight']:
                arg_list.append((directed, size, weight_type))
    arg_list.append((True, 'easy', 'negweight'))
    
    with mp.Pool() as pool:
        results = pool.map(mp_full_datapool, arg_list)

def mp_full_dataset(arg_tuple):
        with semaphore:
            folder_path, datapool_path, num_samples, target_hop, connectivity_type = arg_tuple
            datapool = load_data(datapool_path)

            save_path = f'{folder_path}/{connectivity_type}_connect.pkl'
            plot_path = f'{folder_path}/{connectivity_type}_plot'
            make_class(datapool, save_path, plot_path, num_samples, target_hop, connectivity_type, verbose=False)
            save_path_shot = f'{folder_path}/{connectivity_type}_connect_shot.pkl'
            plot_path_shot = f'{folder_path}/{connectivity_type}_shot_plot'
            make_class(datapool, save_path_shot, plot_path_shot, num_samples, target_hop, connectivity_type, filt_isomorphic_from_path=save_path, verbose=False)

def gen_full_dataset():
    arg_list = []
    pool_list = []
    for directed in [False, True]:
        for size in ['easy', 'medium', 'hard']:
            for weight_type in ['unweight', 'posweight']:
                pool_list.append((directed, size, weight_type))
    pool_list.append((True, 'easy', 'negweight'))
    for directed, size, weight_type in pool_list:
        folder_path = f"dataset/{'directed' if directed else 'undirected'}_{size}_{weight_type}"
        datapool_path = f"dataset/datapool/{'directed' if directed else 'undirected'}_{size}_{weight_type}.pkl"
        num_samples_base = 1 if simple else 50
        for k in range(1,6):
            arg_list.append((folder_path, datapool_path, num_samples_base, k, f'hop{k}'))
        if directed:
            for num_samples, target_hop, connectivity_type in [(num_samples_base, 0, 'singleton'), (num_samples_base * 2, 0, 'asymmetric'), (num_samples_base * 2, 0, 'disoc')]:
                arg_list.append((folder_path, datapool_path, num_samples, target_hop, connectivity_type))
        else:
            for num_samples, target_hop, connectivity_type in [(num_samples_base, 0, 'singleton'), (num_samples_base * 4, 0, 'isoc')]:
                arg_list.append((folder_path, datapool_path, num_samples, target_hop, connectivity_type))
    
    with mp.Pool() as pool:
        results = pool.map(mp_full_dataset, arg_list)

if __name__ == '__main__':
    print('generate full datapool')
    gen_full_datapool()
    print('generate full dataset')
    gen_full_dataset()
    print('done')
