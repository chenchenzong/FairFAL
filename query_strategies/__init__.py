import os
import sys
import copy
import pickle
import random
import datetime
import numpy as np

import torch

from models import get_model
from .fairfal_sampling import FairFALSampling

from collections import Counter

def get_label_count(dataset, index_list, args):
    if args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
        labels = np.asarray(dataset.labels)[index_list].ravel().tolist()
    else:
        labels = np.asarray(dataset.targets)[index_list].tolist()
    return Counter(labels)

def random_query_samples(dict_users_train_total, dict_users_test_total, args):
    """ randomly select the labeled samples at the first round
    """
    args.dict_users_total_path = os.path.join(args.dict_user_path, 'dict_users_train_test_total.pkl'.format(args.seed))
            
    with open(args.dict_users_total_path, 'wb') as handle:
        pickle.dump((dict_users_train_total, dict_users_test_total), handle)
        
    dict_users_train_label_path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(args.current_ratio))

    dict_users_train_label = {user_idx: [] for user_idx in dict_users_train_total.keys()}

    # sample n_start example on each client
    for idx in dict_users_train_total.keys():
        dict_users_train_label[idx] = np.random.choice(np.array(list(dict_users_train_total[idx])), int(args.n_data / args.num_users), replace=False)
        
    with open(dict_users_train_label_path, 'wb') as handle:
        pickle.dump(dict_users_train_label, handle)    
    
    return dict_users_train_label, args
    
    
def algo_query_samples(dataset_train, dataset_query, dataset_test, dict_users_train_total, dict_users_test_total, mean_global_ratio, args):
    """ query samples from the unlabeled pool
    """
    previous_ratio = args.current_ratio - args.query_ratio
    path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(previous_ratio))    
    with open(path, 'rb') as f:
        dict_users_train_label = pickle.load(f) 

        print("Before Querying")
        total_data_cnt = 0
        for user_idx in range(args.num_users):
            print(user_idx, len(dict_users_train_label[user_idx]))
            total_data_cnt += len(dict_users_train_label[user_idx])

        print(total_data_cnt)
        print("-" * 20)

    # Build model
    query_net = get_model(args)
    args.raw_ckpt = copy.deepcopy(query_net.state_dict())

    query_net_state_dict = torch.load(args.query_model)
    query_net.load_state_dict(query_net_state_dict)            
    
    # Methods
    if args.al_method == "fairfal":
        strategy = FairFALSampling(dataset_query, dataset_train, dataset_test, query_net, args)
    else:
        exit('There is no al methods')    
    
    results_save_path = os.path.join(args.result_dir, 'query_info.txt')

    time = datetime.timedelta()
    sum_global_ratio, count = 0., 0.
    for user_idx in dict_users_train_total.keys():                
        total_idxs = dict_users_train_total[user_idx]
        label_idxs = dict_users_train_label[user_idx]
        unlabel_idxs = list(set(total_idxs) - set(label_idxs))
        
        count_label = get_label_count(dataset_query, label_idxs, args)
        global_ratio = strategy.query_model_selection(user_idx, label_idxs, unlabel_idxs, count_label)
        
        sum_global_ratio += global_ratio
        count += 1
    
    if mean_global_ratio == None:
        mean_global_ratio = sum_global_ratio / count
    
    print("mean_global_ratio: ", mean_global_ratio)    
    
    for user_idx in dict_users_train_total.keys():                
        total_idxs = dict_users_train_total[user_idx]
        label_idxs = dict_users_train_label[user_idx]
        unlabel_idxs = list(set(total_idxs) - set(label_idxs))
        
        test_idxs = dict_users_test_total[user_idx]
        
        start = datetime.datetime.now()
        count_label = get_label_count(dataset_query, label_idxs, args)
        count_unlabel = get_label_count(dataset_query, unlabel_idxs, args)
        print("user_idx is ", user_idx)
        print("count_label: ", count_label)
        print("count_unlabel: ", count_unlabel)
        new_data, global_acc, local_acc = strategy.query(user_idx, label_idxs, unlabel_idxs, test_idxs, count_label, mean_global_ratio, int(args.n_query / args.num_users))
        count_query = get_label_count(dataset_query, new_data, args)
        print("count_query: ", count_query)
        time += datetime.datetime.now() - start
        
        print(args.al_method, user_idx)
        print("(Before) Label examples: {}".format(len(label_idxs)))
        if len(new_data) < int(args.n_query / args.num_users):
            sys.exit("too few remaining examples to query")

        dict_users_train_label[user_idx] = np.array(list(new_data) + list(label_idxs))   
        print("(After) Label examples: {}".format(len(list(new_data)) + len(label_idxs))) 
        
        with open(results_save_path, "a") as f:
            f.write(f"User {user_idx}\n")
            f.write(f"count_unlabel: {count_unlabel}\n")
            f.write(f"count_query: {count_query}\n")
            f.write(f"global model test accuracy: {global_acc}, local model test accuracy: {local_acc}\n\n")

    time /= len(dict_users_train_total)     
    print('Querying instances takes {}'.format(time))           

    # Save dict_users for next round
    path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(args.current_ratio))
    with open(path, 'wb') as handle:
        pickle.dump(dict_users_train_label, handle)

    return dict_users_train_label, mean_global_ratio
    