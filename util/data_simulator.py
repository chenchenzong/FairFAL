#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import random, math, os, pickle, torch
import numpy as np
from collections import defaultdict

def shard_balance(dataset, args):
    K = args.num_classes
    y_train_dict = {i: [] for i in range(K)}
    
    for idx, d in enumerate(dataset):
        if args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
            y_train_dict[d[1][0]].append(idx)
        else:
            y_train_dict[d[1]].append(idx)
    
    allocate_data_cnt_dict = {i: int(len(y_train_dict[i]) / args.num_classes_per_user)  for i in range(K)}
    
    all_label_lst = list(range(K)) * args.num_classes_per_user
    
    labels_lst = []
    
    for _ in range(args.num_users):
        temp_lst = []
    
        idx = np.random.choice(all_label_lst)
        temp_lst.append(idx)

        while len(temp_lst) != args.num_classes_per_user:
            idx = np.random.choice(all_label_lst)
            if idx not in temp_lst:
                temp_lst.append(idx)

        labels_lst.append(temp_lst)
        
        for idx in temp_lst:
            del all_label_lst[all_label_lst.index(idx)]
        
    net_dataidx_map = {i: [] for i in range(args.num_users)}
    
    for user_idx, labels in enumerate(labels_lst):
        for label in labels:
            allocate_idx = np.random.choice(y_train_dict[label], allocate_data_cnt_dict[label], replace=False)
            y_train_dict[label] = list(set(y_train_dict[label]) - set(allocate_idx))
            net_dataidx_map[user_idx] += list(allocate_idx)
                
    return dict(net_dataidx_map)


def dir_balance(dataset, args, sample=None, seed=0):

    rng_np = np.random.RandomState(seed)
    rng_py = random.Random(seed)
    torch.manual_seed(seed)

    C = args.num_classes
    K = args.num_users
    alpha = getattr(args, 'dd_beta', 1.0)

    total_num = len(dataset)
    total_data = {str(i): [] for i in range(C)}
    data_num = np.zeros(C, dtype=np.int64)

    for idx, data in enumerate(dataset):
        if args.dataset in ['pathmnist','octmnist','organamnist','dermamnist','bloodmnist']:
            y = int(data[1][0]) if hasattr(data[1], '__len__') else int(data[1])
        else:
            y = int(data[1])
        total_data[str(y)].append(idx)
        data_num[y] += 1

    if sample is None:
        diri = torch.distributions.dirichlet.Dirichlet(alpha * torch.ones(C))
        P = torch.stack([diri.sample() for _ in range(K)], dim=0)  # [K, C]

        for _ in range(5):
            P = P / (P.sum(dim=1, keepdim=True) + 1e-12)
            P = P / (P.sum(dim=0, keepdim=True) + 1e-12)
    else:
        P = sample.clone().detach()

    x = torch.zeros((K, C), dtype=torch.long)
    for j in range(C):
        n_j = int(data_num[j])
        if n_j == 0:
            continue
        col = P[:, j].clamp(min=0)
        if col.sum() <= 0:
            col = torch.ones(K) / K
        col = col / (col.sum() + 1e-12)

        real = (col * n_j)                   
        base = torch.floor(real).to(torch.long)
        remain = n_j - int(base.sum().item())
        frac = (real - base.float()).numpy()
        idxs = np.argsort(-frac)
        base_np = base.numpy()
        if remain > 0:
            base_np[idxs[:remain]] += 1
        x[:, j] = torch.from_numpy(base_np)

    target_floor = total_num // K
    r = total_num % K
    q = np.array([target_floor + (1 if i < r else 0) for i in range(K)], dtype=np.int64)

    totals = x.sum(dim=1).numpy()
    deficits = (q - totals).astype(int)

    def sorted_indices_more_less(deficits_arr):
        more = np.where(deficits_arr > 0)[0]
        less = np.where(deficits_arr < 0)[0]

        more = sorted(more, key=lambda i: -deficits_arr[i])
        less = sorted(less, key=lambda i: deficits_arr[i])
        return more, less

    max_moves = int(total_num) + 5
    moves = 0

    while True:
        more_list, less_list = sorted_indices_more_less(deficits)
        if not more_list and not less_list:
            break
        if not more_list or not less_list:
            break

        progressed = False
        for j in range(C):
            if not more_list or not less_list:
                break

            donors = [s for s in less_list if x[s, j] > 0]
            if not donors:
                continue

            receivers = [d for d in more_list if deficits[d] > 0]
            if not receivers:
                continue

            donor_supply = {s: int(min(x[s, j].item(), -deficits[s])) for s in donors if -deficits[s] > 0}
            total_supply = sum(donor_supply.values())
            total_demand = sum(deficits[d] for d in receivers)

            if total_supply <= 0:
                continue

            move_all = min(total_supply, total_demand)

            for d in receivers:
                if move_all <= 0:
                    break
                need = int(deficits[d])
                if need <= 0:
                    continue
                take_left = need

                donors_sorted = sorted(donor_supply.keys(), key=lambda s: -donor_supply[s])
                for s in donors_sorted:
                    if take_left <= 0 or move_all <= 0:
                        break
                    can = donor_supply[s]
                    if can <= 0:
                        continue
                    t = min(can, take_left, move_all)
                    if t <= 0:
                        continue

                    x[s, j] -= t
                    x[d, j] += t
                    donor_supply[s] -= t
                    deficits[s] += t
                    deficits[d] -= t
                    move_all -= t
                    take_left -= t
                    progressed = True
                    moves += t
                    if moves > max_moves:
                        break
                if moves > max_moves:
                    break
            if moves > max_moves:
                break
        if not progressed or moves > max_moves:
            break

    col_ok = (x.sum(dim=0).numpy() == data_num).all()
    row_ok = (x.sum(dim=1).numpy() == q).all()
    assert col_ok, "Column sums must match class totals after fix."
    assert row_ok, "Row sums (per-client totals) must match the target quotas."

    clients_data = {i: [] for i in range(K)}
    clients_data_num = {i: [0]*C for i in range(K)}

    for j in range(C):
        idxs = total_data[str(j)]
        rng_py.shuffle(idxs)
        start = 0
        for i in range(K):
            take = int(x[i, j].item())
            if take > 0:
                part = idxs[start:start+take]
                clients_data[i].extend(part)
                clients_data_num[i][j] += take
                start += take
        assert start == len(idxs), f"class {j}: assigned {start} != total {len(idxs)}"

    div = torch.clamp(torch.tensor(data_num, dtype=torch.float32), min=1.0)
    sample_out = (x.float() / div)

    print('Dataset total num', total_num)
    print('Total dataset class num', data_num.tolist())
    print('per-client target', q.tolist())
    # print('clients totals', x.sum(dim=1).tolist())

    with open(os.path.join(args.result_dir, 'clients_data_num.pickle'), 'wb') as f:
        pickle.dump(clients_data_num, f)

    return clients_data, sample_out
