from __future__ import annotations

import copy
import math
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F

from .strategy import Strategy

from scipy.spatial.distance import jensenshannon

from collections import defaultdict, Counter
from typing import Optional


def proto_probs_from_embeddings(
    E_u: torch.Tensor,         # [N_u, D]
    prototypes: torch.Tensor,  # [C, D]
    counts: torch.Tensor=None, # [C]
    temperature: float = 1.0
) -> torch.Tensor:

    device = E_u.device

    if not torch.allclose(E_u.norm(dim=1), torch.ones(E_u.size(0), device=device), atol=1e-3):
        E_u = F.normalize(E_u, dim=1)
    if not torch.allclose(prototypes.norm(dim=1), torch.ones(prototypes.size(0), device=device), atol=1e-3):
        nz = prototypes.norm(dim=1) > 0
        prototypes = prototypes.clone()
        prototypes[nz] = F.normalize(prototypes[nz], dim=1)

    sim = E_u @ prototypes.t()                   # [N_u, C]

    if counts is not None:
        mask = (counts.to(device) > 0).view(1, -1)     # [1, C]
        neg_inf = torch.finfo(sim.dtype).min
        sim = torch.where(mask, sim, torch.full_like(sim, neg_inf))

    logits = sim / temperature
    probs = F.softmax(logits, dim=1)            # [N_u, C]
    return probs


class FairFALSampling(Strategy):
    
    def query_model_selection(self, user_idx, label_idxs, unlabel_idxs, count_label):
        unlabel_idxs = np.array(unlabel_idxs)
        
        local_net = self.training_local_only(label_idxs)
        self.local_net_dict[user_idx] = local_net

        global_mean_probs = self.predict_balanced_prob_mean(label_idxs, self.net)
        local_mean_probs = self.predict_balanced_prob_mean(label_idxs, local_net)
        global_mean_probs = global_mean_probs.numpy()
        local_mean_probs = local_mean_probs.numpy()
        
        distance = 0.5*(np.abs(global_mean_probs - local_mean_probs)/(0.5*global_mean_probs + 0.5*local_mean_probs)).mean()
        
        idx = np.fromiter(count_label.keys(), dtype=int)
        global_mean_probs = global_mean_probs[idx]
        local_mean_probs = local_mean_probs[idx]
        
        global_ratio = global_mean_probs.min() / global_mean_probs.max()
        local_ratio = local_mean_probs.min() / local_mean_probs.max()
        
        self.ratio_dict[user_idx] = distance
        
        return global_ratio
    
        
    def query(self, user_idx, label_idxs, unlabel_idxs, test_idxs, count_label, mean_global_ratio, n_query=100):
        unlabel_idxs = np.array(unlabel_idxs)
        
        local_net = self.local_net_dict[user_idx]
        
        t = 0.5
        ratio = 1 - (1 - t) * (self.ratio_dict[user_idx] + mean_global_ratio) + (1 - 2*t) * (self.ratio_dict[user_idx] * mean_global_ratio)
        print("ratio: ", ratio, " distance: ", self.ratio_dict[user_idx])
        if ratio > 0.75:
            choose_net1 = self.net
        else:
            choose_net1 = local_net
        
        if mean_global_ratio < 0.5:
            choose_net2 = self.net

        else:
            choose_net2 = local_net
                    
        global_acc = self.test(self.net, test_idxs)
        local_acc = self.test(local_net, test_idxs)

        probs = self.predict_prob(unlabel_idxs, choose_net1)
        
        H = -(probs * (probs + 1e-12).log()).sum(dim=1)                 
        H_tilde = H / torch.log(torch.tensor(float(self.args.num_classes)))
        
        prototypes, counts = self.get_class_prototypes(label_idxs, self.net)
        embeddings = self.get_embedding(unlabel_idxs, self.net)
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=1e-12).to(self.args.device)
        probs = proto_probs_from_embeddings(embeddings, prototypes, counts=counts, temperature=1.5)
        
        feats_unl = self.get_grad_embedding(unlabel_idxs, self.net, choose_net2)
        
        feats_lab, labels = self.get_grad_embedding_v2(label_idxs, self.net, choose_net2)
            
        selected_ids, per_class, quota = balanced_sample_by_pred_kmeans_maxscore(
                probs=probs,
                unlabel_idxs=unlabel_idxs,
                n_total=n_query,
                scores=H_tilde,
                features=feats_unl,
                labeled_kc_features=feats_lab,
                labeled_labels=labels
            )
            
        return np.array(selected_ids), global_acc, local_acc
    

def priors_from_counter(counter: Counter, num_classes: int, alpha: float = 1.0):
    counts = np.array([counter.get(c, 0) for c in range(num_classes)], dtype=np.float64)
    N = counts.sum()
    pi = (counts + alpha) / (N + alpha * num_classes)
    neg_log_pi = -np.log(pi)
    return counts, pi, neg_log_pi

@torch.no_grad()
def balanced_sample_by_pred_kmeans_maxscore(
    probs: torch.Tensor,             
    unlabel_idxs,                   
    n_total: int,                   
    scores: Optional[torch.Tensor] = None,   
    features: Optional[torch.Tensor] = None,
    over_sample_k: float = 4.0,     
    distance: str = "euclidean",    
    labeled_kc_features: Optional[torch.Tensor] = None, 
    labeled_labels:     Optional[torch.Tensor] = None,
    anchor_mode: str = "same_class"                         # "same_class" | "all"
):

    N, C = probs.shape
    n_total = int(min(int(n_total), N))

    scores_vec = scores.detach().cpu().numpy()
    unlabel_idxs = np.asarray(unlabel_idxs, dtype=int)

    X_all = features.detach().cpu().numpy()
    if distance == "cosine":
        X_all = X_all / (np.linalg.norm(X_all, axis=1, keepdims=True) + 1e-12)

    use_anchors = (
        labeled_kc_features is not None and
        labeled_labels is not None and
        labeled_kc_features.shape[0] > 0
    )
    if use_anchors:
        X_lab_kc = labeled_kc_features.detach().cpu().numpy()
        y_lab = labeled_labels.detach().cpu().numpy().astype(int)
        if distance == "cosine":
            X_lab_kc = X_lab_kc / (np.linalg.norm(X_lab_kc, axis=1, keepdims=True) + 1e-12)
 
    pred = probs.argmax(dim=1).cpu().numpy()
    buckets = defaultdict(list)
    for i, c in enumerate(pred):
        buckets[int(c)].append(i)

    base, rem = n_total // C, n_total % C
    quota = {c: base for c in range(C)}
    class_order = sorted(range(C), key=lambda c: len(buckets.get(c, [])), reverse=True)
    for c in class_order[:rem]:
        quota[c] += 1
    leftover = 0
    for c in range(C):
        size_c = len(buckets.get(c, []))
        if quota[c] > size_c:
            leftover += quota[c] - size_c
            quota[c] = size_c
    if leftover > 0:
        can_grow = [c for c in class_order if len(buckets.get(c, [])) > quota[c]]
        j = 0
        while leftover > 0 and can_grow:
            c = can_grow[j % len(can_grow)]
            if quota[c] < len(buckets[c]):
                quota[c] += 1
                leftover -= 1
            j += 1

    def pairwise_distance_kc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
        if distance == "cosine":
            a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
            b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
            S = a_n @ b_n.T
            np.clip(S, -1.0, 1.0, out=S)
            return 1.0 - S
        aa = (a * a).sum(axis=1, keepdims=True)
        bb = (b * b).sum(axis=1, keepdims=True).T
        return aa + bb - 2.0 * (a @ b.T)

    def kcenter_seeded(Xcand: np.ndarray, k: int, seed_scores: Optional[np.ndarray], Xanchors: Optional[np.ndarray]) -> np.ndarray:
        M = Xcand.shape[0]
        if k <= 0 or M == 0:
            return np.empty((0,), dtype=int)
        if M <= k:
            return np.arange(M, dtype=int)

        if Xanchors is not None and Xanchors.shape[0] > 0:
            min_dist = pairwise_distance_kc(Xcand, Xanchors).min(axis=1)
            selected = []
        else:
            first = int(np.argmax(seed_scores)) if seed_scores is not None else 0
            selected = [first]

            min_dist = pairwise_distance_kc(Xcand, Xcand[[first]]).reshape(-1)  # [M]
            min_dist[first] = -np.inf 

        while len(selected) < k:
            j = int(np.argmax(min_dist))
            if not np.isfinite(min_dist[j]):
                break
            selected.append(j)
            d_new = pairwise_distance_kc(Xcand, Xcand[[j]]).reshape(-1)

            min_dist = np.minimum(min_dist, d_new) if (Xanchors is None or Xanchors.size == 0) else np.minimum(min_dist, d_new)

            min_dist[selected] = -np.inf
        return np.asarray(selected, dtype=int)

    per_class_selected = defaultdict(list)
    selected_global_all = []

    for c in range(C):
        idx_list = buckets.get(c, [])
        q_c = quota.get(c, 0)
        if q_c <= 0 or len(idx_list) == 0:
            continue

        idx_arr = np.asarray(idx_list, dtype=int)
        Xc_full = X_all[idx_arr]          # [Nc, Dk]
        Sc_full = scores_vec[idx_arr]
        Nc = len(idx_arr)

        m_c = int(min(Nc, max(1, int(np.floor(over_sample_k * q_c)))))
        cand_loc = np.argsort(-Sc_full)[:m_c]
        Xc_cand = Xc_full[cand_loc]
        Sc_cand = Sc_full[cand_loc]

        Xanchors = None
        if use_anchors:
            mask = (y_lab == c) if anchor_mode == "same_class" else np.ones_like(y_lab, dtype=bool)
            if mask.sum() > 0:
                Xanchors = X_lab_kc[mask]

        pick_loc_in_cand = kcenter_seeded(Xc_cand, q_c, Sc_cand, Xanchors)
        chosen_loc_in_full = cand_loc[pick_loc_in_cand]
        chosen_global_idx = idx_arr[chosen_loc_in_full]

        selected_global_all.extend(chosen_global_idx.tolist())
        per_class_selected[c].extend(unlabel_idxs[chosen_global_idx].tolist())

    selected_global_all = np.unique(np.asarray(selected_global_all, dtype=int))
    selected_global_idxs = unlabel_idxs[selected_global_all].tolist()
    return selected_global_idxs, per_class_selected, quota
