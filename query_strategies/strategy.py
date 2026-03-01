import copy
import numpy as np
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Subset, DataLoader


def make_balanced_indices(y, max_cap=None, seed=42):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    classes, counts = np.unique(y, return_counts=True)
    mask = counts > 0
    classes, counts = classes[mask], counts[mask]

    if len(classes) == 0:
        return []

    target = counts.max() if max_cap is None else min(counts.max(), max_cap)

    all_idx = []
    for c, n in zip(classes, counts):
        idx_c = np.where(y == c)[0]
        need = target - n
        if need <= 0:
            pick = rng.choice(idx_c, size=target, replace=False)
            all_idx.append(pick)
        else:
            extra = rng.choice(idx_c, size=need, replace=True)
            all_idx.append(np.concatenate([idx_c, extra], axis=0))

    balanced_idx = np.concatenate(all_idx, axis=0)
    rng.shuffle(balanced_idx)
    return balanced_idx.tolist()

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.labels = [self.dataset[idx][1] for idx in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label, item
    
    
class Strategy:
    def __init__(self, dataset_query, dataset_train, dataset_test, net, args):
        self.dataset_query = dataset_query
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.net = net
        self.args = args
        self.local_net_dict = {}
        self.loss_func = nn.CrossEntropyLoss()
        self.ratio_dict = {}
        
    def query(self, label_idx, unlabel_idx):
        pass

    
    def test(self, net, test_idx):
        loader_te = DataLoader(DatasetSplit(self.dataset_test, test_idx), shuffle=False)
        data_nums = len(test_idx)

        net.eval()
        correct = 0
        with torch.no_grad():
            for data, target, idx in loader_te:
                if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
                    target = target.squeeze().long()

                if self.args.gpu != -1:
                    data, target = data.to(self.args.device), target.to(self.args.device)
                output, emb = net(data)

                # get the index of the max log-probability
                y_pred = output.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            
        accuracy = 100.00 * float(correct) / data_nums
        return accuracy
    
    def predict_prob(self, unlabel_idxs, net=None):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, unlabel_idxs), shuffle=False)
        
        if net is None:
            net = self.net
            
        net.eval()
        probs = torch.zeros([len(unlabel_idxs), self.args.num_classes])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                output, emb = net(x)
                probs[idxs] = torch.nn.functional.softmax(output, dim=1).cpu().data
        return probs
    
    
    def predict_balanced_prob_mean(self, label_idxs, net=None):
        dataset = DatasetSplit(self.dataset_query, label_idxs)
        balanced_idx = make_balanced_indices(dataset.labels)
        loader_te = DataLoader(Subset(dataset, balanced_idx), shuffle=False)
        
        if net is None:
            net = self.net
            
        net.eval()
        sum_probs = torch.zeros(self.args.num_classes, dtype=torch.float32)
        total = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                output, emb = net(x)
                p = torch.softmax(output, dim=1).cpu()
                sum_probs += p.sum(dim=0)
                total += p.size(0)
        mean_probs = sum_probs / max(total, 1)
        return mean_probs


    def get_embedding(self, data_idxs, net=None):
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        if net is None:
            net = self.net
        
        net.eval()
        embedding = torch.zeros([len(data_idxs), net.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                out, e1 = net(x)
                embedding[idxs] = e1.data.cpu()
        
        return embedding
    
    def get_class_prototypes(self, data_idxs, net=None):
        loader = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        if net is None:
            net = self.net
        
        net.eval()
        device = self.args.device
        C = self.args.num_classes 
        D = net.get_embedding_dim()
        prototypes = torch.zeros(C, D, device=device)
        counts     = torch.zeros(C, device=device)

        with torch.no_grad():
            for x, y, _ in loader:
                x = Variable(x.to(device))

                if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
                    y = y[0].to(device).long()
                else:
                    y = y.to(device).long()

                _, emb = net(x)
                emb = F.normalize(emb, dim=1)

                prototypes.index_add_(0, y, emb)
                counts.index_add_(0, y, torch.ones(y.size(0), device=device, dtype=prototypes.dtype))

        mask = counts > 0
        if mask.any():
            prototypes[mask] = prototypes[mask] / counts[mask].unsqueeze(1)

        return prototypes, counts
    
    # gradient embedding (assumes cross-entropy loss)
    def get_grad_embedding(self, data_idxs, net1, net2):
            
        embDim = net1.get_embedding_dim()
        net1.eval()
        net2.eval()
        
        nLab = self.args.num_classes 
        embedding = np.zeros([len(data_idxs), embDim * nLab])
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                _, out = net1(x)
                cout, _ = net2(x)
                out = out.data.cpu().numpy()
                
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
        
    def get_grad_embedding_v2(self, data_idxs, net1, net2):
            
        embDim = net1.get_embedding_dim()
        net1.eval()
        net2.eval()
        
        nLab = self.args.num_classes 
        embedding = np.zeros([len(data_idxs), embDim * nLab])
        labels = torch.full((len(data_idxs),), -1, dtype=torch.long) 
        loader_te = DataLoader(DatasetSplit(self.dataset_query, data_idxs), shuffle=False)
        
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.args.device)), Variable(y.to(self.args.device))
                _, out = net1(x)
                cout, _ = net2(x)
                out = out.data.cpu().numpy()
                
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)
                
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == y[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                    labels[idxs[j]] = y[j]
            return torch.Tensor(embedding), labels
        
    
    
    def training_local_only(self, label_idxs, finetune=False):
        finetune_ep = 50
        
        local_net = deepcopy(self.net)
        if not finetune: 
            # Training Local Model from the scratch
            local_net.load_state_dict(self.args.raw_ckpt)
        # else: fine-tune from global model checkpoint
        
        # train and update
        label_train = DataLoader(DatasetSplit(self.dataset_train, label_idxs), batch_size=self.args.local_bs, shuffle=True)
        
        optimizer = torch.optim.SGD(local_net.parameters(), 
                                    lr=self.args.lr, 
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(finetune_ep * 3 / 4)], gamma=self.args.lr_decay)
        
        # start = datetime.now()
        for epoch in range(finetune_ep):
            local_net.train()
            for images, labels, _ in label_train:
                if self.args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
                    labels = labels.squeeze().long()
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                optimizer.zero_grad()
                output, emb = local_net(images)
                
                if output.shape[0] == 1:
                    labels = labels.reshape(1,)

                loss = self.loss_func(output, labels)
                loss.backward()
                
                optimizer.step()
                scheduler.step()
                
            correct, cnt = 0., 0.
            local_net.eval()
            with torch.no_grad():
                for images, labels, _ in label_train:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    output, _ = local_net(images)
                    
                    y_pred = output.data.max(1, keepdim=True)[1]
                    correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
                    cnt += len(labels)
        
                acc = correct / cnt
                if acc >= 0.99:
                    break
        
        # time = datetime.now() - start
        # print('Local-only model fine-tuning takes {}'.format(time))

        return local_net