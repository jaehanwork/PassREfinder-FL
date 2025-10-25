import numpy as np
import sys
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import dgl
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm
from random import sample
import torch.nn.functional as F

from .utils import *
from .classifier import *
from .client import load_subtensor
from .evaluate_ranking import *

def get_train_data_loader(g, nodes, batch_size, gnn_depth, suffle=False, device='cpu'):
    reverse_eid = {('website', 'reuse', 'website'): g.edge_ids(g.edges(etype='reuse')[1], g.edges(etype='reuse')[0], etype='reuse')}
    
    ori_id_to_id = dict(zip(g.ndata['_ID'].tolist(), g.nodes().tolist()))
    node_sub = [ori_id_to_id[n.item()] for n in nodes]
    target_eid = torch.unique(torch.cat([g.in_edges(node_sub, etype='reuse', form='eid'), g.out_edges(node_sub, etype='reuse', form='eid')]))

    sampler = dgl.dataloading.as_edge_prediction_sampler(dgl.dataloading.MultiLayerFullNeighborSampler(gnn_depth), exclude='reverse_id', reverse_eids=reverse_eid)

    dataloader = dgl.dataloading.DataLoader(
            g,
            {'reuse': target_eid},
            sampler,
            batch_size=batch_size,
            device=device,
            shuffle=suffle,
            drop_last=False,
            num_workers=0)
    
    return dataloader

def get_test_data_loader(g, test_eids, batch_size, gnn_depth, suffle=False, device='cpu'):
    reverse_eid = {('website', 'reuse', 'website'): g.edge_ids(g.edges(etype='reuse')[1], g.edges(etype='reuse')[0], etype='reuse')}
    
    sampler = dgl.dataloading.as_edge_prediction_sampler(dgl.dataloading.MultiLayerFullNeighborSampler(gnn_depth), exclude='reverse_id', reverse_eids=reverse_eid)

    dataloader = dgl.dataloading.DataLoader(
            g,
            {'reuse': test_eids},
            sampler,
            batch_size=batch_size,
            device=device,
            shuffle=suffle,
            drop_last=False,
            num_workers=0)
    
    return dataloader

def pop_node_feature(g):
    nfeat_n = g.ndata.pop('numerical')
    nfeat_c = g.ndata.pop('categorical')
    nfeat_e = g.ndata.pop('url_encoding')
    nfeat_ip = g.ndata.pop('ip')
    nfeat_text = g.ndata.pop('text')

    return nfeat_n, nfeat_c, nfeat_e, nfeat_ip, nfeat_text

class PassREfinder_Base(object):
    def __init__(self, g_dict, target_eids_dict, node_set_list, args):
        super(PassREfinder_Base, self).__init__()
        self.args = args
        self.g_dict = g_dict
        self.node_set_list = node_set_list
        self.target_eids_dict = target_eids_dict
        
        self.model = GraphSAGE(args)

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.train_nfeat = None
        self.valid_nfeat = None
        self.test_nfeat = None

        self._init_dataloader()
        self._init_nfeat()

        self.optimizer = Adam(self.model.parameters(), lr=float(self.args.max_lr))
        self.scheduler = OneCycleLR(self.optimizer, max_lr=args.max_lr, pct_start=args.warmup,
                           total_steps=len(self.train_loader)*args.max_epoch, epochs=args.max_epoch, anneal_strategy='cos')
        
    def _init_dataloader(self):
        self.train_loader = get_train_data_loader(self.g_dict['train'], self.g_dict['train'].nodes(), self.args.train_batch_size, self.args.gnn_depth, suffle=False, device=self.args.device)
        self.valid_loader = get_test_data_loader(self.g_dict['valid'], self.target_eids_dict['valid'], self.args.eval_batch_size, self.args.gnn_depth, suffle=False, device=self.args.device)
        self.test_loader = get_test_data_loader(self.g_dict['test'], self.target_eids_dict['test'], self.args.eval_batch_size, self.args.gnn_depth, suffle=False, device=self.args.device)

    def _init_nfeat(self):
        self.train_nfeat = pop_node_feature(self.g_dict['train'])
        self.valid_nfeat = pop_node_feature(self.g_dict['valid'])
        self.test_nfeat = pop_node_feature(self.g_dict['test'])

    def train(self):
        device = self.args.device
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'{timestamp} start')
    
        file_path = self.args.output_dir
        
        best_valid_loss = float("Inf")
        running_loss = 0.0
        valid_running_loss = 0.0
        global_step = 0
        train_loss_list = []
        valid_loss_list = []
        training_state_list = []
        early_stop = 0
    
        criterion = nn.CrossEntropyLoss()

        self.model.to(device)
        
        for epoch in range(self.args.max_epoch):
            y_pred = []
            y_true = []
            self.model.train()

            for input_nodes, edge_sub, blocks in tqdm(self.train_loader, total=len(self.train_loader)):
                batch_inputs, batch_labels = load_subtensor(*self.train_nfeat, edge_sub, input_nodes, device)
                blocks = [block.int().to(device) for block in blocks]
                
                
                batch_pred, attn = self.model(edge_sub, blocks, *batch_inputs)
                loss = criterion(batch_pred.squeeze(1), batch_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
    
                running_loss += loss.item()
                global_step += 1

            self.model.eval()
            with torch.no_grad():
                for input_nodes, edge_sub, blocks in tqdm(self.valid_loader):
                    batch_inputs, batch_labels = load_subtensor(*self.valid_nfeat, edge_sub, input_nodes, device)
                    blocks = [block.int().to(device) for block in blocks]
    
                    batch_pred, attn = self.model(edge_sub, blocks, *batch_inputs)
                    loss = criterion(batch_pred.squeeze(1), batch_labels)
    
                    valid_running_loss += loss.item()
    
                    _score_list = batch_pred.detach().cpu().numpy().tolist()
                    score_list = np.argmax(_score_list, axis=1)
                    
                    y_true.extend(batch_labels.detach().cpu().long().numpy().tolist())
                    y_pred.extend(score_list)
    
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
    
            average_train_loss = running_loss / len(self.train_loader)
            average_valid_loss = valid_running_loss / len(self.valid_loader)
            train_loss_list.append(average_train_loss)
            valid_loss_list.append(average_valid_loss)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            training_state_list.append({'epoch': epoch,
                                        'current_time': f'{timestamp}',
                                        'train_loss': average_train_loss,
                                        'valid_loss': average_valid_loss,
                                        'eval_f1': f1,
                                        'eval_recall': recall,
                                        'eval_precision': precision})
    
            running_loss = 0.0                
            valid_running_loss = 0.0
    
            
            print('{} Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'
                  .format(timestamp, epoch + 1, self.args.max_epoch,
                          average_train_loss, average_valid_loss, precision, recall, f1), flush=True)
            if best_valid_loss > average_valid_loss:
                best_valid_loss = average_valid_loss    
                save_model(file_path, f'model.pt', self.model, training_state_list)
                print(f'Save model!')

    def evaluate(self, eval_ranking=False):
        print('Evaluating...')
        
        y_true = []
        y_pred = []

        if eval_ranking == True:
            ori_id_to_id = dict(zip(self.g_dict['test'].ndata['_ID'].tolist(), self.g_dict['test'].nodes().tolist()))
            train_node_dict = {}
            for i, node_set_ori in enumerate(self.node_set_list[:2]):
                node_set = [ori_id_to_id[nid] for nid in node_set_ori.numpy().tolist()]
                train_node_dict.update(dict.fromkeys(node_set, i))
            test_node_dict = {}
            for i, node_set_ori in enumerate(self.node_set_list[2:]):
                node_set = [ori_id_to_id[nid] for nid in node_set_ori.numpy().tolist()]
                test_node_dict.update(dict.fromkeys(node_set, i))
        
            sample_size = 64
            _reuse_rate_top_test = defaultdict(dict)
            for id1, id2, score in zip(self.g_dict['test'].edges(etype='reuse')[0].tolist(), self.g_dict['test'].edges(etype='reuse')[1].tolist(), self.g_dict['test'].edata['rate'][('website', 'reuse', 'website')].tolist()):

                if (id1 not in test_node_dict) and (id2 not in test_node_dict):
                    continue
                elif (id1 in test_node_dict) and (id2 in train_node_dict):
                    _reuse_rate_top_test[id1][id2] = score
                elif (id1 in train_node_dict) and (id2 in test_node_dict):
                    _reuse_rate_top_test[id1][id2] = score
                elif test_node_dict[id1] != test_node_dict[id2]: 
                    _reuse_rate_top_test[id1][id2] = score
                elif test_node_dict[id1] == test_node_dict[id2]:
                    continue
                else:
                    assert(0)
            
            reuse_rate_top_test = {}
            for id1, id2_list in _reuse_rate_top_test.items():
                if len(id2_list) >= sample_size:
                    id2_list = {k: v for k, v in sample(id2_list.items(), sample_size)}
                    id2_list = {k: v for k, v in sorted(id2_list.items(), key=lambda x: x[1], reverse=True)}
                    reuse_rate_top_test[id1] = id2_list

        self.model.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            pair_to_score = defaultdict(dict)
            
            for input_nodes, edge_sub, blocks in tqdm(self.test_loader):
                batch_inputs, batch_labels = load_subtensor(*self.test_nfeat, edge_sub, input_nodes, self.args.device)
                blocks = [block.int().to(self.args.device) for block in blocks]
                
                batch_pred, attn = self.model(edge_sub, blocks, *batch_inputs)

                
                score_values = F.softmax(batch_pred.detach().cpu())[:,1].numpy().tolist()
                if eval_ranking == True:
                    edge_tensor = self.g_dict['test'].find_edges(edge_sub.edata['_ID'][('website', 'reuse', 'website')].detach().cpu().numpy(), etype='reuse')
                    edge_list = [(src, dst) for src, dst in zip(edge_tensor[0].tolist(), edge_tensor[1].tolist())]
                    
                    for edge, score in zip(edge_list, score_values):
                        pair_to_score[edge[0]][edge[1]] = score

                _score_list = batch_pred.detach().cpu().numpy().tolist()
                score_list = np.argmax(_score_list, axis=1)
            
                y_true.extend(batch_labels.detach().cpu().long().numpy().tolist())
                y_pred.extend(score_list)

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        result = {'eval_f1': f1,
                  'eval_precision': precision,
                  'eval_recall': recall,}

        if eval_ranking == True:
            result['prec@k'] = evaluate_precision(pair_to_score, reuse_rate_top_test)
            result['nDCG@k'] = evaluate_ndcg(pair_to_score, reuse_rate_top_test)
            return result, pair_to_score
            
        return result

    def print_graph(self):
        train_g_list = self.g_split['train']
        valid_g = self.g_split['valid']
        test_g = self.g_split['test']
        train_eid_list = self.get_target_eid('train')
        valid_eid = self.get_target_eid('valid')
        test_eid = self.get_target_eid('test')

        train_node_num_list = []
        train_edge_num_list = []
        train_target_edge_num_list = []
        for train_g, train_eid in zip(train_g_list, train_eid_list):
            train_node_num_list.append(train_g.number_of_nodes())
            train_edge_num_list.append(train_g.number_of_edges(etype='reuse'))
            train_target_edge_num_list.append(len(train_eid))
        print(f"""---------------------Data statistics---------------------
#[Train] nodes: {train_node_num_list}, edges: {train_edge_num_list}, target edges: {train_target_edge_num_list}
#[Valid] nodes: {valid_g.number_of_nodes()}, edges: {valid_g.number_of_edges(etype='reuse')}, target edges: {len(valid_eid)}
#[Test]  nodes: {test_g.number_of_nodes()}, edges: {test_g.number_of_edges(etype='reuse')}, target edges: {len(test_eid)}
---------------------------------------------------------\n""")