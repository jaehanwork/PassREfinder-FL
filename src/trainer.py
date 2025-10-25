import numpy as np
import sys
import os
import torch
import torch.nn as nn
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
from .client import *
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

class PassREfinder_FL(object):
    def __init__(self, g_dict, node_set_list, target_eids_dict, args):
        super(PassREfinder_FL, self).__init__()
        self.args = args
        self.g_dict = g_dict
        self.node_set_list = node_set_list
        self.target_eids_dict = target_eids_dict
        
        self.model = GraphSAGE(args)

        self.train_loader_list = []
        self.valid_loader = None
        self.test_loader = None

        self.train_nfeat_list = []
        self.valid_nfeat = None
        self.test_nfeat = None

        self.client_list = []

        self._init_dataloader()
        self._init_nfeat()
        self._init_client()
        
    def _init_dataloader(self):
        for g, nodes in zip(self.g_dict['train'], self.node_set_list):
            self.train_loader_list.append(get_train_data_loader(g, nodes, self.args.train_batch_size, self.args.gnn_depth, suffle=False, device=self.args.device))

        self.valid_loader = get_test_data_loader(self.g_dict['valid'], self.target_eids_dict['valid'], self.args.eval_batch_size, self.args.gnn_depth, suffle=False, device=self.args.device)
        self.test_loader = get_test_data_loader(self.g_dict['test'], self.target_eids_dict['test'], self.args.eval_batch_size, self.args.gnn_depth, suffle=False, device=self.args.device)

    def _init_nfeat(self):
        for train_g in self.g_dict['train']:
            self.train_nfeat_list.append(pop_node_feature(train_g))
        self.valid_nfeat = pop_node_feature(self.g_dict['valid'])
        self.test_nfeat = pop_node_feature(self.g_dict['test'])

    def _init_client(self):
        for i in range(self.args.nclient):
            self.client_list.append(Client(self.train_loader_list[i], self.train_nfeat_list[i], self.args))

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
        
        running_loss_list = []
        for epoch in range(self.args.max_epoch):
            y_pred = []
            y_true = []
            
            params_list = []
            running_loss_client_list = []
            for client_id, client in tqdm(enumerate(self.client_list), total=len(self.client_list)):
                client.set_params(self.model.state_dict())
                
                running_loss_client_list.append(client.train())
                
                client.model.to('cpu')
                params_list.append(client.get_params())

            running_loss_list.append(running_loss_client_list)
    
            # print(f'[Epoch {epoch}] {np.mean(running_loss_client_list)} {running_loss_client_list}')
    
            params_avg = {}
            for key in self.model.state_dict().keys():
                if key == 'batch_norm.num_batches_tracked':
                    torch.mean(torch.stack([params['batch_norm.num_batches_tracked'] for params in params_list]), dim=0, dtype=torch.float).long()
                else:
                    params_avg[key] = torch.mean(torch.stack([params[key] for params in params_list]), dim=0)
    
            self.model.load_state_dict(params_avg)
            
    
    
            self.model.to(device)
            self.model.eval()
            with torch.no_grad():
                pair_to_score = defaultdict(dict)
                for input_nodes, edge_sub, blocks in tqdm(self.valid_loader):
                    batch_inputs, batch_labels = load_subtensor(*self.valid_nfeat, edge_sub, input_nodes, device)
                    blocks = [block.int().to(device) for block in blocks]
    
                    batch_pred, attn = self.model(edge_sub, blocks, *batch_inputs)
                    loss = criterion(batch_pred.squeeze(1), batch_labels)
    
                    valid_running_loss += loss.item()
                    
    #                 edge_tensor = valid_g.find_edges(edge_sub.edata['_ID'].detach().cpu().numpy())
    #                 edge_list = [(valid_id_to_ori_id[src], valid_id_to_ori_id[dst]) for src, dst in zip(edge_tensor[0].tolist(), edge_tensor[1].tolist())]
                
    #                 score_list = batch_pred.squeeze(1).detach().cpu().numpy().tolist()
                
    #                 for edge, score in zip(edge_list, score_list):
    #                     pair_to_score[edge[0]][edge[1]] = score
    
                    _score_list = batch_pred.detach().cpu().numpy().tolist()
                    score_list = np.argmax(_score_list, axis=1)
                    
                    y_true.extend(batch_labels.detach().cpu().long().numpy().tolist())
                    y_pred.extend(score_list)
                
                # evaluate_precision(pair_to_score, reuse_rate_top_valid)
                # evaluate_ndcg(pair_to_score, reuse_rate_top_valid)
                
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
    
            average_train_loss = np.mean(running_loss_client_list)
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
            
        #     if best_valid_loss > average_valid_loss:
        #         best_valid_loss = average_valid_loss
        #         save_checkpoint(os.path.join(file_path, f'model_{epoch}.pt'), model, best_valid_loss)
        #         save_metrics(os.path.join(file_path, f'metrics_{epoch}.pt'), train_loss_list, valid_loss_list)
        #         early_stop = 0
        #     else:
        #         early_stop += 1
        #         if early_stop == 20:
        #             break
    
        # save_metrics(os.path.join(file_path, f'metrics_{epoch}.pt'), train_loss_list, valid_loss_list)
        # print_and_log('Finished Training!')

        with open(os.path.join(file_path, 'running_loss.json'), 'w') as f:
            json.dump(running_loss_list, f)

    def evaluate(self, eval_ranking=False):
        print('Evaluating...')
        
        y_true = []
        y_pred = []

        if eval_ranking == True:
            node_dict = {}
            for i, node_set in enumerate(self.node_set_list):
                node_dict.update(dict.fromkeys(node_set.numpy().tolist(), i))

            sample_size = 64
            _reuse_rate_top_test = defaultdict(dict)
            for id1, id2, score in zip(self.g_dict['test'].edges(etype='reuse')[0].tolist(), self.g_dict['test'].edges(etype='reuse')[1].tolist(), self.g_dict['test'].edata['rate'][('website', 'reuse', 'website')].tolist()):
                if node_dict[id1] != node_dict[id2]:
                    _reuse_rate_top_test[id1][id2] = score
            
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