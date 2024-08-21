import numpy as np
import sys
import os
import torch
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from datetime import datetime
from tqdm import tqdm

import torch.nn as nn

from .utils import *
from .classifier import *
from .client import *



def set_parameters(source_model, target_model):
    params = source_model.state_dict()
    target_model.load_state_dict(params)
    return target_model


def train(client_list, valid_loader, valid_nfeat, args):
    device = args.device
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'{timestamp} start')

    file_path = args.output_dir
    os.makedirs(file_path, exist_ok=True)
    
    best_valid_loss = float("Inf")
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    epoch_list = []
    early_stop = 0

    criterion = nn.CrossEntropyLoss()

    model = GraphSAGE(args)

    from pdb import set_trace
    
    for epoch in range(args.max_epoch):
        y_pred = []
        y_true = []
        
        params_list = []
        running_loss_client_list = []
        for client_id, client in enumerate(client_list):
            client.set_params(model.state_dict())
            
            running_loss_client_list.append(client.train())
            
            client.model.to('cpu')
            params_list.append(client.get_params())

        # print(f'[Epoch {epoch}] {np.mean(running_loss_client_list)} {running_loss_client_list}')

        # set_trace()
        params_avg = {}
        for key in model.state_dict().keys():
            if key == 'batch_norm.num_batches_tracked':
                torch.mean(torch.stack([params['batch_norm.num_batches_tracked'] for params in params_list]), dim=0, dtype=torch.float).long()
            else:
                params_avg[key] = torch.mean(torch.stack([params[key] for params in params_list]), dim=0)

        # set_trace()
        model.load_state_dict(params_avg)
        


        model.to(device)
        model.eval()
        with torch.no_grad():
            pair_to_score = defaultdict(dict)
            
            for input_nodes, edge_sub, blocks in valid_loader:
                batch_inputs, batch_labels = load_subtensor(*valid_nfeat, edge_sub, input_nodes, device)
                blocks = [block.int().to(device) for block in blocks]

                batch_pred, attn = model(edge_sub, blocks, *batch_inputs)
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
        average_valid_loss = valid_running_loss / len(valid_loader)
        train_loss_list.append(average_train_loss)
        valid_loss_list.append(average_valid_loss)
        epoch_list.append(epoch + 1)

        running_loss = 0.0                
        valid_running_loss = 0.0
        model.train()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('{} Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'
              .format(timestamp, epoch + 1, args.max_epoch,
                      average_train_loss, average_valid_loss, precision, recall, f1), flush=True)
        
        
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