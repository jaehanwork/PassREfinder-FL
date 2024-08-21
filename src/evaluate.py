import numpy as np
import sys
import torch
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def load_subtensor(nfeat_n, nfeat_c, nfeat_e, nfeat_ip, nfeat_text, edge_sub, input_nodes, device='cpu'):
    nfeat_n = nfeat_n.to(device)
    nfeat_c = nfeat_c.to(device)
    nfeat_e = nfeat_e.to(device)
    nfeat_ip = nfeat_ip.to(device)
    nfeat_text = nfeat_text.to(device)
    input_nodes = input_nodes.to(device)
    
    batch_inputs_n = nfeat_n[input_nodes].to(device)
    batch_inputs_c = nfeat_c[input_nodes].to(device)
    batch_inputs_e = nfeat_e[input_nodes].to(device)
    batch_inputs_ip = nfeat_ip[input_nodes].to(device)
    batch_inputs_text = nfeat_text[input_nodes].to(device)
    edge_sub = edge_sub.to(device)
    batch_labels = edge_sub.edata['label'][('website', 'reuse', 'website')]
    
    return (batch_inputs_n, batch_inputs_c, batch_inputs_e, batch_inputs_ip, batch_inputs_text), batch_labels

def evaluate(model, data_loader, nfeat, device='cpu'):
    print('Evaluating...')
    
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        pair_to_score = defaultdict(dict)
        
        for input_nodes, edge_sub, blocks in data_loader:
            batch_inputs, batch_labels = load_subtensor(*nfeat, edge_sub, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            
            batch_pred, attn = model(edge_sub, blocks, *batch_inputs)

            _score_list = batch_pred.detach().cpu().numpy().tolist()
            score_list = np.argmax(_score_list, axis=1)
        
            y_true.extend(batch_labels.detach().cpu().long().numpy().tolist())
            y_pred.extend(score_list)
        
    result = classification_report(y_true, y_pred, digits=4)
    
    sys.stdout.write("\033[F")
    print('Evaluating... Done')
    print('----------------Classification report----------------')
    print(result)