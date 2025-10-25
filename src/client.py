import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from .classifier import *

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

class Client(object):
    def __init__(self, train_loader, train_nfeat, args):
        super(Client, self).__init__()
        self.args = args
        self.model = GraphSAGE(args)
        self.train_nfeat = train_nfeat
        self.train_loader = train_loader
        # self.load_subtensor = load_subtensor
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=float(self.args.max_lr))
        self.scheduler = OneCycleLR(self.optimizer, max_lr=args.max_lr, pct_start=args.warmup,
                           total_steps=len(train_loader)*args.max_epoch, epochs=args.max_epoch, anneal_strategy='cos')

    def set_params(self, params):
        self.model.load_state_dict(params)

    def get_params(self):
        return self.model.state_dict()
        
    def train(self):
        running_loss = 0.0
        # global_step = 0
        train_loss_list = []

        self.model.to(self.args.device)
        self.model.train()
        for input_nodes, edge_sub, blocks in self.train_loader:
            batch_inputs, batch_labels = load_subtensor(*self.train_nfeat, edge_sub, input_nodes, self.args.device)
            blocks = [block.int().to(self.args.device) for block in blocks]

            batch_pred, attn = self.model(edge_sub, blocks, *batch_inputs)
            loss = self.criterion(batch_pred.squeeze(1), batch_labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)