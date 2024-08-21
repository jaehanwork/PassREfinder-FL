import numpy as np
from dgl.nn.pytorch import Sequential
from .gnn import SAGEConvN

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GraphSAGE(nn.Module):
    def __init__(self, args):
        super(GraphSAGE, self).__init__()
        self.emb_dim = args.embed_size
        self.n_hidden = args.hidden_size
        
        self.agg_type = args.agg_type
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.LeakyReLU(args.relu)
        self.num_layers = args.gnn_depth
        
        self.batch_norm = nn.BatchNorm1d(5)
        
        self.embed_cat = nn.Embedding(24, self.emb_dim)
        
        self.embed_url = nn.Embedding(40, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, batch_first=True, bidirectional=True)
        self.fc_lstm = nn.Linear(self.emb_dim*2, self.emb_dim)
        
        self.attn_linear = nn.Linear(self.n_hidden, self.n_hidden, bias=False)
        self.attn = nn.Linear(self.n_hidden, 1)
        self.softmax = nn.Softmax(dim=1)
        
        self.conv1 = nn.ModuleList([SAGEConvN(self.emb_dim, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        self.conv2 = nn.ModuleList([SAGEConvN(5, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        self.conv3 = nn.ModuleList([SAGEConvN(32, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        self.conv4 = nn.ModuleList([SAGEConvN(self.emb_dim, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        self.conv5 = nn.ModuleList([SAGEConvN(768, self.n_hidden, aggregator_type=self.agg_type)] + 
                                    [SAGEConvN(self.n_hidden, self.n_hidden, aggregator_type=self.agg_type) for i in range(self.num_layers - 1)])
        
        self.fc = nn.Linear(self.n_hidden*2, self.n_hidden)
        self.fc_out = nn.Linear(self.n_hidden, 2)
        
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.fc(torch.cat([h_u, h_v], 1))
        score = self.relu(score)
        score = self.dropout(score)
        score = self.fc_out(score)
        attn = torch.cat([edges.src['attn'].squeeze(-1).unsqueeze(1), edges.src['attn'].squeeze(-1).unsqueeze(1)], dim=1)
        return {'score': score, 'attn': attn}

    def forward(self, edge_sub, blocks, inputs_n, inputs_c, inputs_e, inputs_ip, inputs_text):
        assert inputs_n.shape[1] == 5
        assert inputs_c.shape[1] == 1
        sec1, sec2, sec3, sec4, https = inputs_n[:, 0].unsqueeze(1), inputs_n[:, 1].unsqueeze(1), inputs_n[:, 2].unsqueeze(1), inputs_n[:, 3].unsqueeze(1), inputs_n[:, 4].unsqueeze(1)
        cat = inputs_c[:, 0]
        
        vec_cat = self.embed_cat(cat)
        
        vec_sec = self.batch_norm(torch.cat([sec1, sec2, sec3, sec4, https], dim=1))
        
        vec_ip = inputs_ip
        
        url_input = inputs_e
        input_lengths = torch.LongTensor([torch.max(url_input[i, :].data.nonzero())+1 for i in range(url_input.size(0))])
        vec_url_input = self.embed_url(url_input)
        packed_input = pack_padded_sequence(vec_url_input, input_lengths.tolist(), batch_first=True, enforce_sorted=False)
        _, (ht, ct) = self.lstm(packed_input)
        vec_url_output = torch.concat([ht[0], ht[-1]], dim=1)
        _vec_url = vec_url_output[packed_input.unsorted_indices]
        _vec_url = self.fc_lstm(_vec_url)
        _vec_url = self.relu(_vec_url)
        _vec_url = self.dropout(_vec_url)
        vec_url = _vec_url

        vec_content = inputs_text

        for block in blocks:
            neg_edge_index = (block.edata['mask'][('website', 'reuse', 'website')] != 0).nonzero().int().squeeze(1)
            block.remove_edges(neg_edge_index, etype='reuse')
        
        h1 = vec_cat
        for i in range(self.num_layers):
            h1 = self.conv1[i](blocks[i], h1)
            h1 = self.relu(h1)
            h1 = self.dropout(h1)

        h2 = vec_sec
        for i in range(self.num_layers):
            h2 = self.conv2[i](blocks[i], h2)
            h2 = self.relu(h2)
            h2 = self.dropout(h2)

        h3 = vec_ip
        for i in range(self.num_layers):
            h3 = self.conv3[i](blocks[i], h3)
            h3 = self.relu(h3)
            h3 = self.dropout(h3)

        h4 = vec_url
        for i in range(self.num_layers):
            h4 = self.conv4[i](blocks[i], h4)
            h4 = self.relu(h4)
            h4 = self.dropout(h4)

        h5 = vec_content
        for i in range(self.num_layers):
            h5 = self.conv5[i](blocks[i], h5)
            h5 = self.relu(h5)
            h5 = self.dropout(h5)
            
        h_stack = torch.stack([h1, h2, h3, h4, h5], dim=1)
        attn = self.softmax(self.attn(self.attn_linear(h_stack)))
        h = torch.sum((self.attn_linear(h_stack) * attn), dim=1)
        
        with edge_sub.local_scope():
            edge_sub.ndata['h'] = h
            edge_sub.ndata['attn'] = attn
            edge_sub.apply_edges(self.apply_edges, etype='reuse')
            return edge_sub.edata['score'][('website', 'reuse', 'website')], edge_sub.edata['attn'][('website', 'reuse', 'website')]