#!/usr/bin/env python
# coding: utf-8

import json
import os
import torch
from torch.optim import Adam
import dgl

from src import *

from argparse import ArgumentParser

parser = ArgumentParser()

import argparse
from pprint import pprint

parser = argparse.ArgumentParser(description="Parser for model configuration")

parser.add_argument('--feature_file', type=str, default="./data/feature_dict.json", help='feature json file')
parser.add_argument('--reuse_rate_file', type=str, default="./data/reuse_rate_dict.json", help='reuse rate json file')
parser.add_argument('--setting', type=str, required=True, help='graph learning setting: inductive/transductive')
parser.add_argument('--model_path', type=str, required=True, default="./model/", help='model file path')
parser.add_argument('--agg_type', type=str, default="attn", help='aggregation type')
parser.add_argument('--valid', type=float, default=0.2, help='split ratio of validation set')
parser.add_argument('--test', type=float, default=0.2, help='split ratio of test set')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--random_seed', type=int, default=1, help='random seed for initialization')
parser.add_argument('--relu', type=float, default=0.2, help='ReLU threshold')
parser.add_argument('--reuse_th', type=float, default=0.5, help='threshold for reuse')
parser.add_argument('--batch_size', type=int, default=65536, help='batch size for training')
parser.add_argument('--embed_size', type=int, default=256, help='size of the embedding layer')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the hidden layer')
parser.add_argument('--gnn_depth', type=int, default=2, help='depth of the GNN')
parser.add_argument('--max_lr', type=float, default=0.001, help='maximum learning rate')
parser.add_argument('--warmup', type=float, default=0.1, help='warmup ratio for learning rate')
parser.add_argument('--max_epoch', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--early_stop', type=int, default=40, help='early stopping epochs')
parser.add_argument('--device', type=int, default=0, help='GPU ID')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

pprint(args)

if __name__ == '__main__':
    with open(args.feature_file, 'r') as f:
        feature_dict = json.load(f)

    with open(args.reuse_rate_file, 'r') as f:
        reuse_rate_dict = json.load(f)
        
    P = PassREfinder_P(feature_dict, reuse_rate_dict, args, device)
    
    P.print_graph()
    
    test_loader = P.get_data_loader('test')
    test_nfeat = P.pop_node_feature('test')

    model=GraphSAGE(args).to(device)
    optimizer = Adam(model.parameters(), lr=float(args.max_lr))
    load_checkpoint(args.model_path, model, optimizer, device)
    
    evaluate(model, test_loader, test_nfeat, device)