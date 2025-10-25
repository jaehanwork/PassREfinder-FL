#!/usr/bin/env python
# coding: utf-8

import json
import os
import random
import torch
from torch.optim import Adam
import dgl

from src import *

from argparse import ArgumentParser

parser = ArgumentParser()

import argparse
from pprint import pprint

parser = argparse.ArgumentParser(description="Parser for model configuration")

parser.add_argument('--output_dir', type=str, default="./output/default", help='output directory')
parser.add_argument('--feature_file', type=str, default="./data/feature_dict.json", help='feature json file')
parser.add_argument('--reuse_rate_file', type=str, default="./data/reuse_rate_dict.json", help='reuse rate json file')
# parser.add_argument('--setting', type=str, required=True, help='graph learning setting: inductive/transductive')
# parser.add_argument('--model_path', type=str, required=True, default="./model/", help='model file path')
parser.add_argument('--agg_type', type=str, default="attn", help='aggregation type')
parser.add_argument('--nclient', type=int, default=10, help='number of clients')
# parser.add_argument('--valid', type=float, default=0.2, help='split ratio of validation set')
# parser.add_argument('--test', type=float, default=0.2, help='split ratio of test set')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--random_seed', type=int, default=1, help='random seed for initialization')
parser.add_argument('--relu', type=float, default=0.2, help='ReLU threshold')
parser.add_argument('--reuse_th', type=float, default=0.5, help='threshold for reuse')
parser.add_argument('--train_batch_size', type=int, default=2048, help='batch size for training')
parser.add_argument('--eval_batch_size', type=int, default=2048, help='batch size for evaluation')
parser.add_argument('--embed_size', type=int, default=256, help='size of the embedding layer')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the hidden layer')
parser.add_argument('--gnn_depth', type=int, default=2, help='depth of the GNN')
parser.add_argument('--max_lr', type=float, default=0.001, help='maximum learning rate')
parser.add_argument('--warmup', type=float, default=0.1, help='warmup ratio for learning rate')
parser.add_argument('--max_epoch', type=int, default=100, help='maximum number of epochs')
# parser.add_argument('--early_stop', type=int, default=40, help='early stopping epochs')
parser.add_argument('--device', type=int, default=0, help='GPU ID')

args = parser.parse_args()
pprint(vars(args))

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
dgl.seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

if __name__ == '__main__':
    with open(args.feature_file, 'r') as f:
        feature_dict = json.load(f)

    with open(args.reuse_rate_file, 'r') as f:
        reuse_rate_dict = json.load(f)

    g = construct_graph(feature_dict, reuse_rate_dict, args.reuse_th)
    g_dict, node_set_list, target_eids_dict = graph_split_FL(g, args.nclient, args.random_seed)

    P = PassREfinder_FL(g_dict, node_set_list, target_eids_dict, args)

    pprint(g_dict)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'arguments.json'), 'w') as f:
        json.dump(vars(args), f)

    P.train()

    P.model = load_model(os.path.join(P.args.output_dir, 'model.pt'), P.model)
    result = P.evaluate()
    f1 = result['eval_f1']
    precision = result['eval_precision']
    recall = result['eval_recall']
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(result, f)