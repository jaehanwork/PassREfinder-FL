import sys
import numpy as np
import torch
import dgl
from sklearn.model_selection import train_test_split

def construct_graph(feature_dict, reuse_rate_dict, reuse_th, hidden_relation=False):
    print('Constructing graph...')
    
    source_to_id = {s: i for i, s in enumerate(feature_dict.keys())}
    id_to_source = {v: k for k, v in source_to_id.items()}

    id_to_feature = {source_to_id[k]: {'numerical': v['numerical'], 'categorical': v['categorical'], 'url_encoding': v['url_encoding'], 'ip': v['ip'], 'text': v['text']} for k, v in feature_dict.items()}

    score_dict = {}
    label_list = []
    src_list = []
    dst_list = []
    rate_list = []

    for s1, id1 in source_to_id.items():
        s2_list = reuse_rate_dict[s1]

        for s2, rate in s2_list.items():
            id2 = source_to_id.get(s2)
            score = 0 if rate < reuse_th else 1

            if id2:
                pair = tuple(sorted((id1, id2)))

                if pair in score_dict:
                    continue

                score_dict[pair] = score

                src_list.append(id1)
                dst_list.append(id2)
                rate_list.append(rate)
                label_list.append(score)

    data_dict = {
        ('website', 'user', 'website'): (src_list, dst_list) if hidden_relation else ([], []),
        ('website', 'reuse', 'website'): (src_list, dst_list),
    }           

    g = dgl.to_bidirected(dgl.heterograph(data_dict))

    g.ndata['numerical'] = torch.tensor([v['numerical'] for v in id_to_feature.values()])
    g.ndata['categorical'] = torch.tensor([v['categorical'] for v in id_to_feature.values()])
    g.ndata['url_encoding'] = torch.tensor([v['url_encoding'] for v in id_to_feature.values()])
    g.ndata['text'] = torch.tensor([v['text'] for v in id_to_feature.values()])
    g.ndata['ip'] = torch.tensor([v['ip'] for v in id_to_feature.values()], dtype=torch.float)

    g.edata['mask'] = {('website', 'reuse', 'website'): torch.zeros(len(src_list)*2, dtype=torch.float)}
    g.edata['mask'][('website', 'reuse', 'website')][g.edge_ids(src_list, dst_list, etype='reuse')] = torch.tensor(label_list, dtype=torch.float)
    g.edata['mask'][('website', 'reuse', 'website')][g.edge_ids(dst_list, src_list, etype='reuse')] = torch.tensor(label_list, dtype=torch.float)

    g.edata['label'] = {('website', 'reuse', 'website'): torch.zeros(len(src_list)*2, dtype=torch.long)}
    g.edata['label'][('website', 'reuse', 'website')][g.edge_ids(src_list, dst_list, etype='reuse')] = torch.tensor(label_list)
    g.edata['label'][('website', 'reuse', 'website')][g.edge_ids(dst_list, src_list, etype='reuse')] = torch.tensor(label_list)

    g.edata['rate'] = {('website', 'reuse', 'website'): torch.zeros(len(src_list)*2, dtype=torch.float)}
    g.edata['rate'][('website', 'reuse', 'website')][g.edge_ids(src_list, dst_list, etype='reuse')] = torch.tensor(rate_list)
    g.edata['rate'][('website', 'reuse', 'website')][g.edge_ids(dst_list, src_list, etype='reuse')] = torch.tensor(rate_list)
    
    sys.stdout.write("\033[F")
    print('Constructing graph... Done')

    return g

def get_test_graph_and_edges_FL(g, node_set_list):
    node_dict = {}
    for i, node_set in enumerate(node_set_list):
        node_dict.update(dict.fromkeys(node_set.numpy().tolist(), i))

    remove_eids = []
    target_eids = []
    for eid, (u, v) in enumerate(zip(g.edges(etype='reuse')[0].numpy(), g.edges(etype='reuse')[1].numpy())):
        if not (u in node_dict and v in node_dict):
            remove_eids.append(eid)

    g = dgl.remove_edges(g, remove_eids, etype='reuse')

    for eid, (u, v) in enumerate(zip(g.edges(etype='reuse')[0].numpy(), g.edges(etype='reuse')[1].numpy())):
        if node_dict[u] != node_dict[v]:
            target_eids.append(eid)
    return g, target_eids

def get_test_edges_base(test_g, train_node_set_list, test_node_set_list):
    ori_id_to_id = dict(zip(test_g.ndata['_ID'].tolist(), test_g.nodes().tolist()))
    train_node_dict = {}
    for i, node_set_ori in enumerate(train_node_set_list):
        node_set = [ori_id_to_id[nid] for nid in node_set_ori.numpy().tolist()]
        train_node_dict.update(dict.fromkeys(node_set, i))
    node_dict = {}
    for i, node_set_ori in enumerate(test_node_set_list):
        node_set = [ori_id_to_id[nid] for nid in node_set_ori.numpy().tolist()]
        node_dict.update(dict.fromkeys(node_set, i))

    target_eids = []
    for eid, (u, v) in enumerate(zip(test_g.edges(etype='reuse')[0].numpy(), test_g.edges(etype='reuse')[1].numpy())):
        if (u not in node_dict) and (v not in node_dict):
            continue
        if (u in node_dict) and (v in train_node_dict):
            target_eids.append(eid)
        elif (u in train_node_dict) and (v in node_dict):
            target_eids.append(eid)
        elif node_dict[u] != node_dict[v]:
            target_eids.append(eid)
        elif node_dict[u] == node_dict[v]:
            continue
        else:
            assert(0)
    return target_eids


# def get_test_graph_and_edges(g, node_set_list, train_node_set):
#     node_dict = {}
#     for i, node_set in enumerate(node_set_list):
#         node_dict.update(dict.fromkeys(node_set.numpy().tolist(), i))
    
#     train_node_dict = dict.fromkeys(train_node_set.numpy().tolist())

#     remove_eids = []
#     target_eids = []
#     for eid, (u, v) in enumerate(zip(g.edges(etype='reuse')[0].numpy(), g.edges(etype='reuse')[1].numpy())):
#         if not (u in node_dict and v in node_dict):
#             remove_eids.append(eid)

#     for eid, (u, v) in enumerate(zip(g.edges(etype='reuse')[0].numpy(), g.edges(etype='reuse')[1].numpy())):
#         if node_dict[u] == node_dict[v] and not (u in train_node_dict and v in train_node_dict):
#             remove_eids.append(eid)

#     g = dgl.remove_edges(g, remove_eids, etype='reuse')

#     for eid, (u, v) in enumerate(zip(g.edges(etype='reuse')[0].numpy(), g.edges(etype='reuse')[1].numpy())):
#         if node_dict[u] != node_dict[v]:
#             target_eids.append(eid)
#     return g, target_eids

def graph_split_FL(g, nclient, random_seed):
    assert(nclient <= 10)
    print('Splitting graph...')

    nodes = g.nodes()
    node_set_list = [nodes[i:i+1000] for i in range(0, len(nodes), 1000)]
    train_g_list = [g.subgraph(node_set) for node_set in node_set_list[:nclient]]

    valid_g, valid_eids = get_test_graph_and_edges_FL(g, node_set_list[:2])
    test_g, test_eids = get_test_graph_and_edges_FL(g, node_set_list)
    
    sys.stdout.write("\033[F")
    print('Splitting graph... Done')

    return {'train': train_g_list, 'valid': valid_g, 'test': test_g}, node_set_list, {'valid': valid_eids, 'test': test_eids}

def get_test_eid(g, test_node):
    ori_id_to_id = dict(zip(g.ndata['_ID'].tolist(), g.nodes().tolist()))
    node_sub = [ori_id_to_id[n.item()] for n in test_node]
    return torch.unique(torch.cat([g.in_edges(node_sub, etype='reuse', form='eid'), g.out_edges(node_sub, etype='reuse', form='eid')]))

def graph_split_base(g, random_seed):
    print('Splitting graph...')

    nodes = g.nodes()
    node_set_list = [nodes[i:i+1000] for i in range(0, len(nodes), 1000)]
    train_g = g.subgraph(nodes[:1000])
    valid_g = g.subgraph(nodes[:2000])
    test_g = g.subgraph(nodes)

    # valid_eids = get_test_eid(valid_g, nodes[1000:2000])
    # test_eids = get_test_eid(test_g, nodes[2000:])

    valid_eids = get_test_edges_base(valid_g, node_set_list[:1], node_set_list[1:2])
    test_eids = get_test_edges_base(test_g, node_set_list[:2], node_set_list[2:])
    
    sys.stdout.write("\033[F")
    print('Splitting graph... Done')

    return {'train': train_g, 'valid': valid_g, 'test': test_g}, {'valid': valid_eids, 'test': test_eids}, node_set_list


    
# def graph_split_no_hidden(g, nclient, random_seed):
#     print('Splitting graph...')

#     nodes = g.nodes()
#     node_set_list = [nodes[i:i+1000] for i in range(0, len(nodes), 1000)]
#     train_g_list = [g.subgraph(node_set) for node_set in node_set_list[:1]]

#     valid_g, valid_eids = get_test_graph_and_edges(g, node_set_list[:2], node_set_list[0])
    
#     test_g, test_eids = get_test_graph_and_edges(g, node_set_list[2:], node_set_list[0])
    
#     sys.stdout.write("\033[F")
#     print('Splitting graph... Done')

#     return {'train': train_g_list, 'valid': valid_g, 'test': test_g}, node_set_list, {'valid': valid_eids, 'test': test_eids}