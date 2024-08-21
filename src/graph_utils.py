import torch
import dgl
import sys
import numpy as np
from sklearn.model_selection import train_test_split

class PassREfinder_P(object):
    def __init__(self, feature_dict, reuse_rate_dict, args):
        super(PassREfinder, self).__init__()
        self.args = args
        self.setting = 'inductive' # args.setting
        
        self.g = self.construct_graph(feature_dict, reuse_rate_dict)
         
        # assert(args.setting in ['transductive', 'inductive'])
        # print('Learning setting:', args.setting)
        
        # if args.setting == 'transductive':
        #     self.edge_split = self.transductive_split()
        # else:
        #     self.g_split, self.node_split = self.inductive_split()

        self.g_split, self.node_split = self.inductive_split()
        
    def construct_graph(self, feature_dict, reuse_rate_dict):
        print('Constructing graph...')
        
        source_to_id = {s: i for i, s in enumerate(feature_dict.keys())}
        id_to_source = {v: k for k, v in source_to_id.items()}

        id_to_feature = {source_to_id[k]: {'numerical': v['numerical'], 'categorical': v['categorical'], 'url_encoding': v['url_encoding'], 'ip': v['ip'], 'text': v['text']} for k, v in feature_dict.items()}

        score_dict = {}
        label_list = []
        src_list = []
        dst_list = []

        for s1, id1 in source_to_id.items():
            s2_list = reuse_rate_dict[s1]

            for s2, score in s2_list.items():
                id2 = source_to_id.get(s2)
                score = 0 if score < self.args.reuse_th else 1

                if id2:
                    pair = tuple(sorted((id1, id2)))

                    if pair in score_dict:
                        continue

                    score_dict[pair] = score

                    src_list.append(id1)
                    dst_list.append(id2)
                    label_list.append(score)

        data_dict = {
            ('website', 'user', 'website'): (src_list, dst_list),
            # ('website', 'user', 'website'): ([], []),
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
        
        sys.stdout.write("\033[F")
        print('Constructing graph... Done')

        return g

    def inductive_split(self):
        print('Splitting graph...')
        
        valid_portion, test_portion = self.args.valid, self.args.test

        p1 = valid_portion + test_portion
        p2 = test_portion / (valid_portion + test_portion)

        train_node, valid_node = train_test_split(self.g.nodes(), test_size=p1, shuffle=True, random_state=self.args.random_seed)
        valid_node, test_node = train_test_split(valid_node, test_size=p2, shuffle=True, random_state=self.args.random_seed)

        train_node_list = np.array_split(train_node, self.args.nclient)

        train_g_list = [self.g.subgraph(train_node) for train_node in train_node_list]
        valid_g = self.g.subgraph(valid_node)
        test_g = self.g.subgraph(test_node)
        
        sys.stdout.write("\033[F")
        print('Splitting graph... Done')

        return {'train': train_g_list, 'valid': valid_g, 'test': test_g}, {'train': train_node_list, 'valid': valid_node, 'test': test_node}

    # def transductive_split(self):
    #     print('Splitting graph...')
        
    #     valid_portion, test_portion = self.args.valid, self.args.test

    #     p1 = valid_portion + test_portion
    #     p2 = test_portion / (valid_portion + test_portion)

    #     train_edge, valid_edge = train_test_split(self.g.edges(etype='reuse', form='eid'), test_size=p1, shuffle=True, random_state=self.args.random_seed)
    #     valid_edge, test_edge = train_test_split(valid_edge, test_size=p2, shuffle=True, random_state=self.args.random_seed)
        
    #     sys.stdout.write("\033[F")
    #     print('Splitting graph... Done')
    #     return {'train': train_edge, 'valid': valid_edge, 'test': test_edge}
    
    def get_reverse_eid(self, g_type):
        if g_type == 'train':
            reverse_eid_list = []
            assert(self.setting == 'inductive')
            for g in self.g_split[g_type]:
                reverse_eid_list.append({('website', 'reuse', 'website'): g.edge_ids(g.edges(etype='reuse')[1], g.edges(etype='reuse')[0], etype='reuse')})
            return reverse_eid_list
        else:
            g = self.g_split[g_type] if self.setting == 'inductive' else self.g
            return {('website', 'reuse', 'website'): g.edge_ids(g.edges(etype='reuse')[1], g.edges(etype='reuse')[0], etype='reuse')}

    def get_target_eid(self, g_type):
        if g_type == 'train':
            target_eid_list = []
            assert(self.setting == 'inductive')
            for g, node in zip(self.g_split[g_type], self.node_split[g_type]):
                ori_id_to_id = dict(zip(g.ndata['_ID'].tolist(), g.nodes().tolist()))
                node_sub = [ori_id_to_id[n.item()] for n in node]
    
                target_eid_list.append(torch.unique(torch.cat([g.in_edges(node_sub, etype='reuse', form='eid'), g.out_edges(node_sub, etype='reuse', form='eid')])))
            return target_eid_list
        else:
            if self.setting == 'transductive':
                return self.edge_split[g_type]
            else:
                g = self.g_split[g_type]
                node = self.node_split[g_type]
    
                ori_id_to_id = dict(zip(g.ndata['_ID'].tolist(), g.nodes().tolist()))
                node_sub = [ori_id_to_id[n.item()] for n in node]
    
                return torch.unique(torch.cat([g.in_edges(node_sub, etype='reuse', form='eid'), g.out_edges(node_sub, etype='reuse', form='eid')]))

    def get_data_loader(self, g_type):
        if self.setting == 'transductive':
            g = self.g
        else:
            g = self.g_split[g_type]
        reverse_eid = self.get_reverse_eid(g_type)
        target_eid = self.get_target_eid(g_type)

        sampler = dgl.dataloading.as_edge_prediction_sampler(dgl.dataloading.MultiLayerFullNeighborSampler(self.args.gnn_depth), exclude='reverse_id', reverse_eids=reverse_eid)

        data_loader = dgl.dataloading.DataLoader(
                g,
                {'reuse': target_eid},
                sampler,
                batch_size=self.args.batch_size,
                device=self.args.device,
                shuffle=False,
                drop_last=False,
                num_workers=0)

        return data_loader

    def get_train_data_loader_list(self):
        if self.setting == 'transductive':
            # g = self.g
            pass
        else:
            g_list = self.g_split['train']
        reverse_eid_list = self.get_reverse_eid('train')
        target_eid_list = self.get_target_eid('train')

        data_loader_list = []
        for g, reverse_eid, target_eid in zip(g_list, reverse_eid_list, target_eid_list):
            sampler = dgl.dataloading.as_edge_prediction_sampler(dgl.dataloading.MultiLayerFullNeighborSampler(self.args.gnn_depth), exclude='reverse_id', reverse_eids=reverse_eid)

            data_loader_list.append(dgl.dataloading.DataLoader(
                    g,
                    {'reuse': target_eid},
                    sampler,
                    batch_size=self.args.batch_size,
                    device=self.args.device,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0))

        return data_loader_list

    def pop_node_feature(self, g_type):
        if self.setting == 'transductive':
            g = self.g
        else:
            g = self.g_split[g_type]
        
        nfeat_n = g.ndata.pop('numerical')
        nfeat_c = g.ndata.pop('categorical')
        nfeat_e = g.ndata.pop('url_encoding')
        nfeat_ip = g.ndata.pop('ip')
        nfeat_text = g.ndata.pop('text')

        return nfeat_n, nfeat_c, nfeat_e, nfeat_ip, nfeat_text

    def pop_train_node_feature(self):
        nfeat_list = []
        for g in self.g_split['train']:
            nfeat_n = g.ndata.pop('numerical')
            nfeat_c = g.ndata.pop('categorical')
            nfeat_e = g.ndata.pop('url_encoding')
            nfeat_ip = g.ndata.pop('ip')
            nfeat_text = g.ndata.pop('text')
            nfeat_list.append([nfeat_n, nfeat_c, nfeat_e, nfeat_ip, nfeat_text])
        return nfeat_list

    def print_graph(self):
        if self.setting == 'transductive':
#             g = self.g
#             train_edge = self.edge_split['train']
#             valid_edge = self.edge_split['valid']
#             test_edge = self.edge_split['test']

#             def get_node_num(g, edges):
#                 node_set = set()
#                 for n1, n2 in zip(g.find_edges(edges, etype='reuse')[0].tolist(), g.find_edges(edges, etype='reuse')[1].tolist()):
#                     node_set.add(n1)
#                     node_set.add(n2)

#                 return len(node_set)

#             print(f"""----------Data statistics----------
# #[Train] nodes: {get_node_num(self.g, train_edge)}, edges: {len(train_edge)}
# #[Valid] nodes: {get_node_num(self.g, valid_edge)}, edges: {len(valid_edge)}
# #[Test]  nodes: {get_node_num(self.g, test_edge)}, edges: {len(test_edge)}
# -----------------------------------\n""")
            pass
        else:
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