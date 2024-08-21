"""Torch Module for PassREfinder GNN layer, modified from dgl sageconv.py"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
from torch.nn import functional as F

from dgl import function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair, check_eq_shape, dgl_warning
from dgl.ops import edge_softmax

class SAGEConvN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConvN, self).__init__()
        valid_aggre_types = {'mean', 'gcn', 'attn', 'no_neighbor', 'no_hidden'}
        if aggregator_type not in valid_aggre_types:
            raise DGLError(
                'Invalid aggregator_type. Must be one of {}. '
                'But got {!r} instead.'.format(valid_aggre_types, aggregator_type)
            )

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/gcn/attn/no_neighbor/no_hidden
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        
        if aggregator_type == 'attn':
            self.fc_manual_u = nn.Linear(self._in_src_feats, out_feats, bias=False)
            self.attn_l_u = nn.Linear(out_feats, 1, bias=False)
            self.attn_r_u = nn.Linear(out_feats, 1, bias=False)
            self.fc_manual_r = nn.Linear(self._in_src_feats, out_feats, bias=False)
            self.attn_l_r = nn.Linear(out_feats, 1, bias=False)
            self.attn_r_r = nn.Linear(out_feats, 1, bias=False)
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.fc_neigh = nn.Linear(out_feats, out_feats)
        elif aggregator_type == 'no_hidden':
            self.fc_manual_r = nn.Linear(self._in_src_feats, out_feats, bias=False)
            self.attn_l_r = nn.Linear(out_feats, 1, bias=False)
            self.attn_r_r = nn.Linear(out_feats, 1, bias=False)
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.fc_neigh = nn.Linear(out_feats, out_feats)
        elif aggregator_type == 'no_neighbor':
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)
        else:
            self.fc_neigh = nn.Linear(self._in_src_feats*2, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The linear weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        """
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        if self._aggre_type == 'attn':
            nn.init.xavier_uniform_(self.fc_manual_u.weight, gain=gain)
            nn.init.xavier_uniform_(self.attn_l_u.weight, gain=gain)
            nn.init.xavier_uniform_(self.attn_r_u.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_manual_r.weight, gain=gain)
            nn.init.xavier_uniform_(self.attn_l_r.weight, gain=gain)
            nn.init.xavier_uniform_(self.attn_r_r.weight, gain=gain)
        if self._aggre_type == 'no_hidden':
            nn.init.xavier_uniform_(self.fc_manual_r.weight, gain=gain)
            nn.init.xavier_uniform_(self.attn_l_r.weight, gain=gain)
            nn.init.xavier_uniform_(self.attn_r_r.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            dgl_warning("You are loading a GraphSAGE model trained from a old version of DGL, "
                        "DGL automatically convert it to be compatible with latest version.")
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn_ori = fn.copy_u('h', 'm')
            msg_fn = fn.copy_u('h', 'm')
            if edge_weight is not None:
                assert edge_weight[('website', 'reuse', 'website')].shape[0] == graph.number_of_edges('reuse')
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            def concat(flist):
                    return torch.cat(flist, dim=1)

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src

                graph.multi_update_all(
                    {'user': (msg_fn_ori, fn.mean('m', 'neigh')),
                     'reuse': (msg_fn_ori, fn.mean('m', 'neigh'))},
                concat)
                h_neigh = graph.dstdata['neigh']
                h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'attn':
                h_src_u = self.fc_manual_u(feat_src)
                h_dst_u = self.fc_manual_u(feat_dst)
                h_src_r = self.fc_manual_r(feat_src)
                h_dst_r = self.fc_manual_r(feat_dst)

                el_u = self.attn_l_u(h_src_u)
                er_u = self.attn_r_u(h_dst_u)
                el_r = self.attn_l_r(h_src_r)
                er_r = self.attn_r_r(h_dst_r)
                
                
                graph.srcdata.update({'ft_u':h_src_u, 'ft_r': h_src_r, 'el_u': el_u, 'el_r': el_r})
                graph.dstdata.update({'er_u': er_u, 'er_r': er_r})

                graph.apply_edges(fn.u_add_v('el_u', 'er_u', 'e_u'), etype='user')
                graph.apply_edges(fn.u_add_v('el_r', 'er_r', 'e_r'), etype='reuse')
                e_u = self.leaky_relu(graph.edata.pop('e_u')[('website', 'user', 'website')])
                e_r = self.leaky_relu(graph.edata.pop('e_r')[('website', 'reuse', 'website')])

                graph.edata['a_u'] = {('website', 'user', 'website'): edge_softmax(graph['user'], e_u)}
                graph.edata['a_r'] = {('website', 'reuse', 'website'): edge_softmax(graph['reuse'], e_r)}
                
                graph.multi_update_all({'user': (fn.u_mul_e('ft_u', 'a_u', 'm_u'), fn.sum('m_u', 'neigh')),
                                        'reuse': (fn.u_mul_e('ft_r', 'a_r', 'm_r'), fn.sum('m_r', 'neigh'))
                                        },
                                        'mean')
                h_neigh = graph.dstdata['neigh']
                h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'no_hidden':
                h_src_r = self.fc_manual_r(feat_src)
                h_dst_r = self.fc_manual_r(feat_dst)
                
                el_r = self.attn_l_r(h_src_r)
                er_r = self.attn_r_r(h_dst_r)
                
                graph.srcdata.update({'ft_r': h_src_r, 'el_r': el_r})
                graph.dstdata.update({'er_r': er_r})

                graph.apply_edges(fn.u_add_v('el_r', 'er_r', 'e_r'))
                e_r = self.leaky_relu(graph.edata.pop('e_r'))

                graph.edata['a_r'] = edge_softmax(graph, e_r)
                
                graph.update_all(fn.u_mul_e('ft_r', 'a_r', 'm_r'), fn.sum('m_r', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                h_neigh = F.normalize(self.leaky_relu(self.fc_neigh(h_neigh)))
            elif self._aggre_type == 'no_neighbor':
                graph.srcdata['h'] = feat_src

                graph.multi_update_all({'user': (fn.copy_src('h', 'm'), fn.mean('m', 'neigh')),
                                        'reuse': (fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
                                        },
                                        'mean')
                h_neigh = graph.dstdata['neigh']
                h_neigh = F.normalize(self.leaky_relu(self.fc_neigh(h_neigh)))
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    if graph.is_block:
                        graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                    else:
                        graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = h_neigh
            if self._aggre_type == 'attn':
                rst = h_neigh
            else:
                rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
