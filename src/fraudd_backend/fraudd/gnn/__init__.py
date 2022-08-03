import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

from torch import tensor
from dgl import DGLHeteroGraph, heterograph
from dgl import function as fn
from dgl.utils import check_eq_shape
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

import tqdm

# https://github.com/wey-gu/NebulaGraph-Fraud-Detection-GNN/blob/main/notebooks/Inference_API.ipynb

def get_subgraph(session, vertex_id):
    r = session.execute_json(
        "USE yelp;"
        f"GET SUBGRAPH WITH PROP 2 STEPS FROM {vertex_id} "
        "YIELD VERTICES AS nodes, EDGES AS relationships;")
    r = json.loads(r)
    return r.get('results', [{}])[0].get('data')


def to_dgl(data):
    node_id_map = {} # key: vertex id in NebulaGraph, value: node id in dgl_graph
    node_idx = 0
    features = [[] for _ in range(32)] + [[]]
    for i in range(len(data)):
        for index, node in enumerate(data[i]['meta'][0]):
            nodeid = data[i]['meta'][0][index]['id']
            if nodeid not in node_id_map:
                node_id_map[nodeid] = node_idx
                node_idx += 1
                for f in range(32):
                    features[f].append(data[i]['row'][0][index][f"review.f{f}"])
                features[32].append(data[i]['row'][0][index]['review.is_fraud'])


    rur_start, rur_end, rsr_start, rsr_end, rtr_start, rtr_end = [], [], [], [], [], []
    for i in range(len(data)):
        for edge in data[i]['meta'][1]:
            edge = edge['id']
            if edge['name'] == 'shares_user_with':
                rur_start.append(node_id_map[edge['src']])
                rur_end.append(node_id_map[edge['dst']])
            elif edge['name'] == 'shares_restaurant_rating_with':
                rsr_start.append(node_id_map[edge['src']])
                rsr_end.append(node_id_map[edge['dst']])
            elif edge['name'] == 'shares_restaurant_in_one_month_with':
                rtr_start.append(node_id_map[edge['src']])
                rtr_end.append(node_id_map[edge['dst']])

    data_dict = {}
    if rur_start:
        data_dict[('review', 'shares_user_with', 'review')] = tensor(rur_start), tensor(rur_end)
    if rsr_start:
        data_dict[('review', 'shares_restaurant_rating_with', 'review')] = tensor(rsr_start), tensor(rsr_end)
    if rtr_start:
        data_dict[('review', 'shares_restaurant_in_one_month_with', 'review')] = tensor(rtr_start), tensor(rtr_end)

    # construct a dgl_graph
    dgl_graph: DGLHeteroGraph = heterograph(data_dict)

    # load node features to dgl_graph
    for i in range(32):
        dgl_graph.ndata[f"f{i}"] = tensor(features[i])
    dgl_graph.ndata['label'] = tensor(features[32])

    # to homogeneous graph
    features = []
    for i in range(32):
        features.append(dgl_graph.ndata[f"f{i}"])

    dgl_graph.ndata['feat'] = torch.stack(features, dim=1)

    dgl_graph.edges['shares_restaurant_in_one_month_with'].data['he'] = torch.ones(
        dgl_graph.number_of_edges('shares_restaurant_in_one_month_with'), dtype=torch.float32)
    dgl_graph.edges['shares_restaurant_rating_with'].data['he'] = torch.full(
        (dgl_graph.number_of_edges('shares_restaurant_rating_with'),), 2, dtype=torch.float32)
    dgl_graph.edges['shares_user_with'].data['he'] = torch.full(
        (dgl_graph.number_of_edges('shares_user_with'),), 4, dtype=torch.float32)

    hg = dgl.to_homogeneous(dgl_graph, edata=['he'], ndata=['feat', 'label'])
    return hg, node_id_map


def do_inference(graph, node_idx, model, batch_size=4096):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        return pred[node_idx]


class SAGEConv(dglnn.SAGEConv):
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
            msg_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                # graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                #########################################################################
                # consdier datatype with different weight, g.edata['he'] as weight here
                g.update_all(fn.u_mul_e('h', 'he', 'm'), fn.mean('m', 'h'))
                #########################################################################
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
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
                graph.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                graph.update_all(fn.copy_e('he', 'm'), fn.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
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


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y


device = torch.device('cpu')
gnn_model = SAGE(32, 256, 2).to(device)
