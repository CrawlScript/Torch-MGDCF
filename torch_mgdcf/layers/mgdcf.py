# coding=utf-8

import torch
import torch.nn as nn
import dgl
import numpy as np
import dgl.function as fn






class MGDCF(nn.Module):

    CACHE_KEY = "mgdcf_weight"

    def __init__(self, k, alpha, beta, x_drop_rate, edge_drop_rate, z_drop_rate, *args, **kwargs):
        """
        :param k: number of iterations.
        :param alpha: a hyperparameter of MGDCF.
        :param beta: a hyperparameter of MGDCF.
        :param x_drop_rate: dropout rate of input embeddings.
        :param edge_drop_rate: dropout rate of edge weights.
        :param z_drop_rate: dropout rate of output embeddings.
        """

        super().__init__(*args, **kwargs)

        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = torch.tensor(MGDCF.compute_gamma(alpha, beta, k)).float()

        self.x_drop_rate = x_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.z_drop_rate = z_drop_rate

        self.x_dropout = nn.Dropout(x_drop_rate)
        self.edge_dropout = nn.Dropout(edge_drop_rate)
        self.z_dropout = nn.Dropout(z_drop_rate)


    @classmethod
    def compute_gamma(cls, alpha, beta, k):
        return np.power(beta, k) + alpha * np.sum([np.power(beta, i) for i in range(k)])

    @classmethod
    def build_homo_graph(cls, user_item_edges, num_users=None, num_items=None):

        user_index, item_index = user_item_edges.T

        if num_users is None:
            num_users = np.max(user_index) + 1

        if num_items is None:
            num_items = np.max(item_index) + 1

        num_homo_nodes = num_users + num_items
        homo_item_index = item_index + num_users
        src = user_index
        dst = homo_item_index

        g = dgl.graph((src, dst), num_nodes=num_homo_nodes)
        g =  dgl.add_reverse_edges(g)
        # Different from LightGCN, MGDCF considers self-loop
        g = dgl.add_self_loop(g)
        g = dgl.to_simple(g)
        
        return g

    @classmethod
    @torch.no_grad()
    def norm_adj(cls, g):

        CACHE_KEY = MGDCF.CACHE_KEY

        if CACHE_KEY in g.edata:
            return
        
        degs = g.in_degrees()
        src_norm = degs.pow(-0.5)
        dst_norm = src_norm

        with g.local_scope():
            g.ndata["src_norm"] = src_norm
            g.ndata["dst_norm"] = dst_norm
            g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", CACHE_KEY))
            gcn_weight = g.edata[CACHE_KEY]

        g.edata[CACHE_KEY] = gcn_weight


    def forward(self, g, x):

        CACHE_KEY = MGDCF.CACHE_KEY
        MGDCF.norm_adj(g)

        edge_weight = g.edata[CACHE_KEY]
        dropped_edge_weight = self.edge_dropout(edge_weight)


        h0 = self.x_dropout(x)
        h = h0

        with g.local_scope():
            g.edata[CACHE_KEY] = dropped_edge_weight
            
            for _ in range(self.k):

                g.ndata["h"] = h
                g.update_all(fn.u_mul_e("h", CACHE_KEY, "m"), fn.sum("m", "h"))
                h = g.ndata.pop("h")

                h = h * self.beta + h0 * self.alpha

        h = h / self.gamma
        h = self.z_dropout(h)

        return h
       