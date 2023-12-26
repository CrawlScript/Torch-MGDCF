# coding=utf-8

import torch
import torch.nn as nn
import dgl
import numpy as np
import dgl.function as fn


class LightGCN(nn.Module):

    CACHE_KEY = "light_gcn_weight"

    def __init__(self, k, edge_drop_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.k = k
        self.edge_drop_rate = edge_drop_rate

        self.edge_dropout = nn.Dropout(edge_drop_rate)

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
        # LightGCN does not consider self-loop
        # g = dgl.add_self_loop(g)
        g = dgl.to_simple(g)
        
        return g

    @classmethod
    @torch.no_grad()
    def norm_adj(cls, g):

        CACHE_KEY = LightGCN.CACHE_KEY

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

        CACHE_KEY = LightGCN.CACHE_KEY
        LightGCN.norm_adj(g)

        edge_weight = g.edata[CACHE_KEY]
        dropped_edge_weight = self.edge_dropout(edge_weight)


        with g.local_scope():
            g.edata[CACHE_KEY] = dropped_edge_weight
            g.ndata["h"] = x

            h = x
            h_list = [h]

            for _ in range(self.k):
                g.update_all(fn.u_mul_e("h", CACHE_KEY, "m"), fn.sum("m", "h"))
                h = g.ndata["h"]
                h_list.append(h)

        h = torch.stack(h_list, dim=1).mean(dim=1)

        return h
       