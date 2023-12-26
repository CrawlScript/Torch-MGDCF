# coding=utf-8
import torch
import numpy as np
import torch.nn.functional as F

def info_bpr(a_embeddings, b_embeddings, pos_edges, num_negs=300, reduction='mean'):

    if isinstance(pos_edges, list):
        pos_edges = np.array(pos_edges)

    device = a_embeddings.device

    a_indices = pos_edges[:, 0]
    b_indices = pos_edges[:, 1]

    if isinstance(pos_edges, torch.Tensor):
        num_pos_edges = pos_edges.size(0)
    else:
        num_pos_edges = len(pos_edges)


    num_b = b_embeddings.size(0)
    neg_b_indices = torch.randint(0, num_b, [num_pos_edges, num_negs]).to(device)

    embedded_a = a_embeddings[a_indices]
    embedded_b = b_embeddings[b_indices]
    embedded_neg_b = b_embeddings[neg_b_indices]
    

    embedded_combined_b = torch.cat([embedded_b.unsqueeze(1), embedded_neg_b], 1)

    logits = (embedded_combined_b @ embedded_a.unsqueeze(-1)).squeeze(-1)

    info_bpr_loss = F.cross_entropy(logits, torch.zeros([num_pos_edges], dtype=torch.int64).to(device), reduction=reduction)

    return info_bpr_loss


def bpr(a_embeddings, b_embeddings, pos_edges, reduction='mean'):
    """
    bpr is a special case of info_bpr, where num_negs=1
    """
    return info_bpr(a_embeddings, b_embeddings, pos_edges, num_negs=1, reduction=reduction)

    

    