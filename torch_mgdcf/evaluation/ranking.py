# coding=utf-8
# The code is from our another project GRecX: https://github.com/maenzhier/grecx_datasets

from tqdm import tqdm
import numpy as np
import torch
from torch_mgdcf.metrics.ranking import ndcg_score, precision_score, recall_score
from torch_mgdcf.vector_search.vector_search import VectorSearchEngine



def score(ground_truth, pred_items, k_list, metrics):
    pred_match = [1 if item in ground_truth else 0 for item in pred_items]

    max_k = k_list[-1]
    if len(ground_truth) > max_k:
        ndcg_gold = [1] * max_k
    else:
        ndcg_gold = [1] * len(ground_truth) + [0] * (max_k - len(ground_truth))

    res_score = []
    for metric in metrics:
        if metric == "ndcg":
            score_func = ndcg_score
        elif metric == "precision":
            score_func = precision_score
        elif metric == "recall":
            score_func = recall_score
        else:
            raise Exception("Not Found Metric : {}".format(metric))

        for k in k_list:
            if metric == "ndcg":
                res_score.append(score_func(ndcg_gold[:k], pred_match[:k]))
            else:
                res_score.append(score_func(ground_truth, pred_match[:k]))

    return res_score


def evaluate_mean_global_metrics(user_items_dict, user_mask_items_dict,
                                 user_embedding, item_embedding,
                                 k_list=[10, 20], metrics=["ndcg"]):

    v_search = VectorSearchEngine(item_embedding)

    if isinstance(user_embedding, torch.Tensor):
        user_embedding = user_embedding.detach().cpu().numpy()
    else:
        user_embedding = np.asarray(user_embedding)

    user_indices = list(user_items_dict.keys())
    embedded_users = user_embedding[user_indices]
    max_mask_items_length = max(len(user_mask_items_dict[user]) for user in user_indices)

    _, user_rank_pred_items = v_search.search(embedded_users, k_list[-1] + max_mask_items_length)

    res_scores = []
    for user, pred_items in tqdm(zip(user_indices, user_rank_pred_items)):

        items = user_items_dict[user]
        mask_items = user_mask_items_dict[user]
        pred_items = [item for item in pred_items if item not in mask_items][:k_list[-1]]

        res_score = score(items, pred_items, k_list, metrics)

        res_scores.append(res_score)

    res_scores = np.asarray(res_scores)
    names = []
    for metric in metrics:
        for k in k_list:
            names.append("{}@{}".format(metric, k))

    # return list(zip(names, np.mean(res_scores, axis=0, keepdims=False)))
    return dict(zip(names, np.mean(res_scores, axis=0, keepdims=False)))

