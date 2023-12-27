
import os
# set gpu id
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from torch_mgdcf.losses import compute_bpr_loss, compute_info_bpr_loss, compute_l2_loss
from torch_mgdcf.utils import create_tensor_dataloader
from torch_mgdcf.datasets import load_dataset
from torch_mgdcf.layers.mgdcf import MGDCF
import torch
import torch.nn.functional as F
import numpy as np
import time
from torch_mgdcf.evaluation.ranking import evaluate_mean_global_metrics

np.set_printoptions(precision=4)

parser = argparse.ArgumentParser(description='Argument parser for the program.')

parser.add_argument('--dataset', type=str, default='light_gcn_yelp', help='Dataset name')
parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
parser.add_argument('--l2_coef', type=float, default=1e-4, help='L2 regularization coefficient')
parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning rate decay')
parser.add_argument('--k', type=int, default=4, help='Number of iterations')
parser.add_argument('--alpha', type=float, default=0.1, help='A hyperparameter of MGDCF')
parser.add_argument('--beta', type=float, default=0.9, help='A hyperparameter of MGDCF')
parser.add_argument('--x_drop_rate', type=float, default=0.3, help='Dropout rate of input embeddings')
parser.add_argument('--edge_drop_rate', type=float, default=0.5, help='Dropout rate of edge weights')
parser.add_argument('--z_drop_rate', type=float, default=0.1, help='Dropout rate of output embeddings')
parser.add_argument('--num_negs', type=int, default=300, help='Number of negative samples for InfoBPR loss')
parser.add_argument('--batch_size', type=int, default=8000, help='Batch size')
parser.add_argument('--num_epochs', type=int, default=2000, help='Number of epochs')

args = parser.parse_args()
 
dataset_name = args.dataset
embedding_size = args.embedding_size
lr = args.lr
l2_coef = args.l2_coef
lr_decay = args.lr_decay
k = args.k
alpha = args.alpha
beta = args.beta
x_drop_rate = args.x_drop_rate
edge_drop_rate = args.edge_drop_rate
z_drop_rate = args.z_drop_rate
num_negs = args.num_negs
batch_size = args.batch_size
num_epochs = args.num_epochs
device = "cuda"


num_users, num_items, user_item_edges, train_index, test_index, train_user_items_dict, test_user_items_dict = load_dataset(dataset_name)
train_user_item_edges = user_item_edges[train_index]


g = MGDCF.build_homo_graph(train_user_item_edges, num_users=num_users, num_items=num_items).to(device)
num_nodes = g.num_nodes()

embeddings = np.random.randn(num_nodes, embedding_size) / np.sqrt(embedding_size)
embeddings = torch.tensor(embeddings, dtype=torch.float32, requires_grad=True, device=device)

model = MGDCF(
    k=k, alpha=alpha, beta=beta, 
    x_drop_rate=x_drop_rate, edge_drop_rate=edge_drop_rate, z_drop_rate=z_drop_rate
)

def forward():
    virtual_h = model(g, embeddings)
    user_h = virtual_h[:num_users]
    item_h = virtual_h[num_users:]
    return user_h, item_h

def evaluate():
    model.eval()
    user_h, item_h = forward()
    user_h = user_h.detach().cpu().numpy()
    item_h = item_h.detach().cpu().numpy()

    mean_results_dict = evaluate_mean_global_metrics(test_user_items_dict, train_user_items_dict,
                                                    user_h, item_h, k_list=[10, 20], metrics=["precision","recall", "ndcg"])
    return mean_results_dict


def update_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay



train_edges_data_loader = create_tensor_dataloader(torch.tensor(train_user_item_edges), batch_size=batch_size, shuffle=True)

optimizer = torch.optim.Adam([embeddings], lr=lr)


for epoch in range(num_epochs):

    if epoch % 20 == 0:
        print("\nEvaluation before epoch {} ......".format(epoch))
        mean_results_dict = evaluate()
        print(mean_results_dict)

    start_time = time.time()

    for step, batch_edges in enumerate(train_edges_data_loader):

        model.train()
        user_h, item_h = forward()

        # Using MGDCF's InfoBPR as ranking loss
        mf_losses = compute_info_bpr_loss(user_h, item_h, batch_edges, num_negs=num_negs, reduction="none")

        # MGDCF applies L2 Regularization on the output embeddings
        l2_loss = compute_l2_loss([user_h, item_h])
        # You can also apply L2 Regularization on the input embeddings instead
        # l2_loss = compute_l2_loss([embeddings])

        loss = mf_losses.sum() + l2_loss * l2_coef

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    update_learning_rate(optimizer, lr_decay)

    end_time = time.time()

    print("epoch = {}\tloss = {:.4f}\tmf_loss = {:.4f}\tl2_loss = {:.4f}\tupdated_lr = {:.4f}\tepoch_time = {:.4f}s"
          .format(epoch, loss.item(), mf_losses.mean().item(), l2_loss.item(), optimizer.param_groups[0]['lr'], end_time-start_time))









