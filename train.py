import argparse
import os.path as osp
import time
import pdb
import hashlib
import sys

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
import wandb
from torch_geometric.data import Data
import torch_geometric.transforms as T

from model import GIN, DiffGIN, DenseGAT, DeepDiff, DeepSet, DeepCount

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--transform', type=str, default=None, help='random, hash, diff, diff_one, counts')
parser.add_argument('--online_trans', action='store_true', help='perform hash/random transformations online')
parser.add_argument('--model', type=str, default='DenseGAT', help='DenseGAT, GIN, DiffGIN, DeepSet, DeepCount')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    # MPS is currently slower than CPU due to missing int64 min/max ops
    device = torch.device('cpu')
else:
    device = torch.device('cpu')



wandb.init(
    project="lexiinv_9_16",
    name=f'{args.dataset}-{args.transform}-{args.model}' + args.online_trans * '-online_trans',
    config={
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "device": device,
        "transform": args.transform, 
        "online_trans": args.online_trans, 
    }
)

print("Command line: ", 'python ' + " ".join(sys.argv))

class ReplaceNodeAttributesWithRandomVectors(object):
    def __init__(self, vector_dim):
        self.vector_dim = vector_dim

    def __call__(self, data):
        num_nodes = data.num_nodes
        # Generate random vectors for each node
        random_vectors = torch.rand((num_nodes, self.vector_dim))
        # Replace the node attributes with the random vectors
        data.x = random_vectors
        return data


class ReplaceNodeAttributesWithHash(object):
    def __init__(self, hash_func=hashlib.sha256, output_dim=16):
        self.hash_func = hash_func
        self.output_dim = output_dim

    def vector_to_hash(self, vector):
        # Convert the vector to a string and encode it
        vector_str = vector.numpy().tobytes()
        # Create the hash
        hash_value = self.hash_func(vector_str).hexdigest()
        # Convert the hash to a fixed-size tensor (e.g., first 64 characters)
        hash_tensor = torch.tensor([int(hash_value[i:i+2], 16) for i in range(0, self.output_dim, 2)], dtype=torch.float32)
        return hash_tensor

    def __call__(self, data):
        # Append a fixed random vector to each node
        rand_vec = torch.randn(8).to(data.x.device)
        # Apply hashing to each node attribute vector
        hashed_vectors = torch.stack([self.vector_to_hash(torch.cat([vec, rand_vec])) for vec in data.x])
        # Replace the node attributes with hashed vectors
        data.x = hashed_vectors
        return data


class ReplaceNodeAttributesWithOne(object):
    def __call__(self, data):
        num_nodes = data.num_nodes
        data.x = torch.ones([num_nodes, 1])
        return data


class CreateSparseDifferenceMatrix(object):
    def __call__(self, data):
        diff = (data.x.unsqueeze(1) == data.x).all(-1)
        edge_index_diff = torch.nonzero(diff, as_tuple=False).t()
        data.edge_index_diff = edge_index_diff
        return data


class CreateAdjacencyDifferenceMatrix(object):
    def __init__(self, max_node_num):
        self.max_node_num = max_node_num

    def __call__(self, data):
        n = data.x.size(0)
		# Initialize the matrices
        diff = torch.zeros((1, self.max_node_num, self.max_node_num), dtype=torch.float32)
        adj = torch.zeros((1, self.max_node_num, self.max_node_num), dtype=torch.float32)
		# Fill the matrices
        for i in range(n):
            for j in range(n):
                if torch.equal(data.x[i], data.x[j]):
                    diff[0, i, j] = 1.0
        for k in range(data.edge_index.size(1)):
            i, j = data.edge_index[0, k], data.edge_index[1, k]
            adj[0, i, j] = 1.0
        data.diff = diff
        data.adj = adj
        data.pad_feat = torch.cat((data.x, torch.zeros((self.max_node_num - n, data.x.size(1)))), 0).unsqueeze(0)
        return data

class CreateUniqueCounts(object):
    def __call__(self, data):
        _, data.x = data.x.unique(return_counts=True, dim=0)
        data.x = data.x.unsqueeze(1).to(torch.float32)
        data.num_nodes = data.x.shape[0]
        data.edge_index = None
        return data


# maximum node number
if True:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'TU_tmp')
    dataset = TUDataset(path, name=args.dataset)
    max_node_num = max([data.x.size(0) for data in dataset])
    print('Maximum node number is ' + str(max_node_num))

# Define the pre_transform
# todo: still transform to random hash, then handle that zeros are still zeros
if args.transform == 'random':
    transform = None
    pre_transform = ReplaceNodeAttributesWithRandomVectors(vector_dim=16)
    #pre_transform = CreateAdjacencyDifferenceMatrix(max_node_num)
    pre_transform = T.Compose([pre_transform, CreateAdjacencyDifferenceMatrix(max_node_num)])
    if args.online_trans:
        transform, pre_transform = pre_transform, None
elif args.transform == 'hash':
    transform = None
    pre_transform = ReplaceNodeAttributesWithHash()
    if args.online_trans:
        transform, pre_transform = pre_transform, None
    #pre_transform = CreateAdjacencyDifferenceMatrix(max_node_num)
    #pre_transform = T.Compose([pre_transform, CreateAdjacencyDifferenceMatrix(max_node_num)])
elif args.transform == 'one':
    transform = None
    pre_transform = ReplaceNodeAttributesWithOne()
elif args.transform == 'diff':
    transform = None
    pre_transform = CreateAdjacencyDifferenceMatrix(max_node_num)
elif args.transform == 'diff_one':
    transform = None
    pre_transform = T.Compose([CreateSparseDifferenceMatrix(), ReplaceNodeAttributesWithOne()])
elif args.transform == 'counts':
    transform = None
    pre_transform = CreateUniqueCounts()


path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'TU_' + args.transform)
if args.online_trans:
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'TU')
dataset = TUDataset(path, name=args.dataset, transform=transform, 
    pre_transform=pre_transform, use_node_attr=True).shuffle()

train_loader = DataLoader(dataset[:0.9], args.batch_size, shuffle=True)
test_loader = DataLoader(dataset[0.9:], args.batch_size)


if args.model == 'GIN':
    model = GIN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
    ).to(device)
if args.model == 'DiffGIN':
    model = DiffGIN(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
    ).to(device)
elif args.model == 'DenseGAT':
    model = DenseGAT(
        in_channels=dataset[0].pad_feat.shape[-1],
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
    ).to(device)
elif args.model == 'DeepDiff':
    model = DeepDiff(
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
    ).to(device)
elif args.model == 'DeepSet':
    model = DeepSet(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
    ).to(device)
elif args.model == 'DeepCount':
    model = DeepCount(
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        #pdb.set_trace()
        optimizer.zero_grad()
        if args.model == 'GIN':
            out = model(data.x, data.edge_index, data.batch)
        elif args.model == 'DiffGIN':
            out = model(data.x, data.edge_index, data.edge_index_diff, data.batch)
        elif args.model == 'DenseGAT':
            out = model(data.pad_feat, data.adj, data.diff)
        elif args.model == 'DeepDiff':
            out = model(data.adj, data.diff)
        elif args.model in ['DeepSet', 'DeepCount']:
            out = model(data.x, data.batch)

        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        if args.model == 'GIN':
            out = model(data.x, data.edge_index, data.batch)
        elif args.model == 'DiffGIN':
            out = model(data.x, data.edge_index, data.edge_index_diff, data.batch)
        elif args.model == 'DenseGAT':
            out = model(data.pad_feat, data.adj, data.diff)
        elif args.model == 'DeepDiff':
            out = model(data.adj, data.diff)
        elif args.model in ['DeepSet', 'DeepCount']:
            out = model(data.x, data.batch)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    wandb.log({"Loss": loss, "Train": train_acc, "Test": test_acc})
    log(Epoch = epoch, Loss = loss, Train = train_acc, Test = test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
