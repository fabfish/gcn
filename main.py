import torch
import torch.nn as nn
import random
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, PairNorm

dataset = Planetoid(root='/tmp/Cora', name='Cora')
graph = dataset[0]
split = T.RandomNodeSplit(num_val=0.1, num_test=0.2)
graph = split(graph)

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

def get_mask(y:torch.tensor):
    train_mask = torch.tensor([False] * y.shape[0])
    for i in torch.unique(y).unbind():
        temp = torch.arange(0, y.shape[0])[y == i].tolist()
        random.shuffle(temp)
        train_mask[temp[:30]] = True
    
    train_mask = torch.tensor(train_mask)
    test_mask = train_mask == False
    return train_mask, test_mask

def drop_edge(edge_index, keep_ratio:float=1.):
    num_keep = int(keep_ratio * edge_index.shape[1])
    temp = [True] * num_keep + [False] * (edge_index.shape[1] - num_keep)
    random.shuffle(temp)
    return edge_index[:, temp]

class GCN(torch.nn.Module):
    def __init__(self, 
        dim_features, 
        num_classes, 
        num_layers, 
        add_self_loops:bool=True, 
        use_pairnorm:bool=False, 
        drop_edge:float=1., 
        activation:str='relu', 
        undirected:bool=False
    ):
        super(GCN, self).__init__()
        dim_hidden = 32

        self.gconvs = torch.nn.ModuleList(
            [GCNConv(in_channels=dim_features, out_channels=dim_hidden, add_self_loops=add_self_loops)] 
            + [GCNConv(in_channels=dim_hidden, out_channels=dim_hidden, add_self_loops=add_self_loops) for i in range(num_layers - 2)]
        )
        self.final_conv = GCNConv(in_channels=dim_hidden, out_channels=num_classes, add_self_loops=add_self_loops)

        self.use_pairnorm = use_pairnorm
        if self.use_pairnorm:
            self.pairnorm = PairNorm()
        self.drop_edge = drop_edge
        activations_map = {'relu':torch.relu, 'tanh':torch.tanh, 'sigmoid':torch.sigmoid, 'leaky_relu':torch.nn.LeakyReLU(0.1)}
        self.activation_fn = activations_map[activation]
        
    def forward(self, x, edge_index):
        for l in self.gconvs:
            edges = drop_edge(edge_index, self.drop_edge)
            x = l(x, edges)
            if self.use_pairnorm:
                x = self.pairnorm(x)
            x = self.activation_fn(x)
        x = self.final_conv(x, edge_index)

        return x

def train_node_classifier(model, graph, optimizer, criterion, n_epochs=200):

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph)
        loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = eval_node_classifier(model, graph, graph.val_mask)

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val Acc: {acc:.3f}')

    return model


def eval_node_classifier(model, graph, mask):

    model.eval()
    pred = model(graph).argmax(dim=1)
    correct = (pred[mask] == graph.y[mask]).sum()
    acc = int(correct) / int(mask.sum())

    return acc
    
device = 'cuda:0'

num_epochs = 100
test_cases = [
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    # num layers
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    {'num_layers':6, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    # self loop
    {'num_layers':2, 'add_self_loops':False, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    # pair norm
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    {'num_layers':6, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu', 'undirected':False}, 
    # drop edge
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.6, 'activation':'relu', 'undirected':False}, 
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.6, 'activation':'relu', 'undirected':False}, 
    # activation fn
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'tanh', 'undirected':False}, 
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'leaky_relu', 'undirected':False}, 
    # undirected
    {'num_layers':2, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':1., 'activation':'relu', 'undirected':True}, 
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':True, 'drop_edge':1., 'activation':'relu', 'undirected':True}, 
    {'num_layers':4, 'add_self_loops':True, 'use_pairnorm':False, 'drop_edge':0.8, 'activation':'relu', 'undirected':True}, 
]

for i_case, kwargs in enumerate(test_cases):
    print(f'Test Case {i_case:>2}')
    model = GCN(x.shape[1], len(labels), **kwargs)

gcn = GCN().to(device)
optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
gcn = train_node_classifier(gcn, graph, optimizer_gcn, criterion)

test_acc = eval_node_classifier(gcn, graph, graph.test_mask)
print(f'Test Acc: {test_acc:.3f}')