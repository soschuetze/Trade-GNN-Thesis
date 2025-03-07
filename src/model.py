import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.nn import SAGEConv, GINConv, GATConv, GCNConv, global_sort_pool
import torch.nn as nn

#Embedding options
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim*2)
        self.conv2 = GCNConv(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index.to(torch.int64)

        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim*2)
        self.conv2 = SAGEConv(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index.to(torch.int64)

        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2)
        self.conv2 = GATConv(hidden_dim*2, hidden_dim, heads=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index.to(torch.int64)

        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GIN, self).__init__()
        
        # Define MLP for GINConv layers
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Define GINConv layers
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        
        self.linear = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, data):
        # First GIN layer
        x, edge_index = data.x.float(), data.edge_index.to(torch.int64)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GIN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.linear(x)
        
        return x

#Classifier
class SiameseGNN(torch.nn.Module):
    def __init__(self, encoder, top_k, input_dim, dropout, nhidden):
        super(SiameseGNN, self).__init__()

        self.encoder = encoder
        self.topk_layer = torch.topk
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.similarity = nn.PairwiseDistance()

        self.fc1 = Linear(top_k, nhidden*2)
        self.norm1 = LayerNorm(nhidden*2)
        self.relu1 = ReLU()

        self.fc2 = Linear(nhidden*2, nhidden)
        self.norm2 = LayerNorm(nhidden)
        self.relu2 = ReLU()

        self.fc3 = Linear(nhidden, 1)

    def forward(self, data1, data2):
        
        out1 = self.encoder(data1)
        out2 = self.encoder(data2)

        similarity = self.similarity(out1, out2)
        out, _  = self.topk_layer(similarity, k=self.top_k)

        #Fully Connected Layer 1
        out = self.fc1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        #Fully Connected Layer 2
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out