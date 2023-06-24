from sklearn.metrics import r2_score
from torch_geometric.nn import GATv2Conv, GCNConv, HANConv
import torch.nn.functional as F
import torch
import os
from typing import Dict, List, Union
from Encoder import *
from torch import nn
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData, download_url, extract_zip
import pandas as pd
import numpy as np
import geopandas as gpd
from torch_geometric.data import Data
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_default_tensor_type(torch.FloatTensor)

dataset = torch.load('./graph_dataset/processed/graph.pt')

tw_data = pd.read_csv('./Baseline/Hoorn_ML.csv')
tw_train = tw_data.iloc[:, 1:-1].reset_index(drop=True)
tw_train = np.array(tw_train)
tw_train = torch.tensor(tw_train, dtype=torch.float32)
y = tw_data.iloc[:, -1][:210].reset_index(drop=True)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


class Spatial_Net1(torch.nn.Module):
    def __init__(self, num_node_features1, num_node_features2):
        super(Spatial_Net1, self).__init__()
        self.gcn1 = GCNConv(num_node_features1, out_channels=20)

        self.re = nn.ReLU()
        self.fcn = nn.Linear(17, 7)

    def forward(self, data):
        x1 = self.gcn1(data[0].x, data[0].edge_index)
        x1 = torch.reshape(x1, (100, 17))

        x = self.re(x1)
        x = self.fcn(x)

        return x


class Spatial_Net2(torch.nn.Module):
    def __init__(self, num_node_features1, num_node_features2):
        super(Spatial_Net2, self).__init__()
        self.gcn1 = GCNConv(num_node_features1, out_channels=20)

        self.re = nn.ReLU()
        self.fcn = nn.Linear(17, 7)

    def forward(self, data):
        x1 = self.gcn1(data[0].x, data[0].edge_index)
        x1 = torch.reshape(x1, (100, 17))

        x = self.re(x1)
        x = self.fcn(x)

        return x


class Spatial_Net3(torch.nn.Module):
    def __init__(self, num_node_features1, num_node_features2):
        super(Spatial_Net3, self).__init__()
        self.gcn1 = GCNConv(num_node_features1, out_channels=20)

        self.re = nn.ReLU()
        self.fcn = nn.Linear(17, 7)

    def forward(self, data):
        x1 = self.gcn1(data[0].x, data[0].edge_index)
        x1 = torch.reshape(x1, (100, 17))

        x = self.re(x1)
        x = self.fcn(x)

        return x


class Spatial_Net4(torch.nn.Module):
    def __init__(self, num_node_features1, num_node_features2):
        super(Spatial_Net4, self).__init__()
        self.gcn1 = GCNConv(num_node_features1, out_channels=20)

        self.re = nn.ReLU()
        self.fcn = nn.Linear(17, 7)

    def forward(self, data):
        x1 = self.gcn1(data[0].x, data[0].edge_index)
        x1 = torch.reshape(x1, (100, 17))

        x = self.re(x1)
        x = self.fcn(x)

        return x


class Tw_Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 14)

    def forward(self, data):
        x = self.fc1(data)

        return x


class Model(torch.nn.Module):
    def __init__(self, num_node_features_cluster1_1, num_node_features_cluster1_2,
                 num_node_features_cluster2_1, num_node_features_cluster2_2,
                 num_node_features_cluster3_1, num_node_features_cluster3_2,
                 num_node_features_cluster4_1, num_node_features_cluster4_2):
        super(Model, self).__init__()
        self.s_model1 = Spatial_Net1(num_node_features_cluster1_1, num_node_features_cluster1_2)
        self.s_model2 = Spatial_Net2(num_node_features_cluster2_1, num_node_features_cluster2_2)
        self.s_model3 = Spatial_Net3(num_node_features_cluster3_1, num_node_features_cluster3_2)
        self.s_model4 = Spatial_Net4(num_node_features_cluster4_1, num_node_features_cluster4_2)
        self.t_model = Tw_Linear()
        self.re = nn.ReLU()
        self.s_fcn = nn.Linear(28, 1)

        self.fcn1 = nn.Linear(57, 1)
        self.fcn2 = nn.Linear(2, 1)

    def forward(self, s_data, t_data):
        x1 = self.s_model1(s_data[0])
        x2 = self.s_model2(s_data[1])
        x3 = self.s_model3(s_data[2])
        x4 = self.s_model4(s_data[3])
        x = torch.cat((x1, x2, x3, x4), 1)

        x_t = self.t_model(t_data)  # 14
        x_s = self.s_fcn(x).reshape(2, -1)  # 2*50
        x_t = x_t.reshape(2, -1)  # 2*7
        x = torch.cat((x_s, x_t), 1)
        x = self.fcn1(x)
        x = x.reshape(1, -1)
        x = self.fcn2(x)
        x = x.squeeze(1)
        return x
