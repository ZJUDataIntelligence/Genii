from sklearn.metrics import r2_score
from torch_geometric.nn import GATv2Conv, GATConv, HANConv, GCNConv
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

dataset = torch.load('./Krommenie_graph/processed/Krommenie_litter_graph.pt')
# print('litter: ',dataset[0][0][0].x, dataset[0][0][0].edge_index)
# print('poi: ',dataset[0][0][1].x, dataset[0][0][1].edge_index)
# print('poi-litter: ', dataset[0][0][2].x_dict, dataset[0][0][2].edge_index_dict)
# print('metadata: ', dataset[0][0][2].metadata())

# 合并数据集
# data_set3 = {'data1': dataset_poi, 'data2': dataset_litter, 'data3': data_poi_litter}
# data_set3 = {key: data_set3[key].cuda() for key in data_set3}

tw_data = pd.read_csv('./Baseline/Krommenie_onlylitter.csv')
tw_train = tw_data.iloc[:, 1:-1].reset_index(drop=True)
tw_train = np.array(tw_train)
tw_train = torch.tensor(tw_train, dtype=torch.float32)
y = tw_data.iloc[:, -1][:181].reset_index(drop=True)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


class Spatial_Net1(torch.nn.Module):
    def __init__(self, num_node_features1, num_node_features2):
        super(Spatial_Net1, self).__init__()
        self.gcn1 = GCNConv(num_node_features1, out_channels=120)
        self.re = nn.ReLU()
        self.fcn = nn.Linear(85, 5)

    def forward(self, data):
        x1 = self.gcn1(data[0].x, data[0].edge_index)
        x1 = torch.reshape(x1, (120, 85))

        x = self.re(x1)
        x = self.fcn(x)

        return x


class Spatial_Net2(torch.nn.Module):
    def __init__(self, num_node_features1, num_node_features2):
        super(Spatial_Net2, self).__init__()
        self.gcn1 = GCNConv(num_node_features1, out_channels=120)
        self.re = nn.ReLU()
        self.fcn = nn.Linear(85, 5)

    def forward(self, data):
        x1 = self.gcn1(data[0].x, data[0].edge_index)
        x1 = torch.reshape(x1, (120, 85))

        x = self.re(x1)
        x = self.fcn(x)

        return x


class Spatial_Net3(torch.nn.Module):
    def __init__(self, num_node_features1, num_node_features2):
        super(Spatial_Net3, self).__init__()
        self.gcn1 = GCNConv(num_node_features1, out_channels=120)
        self.re = nn.ReLU()
        self.fcn = nn.Linear(85, 5)

    def forward(self, data):
        x1 = self.gcn1(data[0].x, data[0].edge_index)
        x1 = torch.reshape(x1, (120, 85))

        x = self.re(x1)
        x = self.fcn(x)

        return x


class Spatial_Net4(torch.nn.Module):
    def __init__(self, num_node_features1, num_node_features2):
        super(Spatial_Net4, self).__init__()
        self.gcn1 = GCNConv(num_node_features1, out_channels=120)
        self.re = nn.ReLU()
        self.fcn = nn.Linear(85, 5)

    def forward(self, data):
        x1 = self.gcn1(data[0].x, data[0].edge_index)
        x1 = torch.reshape(x1, (120, 85))

        x = self.re(x1)
        x = self.fcn(x)

        return x


class Tw_Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 6)

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
        self.s_fcn = nn.Linear(20, 1)  # 120*1
        # self.t_fcn = nn.Linear(5, 1)  # 25*1
        self.fcn1 = nn.Linear(63, 1)
        self.fcn2 = nn.Linear(2, 1)

    def forward(self, s_data, t_data):
        x1 = self.s_model1(s_data[0])
        x2 = self.s_model2(s_data[1])
        x3 = self.s_model3(s_data[2])
        x4 = self.s_model4(s_data[3])
        x = torch.cat((x1, x2, x3, x4), 1)  # 120*20
        x_t = self.t_model(t_data)
        x_s = self.s_fcn(x).reshape(2, -1)
        x_t = x_t.reshape(2, -1)
        # x_t = x_t.unsqueeze(1)
        x = torch.cat((x_s, x_t), 1)
        x = self.fcn1(x)
        x = x.reshape(1, -1)
        x = self.fcn2(x)
        x = x.squeeze(1)
        return x
