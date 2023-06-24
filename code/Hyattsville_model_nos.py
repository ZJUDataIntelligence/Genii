from sklearn.metrics import r2_score
from torch_geometric.nn import GATv2Conv, GATConv, HANConv
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

dataset = torch.load('./Hyattsville_graph/processed/Hyattsville_litter_graph.pt')
# print('litter: ',dataset[0][0][0].x, dataset[0][0][0].edge_index)
# print('poi: ',dataset[0][0][1].x, dataset[0][0][1].edge_index)
# print('poi-litter: ', dataset[0][0][2].x_dict, dataset[0][0][2].edge_index_dict)
# print('metadata: ', dataset[0][0][2].metadata())

# 合并数据集
# data_set3 = {'data1': dataset_poi, 'data2': dataset_litter, 'data3': data_poi_litter}
# data_set3 = {key: data_set3[key].cuda() for key in data_set3}

tw_data = pd.read_csv('./Hyattsville_baseline/Hyattsville_ML.csv')
tw_train = tw_data.iloc[:, 1:-1].reset_index(drop=True)
tw_train = np.array(tw_train)
tw_train = torch.tensor(tw_train, dtype=torch.float32)
y = tw_data.iloc[:, -1][:116].reset_index(drop=True)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


class Tw_Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(23, 10)

    def forward(self, data):
        x = self.fc1(data)

        return x


class Model(torch.nn.Module):
    def __init__(self, num_node_features_cluster1_1, num_node_features_cluster1_2,
                 num_node_features_cluster2_1, num_node_features_cluster2_2,
                 num_node_features_cluster3_1, num_node_features_cluster3_2,
                 num_node_features_cluster4_1, num_node_features_cluster4_2):
        super(Model, self).__init__()

        self.t_model = Tw_Linear()
        self.re = nn.ReLU()
        self.fcn = nn.Linear(10, 1)

    def forward(self, s_data, t_data):
        x_t = self.t_model(t_data)
        x_t = self.re(x_t)
        x = self.fcn(x_t)
        return x
