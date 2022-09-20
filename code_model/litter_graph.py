from torch_geometric.data import HeteroData, download_url, extract_zip
import pandas as pd
import numpy as np
import geopandas as gpd
from haversine import haversine
from Encoder import *


def litter_distance(d, litter1, litter_ad, size):
    id_s = []
    id_d = []
    distance = []

    litter1 = litter1[:size]
    # 在原图结点之间构造边
    for id1, l1_row in litter1.iterrows():
        for id2, l2_row in litter1.iterrows():
            l1 = (l1_row['lat'], l1_row['lon'])
            l2 = (l2_row['lat'], l2_row['lon'])
            dis = haversine(l1, l2)
            # if dis <= d and dis != 0:
            if dis <= d:
                dis = 0
            id_s.append(l1_row['id'])
            id_d.append(l2_row['id'])
            distance.append(dis)
    if len(litter1) < size:
        # 补充图自身的结点之间构造边
        for i in range(size - len(litter1)):
            for j in range(size - len(litter1)):
                id_s.append(litter_ad.iloc[i]['id'])
                id_d.append(litter_ad.iloc[j]['id'])
                distance.append(0)
        # 源：补充图 目：原图
        for i in range(size - len(litter1)):
            for j in range(len(litter1)):
                id_s.append(litter_ad.iloc[i]['id'])
                id_d.append(litter1.iloc[j]['id'])
                distance.append(0)
        # 源：原图 目：补充图
        for i in range(len(litter1)):
            for j in range(size - len(litter1)):
                id_s.append(litter1.iloc[i]['id'])
                id_d.append(litter_ad.iloc[j]['id'])
                distance.append(0)
    print(len(id_s), len(id_d), len(distance))
    id_s = np.array(id_s).reshape(-1, 1)
    id_d = np.array(id_d).reshape(-1, 1)
    distance = np.array(distance).reshape(-1, 1)

    litter_edge_array = np.concatenate([id_s, id_d, distance], axis=1)
    litter_edge = pd.DataFrame(litter_edge_array, columns=['id_s', 'id_d', 'distance'])
    return litter_edge


if __name__ == '__main__':
    litter_ad = pd.read_csv('./litter_days/day75_litter2.csv')
    litter_ad_df = litter_ad[['id', 'lat', 'lon', 'total_litter']]
    litter_ad_df.to_csv('./litter_days_df/litter_ad.csv')
    for i in range(210):
        for j in range(1, 5):
            litter = pd.read_csv('./litter_days/day{0}_litter{1}.csv'.format(i, j))
            litter_df = litter[['id', 'lat', 'lon', 'total_litter']]
            litter_edge = litter_distance(0.05, litter_df, litter_ad, 85)
            litter_df.to_csv('./litter_days_df/day{0}_litter{1}.csv'.format(i, j))
            litter_path = './litter_days_df/day{0}_litter{1}.csv'.format(i, j)
            litter_edge.to_csv('./litter_days_edge/day{0}_litter{1}.csv'.format(i, j))
            litter_edge_path = './litter_days_edge/day{0}_litter{1}.csv'.format(i, j)

# test
for i in range(210):
    for j in range(1, 5):
        litter = pd.read_csv('./litter_days_df/day{0}_litter{1}.csv'.format(i, j))

        # litter_ad = pd.read_csv('./litter_days/day75_litter2.csv')
        # litter_ad_df = litter_ad[['id', 'lat', 'lon', 'total_litter']]
        # litter_ad_df.to_csv('./litter_days_df/litter_ad.csv')
        litter_path = './litter_days_df/day{0}_litter{1}.csv'.format(i, j)
        litter_ad_path = './litter_days_df/litter_ad.csv'
        litter_edge_path = './litter_days_edge/day0_litter1.csv'
        litter_x, litter_mapping = load_add_node_csv(litter_path, litter_ad_path, size=85, index_col='id', encoders={
            #     'geometry': SequenceEncoder(),
            'total_litter': IdentityEncoder()
        })
        if litter_x.shape[0] != 85:
            print((i,j))
        print('over!')
        # edge_index, edge_attr = load_edge_csv(
        #     litter_edge_path,
        #     src_index_col='id_s',
        #     src_mapping=litter_mapping,
        #     dst_index_col='id_d',
        #     dst_mapping=litter_mapping,
        #     encoders={'distance': IdentityEncoder(dtype=torch.long)},
        # )
        # data = HeteroData()
        # data['litter'].x = litter_x
        # print(data)
        #
        # data['litter', 'distance', 'litter'].edge_index = edge_index
        # data['litter', 'distance', 'litter'].edge_attr = edge_attr
        # print(data)
        # print(data['litter'].x[0])
        # print(data['litter', 'distance', 'litter'].edge_index[0])


