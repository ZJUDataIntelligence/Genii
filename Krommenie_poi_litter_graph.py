from torch_geometric.data import HeteroData, download_url, extract_zip
import pandas as pd
import numpy as np
import geopandas as gpd
from haversine import haversine
from Encoder import *


def poi_litter_distance(poi, litter, litter_ad, size):
    id_s = []
    id_d = []
    distance = []
    litter = litter[:size]
    # 原图--poi
    for id1, l1_row in poi.iterrows():
        for id2, l2_row in litter.iterrows():
            l1 = (l1_row['lat'], l1_row['lon'])
            l2 = (l2_row['lat'], l2_row['lon'])
            dis = haversine(l1, l2)
            if dis != 0:
                dis = 1 / dis
                id_s.append(l1_row['id'])
                id_d.append(l2_row['id'])
                distance.append(dis)
            else:
                id_s.append(l1_row['id'])
                id_d.append(l2_row['id'])
                distance.append(dis)
    if len(litter) < size:
        # 补充图--poi
        for i in range(len(poi)):
            for j in range(size - len(litter)):
                id_s.append(poi.iloc[i]['id'])
                id_d.append(litter_ad.iloc[j]['id'])
                distance.append(0)

    id_s = np.array(id_s).reshape(-1, 1)
    id_d = np.array(id_d).reshape(-1, 1)
    print(len(id_s), len(id_d))
    distance = np.array(distance).reshape(-1, 1)
    poi_litter_edge_array = np.concatenate([id_s, id_d, distance], axis=1)
    poi_litter_edge = pd.DataFrame(poi_litter_edge_array, columns=['poi_id', 'litter_id', 'distance'])
    return poi_litter_edge


# if __name__ == '__main__':
#     # 合并位置信息
#     for i in range(181):
#         for j in range(1, 5):
#             poi = gpd.read_file('Krommenie_POI_all/points.shp')
#             for rowid, row in poi.iterrows():
#                 poi.loc[rowid, 'lat'] = list(row['geometry'].coords)[0][1]
#                 poi.loc[rowid, 'lon'] = list(row['geometry'].coords)[0][0]
#             # poi1_df = poi1[['osm_id', 'lat', 'lon']]
#             poi_df = poi.loc[:, ['osm_id', 'lat', 'lon']]
#             poi_df.rename(columns={'osm_id': 'id'}, inplace=1)
#             litter_df = pd.read_csv('./Krommenie_litter_days_df/day{0}_litter{1}.csv'.format(i, j))
#             # litter_df = litter[['id', 'lat', 'lon']]
#             litter_ad_path = './Krommenie_litter_days_df/litter_ad.csv'
#             litter_ad = pd.read_csv(litter_ad_path)
#             # poi_litter_dtdf = pd.concat([poi1_df, litter_df], axis=1).reset_index(drop=True)
#             # 构造边表
#             poi_litter_edge = poi_litter_distance(poi_df, litter_df, litter_ad, 85)
#             poi_litter_edge.to_csv('./Krommenie_poi_litter_edge/poi_litter_day{0}_cluster{1}.csv'.format(i, j))
#     print('over')
# test
# 路径
poi_litter_edge_path = './Krommenie_poi_litter_edge/poi_litter_day0_cluster1.csv'
litter_ad_path = './Krommenie_litter_days_df/litter_ad.csv'
litter_path = './Krommenie_litter_days_df/day0_litter1.csv'
litter_edge_path = './Krommenie_litter_days_edge/day0_litter1.csv'
poi_path = './Krommenie_POI_all/poi.csv'
poi_edge_path = './Krommenie_POI_all/points_edge.csv'

# 加载点
litter_x, litter_mapping = load_add_node_csv(litter_path, litter_ad_path, size=85, index_col='id', encoders={
    #     'geometry': SequenceEncoder(),
    'total_litter': IdentityEncoder()
})
# print(litter_x[0])
poi_x, poi_mapping = load_node_csv(poi_path, index_col='osm_id', encoders={
    #     'geometry': SequenceEncoder(),
    'type': GenresEncoder()
})
# print(poi_x[0])

# 加载边
poi_litter_edge_index0, poi_litter_edge_attr0 = load_edge_csv(
    poi_litter_edge_path,
    src_index_col='poi_id',
    src_mapping=poi_mapping,
    dst_index_col='litter_id',
    dst_mapping=litter_mapping,
    encoders={'distance': IdentityEncoder(dtype=torch.long)},
)
poi_litter_edge_index1, poi_litter_edge_attr1 = load_edge_csv(
    poi_litter_edge_path,
    src_index_col='litter_id',
    src_mapping=litter_mapping,
    dst_index_col='poi_id',
    dst_mapping=poi_mapping,
    encoders={'distance': IdentityEncoder(dtype=torch.long)},
)
# 合并不同方向的边
poi_litter_edge_index = torch.cat([poi_litter_edge_index0, poi_litter_edge_index1], dim=1)
poi_litter_edge_attr = torch.cat([poi_litter_edge_attr0, poi_litter_edge_attr1], dim=0)

poi_edge_index, poi_edge_attr = load_edge_csv(
    poi_edge_path,
    src_index_col='osm_id_s',
    src_mapping=poi_mapping,
    dst_index_col='osm_id_d',
    dst_mapping=poi_mapping,
    encoders={'distance': IdentityEncoder(dtype=torch.long)},
)
litter_edge_index, litter_edge_attr = load_edge_csv(
    litter_edge_path,
    src_index_col='id_s',
    src_mapping=litter_mapping,
    dst_index_col='id_d',
    dst_mapping=litter_mapping,
    encoders={'distance': IdentityEncoder(dtype=torch.long)},
)

data = HeteroData()
data['litter'].x = litter_x
data['poi'].x = poi_x
print(data)

data['litter', 'distance', 'litter'].edge_index = litter_edge_index
data['litter', 'distance', 'litter'].edge_attr = litter_edge_attr
data['poi', 'distance', 'poi'].edge_index = poi_edge_index
data['poi', 'distance', 'poi'].edge_attr = poi_edge_attr
data['poi', 'distance', 'litter'].edge_index = poi_litter_edge_index
data['poi', 'distance', 'litter'].edge_attr = poi_litter_edge_attr
print(data)
print(data.num_nodes)
print(data.num_edges)