import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import geopandas as gpd
from haversine import haversine
from torch_geometric.data import HeteroData
from Encoder import *


def poi_distance(d, point1, point2):
    osm_id_s = []
    osm_id_d = []
    distance = []
    # poi_edge = pd.DataFrame([], columns=['osm_id_s', 'osm_id_d', 'distance'])
    # i = 0
    for id1, p1_row in point1.iterrows():
        for id2, p2_row in point2.iterrows():
            l1 = list(p1_row['geometry'].coords)[0]
            l2 = list(p2_row['geometry'].coords)[0]
            dis = haversine(l1, l2)
            if dis <= d and dis != 0.0:
                # 耗时更多的精简方案 ：
                # poi_edge.loc[i] = [p1_row['osm_id'], p2_row['osm_id'], p1_row['geometry'].distance(
                # p2_row['geometry'])] i += 1
                osm_id_s.append(p1_row['osm_id'])
                osm_id_d.append(p2_row['osm_id'])
                distance.append(dis)

    osm_id_s = np.array(osm_id_s).reshape(-1, 1)
    osm_id_d = np.array(osm_id_d).reshape(-1, 1)
    distance = np.array(distance).reshape(-1, 1)
    poi_edge_array = np.concatenate([osm_id_s, osm_id_d, distance], axis=1)
    poi_edge = pd.DataFrame(poi_edge_array, columns=['osm_id_s', 'osm_id_d', 'distance'])
    return poi_edge


# poi = gpd.read_file('Hyattsville_POI_all/points.shp')
# poi_df = poi[['osm_id', 'type', 'geometry']]
# points_edge = poi_distance(0.1, poi_df, poi_df)
# poi_df.to_csv('./Hyattsville_POI_all/poi.csv')
poi_path = './Hyattsville_POI_all/poi.csv'
# points_edge.to_csv('./Hyattsville_POI_all/points_edge.csv')
poi_edge_path = './Hyattsville_POI_all/points_edge.csv'
poi_x, poi_mapping = load_node_csv(poi_path, index_col='osm_id', encoders={
    #     'geometry': SequenceEncoder(),
    'type': GenresEncoder()
})
print(poi_x[0])

poi_edge_index, poi_edge_attr = load_edge_csv(
    poi_edge_path,
    src_index_col='osm_id_s',
    src_mapping=poi_mapping,
    dst_index_col='osm_id_d',
    dst_mapping=poi_mapping,
    encoders={'distance': IdentityEncoder(dtype=torch.long)},
)

data = HeteroData()
data['poi'].x = poi_x
print(data)

data['poi', 'distance', 'poi'].edge_index = poi_edge_index
data['poi', 'distance', 'poi'].edge_attr = poi_edge_attr
print(data)
# print(data['poi'].x[0])
# print(data['poi', 'distance', 'poi'].edge_index[0][0:10])
