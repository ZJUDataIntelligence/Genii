import torch
from torch_geometric.data import InMemoryDataset, download_url
# from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader
from Encoder import *
from torch_geometric.data import HeteroData
from torch_geometric.data import Data


class Dataset_Net1(InMemoryDataset):
    def download(self):
        pass

    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # names = self.__dict__  # 类对象的属性储存在的__dict__中。__dict__是一个词典，键为属性名，值对应属性的值。
        # for i in range(840):
        #     names['data'+str(i)] = torch.load(self.processed_paths[i])
        self.data = torch.load(self.processed_paths[0])

    # print(root) # MYdata
    # print(self.data) # Data(x=[3, 1], edge_index=[2, 4], y=[3])
    # print(self.slices) # defaultdict(<class 'dict'>, {'x': tensor([0, 3, 6]), 'edge_index': tensor([ 0,  4, 10]), 'y': tensor([0, 3, 6])})
    # print(self.processed_paths[0]) # MYdata\processed\data.pt

    # 返回数据集原始文件名
    @property
    def raw_file_names(self):
        return []

    # 返回process方法所需的保存文件名。之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['litter_graph.pt']
        # filename_list = []
        # for i in range(840):
        #     filename = 'litter_graph{0}.pt'.format(i)
        #     filename_list.append(filename)
        # return filename_list

        # #用于从网上下载数据集
        # def download(self):
        #     # Download to `self.raw_dir`.
        #     download_url(url, self.raw_dir)

    # 生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        # Read data into huge `Data` list.
        # 这里用于构建data

        data_list = []
        for i in range(210):
            data_day = []
            for j in range(1, 5):
                # litter
                litter_ad_path = './litter_days_df/litter_ad.csv'
                litter_path = './litter_days_df/day{0}_litter{1}.csv'.format(i, j)
                litter_edge_path = './litter_days_edge/day{0}_litter{1}.csv'.format(i, j)
                litter_x, litter_mapping = load_add_node_csv(litter_path, litter_ad_path, size=85, index_col='id',
                                                             encoders={
                                                                 #     'geometry': SequenceEncoder(),
                                                                 'total_litter': IdentityEncoder(dtype=torch.float32)
                                                             })
                print(litter_x[0])

                litter_edge_index, litter_edge_attr = load_edge_csv(
                    litter_edge_path,
                    src_index_col='id_s',
                    src_mapping=litter_mapping,
                    dst_index_col='id_d',
                    dst_mapping=litter_mapping,
                    encoders={'distance': IdentityEncoder(dtype=torch.float32)},
                )
                dataset_litter = Data(x=litter_x, edge_index=litter_edge_index, edge_attr=litter_edge_attr)

                # POI
                poi_path = './POI_all/poi.csv'
                poi_edge_path = './POI_all/points_edge.csv'
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
                    encoders={'distance': IdentityEncoder(dtype=torch.float32)},
                )
                dataset_poi = Data(x=poi_x, edge_index=poi_edge_index, edge_attr=poi_edge_attr)

                # POI_litter
                # 路径
                poi_litter_edge_path = './poi_litter_edge/poi_litter_day{0}_cluster{1}.csv'.format(i, j)
                litter_path = './litter_days/day{0}_litter{1}.csv'.format(i, j)
                litter_edge_path = './litter_days_edge/day{0}_litter{1}.csv'.format(i, j)
                litter_ad_path = './litter_days_df/litter_ad.csv'
                poi_path = 'POI_all/poi.csv'
                poi_edge_path = 'POI_all/points_edge.csv'

                # 加载点
                litter_x, litter_mapping = load_add_node_csv(litter_path, litter_ad_path, size=85, index_col='id',
                                                             encoders={
                                                                 #     'geometry': SequenceEncoder(),
                                                                 'total_litter': IdentityEncoder(dtype=torch.float32)
                                                             })
                print(litter_x[0])
                poi_x, poi_mapping = load_node_csv(poi_path, index_col='osm_id', encoders={
                    #     'geometry': SequenceEncoder(),
                    'type': GenresEncoder()
                })
                print(poi_x[0])

                # 加载边
                poi_litter_edge_index0, poi_litter_edge_attr0 = load_edge_csv(
                    poi_litter_edge_path,
                    src_index_col='poi_id',
                    src_mapping=poi_mapping,
                    dst_index_col='litter_id',
                    dst_mapping=litter_mapping,
                    encoders={'distance': IdentityEncoder(dtype=torch.float32)},
                )
                poi_litter_edge_index1, poi_litter_edge_attr1 = load_edge_csv(
                    poi_litter_edge_path,
                    src_index_col='litter_id',
                    src_mapping=litter_mapping,
                    dst_index_col='poi_id',
                    dst_mapping=poi_mapping,
                    encoders={'distance': IdentityEncoder(dtype=torch.float32)},
                )
                # # 合并不同方向的边
                # poi_litter_edge_index = torch.cat([poi_litter_edge_index0, poi_litter_edge_index1], dim=1)
                # poi_litter_edge_attr = torch.cat([poi_litter_edge_attr0, poi_litter_edge_attr1], dim=0)

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

                data_poi_litter = HeteroData()
                data_poi_litter['litter'].x = litter_x
                data_poi_litter['poi'].x = poi_x
                # print(data_poi_litter)

                data_poi_litter['litter', 'distance', 'litter'].edge_index = litter_edge_index
                data_poi_litter['litter', 'distance', 'litter'].edge_attr = litter_edge_attr
                data_poi_litter['poi', 'distance', 'poi'].edge_index = poi_edge_index
                data_poi_litter['poi', 'distance', 'poi'].edge_attr = poi_edge_attr
                data_poi_litter['poi', 'distance', 'litter'].edge_index = poi_litter_edge_index0
                data_poi_litter['poi', 'distance', 'litter'].edge_attr = poi_litter_edge_attr0
                data_poi_litter['litter', 'distance', 'poi'].edge_index = poi_litter_edge_index1
                data_poi_litter['litter', 'distance', 'poi'].edge_attr = poi_litter_edge_attr1
                print(data_poi_litter)
                # # POI_cluster4
                # litter_ad_path = './litter_days_df/litter_ad.csv'
                # litter_path = './litter_days_df/day{0}_litter4.csv'.format(i)
                # litter_edge_path = './litter_days_edge/day{0}_litter4.csv'.format(i)
                # litter_x, litter_mapping = load_add_node_csv(litter_path, litter_ad_path, index_col='id', encoders={
                #     #     'geometry': SequenceEncoder(),
                #     'total_litter': IdentityEncoder()
                # })
                # print(litter_x[0])
                #
                # litter_edge_index, litter_edge_attr = load_edge_csv(
                #     litter_edge_path,
                #     src_index_col='id_s',
                #     src_mapping=litter_mapping,
                #     dst_index_col='id_d',
                #     dst_mapping=litter_mapping,
                #     encoders={'distance': IdentityEncoder(dtype=torch.long)},
                # )
                #
                # dataset_litter4 = Data(x=litter_x, edge_index=litter_edge_index, edge_attr=litter_edge_attr)
                # 放入datalist
                data_day.append([dataset_litter, dataset_poi, data_poi_litter])
            data_list.append(data_day)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # for i in range(840):
        #     torch.save((data_list[i]), self.processed_paths[i])

        # data, slices = self.collate(data_list)
        torch.save(data_list, self.processed_paths[0])
        # torch.save((data_list[2]), self.processed_paths[2])


b = Dataset_Net1("graph")
print(b.data[0])

# data_loader = DataLoader(b, batch_size=1, shuffle=False)  # 加载数据进行处理，每批次数据的数量为1
# for i, data in enumerate(data_loader, 1):
#     print(data)
b1 = torch.load('./graph/processed/litter_graph.pt')
print(b1[0][0][2].edge_index_dict['litter', 'distance', 'poi'].size())
print(b1[0][1][2].edge_index_dict['litter', 'distance', 'poi'].size())
print(b1[0][2][2].edge_index_dict['litter', 'distance', 'poi'].size())
print(b1[0][3][2].edge_index_dict['litter', 'distance', 'poi'].size())
print(b1[1][0][2].edge_index_dict['litter', 'distance', 'poi'].size())
print(b1[1][0][2].metadata())
# print(b1[0][0][0].x.size(), b1[0][0][0].x.size())
# print(b1[0][1][0].x.size(), b1[0][1][0].x.size())
# print(b1[0][2][0].x.size(), b1[0][2][0].x.size())
# print(b1[0][3][0].x.size(), b1[0][3][0].x.size())
