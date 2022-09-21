import torch
from sentence_transformers import SentenceTransformer
import pandas as pd


# SequenceEncoder 类加载由 model_name 给定的预训练 NLP 模型，并使用它将字符串列表编码为形状为 [num_strings, embedding_dim]的 PyTorch 张量
class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()


# 转换类型为分类标签的编码器类
class GenresEncoder(object):
    def __init__(self, sep='|'):
        self.sep = sep

    def __call__(self, df):
        genres = set(g for col in df.values for g in col.split(self.sep))
        mapping = {genre: i for i, genre in enumerate(genres)}

        x = torch.zeros(len(df), len(mapping))
        for i, col in enumerate(df.values):
            for genre in col.split(self.sep):
                x[i, mapping[genre]] = 1
        return x


# IdentityEncoder 编码器类将浮点值列表转换为 PyTorch 张量：
class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


# 加载结点
def load_add_node_csv(path1, path2, size, index_col, encoders=None, **kwargs):
    df1 = pd.read_csv(path1, index_col=index_col, **kwargs)
    df_ad = pd.read_csv(path2, index_col=index_col, **kwargs)
    len1 = len(df1)
    if len1 <= size:
        df_ad = df_ad[:size-len1]
        df = pd.concat([df1, df_ad], axis=0)
    else:
        df = df1[:size]
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_node_csv(path1, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path1, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


# 加载边
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr
