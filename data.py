import torch
import dgl
import numpy as np

from torch.utils.data import Dataset
from dgl.data import DGLDataset


def generate_random_dict(data_types, min_len=30, max_len=50, order=5, edge_dim=32):
    x_len = np.random.randint(min_len, max_len)
    sample = {}
    if 'one_hot' in data_types:
        sample['one_hot'] = torch.tensor(np.random.randint(0, 2, (x_len, 20)).astype(np.float32))
    if 'position' in data_types:
        sample['position'] = torch.tensor(np.random.random((x_len, 3)).astype(np.float32))
    if 'edge' in data_types and edge_dim > 0:
        sample['edge'] = torch.tensor(np.random.random((x_len, x_len, edge_dim)).astype(np.float32))
    if 'features' in data_types:
        sample['features'] = torch.tensor(np.random.random((x_len, 3)).astype(np.float32))
    if 'sh' in data_types:
        sample['sh'] = [torch.tensor(np.random.random((x_len, x_len)).astype(np.float32)) for i in
                        range(np.square(order))]
    if 'cad' in data_types or 'lddt' in data_types:
        sample['y'] = torch.tensor(np.random.random(x_len).astype(np.float32))
        sample['y_mask'] = torch.tensor(np.random.randint(0, 2, x_len).astype(np.bool_))
    return sample


def dict2dgl(data_x, neighbor_val=6, edge_limit=2000, max_dist=15):
    dist_map = torch.cdist(data_x['position'], data_x['position']).squeeze()
    cmap = dist_map <= max_dist
    if torch.sum(cmap) > edge_limit:
        dist_cutoff = torch.topk(torch.flatten(dist_map), edge_limit,
                                 largest=False)[0][-1].item()
        cmap = dist_map <= dist_cutoff
    el = torch.where(cmap)
    # ew = torch.abs(el[0] - el[1]) > neighbor_val
    u = el[0]
    v = el[1]
    # u = el[0][ew]
    # v = el[1][ew]
    g = dgl.graph((u, v), num_nodes=dist_map.shape[0])
    g.ndata['x'] = data_x['position'].squeeze()
    g.ndata['f'] = torch.cat((data_x['one_hot'].squeeze(),
                              data_x['features'].squeeze()), 1).type_as(data_x['position'])
    g.ndata['f'] = torch.reshape(g.ndata['f'], (g.ndata['f'].shape[0],
                                                g.ndata['f'].shape[1], 1))
    u, v = g.edges()
    ew = torch.abs(u - v) > neighbor_val
    ed = dist_map[u, v]
    g.edata['w'] = torch.stack((ed, ew)).type_as(data_x['position']).T
    if 'edge' in data_x.keys():
        g.edata['w'] = torch.cat((g.edata['w'], data_x['edge'].squeeze()[u, v, :]),
                                 dim=-1).type_as(data_x['position'])
    g.edata['d'] = g.ndata['x'][g.edges()[1], :] - g.ndata['x'][g.edges()[0], :]

    g.ndata['y'] = data_x['y'].squeeze()
    g.ndata['y_mask'] = data_x['y_mask'].squeeze()
    return g


class Random_dgl_Dataset(DGLDataset):

    def __init__(self, n, data_types='one_hot,position,features,lddt,edge',
                 neighbor_val=6, dist_cutoff=15, edge_dim=32,
                 min_len=30, max_len=50, edge_limit=2000):
        self.n = n
        self.data_types = data_types.split(',')
        self.dic_list = [generate_random_dict(self.data_types,
                                              edge_dim=edge_dim, min_len=min_len,
                                              max_len=max_len) for i in range(n)]
        self.dist_cutoff = dist_cutoff
        self.neighbor_val = neighbor_val
        self.edge_dim = edge_dim
        self.edge_limit = edge_limit

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        data_x = self.dic_list[idx]
        g = dict2dgl(data_x, self.neighbor_val, edge_limit=self.edge_limit)
        return g


class Random_dict_Dataset(Dataset):

    def __init__(self, n, data_types='one_hot,position,features,lddt,cad,sh,edge',
                 edge_dim=32, min_len=30, max_len=50, edge_limit=2000):
        self.n = n
        self.data_types = data_types.split(',')
        self.dic_list = [generate_random_dict(self.data_types, edge_dim=edge_dim,
                                              min_len=min_len,
                                              max_len=max_len) for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.dic_list[idx]


def task_loss(pred, target, use_mean=True):
    l1_loss = torch.sum(torch.abs(pred - target))
    l2_loss = torch.sum((pred - target) ** 2)
    if use_mean:
        l1_loss /= pred.shape[0]
        l2_loss /= pred.shape[0]
    return l1_loss, l2_loss


def task_corr(pred, target):
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr
