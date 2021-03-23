import time
import torch

import torch.nn as nn
import pytorch_lightning as pl

from equivariant_attention.fibers import Fiber
from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r
from data import dict2dgl, task_loss, task_corr


class TFN_Scorer(pl.LightningModule):
    def __init__(self, atom_feature_size=23, order=4, non_linearity='elu', dropout=0.1, num_layers=7, num_channels=16,
                 edge_dim=2, out_dim=1, num_nlayers=0, edge_limit=2000, max_dist=15,
                 neighbor_val=3, step=1, time_limit=0, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_nlayers = num_nlayers
        self.num_channels = num_channels
        self.num_degrees = order
        self.time_start = time.time()
        self.time_limit = time_limit
        self.num_channels_out = num_channels * order
        self.edge_dim = edge_dim
        self.edge_limit = edge_limit
        self.neighbor_val = neighbor_val
        self.max_dist = max_dist
        self.step = step

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(order, self.num_channels),
                       'out': Fiber(1, self.num_channels_out)}

        self.block0 = self._build_gcn(self.fibers)
        self.fc1 = nn.Linear(self.num_channels_out, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_dim)
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.functional.sigmoid
        elif non_linearity == 'tahn':
            self.non_linearity = nn.functional.tanh
        elif non_linearity == 'none':
            self.non_linearity = None

        self.fc_drp = nn.Dropout(dropout)

    def _build_gcn(self, fibers):

        block0 = []
        fin = fibers['in']
        for i in range(self.num_layers - 1):
            block0.append(GConvSE3(fin, fibers['mid'], self_interaction=True, edge_dim=self.edge_dim))
            block0.append(GNormSE3(fibers['mid'], num_layers=self.num_nlayers))
            fin = fibers['mid']
        block0.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))
        return nn.ModuleList(block0)

    def network_step(self, g):
        G = dict2dgl(g, self.neighbor_val, self.edge_limit, self.max_dist)
        basis, r = get_basis_and_r(G, self.num_degrees - 1)
        h = {'0': G.ndata['f']}
        score_mask = G.ndata['y_mask']
        for layer in self.block0:
            h = layer(h, G=G, r=r, basis=basis)

        x = h['0'][..., -1]
        x = self.fc_drp(self.non_linearity(self.fc1(x)))
        x = self.fc_drp(self.non_linearity(self.fc2(x)))
        x = self.non_linearity(self.fc3(x))
        x = torch.mul(x, score_mask.unsqueeze(1))
        return x

    def forward(self, x):
        y_hat = self.network_step(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x = batch
        pred = self.network_step(x).squeeze()
        y = x['y'].squeeze()
        x_mask_np = x['y_mask'].squeeze()
        loss, _ = task_loss(pred[x_mask_np], y[x_mask_np])
        corr = task_corr(pred[x_mask_np], y[x_mask_np])
        batch_dictionary = {'loss': loss,
                            'corr': corr
                            }
        self.log('train_loss', loss, on_epoch=True, sync_dist=True, on_step=False)
        self.log('train_corr', corr, on_epoch=True, sync_dist=True, on_step=False)
        return batch_dictionary

    def validation_step(self, batch, batch_idx):
        x = batch
        pred = self.network_step(x).squeeze()
        y = x['y'].squeeze()
        x_mask_np = x['y_mask'].squeeze()
        loss, _ = task_loss(pred[x_mask_np], y[x_mask_np])
        corr = task_corr(pred[x_mask_np], y[x_mask_np])
        batch_dictionary = {'val_loss': loss,
                            'val_corr': corr
                            }
        self.log('val_loss', loss, on_epoch=True, sync_dist=True, on_step=False)
        self.log('val_corr', corr, on_epoch=True, sync_dist=True, on_step=False)
        return batch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_epoch_end(self):
        time_elapsed = time.time() - self.time_start
        time_left = self.time_limit - time_elapsed
        avg_epoch_time = time_elapsed / (self.current_epoch + 1)
        if self.time_limit > 0 and time_left < avg_epoch_time * 1.5:
            print('At epoch: {}'.format(self.current_epoch))
            print('Time left: {}s less than 1.5 times of average epoch time: {}s, exiting...'.format(time_left,
                                                                                                     avg_epoch_time))
            exit()