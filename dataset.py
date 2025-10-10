import torch
from torch_geometric.data import Data, Dataset
import random
import numpy as np
from tqdm import tqdm
from array import array
import ROOT
from ROOT import TFile, TTree, vector


class TpcGraphDataset:
    def __init__(self, file_name, nevt=-1, 
                 outlier_distance=-1, 
                 additional_info=False,
                 device=torch.device('cuda')):
        f = TFile(file_name)
        t = f.Get('TpcTrack')
        coord_x = vector['double'](0)
        coord_y = vector['double'](0)
        coord_z = vector['double'](0)
        grid_coord_x = vector['int'](0)
        grid_coord_y = vector['int'](0)
        coord_x_truth = vector['double'](0)
        coord_y_truth = vector['double'](0)
        coord_z_truth = vector['double'](0)
        delta = vector['double'](0)
        length = vector['double'](0)
        drift = vector['double'](0)
        Q = vector['double'](0)
        T = vector['double'](0)
        label = vector['int'](0)
        label_trk = array('i', [-1])
        t.SetBranchAddress('coord_x', coord_x)
        t.SetBranchAddress('coord_y', coord_y)
        t.SetBranchAddress('coord_z', coord_z)
        t.SetBranchAddress('grid_coord_x', grid_coord_x)
        t.SetBranchAddress('grid_coord_y', grid_coord_y)
        t.SetBranchAddress('coord_x_truth', coord_x_truth)
        t.SetBranchAddress('coord_y_truth', coord_y_truth)
        t.SetBranchAddress('coord_z_truth', coord_z_truth)
        t.SetBranchAddress('delta', delta)
        t.SetBranchAddress('length', length)
        t.SetBranchAddress('drift', drift)
        t.SetBranchAddress('Q', Q)
        t.SetBranchAddress('T', T)
        t.SetBranchAddress('ncls', label)
        t.SetBranchAddress('ncls_trk', label_trk)

        self.data = []

        ### Read data
        ntot = nevt if nevt > 0 else t.GetEntries()
        for i in tqdm(range(ntot)):
            t.GetEntry(i)

            feat = []
            pos = []
            len = []
            node_label = []
            pos_grid = []
            chg = []
            for j in range(coord_x.size()):
                if outlier_distance > 0. and delta[j] > outlier_distance: continue

                _x = coord_x[j] / 180.
                _y = coord_y[j] / 180.
                _z = coord_z[j] / 290.
                _q = np.log(Q[j])
                _t = T[j] / 290.

                feat.append((_q, _t))
                pos.append((_x, _y, _z))
                len.append(length[j])
                if additional_info:
                    pos_grid.append((grid_coord_x[j], grid_coord_y[j]))
                    chg.append(Q[j])

                _node_label = label[j] if label[j] <= 1 else 1
                node_label.append(_node_label)

            x = torch.tensor(feat, dtype=torch.float, device=device)
            y = torch.tensor(node_label, dtype=torch.long, device=device)
            pos = torch.tensor(pos, dtype=torch.float, device=device) # pos = [x, y, z, idx]
            len = torch.tensor(len, dtype=torch.float, device=device)
            if additional_info:
                pos_grid = torch.tensor(pos_grid, dtype=torch.long, device=device) # pos_grid = [x, y]
                chg = torch.tensor(chg, dtype=torch.float, device=device)
                self.data.append(Data(x=x, y=y, pos=pos, len=len, pos_grid=pos_grid, charge=chg))
            else:
                self.data.append(Data(x=x, y=y, pos=pos, len=len))

         # Calculate class weights
        _, counts = np.unique(node_label, return_counts=True)
        weights = 1.0 / counts
        self.class_weights = torch.tensor(weights / np.sum(weights), dtype=torch.float, device=device)  # Normalize weights
        print('class_weights = ', self.class_weights)
   


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_class_weights(self):
        return self.class_weights
    
