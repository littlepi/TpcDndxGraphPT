"""
Author: Guang Zhao
Email: zhaog@ihep.ac.cn
Created: 2026-01-16

Description: Reconstruction code for GraphPT
"""

import torch
from dataset import *
from model import *
from tqdm import tqdm
from array import array
import numpy as np
from ROOT import TFile, TTree, vector

def trunc_mean(hits_pos, length, trunc=0.65, layer=2):
    sorted_idx = np.argsort(hits_pos)

    dndx_hits = []
    dndx_used_idx = []
    last_layer_id = hits_pos[sorted_idx[0]]
    ncls = 0
    for idx in sorted_idx:
        layer_id = hits_pos[idx]
        if layer_id - last_layer_id >= layer:
            skim_layer = 0
            for i in range(last_layer_id+layer, layer_id, layer):
                skim_layer += 1
            dndx_hits.append((ncls, list(dndx_used_idx)))
            last_layer_id += layer * (skim_layer+1)
            ncls = 0
            dndx_used_idx.clear()
        ncls += 1
        dndx_used_idx.append(idx)
    dndx_hits.append((ncls, list(dndx_used_idx)))

    dndx_hits.sort(key=lambda x: x[0])
    ndndx_hits = int(len(dndx_hits)*trunc)
    dndx = 0.
    dndx_hit_flags = vector['int'](len(hits_pos), 0)
    for i in range(ndndx_hits):
        dndx += dndx_hits[i][0]
        for idx in dndx_hits[i][1]:
            dndx_hit_flags[int(idx)] = 1
    dndx /= length

    return dndx, dndx_hit_flags

def trunc_mean_charge(hits_pos, hits_charge, length, trunc=0.6, layer=1):
    sorted_idx = np.argsort(hits_pos)

    dedx_hits = []
    last_layer_id = hits_pos[sorted_idx[0]]
    etot = 0
    for idx in sorted_idx:
        layer_id = hits_pos[idx]
        if layer_id - last_layer_id >= layer:
            skim_layer = 0
            for i in range(last_layer_id+layer, layer_id, layer):
                skim_layer += 1
            dedx_hits.append(etot)
            last_layer_id += layer * (skim_layer+1)
            etot = 0
        etot += hits_charge[idx]
    dedx_hits.append(etot)

    dedx_hits.sort()
    ndedx_hits = int(len(dedx_hits)*trunc)
    dedx = 0.
    for i in range(ndedx_hits):
        dedx += dedx_hits[i]
    dedx /= length

    return dedx

def rec(model, input, output, args):
    model = model.to(args.device)
    dataset = TpcGraphDataset(input, device=args.device, nevt=args.nevt_rec, 
                              outlier_distance=args.outlier_distance, 
                              additional_info=True)
    
    outFile = TFile(output, 'recreate')
    outTree = TTree('TpcTrack', 'TpcTrack')
    dndx_rec = array('i', [-1])
    dndx_hit_rec = vector['double'](0)
    dndx_hit_truth = vector['int'](0)
    dndx_rec_trad = array('d', [-999.])
    dndx_hit_rec_trad = vector['int'](0)
    dedx_rec_trad = array('d', [-999.])
    track_length = array('d', [-999.])
    outTree.Branch('dndx_rec', dndx_rec, 'dndx_rec/I')
    outTree.Branch('dndx_hit_rec', dndx_hit_rec)
    outTree.Branch('dndx_hit_truth', dndx_hit_truth)
    outTree.Branch('track_length', track_length, 'track_length/D')
    if args.enable_trad_rec:
        outTree.Branch('dndx_rec_trad', dndx_rec_trad, 'dndx_rec_trad/D')
        outTree.Branch('dndx_hit_rec_trad', dndx_hit_rec_trad)
        outTree.Branch('dedx_rec_trad', dedx_rec_trad, 'dedx_rec_trad/D')

    model.eval()
    for i in tqdm(range(args.nevt_rec)):
        track_length[0] = np.max(dataset[i].len.numpy())
        if args.enable_trad_rec:
            pos_idx = 0 if np.max(dataset[i].pos_grid[:, 0].numpy()) - np.min(dataset[i].pos_grid[:, 0].numpy()) > np.max(dataset[i].pos_grid[:, 1].numpy()) - np.min(dataset[i].pos_grid[:, 1].numpy()) else 1
            _dndx_trad, _dndx_trad_layers = trunc_mean(dataset[i].pos_grid[:, pos_idx], track_length[0], trunc=args.dndx_trunc_ratio, layer=args.dndx_trunc_layers)
            dndx_rec_trad[0] = _dndx_trad

            _dedx_trad = trunc_mean_charge(dataset[i].pos_grid[:, pos_idx], dataset[i].charge.numpy(), track_length[0], trunc=args.dedx_trunc_ratio, layer=args.dedx_trunc_layers)
            dedx_rec_trad[0] = _dedx_trad

            dndx_hit_rec_trad.clear()
            for values in _dndx_trad_layers:
                dndx_hit_rec_trad.push_back(values)

        dndx_rec[0] = 0
        dndx_hit_rec.clear()
        dndx_hit_truth.clear()
        with torch.no_grad():
            out = model(dataset[i])
            # pred = None
            for prob in out:
                prob_hit = prob[1].item()
                dndx_hit_rec.push_back(prob_hit)
                if prob_hit > args.prob_cut:
                    dndx_rec[0] += 1
            for y in dataset[i].y:
                dndx_hit_truth.push_back(y.item())

        outTree.Fill()
    outFile.WriteTObject(outTree)


    return

def main(args):
    model = None
    if not args.rec_with_checkpoint:
        model = torch.load(args.model, weights_only=False, map_location=torch.device(args.device))
    else:
        model = GraphPointTransformer(in_channels=2, 
                                      out_channels=2, 
                                      debug=args.enable_debug, 
                                      dim_model=args.model_layers, 
                                      down_ratio=args.model_down_ratio, k=args.knn, 
                                      undirected=args.undirected, 
                                      reduce=args.reduce, 
                                      self_attention=args.enable_self_attention)
        checkpoint = torch.load(args.load_checkpoint, map_location=torch.device(args.device))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    rec(model, args.dataset_rec, './data/rec_{}.root'.format(args.tag), args)
