import os
import torch
import torch.distributed
import logging
import dataclasses
import open3d as o3d

from pytorch_lightning import Trainer, plugins
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from typing import Dict, Any, List, Optional, Union
from argparse import ArgumentParser
from util.callbacks_loggers import BndaryJncDet_ClbLogger, JncDet_ClbLogger
from util.lightning import add_parser_arguments, configure_logger
from model import load_model
from dataproc import load_data_module
from torch.utils.data import Dataset
from random import Random
import logging
logger = logging.getLogger(__name__)
from sklearn.metrics import classification_report
import torchvision

seed=42
random = Random(seed) 
import numpy as np
from torch import nn

from dataproc import transforms as Transforms


class ResamplerForScan(Transforms.Resampler):        
    def __call__(self, sample):
        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])
            
        sampledIDx = self._resample(np.arange(sample['scan_pts'].shape[0]), self.num, False)        
        sample['scan_pts'] = sample['scan_pts'][sampledIDx]
        sample['sampledIDx'] = sampledIDx
        return sample
    

def main(model_name, checkpoint_path, scan_file=None, scan_np_array=None):
    assert not (scan_file is None and  scan_np_array is None)

    num_points = 10000 if model_name == "BRepEd" else 4192
    transform = torchvision.transforms.Compose([
        ResamplerForScan(num_points, upsampling=True),
        Transforms.Normalizer(),
        Transforms.SetDeterministic(),
        ])
    
    if scan_file:
        scan = o3d.io.read_triangle_mesh(scan_file)    
        n = len(scan.vertices)
        sample = {'scan_pts': np.asarray(scan.vertices).reshape((-1, 3)), }
    else:
        n = len(scan_np_array)
        sample = {'scan_pts': scan_np_array.reshape((-1, 3)), }

    sample = transform(sample)    
    kwarsgs = {
        "checkpoint_path": checkpoint_path,
    }
    model = load_model(model_name, checkpoint_path, kwarsgs)

    
    model.eval()
    with torch.no_grad():                
        # Add simple sample into a batch 
        sample["scan_pts"] = sample["scan_pts"].reshape(1,-1,3)
        preds = model.predict_step(sample, None,None)[0]

    SfMx = nn.Sigmoid()
    preds = (SfMx(preds.detach())[:, :, :1] > 0.55).int().flatten().cpu().numpy()
    sampledIDx = sample["sampledIDx"]
    all_preds = np.zeros(n, dtype=int) + -1
    all_preds[sampledIDx] = preds

    with open('output.npy', 'wb') as fp:
        np.save(fp, all_preds)        
        print ("Saved to output.npy...")

if __name__ == '__main__':  
    model_name = "BRepEd" # One of [BRepEd, BRepJd]
    checkpoint_path = "../BRepDetNet_CheckPoints/CC3D/BRepEd/train/version_5/checkpoint/last.ckpt"
    scan_file = "/home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D/Scan/User Library-8mm motor.ply"
    
    # main(model_name, checkpoint_path, scan_file)
    main(model_name, checkpoint_path, scan_np_array=np.random.randn(10000,3))