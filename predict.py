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

seed=42
random = Random(seed) 
import numpy as np

# sample = transform(sample)               
# logger.info("Done")

"""
1. How to Undersample for inference 
    1.1 Training used 10,000 Boundary and 4192 for Junction Detection
"""
from dataproc import transforms as Transforms
import torchvision
num_points = 10000
num_points = 4192
class ResamplerForScan(Transforms.Resampler):        
    def __call__(self, sample):
        # try:
        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])
            
       
        sampledIDx = self._resample(np.arange(sample['scan_pts'].shape[0]), self.num, False)        
        sample['scan_pts'] = sample['scan_pts'][sampledIDx]
        return sample
    


def main(args):
    print(torch.cuda.is_available())
    dict_args = vars(args)
    dict_args = {k: v for k, v in dict_args.items() if v is not None}
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(args.log_dir, 'test'), 
                                             name=args.experiment if args.experiment \
                                                else args.model_name,)
    logger = configure_logger(tb_logger.log_dir)    
    transform = torchvision.transforms.Compose([
        ResamplerForScan(num_points, upsampling=True),
        Transforms.Normalizer(),
        Transforms.SetDeterministic(),
        ])


    scan_file = "/home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D/Scan/train/User Library-8mm motor.ply"
    scan = o3d.io.read_triangle_mesh(scan_file)
    sample = {'scan_pts': np.asarray(scan.vertices).reshape((-1, 3)), }

    bndry_labels_file = "/home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D/Bndry/train/User Library-8mm motor.npz"
    brepBndryAnnot = np.load(bndry_labels_file, allow_pickle=True)       
    import pdb; pdb.set_trace()

    bndry_vIdxs = np.array(brepBndryAnnot["v_vertex_Ids"], dtype=np.int32)
    assert np.max(bndry_vIdxs) < len(sample['scan_pts'])
    vIdxs = np.zeros(shape=len(scan.vertices), dtype=np.int32)
    vIdxs[bndry_vIdxs] = 1

    sample = transform(sample)    
    vIdxs = vIdxs[:len(sample["scan_pts"])]           
    
    # NOTE:: Load model and Data set
    if not args.checkpoint_path:
        logger.warning('No checkpoint given!')
    model = load_model(args.model_name, args.checkpoint_path, dict_args)

    
    model.eval()
    with torch.no_grad():                
        # Add simple sample into a batch 
        sample["scan_pts"] = sample["scan_pts"].reshape(1,-1,3)
        preds = model.predict_step(sample, None,None)[0]


    from torch import nn
    SfMx = nn.Sigmoid()
    preds = (SfMx(preds.detach())[:, :, :1] > 0.55).int().flatten().cpu().numpy()
    acc = (preds==vIdxs).sum()
    
    print (f"Accuracy: {acc}...")
    print (classification_report(vIdxs, preds))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_parser_arguments(parser, require_checkpoint=True)
    args = parser.parse_args()
    print (args)    
    main(args)
    pass



#  python predict.py --checkpoint_path ../BRepDetNet_CheckPoints/CC3D/BRepEd/train/version_5/checkpoint/last.ckpt --model_name='BRepEd'
