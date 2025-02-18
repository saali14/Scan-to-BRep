import torch
import logging
import torchvision
import numpy as np

from argparse import ArgumentParser
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataproc.CC3DDataSet import CC3DDataSet
from dataproc import transforms as Transforms

class CC3DDataModule(LightningDataModule):
    def __init__(self,
                 data_root_dir="/data/3d_cluster/scan2brep",
                 data_scan_dir="/data/3d_cluster/scan2brep/cc3d_v1.0_scans",
                 data_annot_bndry_dir="/data/3d_cluster/scan2brep/cc3d_v1.0_BoundaryLabels",
                 data_annot_jnc_dir="/data/3d_cluster/scan2brep/cc3d_v1.0_JunctionLabels",
                 train_batch_size=4,
                 eval_batch_size=4,
                 sampling_type='downsample',
                 num_points=1024,
                 shuffle_train=False,
                 **kwargs):
        super().__init__()
        print(data_root_dir)
        self.root_dir = data_root_dir
        self.scan_dir = data_scan_dir
        self.annotBndry_dir = data_annot_bndry_dir
        self.annotJnc_dir = data_annot_jnc_dir

        self.train_batch_size = train_batch_size
        self.test_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.sampling_type = sampling_type
        self.num_points = num_points
        self.shuffle_train = shuffle_train
        self.kwargs = kwargs
        self._logger = logging.getLogger('core.data')

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--data_root_dir', type=str, metavar='PATH', help='path to the processed dataset')  
        parser.add_argument('--data_scan_dir', type=str, metavar='PATH', help='path to the processed dataset') 
        parser.add_argument('--data_annot_bndry_dir', type=str, metavar='PATH', help='path to the processed dataset')
        parser.add_argument('--data_annot_jnc_dir', type=str, metavar='PATH', help='path to the processed dataset') 
    
        parser.add_argument('--num_points', type=int, metavar='N', help='points in point-cloud (default: 1024)')
        parser.add_argument('--sampling_type', choices=['clean', 'downsample', 'sparse', 'bjsample', 'facesample'],)
        parser.add_argument('--train_batch_size', type=int, metavar='N', help='train batch size (default: 4)')
        parser.add_argument('--eval_batch_size', type=int, metavar='N', help='validation/test batch size (default: 4)')
        return parser

    # NOTE: If I change samples noise-type "under data disturbances" on the fly for the 3D Scans of CAD models, 
    # Then load and change here 
    def get_transforms(self):
        if self.sampling_type is None or self.sampling_type == "clean":
            train_transforms = [Transforms.Resampler(self.num_points, upsampling=True)]
            test_transforms = [Transforms.Resampler(self.num_points, upsampling=True),
                               Transforms.SetDeterministic(), 
                               Transforms.ShufflePoints() # FIXME <-- Remember to make a Sanity Check Here
                               ]
        elif self.sampling_type == "downsample":
            train_transforms = [Transforms.Resampler(self.num_points, upsampling=True), 
                                Transforms.Normalizer(),
                                #Transforms.AddBatchDimension(),
                                ]
            test_transforms = [Transforms.Resampler(self.num_points, upsampling=True),
                               Transforms.Normalizer(),
                               Transforms.SetDeterministic(), 
                               #Transforms.AddBatchDimension(), # FIXME <-- Remember to make a Sanity Check Here
                               ]
            eval_transforms = [Transforms.Resampler(self.num_points, upsampling=True),
                               Transforms.Normalizer(),
                               Transforms.SetDeterministic(), 
                               #Transforms.AddBatchDimension(), # FIXME <-- Remember to make a Sanity Check Here
                               ]
        elif self.sampling_type == "bjsample":
            train_transforms = [Transforms.ResamplerBoundary(self.num_points, upsampling=True), 
                                Transforms.Normalizer(),
                                #Transforms.AddBatchDimension(),
                                ]
            test_transforms = [Transforms.ResamplerBoundary(self.num_points, upsampling=True),
                               Transforms.Normalizer(),
                               Transforms.SetDeterministic(), 
                               #Transforms.AddBatchDimension(), # FIXME <-- Remember to make a Sanity Check Here
                               ]
            eval_transforms = [Transforms.ResamplerBoundary(self.num_points, upsampling=True),
                               Transforms.Normalizer(),
                               Transforms.SetDeterministic(), 
                               #Transforms.AddBatchDimension(), # FIXME <-- Remember to make a Sanity Check Here
                               ]
        else:
            raise Exception("Not Implemented Error in File CC3DDataModule.py")

        train_transforms = torchvision.transforms.Compose(train_transforms)
        test_transforms = torchvision.transforms.Compose(test_transforms)
        eval_transforms = torchvision.transforms.Compose(eval_transforms)
        return train_transforms, test_transforms, eval_transforms

    def prepare_data(self):
        # do only on one process, e.g. download dataset
        pass

    
    def setup(self, stage: Optional[str] = None):
        # do for every gpu
        self.train_transform, self.test_transform,self.eval_transforms = self.get_transforms()

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_set = CC3DDataSet(self.root_dir, self.scan_dir, self.annotBndry_dir, self.annotJnc_dir, subset='train', transform=self.train_transform, **self.kwargs)
            self.test_set = CC3DDataSet(self.root_dir, self.scan_dir, self.annotBndry_dir, self.annotJnc_dir, self.annotFace_dir, subset='val', transform=self.test_transform, **self.kwargs)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_set = CC3DDataSet(self.root_dir, self.scan_dir, self.annotBndry_dir, self.annotJnc_dir, subset='test', transform=self.test_transform, **self.kwargs)
            self.sampleIds = self.test_set.samples

        # Apply no transformations when evaluating dataset
        if stage == 'predict' or stage is None:
            self.eval_set = CC3DDataSet(self.root_dir, self.scan_dir,self.annotBndry_dir, self.annotJnc_dir, subset='test',transform=self.eval_transforms, **self.kwargs)
            self.sampleIds = self.eval_set.samples
            
        self.collate_fn = None

        if self.sampling_type == 'sparse':
            from torchsparse.utils.collate import sparse_collate_fn
            self.collate_fn = sparse_collate_fn

    def train_dataloader(self):
        self._logger.info('Loading training Set .{}'.format(len(self.train_set)))
        return DataLoader(self.train_set, batch_size=self.train_batch_size, num_workers=8, pin_memory=True, shuffle=self.shuffle_train, collate_fn=self.collate_fn, drop_last=(True if self.train_batch_size is not None else False))

    def eval_dataloader(self):
        self._logger.info('Loading validation Set .{}'.format(len(self.eval_set)))
        return DataLoader(self.eval_set, batch_size=self.eval_batch_size, num_workers=8, pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        self._logger.info('Loading testing Set .{}'.format(len(self.test_set)))
        return DataLoader(self.test_set, batch_size=self.test_batch_size, num_workers=8, pin_memory=True, collate_fn=self.collate_fn, drop_last=(True if self.test_batch_size is not None else False))

