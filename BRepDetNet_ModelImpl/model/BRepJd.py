import copy
import math
import os

import scipy.sparse # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser
from pytorch_lightning.core.module import LightningModule  # type: ignore
from torch.optim.lr_scheduler import MultiStepLR
from model.backbones import DGCNN, PointNet, PointNet2
from loss.focal_loss import FocalLossJnc
from loss.total_loss import TotalLoss, TotalAccuracy
from util.file_helper import create_dir
from util.pointcloud import nested_list_to_tensor
from util.timer import Timer
from util.metrics import PrecisionRecallMetricsJnc, DetLossAcccMetricsJnc
from util.visualization import localNMS
from dataproc import transforms as Transforms

class BRepJd(LightningModule):
    def __init__(self, 
                 lr=1e-3,
                 lr_steps=(15, 30, 45),
                 emb_dims=128,
                 emb_nn='dgcnn',
                 pointer='transformer',
                 n_blocks=1,
                 dropout=0.0,
                 ff_dims=1024,
                 n_heads=4,
                 det_loss='focal',
                 device='cuda:0',
                 train_only_descriptor=False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.lr_steps = lr_steps
        self.emb_dims = emb_dims
        self.devc = device
        self.train_only_descriptor = train_only_descriptor
        self.forward_time = Timer()
        self.emb_nn_type = emb_nn
        self.dropout = dropout
        if emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif emb_nn == 'pointnet++':
            self.emb_nn = PointNet2(emb_dims=self.emb_dims)
        elif emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(emb_dims=self.emb_dims, dropout=self.dropout)
        else:
            raise Exception('Not implemented')
        self.emb_nn = self.emb_nn.to(self.devc)

        loss_weights = {'jnc_focal_loss':1.0,}
        accuracy_weights = {'jnc_accuracy': 1.0}
        self.detection_loss = None
        if det_loss == 'focal':
            self.detection_loss_type = 'focal'
            self.detection_loss = FocalLossJnc()
        elif det_loss == 'adaptive':
            raise NotImplementedError 
        else:
            raise NotImplementedError
        
        
        self.test_lossAcc = DetLossAcccMetricsJnc(prefix='test')
        self.test_metrics = PrecisionRecallMetricsJnc(num_threshold = 20, prefix='test')
        self.loss = TotalLoss(weights=loss_weights)
        self.accuracy = TotalAccuracy(weights=accuracy_weights)   
        self.ll1 = nn.Conv1d(self.emb_dims, 1, 1, stride=1, padding=0, device=self.devc)
        self.ll2 = nn.Conv1d(self.emb_dims, 1, 1, stride=1, padding=0, device=self.devc)

        self.ll1 = self.ll1.to(self.devc)
        self.ll2 = self.ll2.to(self.devc)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--emb_nn', type=str, metavar='N', choices=['pointnet++', 'dgcnn', 'pointnet', 'sparse'], help='Embedding nn to use, [pointnet, dgcnn, pointnet++]')
        parser.add_argument('--pointer', type=str, metavar='N', choices=['identity', 'transformer'], help='Attention-based pointer gen')
        parser.add_argument('--emb_dims', type=int, metavar='N', help='Dimension of embeddings')
        parser.add_argument('--n_blocks', type=int, metavar='N', help='Num of blocks of encoder&decoder')
        parser.add_argument('--n_heads', type=int, metavar='N', help='Num of heads in multiheadedattention')
        parser.add_argument('--ff_dims', type=int, metavar='N', help='Num of dimensions of fc in transformer')
        parser.add_argument('--dropout', type=float, metavar='N', help='Dropout ratio in embedding')
        parser.add_argument('--device', type=str, choices=['cuda:0'], help='name of the compute device tool, e.g., cuda:0')
        parser.add_argument('--det_loss', type=str, choices=['focal', 'adaptive', 'mse??']) # TODO <-- need other Losses?
        parser.add_argument('--lr', type=float, help='Learning rate during training')
        parser.add_argument('--lr_steps', type=int, nargs='+', help='Steps to decrease lr')        
        parser.add_argument('--train_only_Junction', action='store_true', help='Predict Boundary-to-Junction.')
        parser.add_argument('--train_only_Boundary', action='store_true', help='Predict Scan-to-Boundary.')

        return parser

    @staticmethod
    def get_default_batch_transform(num_points=1024, voxel=False, **kwargs):
        return [ Transforms.ResamplerBoundary(num_points, upsampling=True),
                 Transforms.SetDeterministic(),]
    
    def forward(self, scan, neg=None):
        info = dict()
        scan_embedding1 = self.emb_nn(scan)
        scan_embedding2 = self.emb_nn(scan)
        jnc_embedding = self.ll1(scan_embedding1).transpose(2,1)
        posOffset_embedding = self.ll2(scan_embedding2).transpose(2,1)
        
        info['embedding'] = scan_embedding1
        info['transport'] = None  
        info['masses'] = None 
        return jnc_embedding, posOffset_embedding, info

    def training_step(self, batch, _):
        losses_accr = {}
        scan_pts = torch.tensor(batch['scan_pts'][:, :, :3], dtype=torch.float32, requires_grad=True).transpose(2, 1)
        gt_labels_jnc = torch.tensor(batch['BRepAnnot_jIds'][:, :, :1], dtype=torch.float32)
        pred_labels_jnc, _, _ = self(scan_pts, neg=None)
                
        if self.detection_loss_type == 'focal':
            detLoss, detAcc = self.detection_loss(pred_labels_jnc, gt_labels_jnc, prefix='train') 
        
        losses_accr.update(detLoss)
        losses_accr.update(detAcc)
        losses_accr.update(self.accuracy(detAcc, prefix='train'))
        losses_accr.update(self.loss(detLoss, prefix='train'))
        self.log_dict(losses_accr)
        return losses_accr['train_total_loss']
    
    def validation_step(self, batch, batch_idx):
        #losses_accr = {}
        lossAcc_Pred = {}
        
        scan_pts = torch.tensor(batch['scan_pts'][:, :, :3], dtype=torch.float32, requires_grad=True).transpose(2, 1)
        gt_labels_jnc = torch.tensor(batch['BRepAnnot_jIds'][:, :, :1], dtype=torch.float32)
        pred_labels_jnc, _, _ = self(scan_pts, neg=None)
        
        if self.detection_loss_type == 'focal' or self.detection_loss_type == 'adaptive':
            detLoss, detAcc = self.detection_loss(pred_labels_jnc, gt_labels_jnc, prefix='val') 
            
        lossAcc_Pred.update(detLoss)
        lossAcc_Pred.update(detAcc)
        lossAcc_Pred.update(self.loss(detLoss, prefix='val'))
        lossAcc_Pred.update(self.accuracy(detAcc, prefix='val'))        
        
        self.log_dict(lossAcc_Pred)#, on_epoch=True, sync_dist=True)
        
        return self.log_dict 

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        scan_pts = torch.tensor(batch['scan_pts'][:, :, :3], dtype=torch.float32, requires_grad=True).transpose(2, 1)
        self.forward_time.tic()
        pred_labels_jnc, pred_labels_posOffset, info = self(scan_pts)
        self.forward_time.toc()        
        self.test_metrics.update(torch.sigmoid(pred_labels_jnc.detach()), batch['BRepAnnot_jIds'][:, :, :1])
        
        output = {
            'dataloader_idx': dataloader_idx,
            'scan_pts' : scan_pts.transpose(2, 1),
            'pred_labels_jnc': torch.sigmoid(pred_labels_jnc), 
        }
        return output

    def predict_step(self, batch, batch_idx, dataloader_idx = None):
        scan_pts = torch.tensor(batch['scan_pts'][:, :, :3], dtype=torch.float32, requires_grad=True).transpose(2, 1)
        scan_pts = scan_pts.to(self.devc)
        return self(scan_pts)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.lr_steps, gamma=0.1)
        return [optimizer], [scheduler]

