import os
import math
import numpy as np
import logging
import torch

from torch import nn
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from util.file_helper import create_dir
from util.timer import Timer
from util.visualization import get_colored_pcd, get_colored_pcd2, apply_colormap2scalar, localNMS


def ensure_dirs(dirs):
    try:
        os.makedirs(dirs, exist_ok = True)
    except Exception as error:
        print(f"Directory {dirs} can not be created. Error {error}")

class BndaryJncDet_ClbLogger(Callback):
    def __init__(self, log_path, dataset="CC3D", save_prediction=True, withNMS=False):
        self.logger = _configure_logger(log_path, 'test', format='%(message)s')
        self.logger.propagate = False
        self.log_path = log_path
        self.log_predictions = False
        self.withNMS=withNMS
        self.dataset = dataset
        self.save_prediction = save_prediction
        self.spts = {}
        self.gtbndry = {}
        self.gtjnc = {}
        self.predbndry = {}
        self.predjnc = {}
        self.evidence = {}
        self.sampleIdxName = {}
        self.nms_bndry_acc = 0.0
        self.nms_jnc_acc = 0.0
        self.total = 0

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        self.samples = getattr(trainer.datamodule, 'sampleIds', None)
        self.totalBatches = trainer.num_test_batches[0]
        if self.samples is not None:
            self.log_predictions = True
            for i in range(0, self.totalBatches):
                self.spts[i] = []
                self.gtbndry[i] = []
                self.gtjnc[i] = []
                self.predbndry[i] = []
                self.predjnc[i] = []
                self.sampleIdxName[i]=[]

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.log_predictions:
            self.spts[batch_idx].extend(torch.unbind(batch['scan_pts'][:,:,:3].cpu())) 
            self.gtbndry[batch_idx].extend(torch.unbind(batch['BRepAnnot_vIds'][:,:,:1].cpu()))
            self.gtjnc[batch_idx].extend(torch.unbind(batch['BRepAnnot_jIds'][:,:,:1].cpu()))
            #self.predbndry[batch_idx].extend(torch.unbind((outputs['pred_labels_bndry'][:, :, :1] > 0.56).int().cpu()))
            #self.predjnc[batch_idx].extend(torch.unbind((outputs['pred_labels_jnc'][:, :, :1] > 0.50).int().cpu()))
            self.predbndry[batch_idx].extend(torch.unbind((outputs['pred_labels_bndry']).cpu()))
            self.predjnc[batch_idx].extend(torch.unbind((outputs['pred_labels_jnc']).cpu()))
            self.sampleIdxName[batch_idx].extend(batch['idx_name'])
            #self.predbndry[batch_idx].extend(np.logical_and(torch.unbind((outputs['pred_labels_bndry'][:, :, :1] > #0.6).int().cpu()), torch.unbind((outputs['pred_labels_bndry'][:, :, :1] < 0.8).int().cpu())))
            #self.predjnc[batch_idx].extend(np.logical_and(torch.unbind((outputs['pred_labels_jnc'][:, :, :1] > 0.5).int().cpu()), torch.unbind((outputs['pred_labels_jnc'][:, :, :1] < 0.6).int().cpu())))
                              
    def on_test_end(self, trainer, pl_module):
        import open3d as o3d
        if self.log_predictions:
            self.testbatchSize = getattr(trainer.datamodule, 'test_batch_size', None)
            tempPCD = o3d.geometry.PointCloud()
            if self.save_prediction:
                for i in range(self.totalBatches):
                    scan_pts = torch.stack(self.spts[i]).numpy()
                    gtb = torch.stack(self.gtbndry[i]).numpy()
                    gtj = torch.stack(self.gtjnc[i]).numpy()
                    pb = torch.stack(self.predbndry[i]).numpy()
                    pj = torch.stack(self.predjnc[i]).numpy()
                    uid=self.sampleIdxName[i]

                    #self.logger.info({'test_sanity': pb})
                    for s in range(self.testbatchSize):
                        uid_=uid[s]
                        uid_root="/".join(uid_.split("/")[-3:-1])
                        uid_name=uid_.split("/")[-1].split(".")[0]

                        # Create the Directory
                        gt_bndry_dir=os.path.join(self.log_path,"bndry/gt",uid_root)
                        pred_bndry_dir=os.path.join(self.log_path,"bndry/pred",uid_root)
                        gt_jnc_dir=os.path.join(self.log_path,"jnc/gt",uid_root)
                        pred_jnc_dir=os.path.join(self.log_path,"jnc/pred",uid_root)
                        scan_dir=os.path.join(self.log_path,"scan",uid_root)

                        ensure_dirs(gt_bndry_dir)
                        ensure_dirs(pred_bndry_dir)
                        ensure_dirs(gt_jnc_dir)
                        ensure_dirs(pred_jnc_dir)
                        ensure_dirs(scan_dir)

                        # Save the files
                        np.save(os.path.join(gt_bndry_dir,uid_name+".npy"),gtb[s,:,:1].flatten())
                        np.save(os.path.join(pred_bndry_dir,uid_name+".npy"),pb[s,:,:1].flatten())
                        np.save(os.path.join(gt_jnc_dir,uid_name+".npy"),gtj[s,:,:1].flatten())
                        np.save(os.path.join(pred_jnc_dir,uid_name+".npy"),pj[s,:,:1].flatten())

                        o3d.io.write_point_cloud(os.path.join(scan_dir,uid_name+".ply"),
                                                 o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts[s,:,:3])))

                        # sID = i*self.testbatchSize + s
                        # fp_fn_b = np.logical_xor(pb[s,:,:1].flatten() , gtb[s,:,:1].flatten()).astype(int)
                        # predPcd = get_colored_pcd2(scan_pts[s, :, :3], pb[s,:,:1].flatten(), pj[s,:,:1].flatten(), fp_fn_b) 
                        # gtPcd =  get_colored_pcd(scan_pts[s, :, :3], gtb[s,:,:1].flatten(), gtj[s,:,:1].flatten()) 
                        # tempPCD = gtPcd + predPcd.translate(np.array([4.0, 0.0, 0.0]))
                        # o3d.io.write_point_cloud(os.path.join(self.log_path, 'mix/gt_pbj%s.ply' % sID), tempPCD)
                        # o3d.io.write_point_cloud(os.path.join(self.log_path, 'gt/gt%s.ply' % sID), gtPcd)
                        # o3d.io.write_point_cloud(os.path.join(self.log_path, 'pred/pred%s.ply' % sID), predPcd)
                        
                        if self.withNMS is True:
                            locMx_Idx_b = localNMS(scan_pts[s, :, :3], pb[s,:,:1].flatten(), dist_th=0.01)
                            locMx_Idx_j = localNMS(scan_pts[s, :, :3], pj[s,:,:1].flatten(), dist_th=0.01)
                                
                            mask_edge = gtb[s,:,:1].flatten()
                            neg_mask_edge = np.ones_like(mask_edge) - mask_edge
                            Np_edge = np.sum(mask_edge)
                            Ng_edge = np.sum(neg_mask_edge)
                            
                            pred_b = np.zeros_like(pb[s,:,:1].flatten(), dtype=int)
                            pred_b[locMx_Idx_b] = 1
                            
                            self.nms_bndry_acc += np.sum(np.equal(pred_b, gtb[s,:,:1].flatten()) *  mask_edge)  / np.sum(mask_edge)
                            
                            mask_jnc = gtj[s,:,:1].flatten()
                            neg_mask_jnc = np.ones_like(mask_jnc) - mask_jnc
                            Np_jnc = np.sum(mask_jnc)
                            Ng_jnc = np.sum(neg_mask_jnc)
                            
                            pred_j = np.zeros_like(pj[s,:,:1].flatten(), dtype=int)
                            pred_j[locMx_Idx_j] = 1.0
                            
                            self.nms_jnc_acc += np.sum(np.equal(pred_j, gtj[s,:,:1].flatten()) *  mask_jnc)  / np.sum(mask_edge)
                            self.total += 1
                            
                            fp_fn_b = np.logical_xor(pred_b , gtb[s,:,:1].flatten()).astype(int)
                            print("after NMS")
                            predPcd = get_colored_pcd2(scan_pts[s, :, :3], pred_b, pred_j, fp_fn_b) 
                            gtPcd =  get_colored_pcd(scan_pts[s, :, :3], gtb[s,:,:1].flatten(), gtj[s,:,:1].flatten()) 
                            
                            tempPCD = gtPcd + predPcd.translate(np.array([8.0, 0.0, 0.0]))
                            o3d.io.write_point_cloud(os.path.join(self.log_path, 'mix/gt_pbj_nms%s.ply' % sID), tempPCD)
                            #print("REACH HERE")
                
                if self.withNMS is True:
                    self.logger.info({'Stage': 'nms', 'Epoch': trainer.current_epoch, 'bndry_acc': self.nms_bndry_acc / self.total, 'jnc_acc': self.nms_jnc_acc / self.total, 'total_acc': 0.9 * (self.nms_bndry_acc / self.total) + 0.1*(self.nms_jnc_acc / self.total)})
                        
class JncDet_ClbLogger(Callback):
    def __init__(self, log_path, dataset="CC3D", save_prediction=True, withNMS=False):
        self.logger = _configure_logger(log_path, 'test', format='%(message)s')
        self.logger.propagate = False
        self.log_path = log_path
        self.log_predictions = False
        self.withNMS=withNMS
        self.dataset = dataset
        self.save_prediction = save_prediction
        self.spts = {}
        self.gtjnc = {}
        self.predjnc = {}
        self.evidence = {}
        self.sampleIdxName={}
        self.nms_jnc_acc = 0.0
        self.total = 0

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        self.samples = getattr(trainer.datamodule, 'sampleIds', None)
        self.totalBatches = trainer.num_test_batches[0]
        if self.samples is not None:
            self.log_predictions = True
            for i in range(0, self.totalBatches):
                self.spts[i] = []
                self.gtjnc[i] = []
                self.predjnc[i] = []
                self.sampleIdxName[i]=[]

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.log_predictions:
            self.spts[batch_idx].extend(torch.unbind(batch['scan_pts'][:,:,:3].cpu())) 
            self.gtjnc[batch_idx].extend(torch.unbind(batch['BRepAnnot_jIds'][:,:,:1].cpu()))
            #self.predjnc[batch_idx].extend(torch.unbind((outputs['pred_labels_jnc'][:, :, :1] > 0.6).int().cpu()))
            self.predjnc[batch_idx].extend(torch.unbind((outputs['pred_labels_jnc']).cpu()))
            self.sampleIdxName[batch_idx].extend(batch['idx_name'])
                      
    def on_test_end(self, trainer, pl_module):
        import open3d as o3d
        if self.log_predictions:
            self.testbatchSize = getattr(trainer.datamodule, 'test_batch_size', None)
            tempPCD = o3d.geometry.PointCloud()
            if self.save_prediction:
                for i in range(self.totalBatches):
                    scan_pts = torch.stack(self.spts[i]).numpy()
                    gtj = torch.stack(self.gtjnc[i]).numpy()
                    pj = torch.stack(self.predjnc[i]).numpy()
                    #self.logger.info({'test_sanity': pb})

                    uid=self.sampleIdxName[i]

                    #self.logger.info({'test_sanity': pb})
                    for s in range(self.testbatchSize):
                        uid_=uid[s]
                        uid_root="/".join(uid_.split("/")[-3:-1])
                        uid_name=uid_.split("/")[-1].split(".")[0]

                        # Create the Directory
                        gt_jnc_dir=os.path.join(self.log_path,"jnc/gt",uid_root)
                        pred_jnc_dir=os.path.join(self.log_path,"jnc/pred",uid_root)
                        scan_dir=os.path.join(self.log_path,"scan",uid_root)

                        ensure_dirs(gt_jnc_dir)
                        ensure_dirs(pred_jnc_dir)
                        ensure_dirs(scan_dir)

                        # Save the files
                        np.save(os.path.join(gt_jnc_dir,uid_name+".npy"),gtj[s,:,:1].flatten())
                        np.save(os.path.join(pred_jnc_dir,uid_name+".npy"),pj[s,:,:1].flatten())

                        o3d.io.write_point_cloud(os.path.join(scan_dir,uid_name+".ply"),
                                                 o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts[s,:,:3])))
                        #o3d.io.write_point_cloud(os.path.join(self.log_path, 'gt_%s.ply' % sID), gtPcd)
                        
                        if self.withNMS is True:
                            locMx_Idx_j = localNMS(scan_pts[s, :, :3], pj[s,:,:1].flatten(), dist_th=0.01)
                            mask_jnc = gtj[s,:,:1].flatten()
                            neg_mask_jnc = np.ones_like(mask_jnc) - mask_jnc
                            Np_jnc = np.sum(mask_jnc)
                            Ng_jnc = np.sum(neg_mask_jnc)
                            
                            pred_j = np.zeros_like(pj[s,:,:1].flatten(), dtype=int)
                            pred_j[locMx_Idx_j] = 1.0
                            
                            self.nms_jnc_acc += np.sum(np.equal(pred_j, gtj[s,:,:1].flatten()) *  mask_jnc)  / np.sum(mask_edge)
                            self.total += 1
                            
                            print("after NMS")
                            fp_fn_j = np.logical_xor(pj[s,:,:1].flatten() , gtj[s,:,:1].flatten()).astype(int)
                            predPcd = get_colored_pcd2(scan_pts[s, :, :3], pj[s,:,:1].flatten(), fp_fn_j) 
                            gtPcd =  get_colored_pcd(scan_pts[s, :, :3], gtj[s,:,:1].flatten()) 
                            tempPCD = gtPcd + predPcd.translate(np.array([4.0, 0.0, 0.0]))
                            o3d.io.write_point_cloud(os.path.join(self.log_path, 'gt_pj_nms%s.ply' % sID), tempPCD)
                
                if self.withNMS is True:
                    self.logger.info({'Stage': 'nms', 'Epoch': trainer.current_epoch, 'jnc_acc': self.nms_jnc_acc / self.total,})

def format_dict(dict, stage):
    return {k: (v if isinstance(v, torch.Tensor) else v) for k, v in dict.items()}
    #return {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in dict.items()}

def format_dict2(dict, stage, onlyJnc = False):
    d = {k.replace(stage+'_', ''): (v if isinstance(v, torch.Tensor) else v) for k, v in dict.items()}
    if onlyJnc == False:
        return 'stage: %s, epoch: %d, bndry_loss: %f, jnc_loss: %f, total_loss: %f, bndry_acc: %f, jnc_acc: %f, total_acc: %f' % (stage, d['epoch'], d['bndry_loss'], d['jnc_loss'], d['total_loss'], d['bndry_acc'], d['jnc_acc'], d['total_acc'])
    else:
        return 'stage: %s, epoch: %d, jnc_loss: %f, jnc_acc: %f' % (stage, d['epoch'], d['jnc_loss'], d['jnc_acc'])

def format_dict3(dict, stage):
    d = {k.replace(stage+'_', ''): (v if isinstance(v, torch.Tensor) else v) for k, v in dict.items()}
    return 'stage: %s, epoch: %d, Loss: %f, accuracy: %f, boundary_precision: %s, junction_precision: %s, boundary_recall: %s, junction_recall: %s'  % (stage, d['epoch'], d['loss'], d['accuracy'], d['boundary_prec'].detach().cpu().numpy(), d['junction_prec'].detach().cpu().numpy(), d['boundary_recall'].detach().cpu().numpy(), d['junction_recall'].detach().cpu().numpy())

def format_dict4(dict, stage):
    d = {k.replace(stage+'_', ''): (v if isinstance(v, torch.Tensor) else v) for k, v in dict.items()}
    return 'stage: %s, epoch: %d, Loss: %f, accuracy: %f' % (stage, d['epoch'], d['loss'], d['accuracy'])

def _configure_logger(path, subset, format='%(asctime)s %(message)s'):
    if not os.path.exists(path):
        create_dir(path)
    logger = logging.getLogger(subset + '_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(path, subset + '.log'))
    formatter = logging.Formatter(fmt=format, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)   
    return logger
    

