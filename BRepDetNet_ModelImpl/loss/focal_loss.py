import torch
import torch.nn as nn

class FocalLossJnc(nn.Module):
    def __init__(self,
                 weight=0.25,
                 gamma=1.25,
                 alpha=0.25,
                 device='cuda'):
        super(FocalLossJnc, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = torch.tensor([weight, 1-weight]).cuda()
        self.device = device
        self.epsilon = 1e-10  
        self.SfMx = nn.Sigmoid()
        
    def __call__(self, pred_labels_jnc, gt_labels_jnc, prefix='train'):
        losses = dict()
        acc = dict()
        
        bce_loss_jnc = nn.BCEWithLogitsLoss(reduction='none')
        mask_jnc = gt_labels_jnc 
        neg_mask_jnc = torch.ones_like(mask_jnc) - mask_jnc
        Np_jnc = torch.sum(mask_jnc, dim=1).unsqueeze(1) 
        Ng_jnc = torch.sum(neg_mask_jnc, dim=1).unsqueeze(1) 
        
        bce_loss_jnc = bce_loss_jnc(pred_labels_jnc, gt_labels_jnc)
        p_tj = torch.exp(-bce_loss_jnc)
        modulating_factor_j = torch.pow(1.0 - p_tj, self.gamma)
        focal_cross_entropy_loss = (modulating_factor_j  * bce_loss_jnc) * (mask_jnc * (Np_jnc / (Ng_jnc + self.epsilon)) + 1)
        jnc_acc = torch.sum(torch.eq((self.SfMx(pred_labels_jnc.detach())[:, :, :1] > 0.4).int(), gt_labels_jnc.detach()).type(torch.float32) *  mask_jnc, dim=1)  / torch.sum(mask_jnc, dim=1)
        losses[prefix + '_jnc_focal_loss'] = focal_cross_entropy_loss.mean()
        acc[prefix + '_jnc_accuracy'] = jnc_acc.mean()

        return losses, acc

class FocalLossBndry(nn.Module):
    def __init__(self,
                 weight=0.25,
                 gamma=1.25,
                 alpha=0.25,
                 device='cuda'):
        super(FocalLossBndry, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = torch.tensor([weight, 1-weight]).cuda()
        self.device = device
        self.epsilon = 1e-10 
        self.SfMx = nn.Sigmoid()
        
    def __call__(self, pred_labels_bndry, gt_labels_bndry, prefix='train'):
        losses = dict()
        acc = dict()
        
        bce_loss_bndry = nn.BCEWithLogitsLoss(reduction='none')
        mask_edge = gt_labels_bndry 
        neg_mask_edge = torch.ones_like(mask_edge) - mask_edge
        Np_edge = torch.sum(mask_edge, dim=1).unsqueeze(1) 
        Ng_edge = torch.sum(neg_mask_edge, dim=1).unsqueeze(1) 
        
        bce_loss_bndry = bce_loss_bndry(pred_labels_bndry, gt_labels_bndry)
        p_tb = torch.exp(-bce_loss_bndry)
        
        modulating_factor_b = torch.pow(1.0 - p_tb, self.gamma)
        focal_cross_entropy_loss = (modulating_factor_b * bce_loss_bndry) * (mask_edge * (Ng_edge / (Np_edge+self.epsilon)) + 1)
        bndry_acc = torch.sum(torch.eq((self.SfMx(pred_labels_bndry.detach())[:, :, :1] > 0.55).int(), gt_labels_bndry).type(torch.float32) *  mask_edge, dim=1)  / torch.sum(mask_edge, dim=1)
        
        losses[prefix + '_bndry_focal_loss'] = focal_cross_entropy_loss.mean()
        acc[prefix + '_bndry_accuracy'] = bndry_acc.mean()
        
        return losses, acc







