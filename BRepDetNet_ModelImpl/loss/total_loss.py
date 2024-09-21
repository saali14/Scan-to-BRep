import torch.nn as nn

class TotalLoss(nn.Module):
    def __init__(self, weights):
        super(TotalLoss, self).__init__()
        self.weights = weights

    def __call__(self, losses, prefix='train'):
        total = 0.0
        total_w = 0.0
        for key, weight in self.weights.items():
            if prefix + '_' + key in losses:
                total += weight * losses[prefix + '_' + key]
                total_w += weight
        
        #return {'loss': total}
        return {prefix + '_total_loss': total/total_w}

class TotalAccuracy(nn.Module):
    def __init__(self, weights):
        super(TotalAccuracy, self).__init__()
        self.weights = weights

    def __call__(self, accr, prefix='train'):
        total = 0.0
        total_w = 0.0
        for key, weight in self.weights.items():
            if prefix + '_' + key in accr:
                total += weight * accr[prefix + '_' + key]
                total_w += weight
                
        #return {'loss': total}
        return {prefix + '_total_accuracy': total/total_w}
