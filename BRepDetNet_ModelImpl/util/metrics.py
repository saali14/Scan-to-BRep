import torch
from typing import Any, Callable, Optional
from torchmetrics.metric import Metric
from torchmetrics.classification import BinaryPrecisionRecallCurve

class PrecisionRecallMetricsBndry(Metric):
    """
    Computes MAE, MSE, RMSE and R2 for rotation in degree and for translation of given transformation matrices.
    Following torchmetrics for R2-score.
    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
    """
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        num_threshold: int = 10,
        prefix: str = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if prefix is not None:
            self.prefix = prefix + '_'
        else:
            self.prefix = ''
        
        self.num_threshold = num_threshold
        self.bndry_PrecRecmetric = BinaryPrecisionRecallCurve(thresholds = self.num_threshold)
        
        self.add_state("boundary_precision", default=torch.zeros(num_threshold + 1, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("boundary_recall", default=torch.zeros(num_threshold + 1, dtype=torch.float), dist_reduce_fx="sum")        
        self.add_state("boundary_F1Measure", default=torch.zeros(num_threshold + 1, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predsBndry: torch.Tensor, gtBndry: torch.Tensor):
        """
        Update state with predictions and targets.
        Input shape: (..., 1) 
        Args:
            preds: Predicted labels of Scan Points from model
            gt: Ground labels of Scan Points
        """
        assert predsBndry.shape == gtBndry.shape

        prec_b, rec_b, thr_b = self.bndry_PrecRecmetric(predsBndry, gtBndry)
        
        self.boundary_precision = torch.add(self.boundary_precision, prec_b)
        self.boundary_recall = torch.add(self.boundary_recall, rec_b)
        self.total += 1
        self.thr_b = thr_b

    def compute(self):
        """
        Computes mean absolute error over state.
        """
        return {
            self.prefix + 'boundary_prec': torch.as_tensor(self.boundary_precision) / self.total,
            self.prefix + 'boundary_recall': torch.as_tensor(self.boundary_recall) / self.total,
            self.prefix + 'boundary_threshold': torch.as_tensor(self.thr_b),
        }

class PrecisionRecallMetricsJnc(Metric):
    """
    Computes MAE, MSE, RMSE and R2 for rotation in degree and for translation of given transformation matrices.
    Following torchmetrics for R2-score.
    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
    """
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        num_threshold: int = 10,
        prefix: str = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if prefix is not None:
            self.prefix = prefix + '_'
        else:
            self.prefix = ''
        
        self.num_threshold = num_threshold
        self.jnc_PrecRecmetric = BinaryPrecisionRecallCurve(thresholds = self.num_threshold)
        
        self.add_state("junction_precision", default=torch.zeros(num_threshold + 1, dtype=torch.float), dist_reduce_fx="sum")  
        self.add_state("junction_recall", default=torch.zeros(num_threshold + 1, dtype=torch.float), dist_reduce_fx="sum")      
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predsJnc: torch.Tensor, gtJnc: torch.Tensor):
        """
        Update state with predictions and targets.
        Input shape: (..., 1) 
        Args:
            preds: Predicted labels of Scan Points from model
            gt: Ground labels of Scan Points
        """
        assert predsJnc.shape == gtJnc.shape
        prec_j, rec_j, thr_j = self.jnc_PrecRecmetric(predsJnc, gtJnc)
        
        
        self.junction_precision = torch.add(self.junction_precision, prec_j)
        self.junction_recall = torch.add(self.junction_recall, rec_j)
        self.total += 1
        self.thr_j = thr_j

    def compute(self):
        """
        Computes mean absolute error over state.
        """
        return {
            self.prefix + 'junction_prec': torch.as_tensor(self.junction_precision) / self.total,
            self.prefix + 'junction_recall': torch.as_tensor(self.junction_recall) / self.total,
            self.prefix + 'junction_threshold': torch.as_tensor(self.thr_j),
        }

class DetLossAcccMetricsBndry(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        prefix: str = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if prefix is not None:
            self.prefix = prefix + '_'
        else:
            self.prefix = ''
        

        self.add_state("bndry_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")  
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("bndry_acc", default=torch.tensor(0.0), dist_reduce_fx="sum")     
        self.add_state("total_acc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, 
               BLoss: torch.Tensor, 
               TLoss: torch.Tensor, 
               BAcc: torch.Tensor, 
               TAcc: torch.Tensor):
        """
        Update state with predictions and targets.
        Input shape: (..., 1) 
        Args:
            preds: Predicted labels of Scan Points from model
            gt: Ground labels of Scan Points
        """
        self.bndry_loss = torch.add(self.bndry_loss, BLoss)
        self.total_loss = torch.add(self.total_loss, TLoss)
        self.bndry_acc = torch.add(self.bndry_acc, BAcc)
        self.total_acc = torch.add(self.total_acc, TAcc)
        self.total += 1

    def compute(self):
        """
        Computes mean absolute error over state.
        """
        return {
            self.prefix + 'bndry_loss': torch.as_tensor(self.bndry_loss) / self.total,
            self.prefix + 'total_loss': torch.as_tensor(self.total_loss) / self.total,
            self.prefix + 'bndry_acc': torch.as_tensor(self.bndry_acc) / self.total,
            self.prefix + 'total_acc': torch.as_tensor(self.total_acc) / self.total
        }

class DetLossAcccMetricsJnc(Metric):
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        prefix: str = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        if prefix is not None:
            self.prefix = prefix + '_'
        else:
            self.prefix = ''
        
        self.add_state("jnc_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")        
        self.add_state("jnc_acc", default=torch.tensor(0.0), dist_reduce_fx="sum")      
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, JLoss: torch.Tensor, JAcc: torch.Tensor):
        """
        Update state with predictions and targets.
        Input shape: (..., 1) 
        Args:
            preds: Predicted labels of Scan Points from model
            gt: Ground labels of Scan Points
        """
        self.jnc_loss = torch.add(self.jnc_loss, JLoss)
        self.jnc_acc = torch.add(self.jnc_acc, JAcc)
        self.total += 1

    def compute(self):
        """
        Computes mean absolute error over state.
        """
        return {
            self.prefix + 'jnc_loss': torch.as_tensor(self.jnc_loss) / self.total,
            self.prefix + 'jnc_acc': torch.as_tensor(self.jnc_acc) / self.total,
        }
