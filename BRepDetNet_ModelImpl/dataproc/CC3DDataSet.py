"""Data loader
"""
import os
import logging
import numpy as np
import open3d as o3d
import pickle

from torch.utils.data import Dataset
from random import Random
from tqdm import tqdm

class CC3DDataSet(Dataset):
    def __init__(self, root_path: str, scan_path: str, annotationBndry_path: str, 
                 annotationJnc_path: str, subset: str = 'test', transform=None, selectedSamples=None, seed=42, sequence=None, add_pose=False, **kwargs):
        """args:
            root_path (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self._logger = logging.getLogger('core.data')
        self._root = root_path
        self._scanPath = scan_path
        self._annotBndryPath = annotationBndry_path
        self._annotJncPath = annotationJnc_path
        self._logger.info('Loading data for {}'.format(subset))
        self.all_edgetypes = ["line", "ellipse", "hyperbola", "parabola", "bspline", "bezier", "circle", "offset"]

        if subset!="test":
            if not os.path.exists(os.path.join(root_path)):
                raise Exception("Dataset not found at: {}".format(os.path.join(root_path)))
            if not os.path.exists(os.path.join(scan_path)):
                raise Exception("Scans not found at: {}".format(os.path.join(scan_path)))
            if not os.path.exists(os.path.join(annotationBndry_path)):
                raise Exception("Boundary Annotation not found at: {}".format(os.path.join(annotationBndry_path)))
            if not os.path.exists(os.path.join(annotationJnc_path)):
                raise Exception("Junction Annotation not found at: {}".format(os.path.join(annotationJnc_path)))
            
        self.samples = []
        self.subset = subset
        self.random = Random(seed)
        
        with open(os.path.join(root_path, self.subset + ".txt"), 'r') as f:
            sampleIDs = [line.replace("\n", "") for line in f]
        if len(sampleIDs) == 0:
            raise Exception("Sample IDs not found at: {}.".format(os.path.join(root_path, self.subset + ".txt")))
        else:
            for s in tqdm(sampleIDs): #tqdm(sampleIDs[:int(0.6 * len(sampleIDs))]): #sampleIDs

                corrScanFile = os.path.splitext(os.path.join(self._scanPath, self.subset, s))[0] + ".ply"
                if not os.path.exists(corrScanFile):
                    corrScanFile = os.path.splitext(os.path.join(self._scanPath, self.subset+"_codalab", s))[0] + ".ply"
                    
                corrBndryAnnotFile = os.path.splitext(os.path.join(self._annotBndryPath, self.subset, s))[0] + ".npz"
                corrJncAnnotFile = os.path.splitext(os.path.join(self._annotJncPath, self.subset, s))[0] + ".npz"              
            
                if os.path.exists(corrScanFile):
                    if os.path.exists(corrBndryAnnotFile) and os.path.exists(corrJncAnnotFile):
                        brepAnnot_jncId = np.array(np.load(corrJncAnnotFile, allow_pickle=True)["j_vertex_Ids"].astype(int))
                        brepAnnot_bndryId = np.array(np.load(corrBndryAnnotFile, allow_pickle=True)["v_vertex_Ids"].astype(int))

                        if len(np.where(brepAnnot_jncId)[0]) > 30 and (not np.isnan(brepAnnot_jncId).any()):
                            if not np.any(brepAnnot_jncId < 0):
                                sample = {
                                    'scan': corrScanFile,
                                    'brepBndryAnnot': corrBndryAnnotFile,
                                    'brepJncAnnot': corrJncAnnotFile,
                                    'split': self.subset,
                                }
                                self.samples.append(sample)
            
        self._logger.info('Loaded {} {} instances.'.format(len(self.samples), self.subset))
        self._transform = transform

    def __getitem__(self, item):
        t = self.samples[item]  #NOTE Ali:: <-- Item is the index of the sample from CC3DDataSet
        scan = o3d.io.read_triangle_mesh(t['scan'])
       
        brepBndryAnnot = np.load(t['brepBndryAnnot'], allow_pickle=True)
        brepJncAnnot = np.load(t['brepJncAnnot'], allow_pickle=True)

        vIdxs = np.array(brepBndryAnnot["v_vertex_Ids"], dtype=np.int32)
        jIdxs = np.array(brepJncAnnot["j_vertex_Ids"], dtype=np.int32)

        wrong_vIdxs = np.where(np.array(brepBndryAnnot["v_vertex_Ids"], dtype=np.int32)  >= len(scan.vertices))
        wrong_jIdxs = np.where(np.array(brepJncAnnot["j_vertex_Ids"], dtype=np.int32)  >= len(scan.vertices))

        correct_vIdxs = np.delete(vIdxs, wrong_vIdxs)
        correct_jIdxs = np.delete(jIdxs, wrong_jIdxs)
        
        BRepAnnot_vIds = np.zeros(shape=len(scan.vertices), dtype=np.int32)
        BRepAnnot_vIds[correct_vIdxs] = 1
       
        BRepAnnot_jIds = np.zeros(shape=len(scan.vertices), dtype=np.int32)
        BRepAnnot_jIds[correct_jIdxs] = 1 
       

        sample = {'scan_pts': np.asarray(scan.vertices).reshape((-1, 3)), 
                'BRepAnnot_vIds': BRepAnnot_vIds.reshape((-1, 1)),
                'BRepAnnot_jIds': BRepAnnot_jIds.reshape((-1, 1)),
                'idx': np.array(item, dtype=np.int32),
                'idx_name':t['scan']
                }
        
        if self._transform:
            sample = self._transform(sample)
        
        return sample
        
    def __len__(self):
        return len(self.samples)
    
    
    
