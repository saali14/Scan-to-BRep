import torch
import numpy as np
import open3d as o3d


class Resampler:
    def __init__(self, num: int, upsampling=False):
        """Downsample a point cloud containing N points to one containing M
        Require M <= N.
        Args:
            num (int): Number of points to resample to, i.e. M
        """
        self.num = num
        self.upsampling = upsampling
        self.sanity_checkList = [1] # <-- NOTE add more sample Idx if required

    def __call__(self, sample):
        # try:
        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])
            
        # TODO@Srikanth: Why ? this will remove vertex with ID 0
        bIds = np.where(sample['BRepAnnot_vIds'])[0] # <--- Returns the idx / positions of Boundary Points in Scan
        jIds = np.where(sample['BRepAnnot_jIds'])[0] # <--- Returns the idx / positions of Junction Points in Scan
        neg_bj_id = np.setdiff1d(np.arange(sample['scan_pts'].shape[0]), np.union1d(bIds, jIds))
        
 
        rand_b_idxs = self._resample(bIds, min(int(0.5 * len(bIds)), int(0.4 * self.num)), self.upsampling)
        rand_j_idxs = self._resample(jIds, min(int(0.95 * len(jIds)), int(0.4 * self.num)), self.upsampling)
        
        if self.num <= len(neg_bj_id):
            rand_v_idxs = self._resample(neg_bj_id, self.num - len(np.union1d(rand_b_idxs, rand_j_idxs)), True)
        else:
            #NOTE reduce more boundary points sampling ratio and increase Non-(boundary + Junction) points sampling
            rand_b_idxs = self._resample(bIds, min(int(0.5 * len(bIds)), int(0.4 * self.num)), self.upsampling)
            rand_j_idxs = self._resample(jIds, min(int(0.95 * len(jIds)), int(0.4 * self.num)), self.upsampling)
            
            if len(neg_bj_id) == 0:
                rand_v_idxs = self._resample(np.arange(sample['scan_pts'].shape[0]), self.num - len(np.union1d(rand_b_idxs, rand_j_idxs)), self.upsampling)
            else:
                rand_v_idxs =  self._resample(np.arange(sample['scan_pts'].shape[0]), self.num - len(np.union1d(rand_b_idxs, rand_j_idxs)), self.upsampling)
            
        sampledBJFIDx = list(rand_v_idxs) +  list(np.union1d(rand_b_idxs, rand_j_idxs))
            
        #NOTE -->  Sanity Check 4
        if (sample['idx'] in self.sanity_checkList) and len(sample) > 0:
            sanity_pcd1 = o3d.geometry.PointCloud()
            sanity_pcd1.points = o3d.utility.Vector3dVector(sample['scan_pts'])
            sanity_pcd1.paint_uniform_color([0.3, 0.8, 1.0])
            
            sanity_pcd2 = o3d.geometry.PointCloud()
            sanity_pcd2.points = o3d.utility.Vector3dVector(sample['scan_pts'][sampledBJFIDx])
            sanity_pcd2.paint_uniform_color([0.8, 0.3, 1.0])
            sanity_pcd2.translate([200.5, 0.0, 0.0])
        
            sanity_pcd3 = o3d.geometry.PointCloud()
            sanity_pcd3.points = o3d.utility.Vector3dVector(sample['scan_pts'][np.union1d(rand_b_idxs, rand_j_idxs)])
            sanity_pcd3.paint_uniform_color([1.0, 0.1, 0.5])
            sanity_pcd3.translate([300.5, 0.0, 0.0])
            
            
            sanity_pcd4 = o3d.geometry.PointCloud()
            sanity_pcd4.points = o3d.utility.Vector3dVector(sample['scan_pts'][rand_j_idxs])
            sanity_pcd4.paint_uniform_color([0.5, 1.0, 1.0])
            sanity_pcd4.translate([400.5, 0.0, 0.0])
            
            o3d.visualization.draw_geometries([sanity_pcd1, sanity_pcd2, sanity_pcd3, sanity_pcd4])
            sanity_pcd = sanity_pcd1 + sanity_pcd2 + sanity_pcd3 + sanity_pcd4
            o3d.io.write_point_cloud("./sampled_input_" + str(sample['idx']) + ".ply", sanity_pcd)
            
        sample['scan_pts'] = sample['scan_pts'][sampledBJFIDx]
        sample['BRepAnnot_vIds'] = sample['BRepAnnot_vIds'][sampledBJFIDx]
        sample['BRepAnnot_jIds'] = sample['BRepAnnot_jIds'][sampledBJFIDx]
        return sample


    @staticmethod
    def _resample(points, k, upsampling):
        """
        Resamples the points such that there is exactly k points.
        """
        if k <= points.shape[0]:
            #print("Dimension +ve: ", points.shape)
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs]
        elif k == points.shape[0] or not upsampling:
            #print("Dimension -0+: ", points.shape)
            return points
        else:
            #print("Dimension -ve: ", points.shape)
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs]



class Normalizer:
    def __call__(self, sample):  
        mn = np.min(sample['scan_pts'], axis=0)
        mx = np.max(sample['scan_pts'], axis=0)
        
        xmin = mn[0]
        xmax = mx[0]
        ymin = mn[1]
        ymax = mx[1]
        zmin = mn[2]
        zmax = mx[2]
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        longest_length = dx
        if longest_length < dy:
            longest_length = dy
        if longest_length < dz:
            longest_length = dz

        orig = [0.0, 0.0, 0.0]
        center = [
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
        ]
        sample['scan_pts'] = (2.0/longest_length) * (sample['scan_pts'] - np.asarray(center)) 
        
        return sample


class ResamplerBoundary:
    def __init__(self, num: int, upsampling=False, sbj_perc= None):
        """Downsample a point cloud containing N points to one containing M
        Require M <= N.
        Args:
            num (int): Number of points to resample to, i.e. M
        """
        self.num = num
        self.upsampling = upsampling
        self.sanity_checkList = [] # <-- NOTE add more sample Idx if required
        if sbj_perc is not None:
            self.sbj_percentages = sbj_perc
        else:
            self.sbj_percentages = [0.2, 0.8, 0.95] 

    def __call__(self, sample):
        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])
            
        bIds = np.where(sample['BRepAnnot_vIds'])[0] # <--- Returns the idx / positions of Boundary Points in Scan
        jIds = np.where(sample['BRepAnnot_jIds'])[0] # <--- Returns the idx / positions of Junction Points in Scan
        
        rand_j_idxs = self._resample(jIds, min(int(1.0 * len(jIds)), int(0.3 * self.num)), self.upsampling)
        rand_b_idxs = self._resample(bIds, min(int(0.6 * len(bIds)), int(0.7 * self.num)), self.upsampling)
        bj_id = np.concatenate((rand_b_idxs, rand_j_idxs), axis=None)

        
        if self.num < len(bj_id):
            rand_b_idxs = self._resample(np.setdiff1d(bIds, bj_id), self.num - len(bj_id), self.upsampling)
        elif self.num == len(bj_id):
            rand_b_idxs = rand_b_idxs 
        else:
            #NOTE reduce more boundary points sampling ratio and increase Non-(Junction) points sampling
            rand_b_idxs = self._resample(bIds, self.num - len(rand_j_idxs), True)
            
        sampledBJIDx = list(np.concatenate((rand_b_idxs, rand_j_idxs), axis=None)) 
            
        #NOTE -->  Sanity Check 4
        if (sample['idx'] in self.sanity_checkList) and len(sample) > 0:
            sanity_pcd1 = o3d.geometry.PointCloud()
            sanity_pcd2 = o3d.geometry.PointCloud()
            
            sanity_pcd1.points = o3d.utility.Vector3dVector(sample['scan_pts'][np.union1d(rand_b_idxs, rand_j_idxs)])
            sanity_pcd1.paint_uniform_color([1.0, 0.1, 0.5])
            sanity_pcd1.translate([250.5, 0.0, 0.0])
            
            sanity_pcd2.points = o3d.utility.Vector3dVector(sample['scan_pts'][rand_j_idxs])
            sanity_pcd2.paint_uniform_color([0.5, 1.0, 1.0])
            sanity_pcd2.translate([350.5, 0.0, 0.0])
            
            o3d.visualization.draw_geometries([sanity_pcd1, sanity_pcd2])
            sanity_pcd = sanity_pcd1 + sanity_pcd2
            o3d.io.write_point_cloud("./sampled_input_" + str(sample['idx']) + ".ply", sanity_pcd)

        sample['scan_pts'] = sample['scan_pts'][sampledBJIDx]
        sample['BRepAnnot_vIds'] = sample['BRepAnnot_vIds'][sampledBJIDx]
        sample['BRepAnnot_jIds'] = sample['BRepAnnot_jIds'][sampledBJIDx]
        return sample

    @staticmethod
    def _resample(points, k, upsampling):
        """
        Resamples the points such that there is exactly k points.
        """
        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs]
        elif k == points.shape[0] or not upsampling:
            return points
        else:
            rand_idxs = np.random.choice(points.shape[0], k, replace=True)
            return points[rand_idxs]

class ResamplerBoundary_Old:
    def __init__(self, num: int, upsampling=False, sbj_perc= None):
        """Downsample a point cloud containing N points to one containing M
        Require M <= N.
        Args:
            num (int): Number of points to resample to, i.e. M
        """
        self.num = num
        self.upsampling = upsampling
        self.sanity_checkList = [] # <-- NOTE add more sample Idx if required
        if sbj_perc is not None:
            self.sbj_percentages = sbj_perc
        else:
            self.sbj_percentages = [0.2, 0.8, 0.95] 

    def __call__(self, sample):
        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])
            
        bIds = np.where(sample['BRepAnnot_vIds'])[0] # <--- Returns the idx / positions of Boundary Points in Scan
        jIds = np.where(sample['BRepAnnot_jIds'])[0] # <--- Returns the idx / positions of Junction Points in Scan
        neg_bj_id = np.setdiff1d(np.arange(sample['scan_pts'].shape[0]), np.union1d(bIds, jIds))
        
        rand_b_idxs = self._resample(bIds, min(int(0.6 * len(bIds)), int(0.7 * self.num)), self.upsampling)
        rand_j_idxs = self._resample(jIds, min(int(0.95 * len(jIds)), int(0.2 * self.num)), self.upsampling)
        
        if self.num <= len(neg_bj_id):
            rand_v_idxs = self._resample(neg_bj_id, self.num - len(np.union1d(rand_b_idxs, rand_j_idxs)), True)
        else:
            #NOTE reduce more boundary points sampling ratio and increase Non-(boundary + Junction) points sampling
            rand_b_idxs = self._resample(bIds, min(int(0.5 * len(bIds)), int(0.6 * self.num)), self.upsampling)
            rand_j_idxs = self._resample(jIds, min(int(0.95 * len(jIds)), int(0.15 * self.num)), self.upsampling)
            
            if len(neg_bj_id) == 0:
                rand_v_idxs = self._resample(np.arange(sample['scan_pts'].shape[0]), self.num - len(np.union1d(rand_b_idxs, rand_j_idxs)), self.upsampling)
            else:
                rand_v_idxs =  self._resample(np.arange(sample['scan_pts'].shape[0]), self.num - len(np.union1d(rand_b_idxs, rand_j_idxs)), self.upsampling)
            
        sampledBJIDx = list(np.union1d(rand_b_idxs, rand_j_idxs))
            
        #NOTE -->  Sanity Check 4
        if (sample['idx'] in self.sanity_checkList) and len(sample) > 0:
            sanity_pcd1 = o3d.geometry.PointCloud()
            sanity_pcd2 = o3d.geometry.PointCloud()
            sanity_pcd3 = o3d.geometry.PointCloud()
         
            sanity_pcd1.points = o3d.utility.Vector3dVector(sample['scan_pts'])
            sanity_pcd1.paint_uniform_color([0.3, 0.8, 1.0])
            
            sanity_pcd2.points = o3d.utility.Vector3dVector(sample['scan_pts'][np.union1d(rand_b_idxs, rand_j_idxs)])
            sanity_pcd2.paint_uniform_color([1.0, 0.1, 0.5])
            sanity_pcd2.translate([250.5, 0.0, 0.0])
            
            sanity_pcd3.points = o3d.utility.Vector3dVector(sample['scan_pts'][rand_j_idxs])
            sanity_pcd3.paint_uniform_color([0.5, 1.0, 1.0])
            sanity_pcd3.translate([350.5, 0.0, 0.0])
            
            o3d.visualization.draw_geometries([sanity_pcd1, sanity_pcd2, sanity_pcd3, sanity_pcd4])
            sanity_pcd = sanity_pcd1 + sanity_pcd2 + sanity_pcd3 + sanity_pcd4
            o3d.io.write_point_cloud("./sampled_input_" + str(sample['idx']) + ".ply", sanity_pcd)
            
        sample['scan_pts'] = sample['scan_pts'][sampledBJIDx]
        sample['BRepAnnot_vIds'] = sample['BRepAnnot_vIds'][sampledBJIDx]
        sample['BRepAnnot_jIds'] = sample['BRepAnnot_jIds'][sampledBJIDx]
        return sample

    @staticmethod
    def _resample(points, k, upsampling):
        """
        Resamples the points such that there is exactly k points.
        """
        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs]
        elif k == points.shape[0] or not upsampling:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs]

class FixedResampler(Resampler):
    """Fixed resampling to always choose points from evenly spaced indices.
    Always deterministic regardless of whether the deterministic flag has been set
    """
    @staticmethod
    def _resample(points, k, upsampling=False):
        resampled = points[np.linspace(0, points.shape[0], num=k, endpoint=False, dtype=np.int),:]
        return resampled

class ShufflePoints:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        sample['scan_pts'] = np.random.permutation(sample['scan_pts'])
        return sample

class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['deterministic'] = True
        return sample

class AddBatchDimension():
    def __call__(self, sample):
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                sample[k] = np.expand_dims(v, 0)
        return sample

