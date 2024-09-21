import numpy as np
import torch


def transform(points, trans_mat):
    return (torch.matmul(trans_mat[..., :3, :3], points.transpose(-1,-2)) + trans_mat[..., :3, 3].unsqueeze(-1)).transpose(-1,-2)

def nested_list_to_tensor(pcds, device='cuda'):
    return torch.stack([torch.tensor(x[0], device=device) for x in pcds])

def random_downsampling(points, num_points=1024):
    rand_idxs = np.random.choice(points.shape[0], num_points, replace=False)
    return points[rand_idxs, :]
