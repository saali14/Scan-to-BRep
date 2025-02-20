import logging
from random import Random
from typing import Iterable, Union

import numpy as np
import open3d as o3d
import torch
import torch.distributed
import torchvision
from dataproc import transforms as Transforms
from model import load_model
from torch import nn

logger = logging.getLogger(__name__)

seed = 42
random = Random(seed)


class ResamplerForScan(Transforms.Resampler):
    def __call__(self, sample):
        if "deterministic" in sample and sample["deterministic"]:
            np.random.seed(sample["idx"])

        sampledIDx = self._resample(
            np.arange(sample["scan_pts"].shape[0]), self.num, False
        )
        sample["scan_pts"] = sample["scan_pts"][sampledIDx]
        sample["sampledIDx"] = sampledIDx
        return sample


def detect(
    model_name: str,
    checkpoint_path: str,
    num_points: int,
    scan_file: str = None,
    scan_np_array: np.array = None,
) -> Iterable[Union[np.array, np.array]]:
    assert not (scan_file is None and scan_np_array is None)

    transform = torchvision.transforms.Compose(
        [
            ResamplerForScan(num_points, upsampling=True),
            Transforms.Normalizer(),
            Transforms.SetDeterministic(),
        ]
    )

    if scan_file is not None:
        scan = o3d.io.read_triangle_mesh(scan_file)
        point_cloud = np.asarray(scan.vertices).reshape((-1, 3))
    else:
        point_cloud = scan_np_array.reshape((-1, 3))

    sample = {"scan_pts": point_cloud}
    sample = transform(sample)

    kwarsgs = {
        "checkpoint_path": checkpoint_path,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    model = load_model(model_name, checkpoint_path, kwarsgs)

    with torch.no_grad():
        # Add simple sample into a batch
        sample["scan_pts"] = sample["scan_pts"].reshape(1, -1, 3)
        preds = model.predict_step(sample, None, None)[0]

    SfMx = nn.Sigmoid()
    preds = (SfMx(preds.detach())[:, :, :1] > 0.55).int().flatten().cpu().numpy()
    sampledIDx = sample["sampledIDx"]
    all_preds = np.zeros(len(point_cloud), dtype=int) + -1
    all_preds[sampledIDx] = preds
    return point_cloud, all_preds

def main() :
    logger.info(f"""Using: {"cuda" if torch.cuda.is_available() else "cpu"}...""")

    Ed_checkpoint_path = "BRepDetNet_CheckPoints/ABC/BRepEd/train/version_4/checkpoint/last.ckpt"
    Jd_checkpoint_path = "BRepDetNet_CheckPoints/ABC/BRepJd/train/version_0/checkpoint/last.ckpt"

    scan_file = "/home/srikanth/Documents/bits/BRep/datasets/annotations/CC3D/Scan/User Library-8mm motor.ply"
    # Or a numpy point cloud
    # scan_np_array = np.random.randn(50000,3)

    logger.info("Predicting Edges...")
    point_cloud, edges = detect("BRepEd", Ed_checkpoint_path, 10000, scan_file=scan_file)
    assert len(edges) == len(point_cloud)

    # Select only Detected Edges
    logger.info("Predicting Junctions...")
    _, junctions = detect("BRepJd", Jd_checkpoint_path, 4192, scan_np_array=point_cloud[edges == 1])

    full_junctions_labels = np.zeros(len(point_cloud), dtype=int)  # A non edge is a non junction
    full_junctions_labels[edges == -1] = -1  # Skipped during downsample; so no label
    full_junctions_labels[edges == 1] = junctions  # Junction labels

    # with open("", "wb") as fp:
    np.savez(
        "output.npz", vertices=point_cloud, edges=edges, junctions=full_junctions_labels
    )

    logger.info("Done, output saved to output.npz...")
    # Note: -1 in predicitons represetns no label due point skipped during downsampling

if __name__ == "__main__":
    main()
    