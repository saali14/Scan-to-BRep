import argparse
import numpy as np
import json  
import logging
import os
import sys
import time
import pathlib
import glob
import gc
import copy
import open3d as o3d
import utils.scale_utils as scale_utils
import utils.data_utils as data_utils

from tqdm.auto import tqdm
from tqdm.contrib import tzip
from matplotlib.colors import hsv_to_rgb 
from distinctipy import distinctipy

from analyzer.extract_brepnet_data_from_step import BRepExtractor
from analyzer.entity_mapper import EntityMapper
from analyzer.face_index_validator import FaceIndexValidator
from analyzer.segmentation_file_crosschecker import SegmentationFileCrosschecker
from BRep2CADLabler.dataStruct import LabelBoundaryDataStruct, LabelFaceDataStruct
from BRep2CADLabler.BRep2MeshLabler import BRep2MeshLabler

from OCC.Extend import TopologyUtils
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

def _parse_args():
    brep2meshL =  BRep2MeshLabler("Scan_Labler")
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    parser_labelBRepBndry2Scan = subparsers.add_parser("labelBRepBndry2Scan",
        help="Label the Edges + Faces of CAD w.r.t faces, Edges and Loops of BReps")
    parser_labelBRepBndry2Scan.add_argument("--step_path", type=str, required=True, 
                        help="Path to load the step files from")
    parser_labelBRepBndry2Scan.add_argument("--scan_path", type=str, required=True, 
                        help="Path to load the 3D scan files from")
    parser_labelBRepBndry2Scan.add_argument("--output", type=str, required=True,
        help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points" ,)
    parser_labelBRepBndry2Scan.add_argument("--nProc", type=int, default=1, 
                        help="Number of worker threads",)
    parser_labelBRepBndry2Scan.add_argument("--feature_list", type=str, required=False,
        help="Optional path to the feature lists",)
    parser_labelBRepBndry2Scan.add_argument("--mesh_dir", type=str, required=False,
                        help="Optionally cross check with Fusion Gallery mesh files to check the segmentation labels",)
    parser_labelBRepBndry2Scan.add_argument("--seg_dir", type=str, required=False,
        help="Optionally provide a directory containing segmentation labels seg files.",)
    parser_labelBRepBndry2Scan.add_argument("--fRegen", type=bool, required=False, default=False,
        help="To regenerate the stepFiles",)
    parser_labelBRepBndry2Scan.add_argument("--scale_brep", type=bool, required=False, default=True,
        help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_labelBRepBndry2Scan.add_argument("--infmt", required=False, 
                        choices=(['.stp', '.step']), default='.stp', 
                        help="Format of STEP files. Default is '.step'")
    parser_labelBRepBndry2Scan.add_argument("--ofmt", required=False, 
                        choices=(['.npz', '.pkl', '.ply']), default='.npz', 
                        help="Format of labels of Scan Points / BRep PSE+PSF as Line-Sets. Default is '.npz'")
    parser_labelBRepBndry2Scan.set_defaults(func=brep2meshL.labelBRepBndry2Scan)
     
    parser_labelBRepJunction2Scan = subparsers.add_parser("labelBRepJunction2Scan",
        help="Label the Edges + Faces of CAD w.r.t faces, Edges and Loops of BReps")
    parser_labelBRepJunction2Scan.add_argument("--step_path", type=str, required=True, 
                        help="Path to load the step files from")
    parser_labelBRepJunction2Scan.add_argument("--scan_path", type=str, required=True, 
                        help="Path to load the 3D scan files from")
    parser_labelBRepJunction2Scan.add_argument("--output", type=str, required=True,
        help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points" ,)
    parser_labelBRepJunction2Scan.add_argument("--nProc", type=int, default=1, 
                        help="Number of worker threads",)
    parser_labelBRepJunction2Scan.add_argument("--feature_list", type=str, required=False,
        help="Optional path to the feature lists",)
    parser_labelBRepJunction2Scan.add_argument("--mesh_dir", type=str, required=False,
                        help="Optionally cross check with Fusion Gallery mesh files to check the segmentation labels",)
    parser_labelBRepJunction2Scan.add_argument("--seg_dir", type=str, required=False,
        help="Optionally provide a directory containing segmentation labels seg files.",)
    parser_labelBRepJunction2Scan.add_argument("--fRegen", type=bool, required=False, default=False,
        help="To regenerate the stepFiles",)
    parser_labelBRepJunction2Scan.add_argument("--scale_brep", type=bool, required=False, default=True,
        help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_labelBRepJunction2Scan.add_argument("--infmt", required=False, 
                        choices=(['.stp', '.step']), default='.step', 
                        help="Format of STEP files. Default is '.step'")
    parser_labelBRepJunction2Scan.add_argument("--ofmt", required=False, 
                        choices=(['.npz', '.pkl', '.ply',]), default='.npz', 
                        help="Format of labels of Scan Points / BRep PSE+PSF as Line-Sets. Default is '.npz'")
    parser_labelBRepJunction2Scan.set_defaults(func=brep2meshL.labelBRepJunction2Scan)
    
    
    parser_labelBRepJunction2ScanABC = subparsers.add_parser("labelBRepJunction2ScanABC",
        help="Label the Edges + Faces of CAD w.r.t faces, Edges and Loops of BReps")
    parser_labelBRepJunction2ScanABC.add_argument("--step_path", type=str, required=True, 
                        help="Path to load the step files from")
    parser_labelBRepJunction2ScanABC.add_argument("--scan_path", type=str, required=True, 
                        help="Path to load the 3D scan files from")
    parser_labelBRepJunction2ScanABC.add_argument("--output", type=str, required=True,
        help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points" ,)
    parser_labelBRepJunction2ScanABC.add_argument("--nProc", type=int, default=1, 
                        help="Number of worker threads",)
    parser_labelBRepJunction2ScanABC.add_argument("--feature_list", type=str, required=False,
        help="Optional path to the feature lists",)
    parser_labelBRepJunction2ScanABC.add_argument("--mesh_dir", type=str, required=False,
                        help="Optionally cross check with Fusion Gallery mesh files to check the segmentation labels",)
    parser_labelBRepJunction2ScanABC.add_argument("--seg_dir", type=str, required=False,
        help="Optionally provide a directory containing segmentation labels seg files.",)
    parser_labelBRepJunction2ScanABC.add_argument("--fRegen", type=bool, required=False, default=False,
        help="To regenerate the stepFiles",)
    parser_labelBRepJunction2ScanABC.add_argument("--scale_brep", type=bool, required=False, default=True,
        help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_labelBRepJunction2ScanABC.add_argument("--infmt", required=False, 
                        choices=(['.stp', '.step']), default='.step', 
                        help="Format of STEP files. Default is '.step'")
    parser_labelBRepJunction2ScanABC.add_argument("--ofmt", required=False, 
                        choices=(['.npz', '.pkl', '.ply',]), default='.npz', 
                        help="Format of labels of Scan Points / BRep PSE+PSF as Line-Sets. Default is '.npz'")
    parser_labelBRepJunction2ScanABC.set_defaults(func=brep2meshL.labelBRepJunction2ScanABC)
    
    parser_labelBRepBndry2ScanABC = subparsers.add_parser("labelBRepBndry2ScanABC",
        help="Label the Edges + Faces of CAD w.r.t faces, Edges and Loops of BReps")
    parser_labelBRepBndry2ScanABC.add_argument("--step_path", type=str, required=True, 
                        help="Path to load the step files from")
    parser_labelBRepBndry2ScanABC.add_argument("--scan_path", type=str, required=True, 
                        help="Path to load the 3D scan files from")
    parser_labelBRepBndry2ScanABC.add_argument("--output", type=str, required=True,
        help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points" ,)
    parser_labelBRepBndry2ScanABC.add_argument("--nProc", type=int, default=1, 
                        help="Number of worker threads",)
    parser_labelBRepBndry2ScanABC.add_argument("--feature_list", type=str, required=False,
        help="Optional path to the feature lists",)
    parser_labelBRepBndry2ScanABC.add_argument("--mesh_dir", type=str, required=False,
                        help="Optionally cross check with Fusion Gallery mesh files to check the segmentation labels",)
    parser_labelBRepBndry2ScanABC.add_argument("--seg_dir", type=str, required=False,
        help="Optionally provide a directory containing segmentation labels seg files.",)
    parser_labelBRepBndry2ScanABC.add_argument("--fRegen", type=bool, required=False, default=False,
        help="To regenerate the stepFiles",)
    parser_labelBRepBndry2ScanABC.add_argument("--scale_brep", type=bool, required=False, default=True,
        help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_labelBRepBndry2ScanABC.add_argument("--infmt", required=False, 
                        choices=(['.stp', '.step']), default='.stp', 
                        help="Format of STEP files. Default is '.step'")
    parser_labelBRepBndry2ScanABC.add_argument("--ofmt", required=False, 
                        choices=(['.npz', '.pkl', '.ply']), default='.npz', 
                        help="Format of labels of Scan Points / BRep PSE+PSF as Line-Sets. Default is '.npz'")
    parser_labelBRepBndry2ScanABC.set_defaults(func=brep2meshL.labelBRepBndry2ScanABC)
    
    
    parser_labelBRepFace2ScanABC = subparsers.add_parser("labelBRepFace2ScanABC", 
        help="Label the Edges + Faces of CAD w.r.t faces, Edges of BReps",)
    parser_labelBRepFace2ScanABC.add_argument("--step_path", type=str, required=True, 
        help="Path to load the step files from")
    parser_labelBRepFace2ScanABC.add_argument("--scan_path", type=str, required=True,
        help="Path to load the 3D scan files from",)
    parser_labelBRepFace2ScanABC.add_argument("--output", type=str, required=True,
        help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points",)
    parser_labelBRepFace2ScanABC.add_argument("--nProc", type=int, default=1,
        help="Number of worker threads",)
    parser_labelBRepFace2ScanABC.add_argument("--feature_list", type=str, required=False,
        help="Optional path to the feature lists",)
    parser_labelBRepFace2ScanABC.add_argument("--fRegen", type=bool, required=False, default=False,
        help="To regenerate the stepFiles",)
    parser_labelBRepFace2ScanABC.add_argument("--scale_brep", type=bool, required=False, default=True,
        help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_labelBRepFace2ScanABC.add_argument("--debug", action="store_true",
        help="Whether to save the intermediate point clouds with labels as colors",)
    parser_labelBRepFace2ScanABC.add_argument("--infmt", required=False, choices=([".stp", ".step"]), default=".stp",
        help="Format of STEP files. Default is '.step'",)
    parser_labelBRepFace2ScanABC.add_argument("--infmt_scan", required=False, choices=([".ply", ".obj", ".npz"]), default=".obj",
        help="Format of PCL/mesh. Default is '.obj'",)
    parser_labelBRepFace2ScanABC.add_argument("--ofmt", required=False, choices=([".npz"]), default=".npz",
        help="Format of labels of Scan Points / BRep PSE+PSF as Line-Sets. Default is '.npz'",)
    parser_labelBRepFace2ScanABC.set_defaults(func=brep2meshL.labelBRepFace2ScanABC)


    parser_labelBRepFace2Scan = subparsers.add_parser("labelBRepFace2Scan", help="Label the Edges + Faces of CAD w.r.t faces, Edges of BReps",)
    parser_labelBRepFace2Scan.add_argument("--step_path", type=str, required=True, help="Path to load the step files from")
    parser_labelBRepFace2Scan.add_argument("--scan_path", type=str, required=True, help="Path to load the 3D scan files from",)
    parser_labelBRepFace2Scan.add_argument("--output", type=str, required=True, help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points",)
    parser_labelBRepFace2Scan.add_argument("--nProc", type=int, default=1, help="Number of worker threads",)
    parser_labelBRepFace2Scan.add_argument("--feature_list", type=str, required=False, help="Optional path to the feature lists",)
    parser_labelBRepFace2Scan.add_argument("--fRegen", type=bool, required=False, default=False, help="To regenerate the stepFiles",)
    parser_labelBRepFace2Scan.add_argument("--scale_brep", type=bool, required=False, default=True, help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_labelBRepFace2Scan.add_argument("--debug", action="store_true", help="Whether to save the intermediate point clouds with labels as colors",)
    parser_labelBRepFace2Scan.add_argument("--infmt", required=False, choices=([".stp", ".step"]), default=".stp", help="Format of STEP files. Default is '.step'",)
    parser_labelBRepFace2Scan.add_argument("--infmt_scan", required=False,choices=([".ply", ".stl", ".npz"]), default=".npz", help="Format of PCL/mesh. Default is '.npz'",)
    parser_labelBRepFace2Scan.add_argument("--ofmt", required=False, choices=([".npz"]), default=".npz", help="Format of labels of Scan Points / BRep PSE+PSF as Line-Sets. Default is '.npz'",)
    parser_labelBRepFace2Scan.set_defaults(func=brep2meshL.labelBRepFace2Scan)

    parser_labelBRepFaceTypes = subparsers.add_parser("labelBRepFaceTypes", help="Label the Face types of BReps",)
    parser_labelBRepFaceTypes.add_argument("--step_path", type=str, required=True, help="Path to load the step files from")
    parser_labelBRepFaceTypes.add_argument("--output", type=str, required=True, help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points",)
    parser_labelBRepFaceTypes.add_argument("--nProc", type=int, default=1, help="Number of worker threads",)
    parser_labelBRepFaceTypes.add_argument("--feature_list", type=str, required=False, help="Optional path to the feature lists",)
    parser_labelBRepFaceTypes.add_argument("--fRegen", type=bool, required=False, default=False, help="To regenerate the stepFiles",)
    parser_labelBRepFaceTypes.add_argument("--scale_brep", type=bool, required=False, default=True, help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_labelBRepFaceTypes.add_argument("--debug", action="store_true", help="Whether to save the intermediate point clouds with labels as colors",)
    parser_labelBRepFaceTypes.add_argument("--infmt", required=False, choices=([".stp", ".step"]), default=".stp", help="Format of STEP files. Default is '.step'",)
    parser_labelBRepFaceTypes.add_argument("--infmt_scan", required=False, choices=([".ply", ".stl", ".npz"]), default=".npz", help="Format of PCL/mesh. Default is '.npz'",)
    parser_labelBRepFaceTypes.add_argument("--ofmt", required=False, choices=([".facetype", ".ftype"]), default=".facetype", help="Format of BRep face type label file. Default is '.facetype'",)
    parser_labelBRepFaceTypes.set_defaults(func=brep2meshL.labelBRepFaceTypes)


    parser_labelBRepFaceQualityCheck = subparsers.add_parser("labelBRepFaceQualityCheck", help="Label the Edges + Faces of CAD w.r.t faces, Edges of BReps",)
    parser_labelBRepFaceQualityCheck.add_argument("--step_path", type=str, required=True, help="Path to load the step files from")
    parser_labelBRepFaceQualityCheck.add_argument("--scan_path", type=str, required=True, help="Path to load the 3D scan files from",)
    parser_labelBRepFaceQualityCheck.add_argument("--label_path", type=str, required=True, help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points",)
    parser_labelBRepFaceQualityCheck.add_argument("--nProc", type=int, default=1, help="Number of worker threads",)
    parser_labelBRepFaceQualityCheck.add_argument("--feature_list", type=str, required=False, help="Optional path to the feature lists",)
    parser_labelBRepFaceQualityCheck.add_argument("--fRegen", type=bool, required=False, default=False, help="To regenerate the stepFiles",)
    parser_labelBRepFaceQualityCheck.add_argument("--scale_brep", type=bool, required=False, default=True, help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_labelBRepFaceQualityCheck.add_argument("--debug", action="store_true", help="Whether to save the intermediate point clouds with labels as colors",)
    parser_labelBRepFaceQualityCheck.add_argument("--infmt", required=False, choices=([".stp", ".step"]), default=".step", help="Format of STEP files. Default is '.step'",)
    parser_labelBRepFaceQualityCheck.add_argument("--infmt_scan", required=False, choices=([".ply", ".stl", ".npz"]), default=".ply", help="Format of PCL/mesh. Default is '.npz'",)
    parser_labelBRepFaceQualityCheck.add_argument("--infmt_label", required=False, choices=([".npz"]), default=".npz", help="Format of labels of Scan Points / BRep PSE+PSF as Line-Sets. Default is '.npz'",)
    parser_labelBRepFaceQualityCheck.set_defaults(func=brep2meshL.labelBRepFaceQualityCheck)

    args = parser.parse_args()
    #help message is displayed when no command is provided.
    if "func" not in args:
        parser.print_help()
        sys.exit(1)
        
    return args

def main():
    args = _parse_args()
    args.func(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    
# Example Command --> python3 -m BRep2CADLabler labelBRepJunction2Scan --step_path /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_step/test/batch_01/ --scan_path /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_fusion/test/batch_01/ --output /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_BrepAnnot_JunctionPts/test/batch_01/ --nProc 64 --infmt .step --ofmt .npz
# Example Command --> python3 -m BRep2CADLabler labelBRepJunction2Scan --step_path /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_step/test/batch_01/ --scan_path /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_fusion/test/batch_01/ --output /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_BrepAnnot_JunctionPts/test/batch_01/ --nProc 64 --infmt .step --ofmt .ply




