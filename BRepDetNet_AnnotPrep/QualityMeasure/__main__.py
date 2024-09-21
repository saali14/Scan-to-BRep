import argparse
import logging
import os
import sys
import time
import pathlib
import glob
import open3d as o3d 
import numpy as np
import gc
import copy

sys.path.insert(0, '..')

from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
from tqdm.contrib import tzip
from concurrent.futures import ProcessPoolExecutor

from BRep2CADLabler.BRep2MeshLabler import BRep2MeshLabler

logger = logging.getLogger(__name__)
BRep2ScanAnnot_CDs = []
BRep2Cad_CDs = []



def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist

def scale_mesh_to_unit_box(inMesh):
    iMesh = copy.deepcopy(inMesh)
    
    mn = iMesh.get_min_bound()
    mx = iMesh.get_max_bound()
    
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
    
    iMesh.translate(np.asarray(orig) - np.asarray(center))
    iMesh.scale(2.0/longest_length, [0,0,0])
    
    return iMesh

def run_workerAnnotQuality(worker_args):
    brepfile = worker_args[0]
    cadfile = worker_args[1]
    scanfile = worker_args[2]
    annotFile = worker_args[3]
    outdir = worker_args[4]
    scaleMesh = bool(worker_args[4])
    lf=open('logQuality.txt','a')
    
    if os.path.exists(brepfile) and os.path.exists(scanfile):
        if os.path.exists(cadfile) and os.path.exists(annotFile):
            brepBndryAnnot = np.load(annotFile, allow_pickle=True)
            brep = o3d.io.read_point_cloud(brepfile)
            scan = o3d.io.read_triangle_mesh(scanfile)
            cad =  o3d.io.read_triangle_mesh(cadfile)
            
            if scaleMesh is True:
                scan = scale_mesh_to_unit_box(scan)
                cad = scale_mesh_to_unit_box(cad)
                
            x_scanAnnot = np.asarray(scan.vertices)[np.array(brepBndryAnnot["v_vertex_Ids"], dtype=np.int32)]
            x_cad = np.asarray(cad.vertices)
            y_brep = np.asarray(brep.points)
            BRep2ScanAnnot_CD = chamfer_distance(x_scanAnnot, y_brep, metric='l2', direction='y_to_x')
            BRep2Cad_CD = chamfer_distance(x_cad, y_brep, metric='l2', direction='y_to_x')
            BRep2ScanAnnot_CDs.append(BRep2ScanAnnot_CD)
            BRep2Cad_CDs.append(BRep2Cad_CD)
            lf.write("%1.5f, %1.5f\n"%(BRep2ScanAnnot_CD, BRep2Cad_CD))
        else:
            logger.info("Either {cadfile} Or {annotFile} is NOT found")
            return
    else:
        logger.info("Either {brepFile} Or {scanfile} is NOT found")
        return
    
def ComputeAnnotationQuality(args):
        """
        NOTE: labelling/annotating the scan points w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs 
        """
        brep_dir = args.brep_path
        annot_dir = args.annot_path
        cad_dir = args.cad_path
        scan_dir = args.scan_path
        out_dir  = args.output_path
        nProc = args.nProc
        inBrepfmt = args.inBrepfrmt
        inMeshfrmt = args.inMeshfrmt
        inAnnotfmt = args.inAnnotfrmt
        scale_mesh = args.scale_mesh
            
        logger.info("Computing Annotation Quality...")
        logger.info(f"input BREP files dir = {brep_dir}")
        logger.info(f"input Annotations files dir = {annot_dir}")
        logger.info(f"input Mesh (CAD) files dir = {cad_dir}")
        logger.info(f"input 3D-Scan files dir = {scan_dir}")
        logger.info(f"output dir = {out_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"iBRepfmt = {inBrepfmt}")
        logger.info(f"inAnnotfrmt = {inAnnotfmt}")
        logger.info(f"inMeshfrmt = {inMeshfrmt}")
        logger.info(f"scale_cad = {scale_mesh}")
        
        BRepfiles = glob.glob(str(args.brep_path) + '/**/*{}'.format(inBrepfmt), recursive=True)
        CADFiles = [f.replace(str(args.brep_path), str(cad_dir)).replace(inBrepfmt, inMeshfrmt) for f in BRepfiles]
        ScanFiles = [f.replace(str(args.brep_path), str(scan_dir)).replace(inBrepfmt, inMeshfrmt) for f in BRepfiles]
        AnnotFiles = [f.replace(str(args.brep_path), str(annot_dir)).replace(inBrepfmt, inAnnotfmt) for f in BRepfiles]
        
        logger.info(f" found scans {len(BRepfiles)}")
        
        use_many_threads =  nProc > 1
        if use_many_threads:
            worker_args = [(br, cd, sc, at, out_dir, scale_mesh) for br, cd, sc, at in zip(BRepfiles, CADFiles, ScanFiles, AnnotFiles)]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(tqdm(executor.map(run_workerAnnotQuality, worker_args), total=len(worker_args)))
        else:
            brep2meshL =  BRep2MeshLabler("Scan_Labler")
            for br, cd, sc, at in tzip(BRepfiles, CADFiles, ScanFiles, AnnotFiles):
                brep2meshL.transferBndryLabel_brep2Scan(br, cd, sc, at, out_dir, scale_mesh)
        
        gc.collect()
        
def _parse_args():    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    
    parser_ComputeAnnotationQuality = subparsers.add_parser("ComputeAnnotationQuality",
        help="Label the Edges + Faces of CAD w.r.t faces, Edges and Loops of BReps")
    parser_ComputeAnnotationQuality.add_argument("--brep_path", type=str, required=True, 
                        help="Path to load the step files from")
    parser_ComputeAnnotationQuality.add_argument("--annot_path", type=str, required=True, 
                        help="Path to load the Boundary files from")
    parser_ComputeAnnotationQuality.add_argument("--cad_path", type=str, required=True, 
                        help="Path to load the 3D CAD files from")
    parser_ComputeAnnotationQuality.add_argument("--scan_path", type=str, required=True, 
                        help="Path to load the 3D scan files from")
    parser_ComputeAnnotationQuality.add_argument("--output_path", type=str, required=True,
        help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points" ,)
    parser_ComputeAnnotationQuality.add_argument("--nProc", type=int, default=1, 
                        help="Number of worker threads",)
    parser_ComputeAnnotationQuality.add_argument("--scale_mesh", type=bool, required=False, default=True,
        help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_ComputeAnnotationQuality.add_argument("--inBrepfrmt", required=False, 
                        choices=(['.stp', '.step', '.ply']), default='.ply', 
                        help="Format of STEP files.")
    parser_ComputeAnnotationQuality.add_argument("--inAnnotfrmt", required=False, 
                        choices=(['.npz', '.ply']), default='.npz', 
                        help="Format of Boundary Annotation is '.npz'")
    parser_ComputeAnnotationQuality.add_argument("--inMeshfrmt", required=False, 
                        choices=(['.ply', '.obj']), default='.ply', 
                        help="Format of Scans/CAD Mesh files. Default is '.ply'")
    parser_ComputeAnnotationQuality.add_argument("--ofmt", required=False, 
                        choices=(['.txt', '.png', '.ply']), default='.txt', 
                        help="Format of output score file")
    parser_ComputeAnnotationQuality.set_defaults(func=ComputeAnnotationQuality)
    
    
    parser_ComputeAnnotationQuality = subparsers.add_parser("CheckBndaryFacePtRatio",
        help="Check the boundary vs FacePoint ratio")
    parser_ComputeAnnotationQuality.add_argument("--brep_path", type=str, required=True, 
                        help="Path to load the step files from")
    parser_ComputeAnnotationQuality.add_argument("--annot_path", type=str, required=True, 
                        help="Path to load the Boundary files from")
    parser_ComputeAnnotationQuality.add_argument("--scan_path", type=str, required=True, 
                        help="Path to load the 3D scan files from")
    parser_ComputeAnnotationQuality.add_argument("--output_path", type=str, required=True,
        help="Path to the save intermediate brep data as PSE + PSF & Labelled Scan Points" ,)
    parser_ComputeAnnotationQuality.add_argument("--nProc", type=int, default=1, 
                        help="Number of worker threads",)
    parser_ComputeAnnotationQuality.add_argument("--scale_mesh", type=bool, required=False, default=True,
        help="To normalize the BRep body in [-1, 1]^3 box",)
    parser_ComputeAnnotationQuality.add_argument("--inBrepfrmt", required=False, 
                        choices=(['.stp', '.step', '.ply']), default='.ply', 
                        help="Format of STEP files.")
    parser_ComputeAnnotationQuality.add_argument("--inAnnotfrmt", required=False, 
                        choices=(['.npz', '.ply']), default='.npz', 
                        help="Format of Boundary Annotation is '.npz'")
    parser_ComputeAnnotationQuality.add_argument("--inMeshfrmt", required=False, 
                        choices=(['.ply', '.obj']), default='.ply', 
                        help="Format of Scans/CAD Mesh files. Default is '.ply'")
    parser_ComputeAnnotationQuality.add_argument("--ofmt", required=False, 
                        choices=(['.txt', '.png', '.ply']), default='.txt', 
                        help="Format of output score file")
    parser_ComputeAnnotationQuality.set_defaults(func=ComputeAnnotationQuality)
    
    
    
    args = parser.parse_args()

    #help message is displayed when no command is provided.
    if "func" not in args:
        parser.print_help()
        sys.exit(1)
        
    return args

def main():
    args = _parse_args()
    args.func(args)
    
    np.savetxt('QualityBRep2SA.txt', BRep2ScanAnnot_CDs)
    np.savetxt('QualityBRep2CAD.txt', BRep2Cad_CDs)
    
    #with open('QualityBRep2SA.txt', 'w') as f:
    #    for s in validSampleIds:
    #        f.write(f"{s}\n")
    
    #with open('QualityBRep2CAD.txt', 'w') as f:
    #for s in validSampleIds:
        #f.write(f"{s}\n")
        
    #Threslds = []
    #quality =[]
    #PercOfModels = []
    #for v in 
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()

# Example Command 
# python3 -m QualityMeasure ComputeAnnotationQuality --brep_path /home/ali/Desktop/trial_CC3D/cc3d_v1.0_BREPEdges/test/batch_01/ --annot_path /home/ali/Desktop/trial_CC3D/cc3d_v1.0_BoundaryLabels/test/batch_01/ --cad_path /home/ali/Desktop/trial_CC3D/cc3d_v1.0_cad/test/batch_01/ --scan_path /home/ali/Desktop/trial_CC3D/cc3d_v1.0_scan/test/batch_01/  --output_path /home/ali/Desktop/trial_CC3D/cc3d_v1.0_STATS/test/batch_01/ --nProc 6
# python3 -m QualityMeasure ComputeAnnotationQuality --brep_path /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_BRepBoundary/test/batch_01/ --annot_path /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_BoundaryLabels/test/batch_01/ --cad_path /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_ply/test/batch_01/ --scan_path /data/3d_cluster/SHARP2022/Challenge2/Base-CC3D-PSE/cc3d_v1.0_fusion/test/batch_01/  --output_path ./ --nProc 40

