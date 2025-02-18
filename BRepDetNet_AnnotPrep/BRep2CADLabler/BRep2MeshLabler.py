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

from OCC.Extend import TopologyUtils
from concurrent.futures import ProcessPoolExecutor
import logging
logger = logging.getLogger(__name__)

class BRep2MeshLabler:
    def __init__(self, name):
        self.className = str(name)
        return
        
    def load_json(self, pathname):
        with open(pathname, "r") as fp:
            return json.load(fp)

    def crosscheck_file_correspondences(self, step_file, mesh_file):
        if not os.path.exists(step_file):        
            print(f"Warning!! Step File {step_file} is missing")
            return False
        else:
            if not os.path.exists(mesh_file):
                print(f"Warning!! Mesh  File {mesh_file} is missing")
                return False
            else:
                # Nothing to check
                return True
    
    def check_face_indices(self, step_file, mesh_dir):
        if mesh_dir is None:
            # Nothing to check
            return True
        # Check against the given meshes and Fusion labels
        validator = FaceIndexValidator(step_file, mesh_dir)
        return validator.validate()

    def crosscheck_faces_and_seg_file(self, infile, seg_dir):
        seg_pathname = None
        if seg_dir is None:
            # Look to see if the seg file is in the step dir
            step_dir = infile.parent
            trial_seg_pathname = step_dir / (infile.stem + ".seg")
            if trial_seg_pathname.exists():
                seg_pathname = trial_seg_pathname
        else:
            # We expect to find the segmentation file in the
            # seg dir
            seg_pathname = seg_dir / (infile.stem + ".seg")
            if not seg_pathname.exists():
                print(f"Warning!! Segmentation file {seg_pathname} is missing")
                return False

        if seg_pathname is not None:
            checker = SegmentationFileCrosschecker(infile, seg_pathname)
            data_ok = checker.check_data()
            if not data_ok:
                print(
                    f"Warning!! Segmentation file {seg_pathname} and step file {infile} have different numbers of faces"
                )
            return data_ok

        # In the case where we don't know the seg pathname we don't do
        # any extra checking
        return True

    def transferBndryLabel_brep2Scan(self, brepExtr, inMeshFile, outLabelFile):
        if os.path.exists(inMeshFile) == False:
            logger.info(f"Input Scan file {inMeshFile} Does not exist")
            return

        all_edgetypes = [
            "line",
            "ellipse",
            "hyperbola",
            "parabola",
            "bspline",
            "bezier",
            "circle",
            "offset",
        ]

        # Load the body from the STEP file
        brepBody = brepExtr.load_body_from_step()
        if brepBody is None:
            return 
        
        scanMesh = o3d.io.read_triangle_mesh(inMeshFile)
        
        # ------------------- Folloing Code Generates HD Scan by ubSampling -----------------# 
        # hdScanPath = os.path.split(outLabelFile.replace("abc_v1.0_BRepBoundary", "obj").replace("_step_", "_obj_"))[0] + "/" + os.path.basename(inMeshFile)
        # if pathlib.Path(hdScanPath).parent.exists():
        #     logger.warning(f"folder exists, not creating {pathlib.Path(hdScanPath).parent}")
        #     logger.info(f"generating HDScan {hdScanPath}")
        #     if pathlib.Path(hdScanPath).exists():
        #         logger.warning(f"HDScan Exists, not creating {pathlib.Path(hdScanPath)}")
        #         scanMesh = o3d.io.read_triangle_mesh(hdScanPath) 
        # else:  
        #     scanMesh = scanMesh.subdivide_loop(number_of_iterations=1)
        #     scanMesh.compute_vertex_normals()
        #     pathlib.Path(hdScanPath).parent.mkdir(parents=True, exist_ok=True)
        #     logger.info(f"generating HDScan {hdScanPath}")
        #     o3d.io.write_triangle_mesh(hdScanPath, scanMesh, write_ascii=True, write_triangle_uvs=False)
        # ------------------- Previous Code Generates HD Scan by ubSampling -----------------#

        origScanMesh = copy.deepcopy(scanMesh) 

        # We want to apply a transform so that the solid is centered on the origin and scaled into a box [-1, 1]^3
        if brepExtr.scale_body == True:
            brepBody, center, scale = scale_utils.scale_solid_to_unit_box(brepBody)
            if brepBody is None:
                return
            scanMesh = scale_utils.scale_mesh(scanMesh, center, scale)

        top_exp = TopologyUtils.TopologyExplorer(brepBody, ignore_orientation=True)
        if not brepExtr.check_manifold(top_exp):
            logger.warning(f"Non-manifold bodies are not supported")
            return
        if not brepExtr.check_closed(brepBody):
            logger.warning(f"Bodies which are not closed are not supported")
            return
        if not brepExtr.check_unique_coedges(top_exp):
            logger.warning(f"Bodies where the same coedge is used in multiple loops are not supported")
            return
        if pathlib.Path(outLabelFile).parent.exists():
            logger.warning(f"folder exists, not creating {pathlib.Path(outLabelFile).parent}")
            logger.info(f"generating BRep {outLabelFile}")
            if pathlib.Path(outLabelFile).exists():
                logger.warning(f"Annotation File exists, not creating {pathlib.Path(outLabelFile)}")
                return 
        else:
            pathlib.Path(outLabelFile).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"generating BRep {outLabelFile}")

        entity_mapper = EntityMapper(brepBody)
        coedge_point_grids = brepExtr.extract_coedge_point_grids(brepBody, entity_mapper)
        #face_point_grids = brepExtr.extract_face_point_grids(brepBody, entity_mapper)
        edge_feats = brepExtr.extract_edge_info_from_body(brepBody, entity_mapper)
        
        choiceList = ['parent_edge', 'closed_loop', 'closed_mate_loop', 'mate_face1', 'mate_face2', 'edge_type']
        selected = 'parent_edge' 
        
        if coedge_point_grids is not None: 
            # @Ali NOTE: nxt , mate, face, edge, loops and scales all array has dimension [num_coedges, 1]
            nxt, mate, face, edge, loops = brepExtr.build_incidence_arrays(brepBody, entity_mapper)
            
            brep_v_close_loop_Id = []
            brep_v_close_mate_loop_Id = []
            brep_v_parentedge_Id = []
            brep_v_mateface_Id1 = []
            brep_v_mateface_Id2 = []
            brep_v_edfeats = []
            
            brep_v_parentedge_Color = []
            brep_v_close_loop_Id_Color = []
            brep_v_close_mate_loop_Id_Color = []
            brep_v_mateface_Id1_Color = []
            brep_v_mateface_Id2_Color = []
            brep_v_edgetype_Color = []

            clBreps = o3d.geometry.PointCloud()

            distinct_clrs_map = []
            # if selected == 'parent_edge':
            #     distinct_clrs = distinctipy.get_colors(len(np.unique(edge))) 
            #     distinct_clrs_map = [distinct_clrs[e] for e in edge]
            # if selected == 'closed_loop':
            #     distinct_clrs = distinctipy.get_colors(len(np.unique(loops)))
            #     distinct_clrs_map = [distinct_clrs[l] for l in loops]
            # if selected == 'closed_mate_loop':
            #     distinct_clrs = distinctipy.get_colors(len(np.unique(loops)))
            #     distinct_clrs_map = [distinct_clrs[loops[m]] for m in mate]
            # if selected == 'mate_face1':
            #     distinct_clrs = distinctipy.get_colors(len(np.unique(face)))
            #     distinct_clrs_map = [distinct_clrs[f] for f in face]
            # if selected == 'mate_face2':
            #     distinct_clrs = distinctipy.get_colors(len(np.unique(face)))
            #     distinct_clrs_map = [distinct_clrs[face[m]] for m in mate]
            # if selected == 'edge_type':
            #     distinct_clrs = data_utils.generate_colors(len(all_edgetypes) + 1)
            #     distinct_clrs_map = []
            #     for e in edge:
            #         if ("type" in edge_feats[e]) and (edge_feats[e]["type"] in all_edgetypes):
            #             distinct_clrs_map.append(distinct_clrs[all_edgetypes.index(edge_feats[e]["type"])])
            #         else:
            #             distinct_clrs_map.append(distinct_clrs[len(all_edgetypes)])

            if selected == 'parent_edge':
                distinct_clrs = data_utils.generate_colors(np.asarray(list(set(edge))).shape[0])
                distinct_clrs_map = [distinct_clrs[e] for e in edge]
            if selected == 'closed_loop':
                distinct_clrs = data_utils.generate_colors(np.asarray(list(set(loops))).shape[0])
                distinct_clrs_map = [distinct_clrs[l] for l in loops]
            if selected == 'closed_mate_loop':
                distinct_clrs = data_utils.generate_colors(np.asarray(list(set(loops))).shape[0])
                distinct_clrs_map = [distinct_clrs[loops[m]] for m in mate]
            if selected == 'mate_face1':
                distinct_clrs = data_utils.generate_colors(np.asarray(list(set(face))).shape[0])
                distinct_clrs_map = [distinct_clrs[f] for f in face]
            if selected == 'mate_face2':
                distinct_clrs = data_utils.generate_colors(np.asarray(list(set(face))).shape[0])
                distinct_clrs_map = [distinct_clrs[face[m]] for m in mate]


            #@Ali NOTE: Following For-loop segment all POINTS of B-Rep by loop/mate/edge/parent-edge "Ids"
            for c in range(0, len(coedge_point_grids)): #coedge_point_grids.shape[0]):
                ce = o3d.geometry.PointCloud()
                #ce.points = o3d.utility.Vector3dVector(((coedge_point_grids[c])[:3, :]).transpose(1,0).reshape(-1,3))
                ce.points = o3d.utility.Vector3dVector((coedge_point_grids[c])[:, :3])
                if selected == 'parent_edge':
                    ce.paint_uniform_color(distinct_clrs_map[c])
                if selected == 'closed_loop':
                    ce.paint_uniform_color(distinct_clrs_map[c])
                if selected == 'closed_mate_loop':
                    ce.paint_uniform_color(distinct_clrs_map[c])
                if selected == 'mate_face1':
                    ce.paint_uniform_color(distinct_clrs_map[c])
                if selected == 'mate_face2':
                    ce.paint_uniform_color(distinct_clrs_map[c])
                if selected == 'edge_type':
                    ce.paint_uniform_color(distinct_clrs_map[c])
                
                clBreps += ce
                for pts in range(0, len(np.asarray(ce.points))):
                    brep_v_parentedge_Id.append(edge[c])
                    brep_v_close_loop_Id.append(loops[c])
                    brep_v_close_mate_loop_Id.append(loops[mate[c]])
                    brep_v_mateface_Id1.append(face[c])
                    brep_v_mateface_Id2.append(face[mate[c]])
                    brep_v_edfeats.append(edge_feats[edge[c]]) 
                    
                    if selected == 'parent_edge':
                        brep_v_parentedge_Color.append(distinct_clrs_map[c])
                    if selected == 'closed_loop':
                        brep_v_close_loop_Id_Color.append(distinct_clrs_map[c])
                    if selected == 'closed_mate_loop':
                        brep_v_close_mate_loop_Id_Color.append(distinct_clrs_map[c])
                    if selected == 'mate_face1':
                        brep_v_mateface_Id1_Color.append(distinct_clrs_map[c])
                    if selected == 'mate_face2':
                        brep_v_mateface_Id2_Color.append(distinct_clrs_map[c])
                    if selected == 'edge_type':
                        brep_v_edgetype_Color.append(distinct_clrs_map[c])

            cached_bRepPts = np.asarray(clBreps.points)
            cached_bRepPtClrs = np.asarray(clBreps.colors)
            
            nnIdx, nn_dists = data_utils.find_knn(cached_bRepPts, np.asarray(scanMesh.vertices))
            if nnIdx is None or nn_dists is None:
                return 
            
            # NOTE: These are the  
            scan_bndryIdx = []
            scan_bndryColors = []
            scan_close_loop_Id = []
            scan_close_mate_loop_Id = []
            scan_parentedge_Id = []
            scan_mateface_Id1 = []
            scan_mateface_Id2 = []
            scan_edgefeats = []
            
            for v in range(0, len(scanMesh.vertices)):
                if nn_dists[v] < 0.008:
                    scan_bndryIdx.append(v)
                    scan_close_loop_Id.append(brep_v_close_loop_Id[nnIdx[v]])
                    scan_close_mate_loop_Id.append(brep_v_close_mate_loop_Id[nnIdx[v]])
                    scan_parentedge_Id.append(brep_v_parentedge_Id[nnIdx[v]])
                    scan_mateface_Id1.append(brep_v_mateface_Id1[nnIdx[v]])
                    scan_mateface_Id2.append(brep_v_mateface_Id2[nnIdx[v]])
                    scan_edgefeats.append(brep_v_edfeats[nnIdx[v]])
                    
                    if selected == 'parent_edge':
                        scan_bndryColors.append(brep_v_parentedge_Color[nnIdx[v]])
                    if selected == 'closed_loop':
                        scan_bndryColors.append(brep_v_close_loop_Id_Color[nnIdx[v]])
                    if selected == 'closed_mate_loop':
                        scan_bndryColors.append(brep_v_close_mate_loop_Id_Color[nnIdx[v]])
                    if selected == 'mate_face1':
                        scan_bndryColors.append(brep_v_mateface_Id1_Color[nnIdx[v]])
                    if selected == 'mate_face2':
                        scan_bndryColors.append(brep_v_mateface_Id2_Color[nnIdx[v]])
                    
            
            #NOTE --> some fields are filled by dummy values, when NOT implemented!
            dummy_scan_proximity_weigts = np.zeros(len(scan_bndryIdx))
            dummy_scan_memberedge_type = np.zeros(len(scan_bndryIdx))
            #"""
            LbldataStr = LabelBoundaryDataStruct()
            if len(scan_bndryIdx) > 0:
                directory = os.path.dirname(outLabelFile)
                filename = os.path.splitext(os.path.basename(outLabelFile))[0]
                extension = os.path.splitext(os.path.basename(outLabelFile))[1]

                # @Ali NOTE: Saving the Annotations 
                if extension == '.npz':
                    ## NOTE --> related to Closed-Edges!
                    #"""
                    LbldataStr.dataBndr.update({"v_vertex_Ids": scan_bndryIdx})
                    LbldataStr.dataBndr.update({"v_close_loop_Id": scan_close_loop_Id})
                    LbldataStr.dataBndr.update({"v_close_mate_loop_Id": scan_close_mate_loop_Id})
                    LbldataStr.dataBndr.update({"v_parentedge_Id": scan_parentedge_Id})
                    LbldataStr.dataBndr.update({"v_memberedge_type": dummy_scan_memberedge_type})
                    LbldataStr.dataBndr.update({"v_proximity_weigts": dummy_scan_proximity_weigts})
                    
                    ## NOTE --> related to Closed-Faces!
                    LbldataStr.dataBndr.update({"v_mateface_Id1": scan_mateface_Id1})
                    LbldataStr.dataBndr.update({"v_mateface_Id2": scan_mateface_Id2})
                    LbldataStr.dataBndr.update({"v_edgefeats": scan_edgefeats}) #NOTE <-- newly added
                    LbldataStr.save_npz_data_BRep2CAD_BoundaryVertsLabel(outLabelFile, LbldataStr.dataBndr)
                    #"""
                    return
                elif extension == '.ply':
                    # @Ali NOTE: --> One can choose Any One of the Scalars |--> Color Map
                    #"""
                    #parentEdgeID_cmap = data_utils.apply_colormap2scalar(scan_parentedge_Id)
                    labeledScanPCD = o3d.geometry.PointCloud()
                    labeledScanPCD.points = o3d.utility.Vector3dVector(np.take(np.asarray(origScanMesh.vertices), np.asarray(scan_bndryIdx), axis=0))
                    labeledScanPCD.colors = o3d.utility.Vector3dVector(np.asarray(scan_bndryColors).reshape(-1,3))  
                                      
                    #labeledScanPCD.paint_uniform_color([0.5, 0.5, 0.5])
                    #for ix in range(0, len(labeledScanPCD.points)):
                        #labeledScanPCD.colors[ix] = parentEdgeID_cmap[ix]
                    o3d.io.write_point_cloud(outLabelFile, 
                                             labeledScanPCD, 
                                             write_ascii=False)
                    logger.info(f"Saving only Colored Scan Points {outLabelFile}")
                    #"""
                    return 
                elif extension == '.pkl':
                    logger.info(f"Warning: saving is .pkl format is not supported -- saving only B-Rep Points")
                    logger.info(f"Saving only Colored B-Rep Points {outLabelFile}")
                    o3d.io.write_point_cloud(directory + "/" + filename + ".ply", 
                                             clBreps.scale(1.0/scale, [0, 0, 0]).translate(np.array(center)),
                                             write_ascii=False)
                else:
                    raise NotImplementedError
        
        return

    def transferJunctLabel_brep2Scan(self, brepExtr, inMeshFile, outLabelFile):
        if os.path.exists(inMeshFile) == False:
            logger.info(f"Input Scan file {inMeshFile} Does not exist")
            return
        
        # Load the body from the STEP file
        brepBody = brepExtr.load_body_from_step()
        if brepBody is None:
            return 
        
        scanMesh = o3d.io.read_triangle_mesh(inMeshFile)
        origScanMesh = copy.deepcopy(scanMesh)
        # We want to apply a transform so that the solid is centered on the origin and scaled into a box [-1, 1]^3
        if brepExtr.scale_body == True:
            brepBody, center, scale = scale_utils.scale_solid_to_unit_box(brepBody)
            if brepBody is None:
                return 
            scanMesh = scale_utils.scale_mesh(scanMesh, center, scale)
        
        top_exp = TopologyUtils.TopologyExplorer(brepBody, ignore_orientation=True)
        if not brepExtr.check_manifold(top_exp):
            logger.info(f"Non-manifold bodies are not supported")
            return
        if not brepExtr.check_closed(brepBody):
            logger.info(f"Bodies which are not closed are not supported")
            return
        if not brepExtr.check_unique_coedges(top_exp):
            logger.info(f"Bodies where the same coedge is used in multiple loops are not supported")
            return
        if pathlib.Path(outLabelFile).parent.exists():
            logger.warning(f"folder exists, not creating {pathlib.Path(outLabelFile).parent}")
            logger.info(f"generating BRep {outLabelFile}")
            if pathlib.Path(outLabelFile).exists():
                logger.warning(f"Annotation File exists, not creating {pathlib.Path(outLabelFile)}")
                return 
        else:
            pathlib.Path(outLabelFile).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"generating BRep {outLabelFile}")

        entity_mapper = EntityMapper(brepBody)
        
        coedge_point_grids = brepExtr.extract_coedge_point_grids(brepBody, entity_mapper)
        #face_point_grids = brepExtr.extract_face_point_grids(brepBody, entity_mapper)
        if coedge_point_grids is not None: 
            nxt, mate, face, edge, loops = brepExtr.build_incidence_arrays(brepBody, entity_mapper)  
        
            brep_j_close_loop_Ids = []
            
            brep_j_parentedge_Id = []
            brep_j_nextedge_Id = []
            
            brep_j_mateface_Id1 = []
            brep_j_mateface_Id2 = []
            
            brep_j_nextmateface_Id1 = []
            brep_j_nextmateface_Id2 = []
            
            clBreps = o3d.geometry.PointCloud()
            
            # iterating over different Coed edges  ----> shape[0] denote number of co-edges 
            for c in range(0, len(coedge_point_grids)): 
                ce = o3d.geometry.PointCloud()
                
                # every Co-Edge grid has 100 sampled points as set fixed in "BRepExtractor" 
                # we take the First and the Last point of every Co-Edge
                idxx = [0, len(coedge_point_grids[c])-1]
                ce.points = o3d.utility.Vector3dVector(np.take(np.asarray((coedge_point_grids[c])[:, :3]), idxx, axis=0))
                clBreps += ce
                
                for pts in range(0, len(np.asarray(ce.points))):
                    brep_j_close_loop_Ids.append(loops[c])
                    
                    brep_j_parentedge_Id.append(edge[c])
                    brep_j_nextedge_Id.append(edge[nxt[c]])
                    
                    brep_j_mateface_Id1.append(face[c])
                    brep_j_mateface_Id2.append(face[mate[c]])
                    
                    brep_j_nextmateface_Id1.append(face[nxt[c]])
                    brep_j_nextmateface_Id2.append(face[mate[nxt[c]]])
            
            # Union of incident Faces + Edges + Loops 
            cached_bRepPts = np.asarray(clBreps.points)

            logger.info(f"generating BRep {outLabelFile}")
            nnIdx, nn_dists = data_utils.find_knn(cached_bRepPts, np.asarray(scanMesh.vertices))
            if nnIdx is None or nn_dists is None:
                return 
            
            # NOTE: These are the  
            scan_jncIdx = []
            scan_close_loop_Ids = []
            scan_parentedge_Ids = []
            scan_mateface_Id1s = []
            scan_mateface_Id2s = []
            for v in range(0, len(scanMesh.vertices)):
                if nn_dists[v] < 0.02:
                    scan_jncIdx.append(v)
                    scan_close_loop_Ids.append(brep_j_close_loop_Ids[nnIdx[v]])
                    scan_parentedge_Ids.append(brep_j_parentedge_Id[nnIdx[v]])
                    scan_mateface_Id1s.append(brep_j_mateface_Id1[nnIdx[v]])
                    scan_mateface_Id2s.append(brep_j_mateface_Id2[nnIdx[v]])
                    
            
            #NOTE --> some fields are filled by dummy values, when NOT implemented!
            #dummy_scan_proximity_weigts = np.zeros(len(scan_jncIdx))
            #dummy_scan_memberedge_types = np.zeros(len(scan_jncIdx))
            
            
            if len(scan_jncIdx) > 0:
                directory = os.path.dirname(outLabelFile)
                filename = os.path.splitext(os.path.basename(outLabelFile))[0]
                extension = os.path.splitext(os.path.basename(outLabelFile))[1]

                # @Ali NOTE: Saving the Annotations 
                if extension == '.npz':
                    LbldataStr = LabelBoundaryDataStruct()
                    # NOTE --> related to Closed-Edges!
                    LbldataStr.dataJunc.update({"j_vertex_Ids": scan_jncIdx})
                    LbldataStr.dataJunc.update({"j_close_loop_Ids": scan_close_loop_Ids})
                    LbldataStr.dataJunc.update({"j_parentedge_Ids": scan_parentedge_Ids})
                    #LbldataStr.dataJunc.update({"j_memberedge_types": dummy_scan_memberedge_types})
                    #LbldataStr.dataJunc.update({"j_proximity_weigts": dummy_scan_proximity_weigts})
                    
                    # NOTE --> related to Closed-Faces!
                    LbldataStr.dataJunc.update({"j_mateface_Id1s": scan_mateface_Id1s})
                    LbldataStr.dataJunc.update({"j_mateface_Id2s": scan_mateface_Id2s})
                    LbldataStr.save_npz_data_BRep2CAD_JunctionVertsLabel(outLabelFile, LbldataStr.dataJunc)
                
                elif extension == '.ply':
                    # @Ali NOTE: --> One can choose Any One of the Scalars |--> Color Map
                    labeledScanPCD = o3d.geometry.PointCloud()
                    labeledScanPCD.points = o3d.utility.Vector3dVector(np.take(np.asarray(origScanMesh.vertices), np.asarray(scan_jncIdx), axis=0))
                    labeledScanPCD.paint_uniform_color([0.0, 0.9, 0.0])
                    o3d.io.write_point_cloud(outLabelFile, labeledScanPCD, write_ascii=False)
                    logger.info(f"Saving only Colored Scan Points {outLabelFile}")            
                
                elif extension == '.pkl':
                    logger.info(f"Warning: saving is .pkl format is not supported -- saving only B-Rep Points")
                    logger.info(f"Saving only Colored B-Rep Points {outLabelFile}")
                    o3d.io.write_point_cloud(directory + "/" + filename + ".ply", 
                                             clBreps.scale(1.0/scale, [0, 0, 0]).translate(np.array(center)),
                                             write_ascii=False)
                
                else:
                    # @Ali NOTE: Hidden Flag to Save B-Reps as .ply files 
                    #logger.info(f"Saving only Colored B-Rep Points {outLabelFile}")
                    #o3d.io.write_point_cloud(directory + "/" + filename + ".ply", clBreps, write_ascii=False)
                    raise NotImplementedError
        return

    def transferFaceLabel_brep2Scan(self, brepExtr, inMeshFile, outLabelFile, debug):
        nn_threshold = 0.009  # TODO define as argument
        nb_sampled_pts = 200000  # TODO define as argument
        selected_debug = [
            "face_membership",
            "face_type",
            "face_loop",
        ] 
        all_facetypes = [
            "Plane",
            "Cylinder",
            "Cone",
            "Sphere",
            "Torus",
            "Bezier_surface",
            "BSplineSurface",
            "Others",
        ]
        # Plane --> 0 ; Cylinder --> 1 ; Cone --> 2 ; Sphere --> 3 ; Torus --> 4 ; Bezier_surface --> 5 ; BSplineSurface --> 6 ; Others --> 7
        
        # Load npz/ply pcl and convert it to open3D format
        inMesh = o3d.geometry.PointCloud()
        if inMeshFile.endswith('.npz'): 
            pcl_npz = data_utils.load_npz(inMeshFile)
            pcl_mesh = o3d.geometry.PointCloud()
            pcl_mesh.points = o3d.utility.Vector3dVector(pcl_npz.vertices)
            pcl_mesh.normals = o3d.utility.Vector3dVector(pcl_npz.vertex_normals)
            pcl_mesh.colors = o3d.utility.Vector3dVector(pcl_npz.vertex_colors)
        elif inMeshFile.endswith('.ply'):
            pcl_mesh = o3d.io.read_point_cloud(inMeshFile)
            inMesh.points = pcl_mesh.points
        elif inMeshFile.endswith('.obj'):
            pcl_mesh = o3d.io.read_triangle_mesh(inMeshFile)
            inMesh.points = pcl_mesh.vertices
        else:
            print("Unknown Scan File Format")
            return
        
        # Load the body from the STEP file
        brepBody = brepExtr.load_body_from_step()

        if len(inMesh.points) < 3:
            return
            
        # We want to apply a transform so that the solid
        # is centered on the origin and scaled so it just fits
        # into a box [-1, 1]^3
        if brepExtr.scale_body:
            brepBody, center, scale = scale_utils.scale_solid_to_unit_box(brepBody)
            if brepBody is None:
                return
            scalepcl_mesh = scale_utils.scale_mesh(inMesh, center, scale)
        else:
            scalepcl_mesh = inMesh

        top_exp = TopologyUtils.TopologyExplorer(brepBody, ignore_orientation=True)
        if not brepExtr.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return
        if not brepExtr.check_closed(brepBody):
            print("Bodies which are not closed are not supported")
            return
        if not brepExtr.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is used in multiple loops are not supported")
            return
        if pathlib.Path(outLabelFile).parent.exists():
            logger.info(f"generating BRep {outLabelFile}")
        else:
            logger.info(f"folder does not exist, creating {pathlib.Path(outLabelFile).parent}")
            pathlib.Path(outLabelFile).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"generating BRep {outLabelFile}")

        # Extract the IDs, Parameters, and UV Grids of the faces from the Brep
        entity_mapper = EntityMapper(brepBody)
        face_id, face_feat, face_uv_grid = brepExtr.extract_face_info_from_body(brepBody, entity_mapper)
        # face_uv_grid --> list of len num_faces. Each one 7 x num_pts_u x num_pts_v
        # face_feat --> [ num_faces x dict]
        # face_id --> [ num_faces x 1]
        # Transfer the labels to the scans by NN
        # convert to open3D point cloud the uv grid
        if face_id is not None:
            pts = []
            face_types = []
            face_feats = []
            face_ids = []
            members_clrs_map = []
            types_clrs_map = []
            brep_v_facemember_Color = []
            brep_v_typeface_Color = []

            if "face_membership" in selected_debug:
                distinct_clrs = [
                            [1.00, 0.67, 0.60],
                            [0.00, 0.00, 0.70],
                            [1.00, 1.00, 0.40],
                            [1.00, 0.60, 0.80],
                            [0.10, 1.00, 1.00],
                            [0.75, 0.70, 1.00],
                            [1.00, 0.90, 0.70],
                            [0.40, 0.70, 1.00],
                            [0.60, 0.00, 0.30],
                            [0.90, 1.00, 0.70],
                            [0.40, 0.00, 0.40]]
                if len(face_id) > 11:
                    hsv_colors_extra = [[i / (len(face_id) - 11) + 0.27 , 1, 1] for i in range(len(face_id) - 11)]
                    hsv_colors_extras = [(hsv_to_rgb(hsv_color)).tolist() for hsv_color in hsv_colors_extra]
                    distinct_clrs.extend(hsv_colors_extras)

                members_clrs_map = [
                    distinct_clrs[f] for f in range(len(face_uv_grid))
                ]
            if "face_type" in selected_debug:
                distinct_clrs = [
                [1.00, 0.47, 0.20], # Plane
                [1.00, 0.87, 0.20], # Cylinder
                [0.67, 1.00, 0.00], # Cone
                [0.00, 0.70, 0.58], # Sphere
                [0.10, 1.00, 1.00], # Torus
                [0.00, 0.58, 0.70], # BezierSurface
                [0.00, 0.33, 1.00], # BSplineSurface
                [0.50, 0.40, 1.00], # Other
                ]
                types_clrs_map = [distinct_clrs[e] for e in range(len(all_facetypes))]

            for fidx in range(len(face_uv_grid)):
                for u in range(face_uv_grid[fidx].shape[1]):
                    for v in range(face_uv_grid[fidx].shape[2]):
                        if (face_uv_grid[fidx][6, u, v] == 1):  # Get only non-masked points of the grid
                            pts.append(face_uv_grid[fidx][:3, u, v])
                            face_ids.append(fidx)
                            if "type" in face_feat[fidx]: 
                                if face_feat[fidx]["type"] not in all_facetypes:
                                    f_type = "Others"
                                else:
                                    f_type = face_feat[fidx]["type"]
                            else:
                                f_type = "Others"
                            
                            face_types.append(all_facetypes.index(f_type))
                            face_feats.append(face_feat[fidx])
                            if "face_membership" in selected_debug:
                                brep_v_facemember_Color.append(members_clrs_map[fidx])
                            if "face_type" in selected_debug:
                                brep_v_typeface_Color.append(types_clrs_map[all_facetypes.index(f_type)])

            nnIdx, nn_dists = data_utils.find_knn(np.asarray(pts), np.asarray(scalepcl_mesh.points))
            scan_close_face_ids = []
            scan_close_face_feats = []
            scan_close_face_types = []
            scan_face_v_ID = []
            scan_proximity_weights = []
            scan_faceMembersColors = []
            scan_faceTypesColors = []

            for v in range(0, len(inMesh.points)):
                scan_face_v_ID.append(v)
                scan_close_face_ids.append(face_ids[nnIdx[v]])
                scan_close_face_feats.append(face_feats[nnIdx[v]])
                scan_close_face_types.append(face_types[nnIdx[v]])
                scan_proximity_weights.append(nn_dists[v])
                if "face_membership" in selected_debug:
                    scan_faceMembersColors.append(brep_v_facemember_Color[nnIdx[v]])
                if "face_type" in selected_debug:
                    scan_faceTypesColors.append(brep_v_typeface_Color[nnIdx[v]])

            assert len(scan_face_v_ID) == len(inMesh.points)

            LbldataStr = LabelFaceDataStruct()
            if len(scan_face_v_ID) > 0:
                extension = os.path.splitext(os.path.basename(outLabelFile))[1]
                if extension == ".npz":
                    LbldataStr.data.update({"v_vertex_Ids": scan_face_v_ID})
                    LbldataStr.data.update({"v_face_Ids": scan_close_face_ids})
                    LbldataStr.data.update({"v_face_type": scan_close_face_types})
                    LbldataStr.data.update({"v_face_feat": scan_close_face_feats})
                    LbldataStr.save_npz_data_BRep2CAD_FaceVertsLabel(outLabelFile, LbldataStr.data)
                else:
                    NotImplementedError

                if debug == True:
                    logger.info(f"Debug mode... Saving Colored Scan Points {outLabelFile}")
                    labeledScanPCD_members = o3d.geometry.PointCloud()
                    labeledScanPCD_members.points = o3d.utility.Vector3dVector(np.take(np.asarray(inMesh.points), np.asarray(scan_face_v_ID), axis=0,))
                    labeledScanPCD_members.colors = o3d.utility.Vector3dVector(np.asarray(scan_faceMembersColors).reshape(-1, 3))
                    o3d.io.write_point_cloud(os.path.splitext(outLabelFile)[0] + "-debug-members.ply", labeledScanPCD_members, write_ascii=False,)
                    labeledScanPCD_types = o3d.geometry.PointCloud()
                    labeledScanPCD_types.points = o3d.utility.Vector3dVector(np.take(np.asarray(inMesh.points), np.asarray(scan_face_v_ID), axis=0,))
                    labeledScanPCD_types.colors = o3d.utility.Vector3dVector(np.asarray(scan_faceTypesColors).reshape(-1, 3))
                    o3d.io.write_point_cloud(os.path.splitext(outLabelFile)[0] + "-debug-types.ply", labeledScanPCD_types, write_ascii=False,)
            return
        else:
            return        
    
    def run_worker_face(self, worker_args):
        inStepfile = worker_args[0]
        inMeshFile = worker_args[1]
        outLabelFile = worker_args[2]
        feature_schema = worker_args[3]
        debug = worker_args[4]
        scale_brep = worker_args[5]
        
        #if not check_face_indices(inStepfile, mesh_dir):
        #    return
        #if not crosscheck_faces_and_seg_file(inStepfile, seg_dir):
        #    return
        brepExtr = BRepExtractor(inStepfile, outLabelFile, feature_schema, scale_brep)
        self.transferFaceLabel_brep2Scan(brepExtr, inMeshFile, outLabelFile, debug)
    
    def run_workerBoundaryLabel(self, worker_args):
        inStepfile = worker_args[0]
        inMeshFile = worker_args[1]
        outLabelFile = worker_args[2]
        feature_schema = worker_args[3]
        mesh_dir = worker_args[4]
        seg_dir = worker_args[5]
        scale_brep = worker_args[6]

        if not self.crosscheck_file_correspondences(inStepfile, inMeshFile):
            return
        
        brepExtr = BRepExtractor(inStepfile, outLabelFile, feature_schema, scale_brep)
        self.transferBndryLabel_brep2Scan(brepExtr, inMeshFile, outLabelFile) 
            
        #if not crosscheck_faces_and_seg_file(inStepfile, seg_dir):
        #    return
    
    def run_workerJunctionLabel(self, worker_args):
        inStepfile = worker_args[0]
        inMeshFile = worker_args[1]
        outLabelFile = worker_args[2]
        feature_schema = worker_args[3]
        mesh_dir = worker_args[4]
        seg_dir = worker_args[5]
        scale_brep = worker_args[6]
        
        if not self.crosscheck_file_correspondences(inStepfile, inMeshFile):
            return
        
        brepExtr = BRepExtractor(inStepfile, outLabelFile, feature_schema, scale_brep)
        self.transferJunctLabel_brep2Scan(brepExtr, inMeshFile, outLabelFile)
        
    def run_workerPSELabel(self, worker_args):
        inStepfile = worker_args[0]
        inPSEfile = worker_args[1]
        outLabelFile = worker_args[2]
        feature_schema = worker_args[3]
        scale_brep = worker_args[6]
        
        if not self.crosscheck_file_correspondences(inStepfile, inPSEfile):
            return
        
        brepExtr = BRepExtractor(inStepfile, outLabelFile, feature_schema, scale_brep)
        self.transferBndryLabel_brep2PSE(brepExtr, inPSEfile, outLabelFile)
        data_utils.filterPSECC3D(inPSEfile, outLabelFile)
                
    def filter_out_files_which_are_already_converted(self, files, output_StepFiles):
        files_to_convert = []
        for f, s in zip(files, output_StepFiles):
            if not os.path.exists(s):
                files_to_convert.append(f)    
        return files_to_convert
        
    def labelBRepFaceTypes(self, args):
        """
        NOTE: labelling/annotating the scan points w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs
        """
        step_dir = args.step_path
        out_dir = args.output
        nProc = args.nProc
        inStepfmt = args.infmt
        outLabelFrmt = args.ofmt
        force_regeneration = args.fRegen
        scale_brep = args.scale_brep
        debug = args.debug

        featr_dir = None
        if args.feature_list is not None:
            featr_dir = pathlib.Path(args.feature_list)

        logger.info("generating partial data in directory tree")
        logger.info(f"input STEP files dir = {step_dir}")
        logger.info(f"output LABEL dir = {out_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"infmt = {inStepfmt}")
        logger.info(f"ofmt = {outLabelFrmt}")
        logger.info(f"scale_brep = {scale_brep}")
        logger.info(f"Debug mode is {debug}")

        """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
        parent_folder = pathlib.Path(__file__).parent.parent
        if featr_dir is None:
            featr_dir = parent_folder / "feature_lists/all.json"
        feature_schema = self.load_json(featr_dir)

        Stepfiles = sorted(glob.glob(str(args.step_path) + '/**/*{}'.format(inStepfmt), recursive=True))
        outLabelFiles = [f.replace(str(args.step_path), str(args.output)).replace(args.infmt, args.ofmt) for f in Stepfiles]

        logger.info(f" found steps {len(Stepfiles)}")
        if not force_regeneration:
            Stepfiles = self.filter_out_files_which_are_already_converted(Stepfiles, outLabelFiles)

        use_many_threads = nProc > 1
        if use_many_threads:
            worker_args = [
                (st, ol, feature_schema, debug, scale_brep)
                for st, ol in zip(Stepfiles, outLabelFiles)
            ]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(
                    tqdm(
                        executor.map(self.run_worker_faceTypes, worker_args),
                        total=len(worker_args),
                    )
                )
        else:
            for st, ol in tzip(Stepfiles, outLabelFiles):
                brepExtr = BRepExtractor(st, ol, feature_schema, scale_brep)
                self.transferFacetypeLabel(brepExtr, ol, debug)

        gc.collect()

    def labelBRepFaceQualityCheck(self, args):
        """
        NOTE: labelling/annotating the scan points w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs
        """
        step_dir = args.step_path
        scan_dir = args.scan_path
        label_dir = args.label_path
        nProc = args.nProc
        inStepfmt = args.infmt
        inScanfmt = args.infmt_scan
        inLabelFrmt = args.infmt_label
        force_regeneration = args.fRegen
        scale_brep = args.scale_brep
        debug = args.debug

        featr_dir = None
        if args.feature_list is not None:
            featr_dir = pathlib.Path(args.feature_list)

        logger.info("generating partial data in directory tree")
        logger.info(f"input STEP files dir = {step_dir}")
        logger.info(f"input SCAN files dir = {scan_dir}")
        logger.info(f"input LABEL dir = {label_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"infmt = {inStepfmt}")
        logger.info(f"inlabelfmt = {inLabelFrmt}")
        logger.info(f"scale_brep = {scale_brep}")
        logger.info(f"Debug mode is {debug}")

        """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
        parent_folder = pathlib.Path(__file__).parent.parent
        if featr_dir is None:
            featr_dir = parent_folder / "feature_lists/all.json"
        feature_schema = self.load_json(featr_dir)

        Stepfiles = sorted(glob.glob(str(args.step_path) + '/**/*{}'.format(args.infmt), recursive=True))
        ScanFiles = [f.replace(str(args.step_path), str(args.scan_path)).replace(args.infmt, ".ply") for f in Stepfiles]
        LabelFiles = [f.replace(str(args.step_path), str(args.label_path)).replace(args.infmt, args.infmt_label) for f in Stepfiles]

        logger.info(f" found steps {len(Stepfiles)}")
        logger.info(f" found scans {len(ScanFiles)}")
        logger.info(f" found labels {len(LabelFiles)}")

        use_many_threads = nProc > 1
        if use_many_threads:
            worker_args = [
                (st, sc, il, feature_schema, debug, scale_brep)
                for st, sc, il in zip(Stepfiles, ScanFiles, LabelFiles)
                ]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(
                    tqdm(
                        executor.map(self.run_worker_faceQualityCheck, worker_args),
                        total=len(worker_args),)
                    )
        else:
            for st, sc, il in tzip(Stepfiles, ScanFiles, LabelFiles):
                brepExtr = BRepExtractor(st, il, feature_schema, scale_brep)
                self.FaceLabel_QualityCheck(brepExtr, sc, il, debug)
        gc.collect()

    def labelBRepBndry2Scan(self, args):
        """
        NOTE: labelling/annotating the scan points w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs 
        """
        step_dir = args.step_path
        scan_dir = args.scan_path
        out_dir  = args.output
        nProc = args.nProc
        inStepfmt = args.infmt
        outLabelFrmt = args.ofmt
        force_regeneration = args.fRegen
        scale_brep = args.scale_brep
        
        meshsegm_dir = None
        if args.mesh_dir is not None:
            meshsegm_dir = Path(args.mesh_dir)
        segLabel_dir = None
        if args.seg_dir is not None:
            segLabel_dir = Path(args.seg_dir)
        featr_dir = None
        if args.feature_list is not None:
            featr_dir = Path(args.feature_list)
            
        logger.info("generating partial data in directory tree")
        logger.info(f"input STEP files dir = {step_dir}")
        logger.info(f"input SCAN files dir = {scan_dir}")
        logger.info(f"output LABEL dir = {out_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"infmt = {inStepfmt}")
        logger.info(f"ofmt = {outLabelFrmt}")
        logger.info(f"scale_brep = {scale_brep}")
        
        """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
        parent_folder = pathlib.Path(__file__).parent.parent
        if featr_dir is None:
            featr_dir = parent_folder / "feature_lists/all.json"
        feature_schema = self.load_json(featr_dir)
        
        Stepfiles = glob.glob(str(args.step_path) + '/**/*{}'.format(inStepfmt), recursive=True)
        ScanFiles = [f.replace(str(args.step_path), str(args.scan_path)).replace(args.infmt, ".ply") for f in Stepfiles]
        outLabelFiles = [f.replace(str(args.step_path), str(args.output)).replace(args.infmt, args.ofmt) for f in Stepfiles]
        
        logger.info(f" found scans {len(Stepfiles)}")
        if not force_regeneration:
            Stepfiles = self.filter_out_files_which_are_already_converted(Stepfiles, outLabelFiles)
            ScanFiles = [f.replace(str(args.step_path), str(args.scan_path)).replace(args.infmt, ".ply") for f in Stepfiles]
            outLabelFiles = [f.replace(str(args.step_path), str(args.output)).replace(args.infmt, args.ofmt) for f in Stepfiles]
        
        use_many_threads =  nProc > 1
        if use_many_threads:
            worker_args = [(st, sc, ol, feature_schema, meshsegm_dir, segLabel_dir, scale_brep) for st, sc, ol in zip(Stepfiles, ScanFiles, outLabelFiles)]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(tqdm(executor.map(self.run_workerBoundaryLabel, worker_args), total=len(worker_args)))
        else:
            for st, sc, ol in tzip(Stepfiles, ScanFiles, outLabelFiles):
                self.transferBndryLabel_brep2Scan(st, sc, ol, feature_schema, meshsegm_dir, segLabel_dir, scale_brep)

        gc.collect()
        
    def labelBRepJunction2Scan(self, args):
        """
        NOTE: labelling/annotating the scan points w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs 
        """
        step_dir = args.step_path
        scan_dir = args.scan_path
        out_dir  = args.output
        nProc = args.nProc
        inStepfmt = args.infmt
        outLabelFrmt = args.ofmt
        force_regeneration = args.fRegen
        scale_brep = args.scale_brep
        
        meshsegm_dir = None
        if args.mesh_dir is not None:
            meshsegm_dir = pathlib.Path(args.mesh_dir)
        segLabel_dir = None
        if args.seg_dir is not None:
            segLabel_dir = pathlib.Path(args.seg_dir)
        featr_dir = None
        if args.feature_list is not None:
            featr_dir = pathlib.Path(args.feature_list)
            
        logger.info("generating partial data in directory tree")
        logger.info(f"input STEP files dir = {step_dir}")
        logger.info(f"input SCAN files dir = {scan_dir}")
        logger.info(f"output LABEL dir = {out_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"infmt = {inStepfmt}")
        logger.info(f"ofmt = {outLabelFrmt}")
        logger.info(f"scale_brep = {scale_brep}")
        
        """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
        parent_folder = pathlib.Path(__file__).parent.parent
        if featr_dir is None:
            featr_dir = parent_folder / "feature_lists/all.json"
        feature_schema = self.load_json(featr_dir)
        
        Stepfiles = glob.glob(str(args.step_path) + '/**/*{}'.format(inStepfmt), recursive=True)
        ScanFiles = [f.replace(str(args.step_path), str(args.scan_path)).replace(args.infmt, ".ply") for f in Stepfiles]
        outLabelFiles = [(os.path.splitext(f.replace(str(args.step_path), str(args.output)))[0] + os.path.splitext(f.replace(str(args.step_path), str(args.output)))[1]).replace(args.infmt, args.ofmt) for f in Stepfiles]
        
        logger.info(f" found scans {len(Stepfiles)}")
        if not force_regeneration:
            Stepfiles = self.filter_out_files_which_are_already_converted(Stepfiles, outLabelFiles)
            ScanFiles = [f.replace(str(args.step_path), str(args.scan_path)).replace(args.infmt, ".ply") for f in Stepfiles]
            outLabelFiles = [(os.path.splitext(f.replace(str(args.step_path), str(args.output)))[0] + os.path.splitext(f.replace(str(args.step_path), str(args.output)))[1]).replace(args.infmt, args.ofmt) for f in Stepfiles]
        
        use_many_threads = nProc > 1
        if use_many_threads:
            worker_args = [(st, sc, ol, feature_schema, meshsegm_dir, segLabel_dir, scale_brep) for st, sc, ol in zip(Stepfiles, ScanFiles, outLabelFiles)]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(tqdm(executor.map(self.run_workerJunctionLabel, worker_args), total=len(worker_args)))
        else:
            for st, sc, ol in tzip(Stepfiles, ScanFiles, outLabelFiles):
                brepExtr = BRepExtractor(st, ol, feature_schema, scale_brep)
                self.transferJunctLabel_brep2Scan(brepExtr, sc, ol)

        gc.collect()
    
    def labelBRepFace2Scan(self, args):
        """
        NOTE: labelling/annotating the scan points w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs
        """
        step_dir = args.step_path
        scan_dir = args.scan_path
        out_dir = args.output
        nProc = args.nProc
        inStepfmt = args.infmt
        inScanfmt = args.infmt_scan
        outLabelFrmt = args.ofmt
        force_regeneration = args.fRegen
        scale_brep = args.scale_brep
        debug = args.debug

        featr_dir = None
        if args.feature_list is not None:
            featr_dir = pathlib.Path(args.feature_list)

        logger.info("generating partial data in directory tree")
        logger.info(f"input STEP files dir = {step_dir}")
        logger.info(f"input SCAN files dir = {scan_dir}")
        logger.info(f"output LABEL dir = {out_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"infmt = {inStepfmt}")
        logger.info(f"ofmt = {outLabelFrmt}")
        logger.info(f"scale_brep = {scale_brep}")
        logger.info(f"Debug mode is {debug}")

        """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
        parent_folder = pathlib.Path(__file__).parent.parent
        if featr_dir is None:
            featr_dir = parent_folder / "feature_lists/all.json"
        feature_schema = self.load_json(featr_dir)

        Stepfiles = sorted(glob.glob(str(args.step_path) + '/**/*{}'.format(inStepfmt), recursive=True))
        ScanFiles = [f.replace(str(args.step_path), str(args.scan_path)).replace(args.infmt, ".ply") for f in Stepfiles]
        outLabelFiles = [f.replace(str(args.step_path), str(args.output)).replace(args.infmt, args.ofmt) for f in Stepfiles]

        logger.info(f" found steps {len(Stepfiles)}")
        logger.info(f" found scans {len(ScanFiles)}")
        if not force_regeneration:
            Stepfiles = self.filter_out_files_which_are_already_converted(Stepfiles, outLabelFiles)

        use_many_threads = nProc > 1
        if use_many_threads:
            worker_args = [
                (st, sc, ol, feature_schema, debug, scale_brep)
                for st, sc, ol in zip(Stepfiles, ScanFiles, outLabelFiles)
            ]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(
                    tqdm(
                        executor.map(self.run_worker_face, worker_args),
                        total=len(worker_args),
                    )
                )
        else:
            for st, sc, ol in tzip(Stepfiles, ScanFiles, outLabelFiles):
                brepExtr = BRepExtractor(st, ol, feature_schema, scale_brep)
                self.transferFaceLabel_brep2Scan(brepExtr, sc, ol, debug)

        gc.collect()
    
    def labelBRep2PSE(self, args):
        """
        NOTE: labelling/annotating the Parametric Sharp Edges (PSE) w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs 
        """
        step_dir = args.step_path
        pseScan_dir = args.pse_path
        out_dir  = args.output
        nProc = args.nProc
        inStepfmt = args.infmt
        outLabelFrmt = args.ofmt
        scale_brep = args.scale_brep
        force_regeneration = args.fRegen
        
        meshsegm_dir = None
        if args.mesh_dir is not None:
            meshsegm_dir = pathlib.Path(args.mesh_dir)
        segLabel_dir = None
        if args.seg_dir is not None:
            segLabel_dir = pathlib.Path(args.seg_dir)
        featr_dir = None
        if args.feature_list is not None:
            featr_dir = pathlib.Path(args.feature_list)
        
        logger.info("generating partial data in directory tree")
        logger.info(f"input STEP files dir = {step_dir}")
        logger.info(f"input SCAN files dir = {pseScan_dir}")
        logger.info(f"output LABEL dir = {out_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"infmt = {inStepfmt}")
        logger.info(f"ofmt = {outLabelFrmt}")
        logger.info(f"scale_brep = {scale_brep}")
        
        """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
        parent_folder = pathlib.Path(__file__).parent.parent
        if featr_dir is None:
            featr_dir = parent_folder / "feature_lists/all.json"
        feature_schema = self.load_json(featr_dir)
        
        # NOTE -- reverse order of files because 
        pseScanFiles = glob.glob(str(pseScan_dir) + '/**/*{}'.format(".ply"),recursive=True)
        Stepfiles = [f.replace(str(pseScan_dir), str(step_dir)).replace(".ply", args.infmt) for f in pseScanFiles]
        outLabelFiles = [f.replace(str(pseScan_dir), str(out_dir)).replace(args.infmt, ".ply") for f in pseScanFiles]
        
        logger.info(f" found scans {len(pseScanFiles)}")
        #if not force_regeneration:
            #pseScanFiles = self.filter_out_files_which_are_already_converted(pseScanFiles, outLabelFiles)
        
        use_many_threads =  nProc > 1
        if use_many_threads:
            worker_args = [(st, sc, ol, feature_schema, meshsegm_dir, segLabel_dir, scale_brep) for st, sc, ol in zip(Stepfiles, pseScanFiles, outLabelFiles)]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(tqdm(executor.map(self.run_workerPSELabel, worker_args), total=len(worker_args)))
        else:
            for st, sc, ol in tzip(Stepfiles, pseScanFiles, outLabelFiles):
                brepExtr = BRepExtractor(st, ol, feature_schema, scale_brep)
                self.transferBndryLabel_brep2PSE(brepExtr, sc, ol)
                
        gc.collect()
        
    def labelBRepFace2ScanABC(self, args):
        """
        NOTE: labelling/annotating the scan points w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs
        """
        step_dir = args.step_path
        scan_dir = args.scan_path
        out_dir = args.output
        nProc = args.nProc
        inStepfmt = args.infmt
        inScanfmt = args.infmt_scan
        outLabelFrmt = args.ofmt
        force_regeneration = args.fRegen
        scale_brep = args.scale_brep
        debug = args.debug

        featr_dir = None
        if args.feature_list is not None:
            featr_dir = pathlib.Path(args.feature_list)

        logger.info("generating partial data in directory tree")
        logger.info(f"input STEP files dir = {step_dir}")
        logger.info(f"input SCAN files dir = {scan_dir}")
        logger.info(f"output LABEL dir = {out_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"infmt = {inStepfmt}")
        logger.info(f"ofmt = {outLabelFrmt}")
        logger.info(f"scale_brep = {scale_brep}")
        logger.info(f"Debug mode is {debug}")

        """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
        parent_folder = pathlib.Path(__file__).parent.parent
        if featr_dir is None:
            featr_dir = parent_folder / "feature_lists/all.json"
        feature_schema = self.load_json(featr_dir)

        Stepfiles = glob.glob(str(args.step_path) + '/**/*{}'.format(inStepfmt), recursive=True)
        ScanFiles = [os.path.split(f.replace(str(args.step_path), str(args.scan_path)).replace("_step_", "_obj_"))[0] + "/" + os.path.basename(f).replace("_step_", "_trimesh_").replace(args.infmt, ".obj") for f in Stepfiles]
        outLabelFiles = [f.replace(str(args.step_path), str(args.output)).replace(args.infmt, args.ofmt) for f in Stepfiles]
        
        logger.info(f" found steps {len(Stepfiles)}")
        logger.info(f" found scans {len(ScanFiles)}")
        
        if not force_regeneration:
            Stepfiles = self.filter_out_files_which_are_already_converted(Stepfiles, outLabelFiles)
            ScanFiles = [os.path.split(f.replace(str(args.step_path), str(args.scan_path)).replace("_step_", "_obj_"))[0] + "/" + os.path.basename(f).replace("_step_", "_trimesh_").replace(args.infmt, ".obj") for f in Stepfiles]
            outLabelFiles = [f.replace(str(args.step_path), str(args.output)).replace(args.infmt, args.ofmt) for f in Stepfiles]

        use_many_threads = nProc > 1
        if use_many_threads:
            worker_args = [(st, sc, ol, feature_schema, debug, scale_brep) for st, sc, ol in zip(Stepfiles, ScanFiles, outLabelFiles)]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(tqdm(executor.map(self.run_worker_face, worker_args), total=len(worker_args),))
        else:
            for st, sc, ol in tzip(Stepfiles, ScanFiles, outLabelFiles):
                brepExtr = BRepExtractor(st, ol, feature_schema, scale_brep)
                self.transferFaceLabel_brep2Scan(brepExtr, sc, ol, debug)

        gc.collect()

    def labelBRepBndry2ScanABC(self, args):
        """
        NOTE: labelling/annotating the scan points w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs 
        """
        step_dir = args.step_path
        scan_dir = args.scan_path
        out_dir  = args.output
        nProc = args.nProc
        inStepfmt = args.infmt
        outLabelFrmt = args.ofmt
        force_regeneration = args.fRegen
        scale_brep = args.scale_brep
        
        meshsegm_dir = None
        if args.mesh_dir is not None:
            meshsegm_dir = pathlib.Path(args.mesh_dir)
        segLabel_dir = None
        if args.seg_dir is not None:
            segLabel_dir = pathlib.Path(args.seg_dir)
        featr_dir = None
        if args.feature_list is not None:
            featr_dir = pathlib.Path(args.feature_list)
            
        logger.info("generating partial data in directory tree")
        logger.info(f"input STEP files dir = {step_dir}")
        logger.info(f"input SCAN files dir = {scan_dir}")
        logger.info(f"output LABEL dir = {out_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"infmt = {inStepfmt}")
        logger.info(f"ofmt = {outLabelFrmt}")
        logger.info(f"scale_brep = {scale_brep}")
        
        """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
        parent_folder = pathlib.Path(__file__).parent.parent
        if featr_dir is None:
            featr_dir = parent_folder / "feature_lists/all.json"
        feature_schema = self.load_json(featr_dir)
        
        Stepfiles = glob.glob(str(args.step_path) + '/**/*{}'.format(inStepfmt), recursive=True)
        ScanFiles = [os.path.split(f.replace(str(args.step_path), str(args.scan_path)).replace("_step_", "_obj_"))[0] + "/" + os.path.basename(f).replace("_step_", "_trimesh_").replace(args.infmt, ".obj") for f in Stepfiles]
        outLabelFiles = [f.replace(str(args.step_path), str(args.output)).replace(args.infmt, args.ofmt) for f in Stepfiles]
        
        logger.info(f" found scans {len(Stepfiles)}")
        if not force_regeneration:
            Stepfiles = self.filter_out_files_which_are_already_converted(Stepfiles, outLabelFiles)
            ScanFiles = [os.path.split(f.replace(str(args.step_path), str(args.scan_path)).replace("_step_", "_obj_"))[0] + "/" + os.path.basename(f).replace("_step_", "_trimesh_").replace(args.infmt, ".obj") for f in Stepfiles]
            outLabelFiles = [f.replace(str(args.step_path), str(args.output)).replace(args.infmt, args.ofmt) for f in Stepfiles]
        
        use_many_threads =  nProc > 1
        if use_many_threads:
            worker_args = [(st, sc, ol, feature_schema, meshsegm_dir, segLabel_dir, scale_brep) for st, sc, ol in zip(Stepfiles, ScanFiles, outLabelFiles)]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(tqdm(executor.map(self.run_workerBoundaryLabel, worker_args), total=len(worker_args)))
        else:
            for st, sc, ol in tzip(Stepfiles, ScanFiles, outLabelFiles):
                self.transferBndryLabel_brep2Scan(st, sc, ol, feature_schema, meshsegm_dir, segLabel_dir, scale_brep)

        gc.collect()
        
    def labelBRepJunction2ScanABC(self, args):
        """
        NOTE: labelling/annotating the scan points w.r.t the
        vertex ID ||  Closed Loop ID ||  Is Boundary Point / NOT || Matting Face IDs 
        """
        step_dir = args.step_path
        scan_dir = args.scan_path
        out_dir  = args.output
        nProc = args.nProc
        inStepfmt = args.infmt
        outLabelFrmt = args.ofmt
        force_regeneration = args.fRegen
        scale_brep = args.scale_brep
        
        meshsegm_dir = None
        if args.mesh_dir is not None:
            meshsegm_dir = pathlib.Path(args.mesh_dir)
        segLabel_dir = None
        if args.seg_dir is not None:
            segLabel_dir = pathlib.Path(args.seg_dir)
        featr_dir = None
        if args.feature_list is not None:
            featr_dir = pathlib.Path(args.feature_list)
            
        logger.info("generating partial data in directory tree")
        logger.info(f"input STEP files dir = {step_dir}")
        logger.info(f"input SCAN files dir = {scan_dir}")
        logger.info(f"output LABEL dir = {out_dir}")
        logger.info(f"nProc = {nProc}")
        logger.info(f"infmt = {inStepfmt}")
        logger.info(f"ofmt = {outLabelFrmt}")
        logger.info(f"scale_brep = {scale_brep}")
        
        """Generate Directory Tree of Line Sets Given Directory Tree of Parametric Curve Segments"""
        parent_folder = pathlib.Path(__file__).parent.parent
        if featr_dir is None:
            featr_dir = parent_folder / "feature_lists/all.json"
        feature_schema = self.load_json(featr_dir)
        
        Stepfiles = glob.glob(str(args.step_path) + '/**/*{}'.format(inStepfmt), recursive=True)
        ScanFiles = [os.path.split(f.replace(str(args.step_path), str(args.scan_path)).replace("_step_", "_obj_"))[0] + "/" + os.path.basename(f).replace("_step_", "_trimesh_").replace(args.infmt, ".obj") for f in Stepfiles]
        outLabelFiles = [(os.path.splitext(f.replace(str(args.step_path), str(args.output)))[0] + os.path.splitext(f.replace(str(args.step_path), str(args.output)))[1]).replace(args.infmt, args.ofmt) for f in Stepfiles]
        
        logger.info(f" found scans {len(Stepfiles)}")
        if not force_regeneration:
            Stepfiles = self.filter_out_files_which_are_already_converted(Stepfiles, outLabelFiles)
            ScanFiles = [os.path.split(f.replace(str(args.step_path), str(args.scan_path)).replace("_step_", "_obj_"))[0] + "/" + os.path.basename(f).replace("_step_", "_trimesh_").replace(args.infmt, ".obj") for f in Stepfiles]
            outLabelFiles = [(os.path.splitext(f.replace(str(args.step_path), str(args.output)))[0] + os.path.splitext(f.replace(str(args.step_path), str(args.output)))[1]).replace(args.infmt, args.ofmt) for f in Stepfiles]
        
        use_many_threads = nProc > 1
        if use_many_threads:
            worker_args = [(st, sc, ol, feature_schema, meshsegm_dir, segLabel_dir, scale_brep) for st, sc, ol in zip(Stepfiles, ScanFiles, outLabelFiles)]
            with ProcessPoolExecutor(max_workers=nProc) as executor:
                list(tqdm(executor.map(self.run_workerJunctionLabel, worker_args), total=len(worker_args)))
        else:
            for st, sc, ol in tzip(Stepfiles, ScanFiles, outLabelFiles):
                brepExtr = BRepExtractor(st, ol, feature_schema, scale_brep)
                self.transferJunctLabel_brep2Scan(brepExtr, sc, ol)

        gc.collect()
        

