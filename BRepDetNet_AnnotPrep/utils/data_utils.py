import json
import numpy as np
import glob
import open3d as o3d
import math
import os

try:
    from scipy.spatial import cKDTree as KDTree
except ImportError:
    from scipy.spatial import KDTree

def generate_colors(N):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    import random
    import colorsys
    distinct_clrs = [     # e.g., 
        [1.0, 0.67, 0.6], # "line" 
        [0.0, 0.0, 0.7],  # "ellipse"
        [1.0, 1.0, 0.4],  # "hyperbola"
        [1.0, 0.6, 0.8],  # "parabola"
        [0.1, 1.0, 1.0],  # "bspline"
        [0.75, 0.7, 1.0], # "bezier"
        [1.0, 0.9, 0.7],  # "circle"
        [0.4, 0.7, 1.0],  # "offset"
        [0.6, 0.0, 0.3]]  # "others"
    if N > 9:
        hsv_colors_extra = [[i / (N - 9) + 0.27 , 1, 1] for i in range(N - 9)]
        hsv_colors_extras = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv_colors_extra))
        distinct_clrs.extend(hsv_colors_extras)
    
    return distinct_clrs

def apply_colormap2scalar(inScalar):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.colors import PowerNorm

    fig = plt.figure(1, figsize = (10,10))
    norm = colors.PowerNorm(gamma=0.25)    
    outColorMap = plt.cm.jet(norm(inScalar))
    #color_255 = np.ceil(outColorMap * 255)

    # Saving the Color-Map Bar
    plt.axis('off')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20b, norm=norm)
    sm.set_array(inScalar)
    cbar=fig.colorbar(sm)
    cbar.set_label('Level')
    cbar.set_ticks(np.hstack([0, np.linspace(0.2, 1, 5)]))
    cbar.set_ticklabels( ('0.0', '0.2.', '0.4', '0.6',  '0.8',  '1'))
    plt.savefig('colorbar_v.png')
        
    return np.asarray(outColorMap)[:, :3]

def find_knn(inPts, queryPts, k=1):
    nn_indices = []
    nn_distances = []
    
    kdtreeInPts = KDTree(np.asarray(inPts), leafsize=2)
    if len(queryPts) == 0 or len(inPts) == 0:
        print("Either the query vertices or the input Points are empty...")
        return None, None
    else:
        for q in range(0, len(queryPts)):
            _, idx = kdtreeInPts.query(queryPts[q], k)
            nn_indices.append(idx)
            nn_distances.append(np.linalg.norm(queryPts[q] - inPts[idx]))
            
        return nn_indices, nn_distances

def load_json_data(pathname):
    """Load data from a json file"""
    with open(pathname, encoding="utf8") as data_file:
        return json.load(data_file)

def save_json_data(pathname, data):
    """Export a data to a json file"""
    with open(pathname, "w", encoding="utf8") as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)

def save_npz_data_without_uvnet_features(output_pathname, data):
    num_faces = data["face_features"].shape[0]
    num_coedges = data["coedge_features"].shape[0]

    dummy_face_point_grids = np.zeros((num_faces, 10, 10, 7))
    dummy_coedge_point_grids = np.zeros((num_coedges, 10, 12))
    dummy_coedge_lcs = np.zeros((num_coedges, 4, 4))
    dummy_coedge_scale_factors = np.zeros((num_coedges))
    dummy_coedge_reverse_flags = np.zeros((num_coedges))
    np.savez(
        output_pathname,
        face_features=data["face_features"],
        face_point_grids=dummy_face_point_grids,
        edge_features=data["edge_features"],
        coedge_features=data["coedge_features"],
        coedge_point_grids=dummy_coedge_point_grids,
        coedge_lcs=dummy_coedge_lcs,
        coedge_scale_factors=dummy_coedge_scale_factors,
        coedge_reverse_flags=dummy_coedge_reverse_flags,
        next=data["coedge_to_next"],
        mate=data["coedge_to_mate"],
        face=data["coedge_to_face"],
        edge=data["coedge_to_edge"],
        savez_compressed=True,
    )

def load_npz_data(npz_file):
    with np.load(npz_file) as data:
        npz_data = {
            "face_features": data["face_features"],
            "face_point_grids": data["face_point_grids"],
            "edge_features": data["edge_features"],
            "coedge_features": data["coedge_features"],
            "coedge_point_grids": data["coedge_point_grids"],
            "coedge_lcs": data["coedge_lcs"],
            "coedge_scale_factors": data["coedge_scale_factors"],
            "coedge_reverse_flags": data["coedge_reverse_flags"],
            "coedge_to_next": data["next"],
            "coedge_to_mate": data["mate"],
            "coedge_to_face": data["face"],
            "coedge_to_edge": data["edge"],
        }
    return npz_data

def load_labels(label_pathname):
    labels = np.loadtxt(label_pathname, dtype=np.int64)
    if labels.ndim == 0:
        labels = np.expand_dims(labels, 0)
    return labels

def make_ftype_colors(faces_type):
    labels= []
    for face_type in faces_type:
        if face_type in ['Plane']:
            label = [1 , 0 , 0]
        elif face_type in ['Cylinder']:
            label = [0 , 1 , 0]
        elif face_type in ['Cone']:
            label = [1 , 1 , 0]
        elif face_type in ['Sphere']:
            label = [1 , 0 , 1]
        elif face_type in ['Torus']:
            label = [0 , 1 , 1]
        elif face_type in ['Bezier_surface']:
            label = [0.5, 0.5 , 0]
        elif face_type in ['BSplineSurface']:
            label = [0 , 0.5 , 0.5]
        else:
            label = [0. , 0. , 0.]
        labels.append(label)
    return labels

def filterPSECC3D(pathname, labelOut_pathname, save=False):
    PSE_pcd = o3d.io.read_point_cloud(pathname)
    v = []
    c = []
    ids  = []
    for i in range(0, len(np.asarray(PSE_pcd.points))):
        if PSE_pcd.colors[i][0] != 0.0 and PSE_pcd.colors[i][1] != 0.0 and PSE_pcd.colors[i][2] != 0.0:
            v.append(np.asarray(PSE_pcd.points[i]))
            c.append(np.asarray(PSE_pcd.colors[i]))
            ids.append(i)
    
    filename = os.path.splitext(os.path.basename(labelOut_pathname))[0]
    extension = os.path.splitext(os.path.basename(labelOut_pathname))[1]
    
    if save == True:
        if extension == ".npz":
            np.savez(pathname, vertices=v, colors=c, indices=ids, savez_compressed=True,)
        elif extension == ".ply":
            pse_filtered = o3d.geometry.PointCloud()
            pse_filtered.points = o3d.utility.Vector3dVector(np.array(v).reshape(len(v), 3))
            pse_filtered.colors = o3d.utility.Vector3dVector(np.array(c).reshape(len(c), 3))
            o3d.io.write_point_cloud(labelOut_pathname, pse_filtered)
        else:
            print("The output format is NOT handled")
        
    return v, c, ids
    
def load_BRepBoundarylabels(label_pathname_npz):
    with np.load(label_pathname_npz) as data:
        npz_data = {
            "v_vertex_Ids": data["v_vertex_Ids"], 
            "v_close_loop_Id": data["v_close_loop_Id"],
            "v_close_mate_loop_Id": data["v_close_mate_loop_Id"],
            "v_parentedge_Id": data["v_parentedge_Id"], 
            "v_memberedge_type": data["v_memberedge_type"],
            "v_proximity_weigts": data["v_proximity_weigts"], 
            "v_mateface_Id1": data["v_mateface_Id1"],
            "v_mateface_Id2": data["v_mateface_Id2"], 
            "v_edgefeats": data["v_edgefeats"]}
    return npz_data

def load_BRepJunctionlabels(label_pathname_npz):
    with np.load(label_pathname_npz, allow_pickle=True) as data:
        npz_data = {
            "j_vertex_Ids": data["j_vertex_Ids"],
            "j_close_loop_Ids": data["j_close_loop_Ids"],            
            "j_parentedge_Ids": data["j_parentedge_Ids"],            
            "j_mateface_Id1s": data["j_mateface_Id1s"],
            "j_mateface_Id2s": data["j_mateface_Id2s"]}
        return npz_data

def decodeBndryJncLabels(args):
    scans_files = glob.glob(str(os.path.abspath(args.scan_path)) + '/**/*{}'.format(args.infmt_mesh), recursive=True)
    bndryAnnot_files = glob.glob(str(os.path.abspath(args.bndryAnnot_path)) + '/**/*{}'.format(args.infmt), recursive=True)
    jncAnnot_files = glob.glob(str(os.path.abspath(args.jncAnnot_path)) + '/**/*{}'.format(args.infmt), recursive=True)
    
    assert len(scans_files) == len(bndryAnnot_files)
    assert len(scans_files) == len(jncAnnot_files) 
    
    
    for f in range(len(scans_files)):
        print(scans_files[f])
        print(bndryAnnot_files[f])
        scanPcd = o3d.io.read_triangle_mesh(scans_files[f])
        annotBndry = load_BRepBoundarylabels(bndryAnnot_files[f])
        annotJnc = load_BRepJunctionlabels(jncAnnot_files[f])
        
        bjPcd = o3d.geometry.PointCloud()
        bjPcd.points = o3d.utility.Vector3dVector(np.asarray(scanPcd.vertices)[annotBndry["v_vertex_Ids"]])
        o3d.visualization.draw_geometries([bjPcd])
        
        maxLoops = max(np.asarray(annotBndry['v_close_loop_Id']))
        maxEdges = max(np.asarray(annotBndry['v_parentedge_Id']))
        
        clrRGB = generate_colors(maxLoops)        
        loopPcd = o3d.geometry.PointCloud()
        for l in range(maxLoops):
            tempLoopPcd = o3d.geometry.PointCloud()
            vIds_per_clp = np.argwhere(annotBndry['v_close_loop_Id'] == l)
            tempLoopPcd.points = o3d.utility.Vector3dVector(np.asarray(bjPcd.points)[list(vIds_per_clp.flatten())])
            tempLoopPcd.paint_uniform_color(np.asarray(clrRGB[l]))
            loopPcd += tempLoopPcd
            o3d.io.write_point_cloud(os.path.splitext(bndryAnnot_files[f])[0] + "_CLP_" + str(l) + ".ply", tempLoopPcd)
            
        o3d.visualization.draw_geometries([loopPcd])
    
def reconstrct_BndryFromAnnot(label_pathname):
    BrepLabels = load_BRepBoundarylabels(label_pathname)
    BrepBoundaryPoints = []
    
    if BrepLabels["v_edgefeats"]["type"] == "hyperbola":
        return None
    
    elif BrepLabels["v_edgefeats"]["type"] == "parabola":
        return None
    
    elif BrepLabels["v_edgefeats"]["type"] == "ellipse":
        return None
    
    elif BrepLabels["v_edgefeats"]["type"] == "circle":
        cx, cy, cz = BrepLabels["v_edgefeats"]["center"]
        r = BrepLabels["v_edgefeats"]["radius"]
        ax, ay, az, cx, cy, cz = BrepLabels["v_edgefeats"]["axis"]
        umin, umax = BrepLabels["v_edgefeats"]["u_bounds"]
        """
        # NOTE: testing the reconstruction:
        # https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
        # also 
        # --> https://stackoverflow.com/questions/61047848/plot-a-point-in-3d-space-perpendicular-to-a-line-vector
        A = np.array([ax, ay, az]) - np.array([cx, cy, cz])
        uniA =  A / np.linalg.norm(A)
        
        UniA_abs = [abs(uniA[0]), abs(uniA[1]), abs(uniA[2])]
        maxUniA = max(UniA_abs[0], UniA_abs[1], UniA_abs[2]) 
        minUniA = min(UniA_abs[0], UniA_abs[1], UniA_abs[2]) 
        
        #Initialise 3 variables to store which array indexes contain the (max, medium, min) vector magnitudes.
        maxindex = 0
        medindex = 0
        minindex = 0
        
        # Loop through p_abs array to find which magnitudes are 
        # equal to maxval & minval. Store their indexes for use later
        for p in UniA_abs:
            if p == maxUniA: 
                maxindex = i
            elif p_abs[i] == minval: 
                minindex = i

        for u in range (umin, umax):
            x_u = cx + (r * math.cos(u)) 
            y_u = cy + (r * math.cos(u))
            z_u = cz + (r * math.cos(u))
        
        return None
        
    elif BrepLabels["v_edgefeats"]["type"] == "line":
        return None
    
    elif BrepLabels["v_edgefeats"]["type"] == "bezier":
        return None
    
    elif BrepLabels["v_edgefeats"]["type"] == "offset":
        return None
    
    elif BrepLabels["v_edgefeats"]["type"] == "bspline":
        return None
    
    elif BrepLabels["v_edgefeats"]["type"] == "bspline":
        return None
    
    else:
        properties["center"] = [CircularCurve.Location().X(), 
                                CircularCurve.Location().Y(), 
                                CircularCurve.Location().Z(),]
        
        properties["radius"] = CircularCurve.Radius()
        
        #NOTE: Returns the main axis of the circle. It is the axis perpendicular to the 
        #      plane of the circle, passing through the "Location" point (center) of the circle.
        properties["axis"] = [CircularCurve.Axis().Direction().X(),
                              CircularCurve.Axis().Direction().Y(),
                              CircularCurve.Axis().Direction().Z(),
                              CircularCurve.Axis().Location().X(),
                              CircularCurve.Axis().Location().Y(),
                              CircularCurve.Axis().Location().Z(),]
        
        properties["perimeter"] = CircularCurve.Length() 
        
        properties["u_bounds"] = Edge(edge).u_bounds()"""
        
    return BrepLabels

def reconstruct_FacesFromAnnot():
    return 

def viewSelectCADBRepAnnot(args):
    import glob
    """view mesh and edges pairwise"""
    xoffset = 2.0
    yoffset = 2.5
    
    countBRep = 0
    countAnnot = 0
    
    BReps = sorted(glob.glob(str(args.brep_path) + '/**/*{}'.format(args.infmt_brep), recursive=True))
    for brp in BReps:
        listS = []
        s = os.sep.join(os.path.normpath(brp).split(os.sep)[-3:])
        scan = os.path.split(os.path.join(str(args.scan_path), s.replace('_step_', '_obj_')))[0] + "/" + os.path.splitext(os.path.basename(s).replace('_step_', '_trimesh_'))[0] + str(args.infmt_mesh)
        bndryAnnot = os.path.splitext(os.path.join(str(args.bndryAnnot_path), s))[0] + str(args.infmt_annot)
        jncAnnot = os.path.splitext(os.path.join(str(args.jncAnnot_path), s))[0] + str(args.infmt_annot)
        
        countBRep += 1
        if os.path.exists(bndryAnnot) and os.path.exists(jncAnnot):
            countAnnot += 1
            gtscan_obj = os.path.splitext(scan)[0] + ".obj"
            gtmesh = o3d.io.read_triangle_mesh(gtscan_obj, True)
            
            centroid = np.mean(np.asarray(gtmesh.vertices), axis=0)
            gtmesh.translate(np.array([-centroid[0], -centroid[1], -centroid[2]]))
            furthest_distance = np.max(np.sqrt(np.sum(abs(np.asarray(gtmesh.vertices)**2),axis=-1)))
            gtmesh.scale(1.0 / furthest_distance, center=np.array([0,0,0]))
            gtmesh.compute_vertex_normals()
            
            gtMeshSubdivide = gtmesh.subdivide_loop(number_of_iterations=2)
            gtMeshSubdivide.compute_vertex_normals()

            listS.append(gtMeshSubdivide)
            listS.append(gtmesh.translate(np.array([2*(xoffset), 0.0, 0.0])))

            annotBndry = load_BRepBoundarylabels(bndryAnnot)
            annotJnc = load_BRepJunctionlabels(jncAnnot)

            bPcd = o3d.geometry.PointCloud()
            bPcd.points = o3d.utility.Vector3dVector(np.asarray(gtmesh.vertices)[annotBndry["v_vertex_Ids"]])
            bPcd.paint_uniform_color([1.0, 0.0, 0.0])

            jPcd = o3d.geometry.PointCloud()
            jPcd.points = o3d.utility.Vector3dVector(np.asarray(gtmesh.vertices)[annotJnc["j_vertex_Ids"]])
            jPcd.paint_uniform_color([0.0, 1.0, 0.0])            

            bjPcd = o3d.geometry.PointCloud()
            bjPcd = bPcd + jPcd
            
            centroid = np.mean(np.asarray(bjPcd.points), axis=0)
            bjPcd.translate(np.array([-centroid[0], -centroid[1], -centroid[2]]))
            furthest_distance = np.max(np.sqrt(np.sum(abs(np.asarray(bjPcd.points)**2),axis=-1)))
            bjPcd.scale(1.0 / furthest_distance, center=np.array([0,0,0]))

            listS.append(bjPcd.translate(np.array([4*(xoffset), 0.0, 0.0])))
            o3d.visualization.draw_geometries(listS)

    
    print("Step Files : " + str(countBRep) + " and Annotations : " + str(countAnnot))

    """
    for f in range(len(gtscan)):
        # check if annot exists or continue to next
        print(gtscan[f])
        if os.path.exists(bannots[f]) and os.path.exists(jannots[f]):
                        
            #choice = input("Delete (Y|N) ??")
            #if choice == 'Y' or choice == 'y':
                #_id = input("Id to Keep (e.g., 2 or 3)??")
                
                ##for i in map(int, _ids.split(" ")) : 
                #for i in range(1, len(listS)):
                    #if i != int(_id): 
                        #os.system("rm " + textures[i-1])
                        #os.system("rm " + mtrls[i-1])
                        #os.system("rm " + ptscans[i-1])
    """

def viewAllCAD(args):
    """view scans, cads and edges pairwise"""
    if args.selection is not None:
        selected = read_selection(args.selection)
        cads = [os.path.join(args.input_cads, s.replace('\\', os.sep)) for s in selected]
    else:
        cads = sorted(glob.glob(str(args.input_cads) + '/**/*{}'.format(args.infmt_mesh), recursive=True))
    
    if math.ceil(math.sqrt(len(cads))) % 2 == 0:
        dim = [int(math.ceil(math.sqrt(len(cads)))), int(math.ceil(math.sqrt(len(cads))))]
    else:
        dim = [int(math.ceil(math.sqrt(len(cads)))), int(math.ceil(math.sqrt(len(cads)))) + 1]
    
    print(dim)
    xoffset = 2.0
    yoffset = 2.0
    xtranslation = []
    ytranslation = []    
    cadList = []
    
    colorList = [[1.0, 0.3, 0.1], 
                 [1.0, 0.1, 0.3],
                 [1.0, 0.5, 0.1],
                 [1.0, 0.1, 0.5],
                 [0.5, 0.3, 1.0],
                 [0.3, 0.5, 1.0],
                 [0.1, 1.0, 0.3],
                 [0.3, 1.0, 0.1],]
    
    for fpath in cads:
        s1 = o3d.io.read_triangle_mesh(fpath)
        if not s1.has_vertex_normals():
            s1.compute_vertex_normals()
        cadList.append(s1)
    
    for i in range (0, dim[0]):
        for j in range (0, dim[1]):
            if (i*dim[0] + j) < len(cads):
                centroid = np.mean(np.asarray(cadList[i*dim[0] + j].vertices), axis=0)
                cadList[i*dim[0] + j].translate(np.array([-centroid[0], -centroid[1], -centroid[2]]))
                furthest_distance = np.max(np.sqrt(np.sum(abs(np.asarray(cadList[i*dim[0] + j].vertices)**2),axis=-1)))
                cadList[i*dim[0] + j].scale(1.0 / furthest_distance, center=np.array([0,0,0]))
                cadList[i*dim[0] + j].compute_vertex_normals()
                cadList[i*dim[0] + j].paint_uniform_color(colorList[np.random.randint(7)])
                cadList[i*dim[0] + j].translate(np.array([i * (xoffset), j * (yoffset), 0]))
    o3d.visualization.draw_geometries(cadList)

def viewAll(args):
    """view scans, cads and edges pairwise"""
    if args.selection is not None:
        selected = read_selection(args.selection)
        scans = [os.path.join(args.input_scans, s.replace('\\', os.sep)) for s in selected]
        BReps =  [os.path.join(args.input_breps, s.replace('\\', os.sep)) for s in selected]
        Brp2Scn = [os.path.join(args.input_labels, s.replace('\\', os.sep)) for s in selected]
    else:
        scans = sorted(glob.glob(str(args.input_scans) + '/**/*{}'.format(args.infmt_mesh), recursive=True))
        BReps = sorted(glob.glob(str(args.input_breps) + '/**/*{}'.format(args.infmt_breps), recursive=True))
        Brp2Scn = sorted(glob.glob(str(args.input_labels) + '/**/*{}'.format(args.infmt_labels), recursive=True))
        #scans = glob.glob(str(args.input_scans) + '/**/*{}'.format(args.infmt_mesh), recursive=True)
        #BReps = glob.glob(str(args.input_breps) + '/**/*{}'.format(args.infmt_breps), recursive=True)
        #Brp2Scn = glob.glob(str(args.input_labels) + '/**/*{}'.format(args.infmt_labels), recursive=True)
    if len(scans) != len(BReps) and len(scans) != len(Brp2Scn):
        print("No. of Scans, Step Files, and Annotation Files are NOT equal")
        exit();
    
    if math.ceil(math.sqrt(len(scans))) % 2 == 0:
        dim = [int(math.ceil(math.sqrt(len(scans)))), int(math.ceil(math.sqrt(len(scans))))]
    else:
        dim = [int(math.ceil(math.sqrt(len(scans)))), int(math.ceil(math.sqrt(len(scans)))) + 1]
    
    print(dim)
    xoffset = 2.0
    yoffset = 2.0
    
    xtranslation = []
    ytranslation = []
    
    brepList = []
    scanList = []
    scanLablList = []
    
    for fpath in BReps:
        #logger.info(f"loading, {fpath}")
        b1 = o3d.io.read_point_cloud(fpath)
        if not b1.has_normals():
            b1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        brepList.append(b1)
    
    for fpath in scans:
        #logger.info(f"loading, {fpath}")
        s1 = o3d.io.read_triangle_mesh(fpath)
        if not s1.has_vertex_normals():
            s1.compute_vertex_normals()
        scanList.append(s1)
    
    for fpath in Brp2Scn:
        #logger.info(f"loading, {fpath}")
        sl1 = o3d.io.read_point_cloud(fpath)
        if not sl1.has_normals():
            sl1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        scanLablList.append(sl1)
    
    for i in range (0, dim[0]):
        for j in range (0, dim[1]):
            if (i*dim[0] + j) < len(scans):
                #bbox = brepList[i*dim[0] + j].get_axis_aligned_bounding_box()
                #sbox = scanList[i*dim[0] + j].get_axis_aligned_bounding_box()
                #lbox = scanLablList[i*dim[0] + j].get_axis_aligned_bounding_box()
                
                centroid = np.mean(np.asarray(brepList[i*dim[0] + j].points), axis=0)
                brepList[i*dim[0] + j].translate(np.array([-centroid[0], -centroid[1], -centroid[2]]))
                furthest_distance = np.max(np.sqrt(np.sum(abs(np.asarray(brepList[i*dim[0] + j].points)**2),axis=-1)))
                brepList[i*dim[0] + j].scale(1.0 / furthest_distance, center=np.array([0,0,0]))
                #if brepList[i*dim[0] + j].has_normals() == False:
                    #brepList[i*dim[0] + j].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
                #brepList[i*dim[0] + j].paint_uniform_color([1.0, 0.6, 0.8]) 
                
                
                centroid = np.mean(np.asarray(scanList[i*dim[0] + j].vertices), axis=0)
                scanList[i*dim[0] + j].translate(np.array([-centroid[0], -centroid[1], -centroid[2]]))
                furthest_distance = np.max(np.sqrt(np.sum(abs(np.asarray(scanList[i*dim[0] + j].vertices)**2),axis=-1)))
                scanList[i*dim[0] + j].scale(1.0 / furthest_distance, center=np.array([0,0,0]))
                #if scanList[i*dim[0] + j].has_vertex_normals() == False:
                scanList[i*dim[0] + j].compute_vertex_normals()
                scanList[i*dim[0] + j].paint_uniform_color([1.0, 0.8, 0.6])
                
                centroid = np.mean(np.asarray(scanLablList[i*dim[0] + j].points), axis=0)
                scanLablList[i*dim[0] + j].translate(np.array([-centroid[0], -centroid[1], -centroid[2]]))
                furthest_distance = np.max(np.sqrt(np.sum(abs(np.asarray(scanLablList[i*dim[0] + j].points)**2),axis=-1)))
                scanLablList[i*dim[0] + j].scale(1.0 / furthest_distance, center=np.array([0,0,0]))
                scanLablList[i*dim[0] + j] = scanLablList[i*dim[0] + j].voxel_down_sample(0.05)
                
                #cbox = cadList[i*dim[0] + j].get_axis_aligned_bounding_box()
                #sbox = scanList[i*dim[0] + j].get_axis_aligned_bounding_box()
                #ebox = scanLablList[i*dim[0] + j].get_axis_aligned_bounding_box()
                
                brepList[i*dim[0] + j].translate(np.array([(3*i) * (xoffset) , 
                                                        j * (yoffset), 
                                                        0]))
                
                scanList[i*dim[0] + j].translate(np.array([(3*i + 1) * (xoffset), 
                                                        j * (yoffset), 
                                                        0]))
                
                scanLablList[i*dim[0] + j].translate(np.array([(3*i + 2) * (xoffset), 
                                                        j * (yoffset),
                                                        0]))
    o3d.visualization.draw_geometries(brepList + scanList + scanLablList)

def viewAllObjs(args):
    from PIL import Image
    """view scans, cads and edges pairwise"""
    if args.selection is not None:
        selected = read_selection(args.selection)
        scans = [os.path.join(args.input_scans, s.replace('\\', os.sep))
                  for s in selected]
    else:
        scans = glob.glob(str(args.input_scans) +
                           '/**/*{}'.format(args.infmt_mesh), recursive=True)
        textures = glob.glob(str(args.input_scans) +
                           '/**/*{}'.format(args.infmt_tex), recursive=True)
        
    if math.ceil(math.sqrt(len(scans))) % 2 == 0:
        dim = [int(math.ceil(math.sqrt(len(scans)))), int(math.ceil(math.sqrt(len(scans))))]
    else:
        dim = [int(math.ceil(math.sqrt(len(scans)))), int(math.ceil(math.sqrt(len(scans)))) + 1]
    
    print(scans)
    xoffset = 1.0
    yoffset = 2.5
    
    xtranslation = []
    ytranslation = []
    scanList = []
    
    for spath, tpath in zip(scans, textures):
        #logger.info(f"loading, {fpath}")
        im = Image.open(tpath)
        tx = trimesh.visual.TextureVisuals(image=im)
        msh = trimesh.load(spath)
        msh.visual.texture = tx
        scanList.append(msh)
    
    for i in range (0, dim[0]):
        for j in range (0, dim[1]):
            if (i*dim[0] + j) < len(scans):
                
                print(str(i) + " " + str(j))
                # if face Normal are not available, one can compute it
                #normals, valid = scanList.triangles.normals(triangles=scanList.triangles, crosses=scanList.triangles_cross)
    
                centroid = np.mean(np.asarray(scanList[i*dim[0] + j].vertices), axis=0)
                trns = np.eye(4)
                trns[:3, 3] = np.array([-centroid[0], -centroid[1], -centroid[2]])
                scanList[i*dim[0] + j].apply_transform(trns)
                
                furthest_distance = np.max(np.sqrt(np.sum(abs(np.asarray(scanList[i*dim[0] + j].vertices)**2),axis=-1)))
                trns = np.eye(4)
                trns[0][0] *= (1.0 / furthest_distance)
                trns[1][1] *= (1.0 / furthest_distance)
                trns[2][2] *= (1.0 / furthest_distance)
                scanList[i*dim[0] + j].apply_transform(trns)
                
                trns = np.eye(4)
                trns[:3, 3] = np.array([(3*i + 1) * (xoffset), j * (yoffset), 0])
                scanList[i*dim[0] + j].apply_transform(trns)
                
    scene = trimesh.Scene(scanList)
    scene.show()