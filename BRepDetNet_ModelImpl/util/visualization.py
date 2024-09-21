import open3d as o3d
import scipy.io as sio
import logging
import time
import argparse
import numpy as np

from pathlib import Path

logger = logging.getLogger(__name__)

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
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    sm.set_array(inScalar)
    cbar=fig.colorbar(sm)
    cbar.set_label('Level')
    cbar.set_ticks(np.hstack([0, np.linspace(0.2, 1, 5)]))
    cbar.set_ticklabels( ('0.0', '0.2.', '0.4', '0.6',  '0.8',  '1'))
    plt.savefig('colorbar_v.png')
    
    return outColorMap

def distance_field_point2point(point, points):
    """calculate distance field for point to points"""
    return points - point

def get_colored_pcd2(pts, bndry_labels, junction_labels, FP_bndry_labels, color_bndry=None, color_jnc=None, color_bndry_FP=None):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    colors = np.tile([0.5, 0.5, 0.5], (len(pcd.points), 1))
    
    if color_bndry is None:
        color_bndry = np.tile([1.0, 0.0, 0.0], (len(pcd.points), 1))
    if color_jnc is None:
        color_jnc = np.tile([0.2, 1.0, 0.0], (len(pcd.points), 1))
    if color_bndry_FP is None:
        color_bndry_FP = np.tile([0.3, 0.65, 1.0], (len(pcd.points), 1))
        
    if FP_bndry_labels is not None and len(FP_bndry_labels) > 0:
        edgeFP_indices = np.argwhere(FP_bndry_labels == 1)
        colors[edgeFP_indices] = color_bndry_FP[edgeFP_indices]
        
    edge_indices = np.argwhere(bndry_labels == 1)
    colors[edge_indices] = color_bndry[edge_indices]
    
    #corner_indices = np.argwhere(junction_labels == 1)
    #colors[corner_indices] = color_jnc[corner_indices]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f'Pred --> points {len(pcd.points)}, edge points {len(edge_indices)}') #, corner points {len(corner_indices)}')
    return pcd

def get_colored_pcd(pts, bndry_labels, junction_labels):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    colors = np.tile([0.5, 0.5, 0.5], (len(pcd.points), 1))
    edge_indices = np.argwhere(bndry_labels == 1)
    colors[edge_indices] = np.array([1.0, 0.0, 0.0])

    corner_indices = np.argwhere(junction_labels == 1)
    colors[corner_indices] = np.array([0.0, 1.0, 0.0])

    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f'GT --> points {len(pcd.points)}, edge points {len(edge_indices)}, corner points {len(corner_indices)}')
    return pcd

def get_final_pcd(pts, offsets, edge_labels, corner_labels):
    edge_indices = np.argwhere(edge_labels == 1)
    pts[edge_indices] = pts[edge_indices] + offsets[edge_indices]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)) 
    ## To complete coreners

    print(f'points {len(pcd.points)}, edge points {len(edge_indices)}, corner points {len(corner_indices)}')
    return pcd

def get_colored_pcd_offsets(pts, offsets, edge_labels, corner_labels):
    pts = pts + offsets
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # colors = np.zeros((len(pcd.points), 3))
    edge_indices = np.argwhere(edge_labels == 0)
    pts = np.delete(pts,edge_indices,axis=0)
    print(offsets)
    # colors[edge_indices] = np.array([1.0, 0.0, 0.0])
    # corner_indices = np.argwhere(corner_labels == 1)
    # colors[corner_indices] = np.array([0.0, 1.0, 0.0])
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    # print(f'points {len(pcd.points)}, edge points {len(edge_indices)}, corner points {len(corner_indices)}')
    return pcd

def visualize_pienet_mat(inpath):
    start = time.time()
    data = sio.loadmat(inpath)['Training_data']
    logger.info(data.shape)
    load_data_duration = time.time() - start
    logger.info(f'{inpath} load time: {load_data_duration}')
    nsamples = data.shape[0]
    
    for x in range(nsamples):
        sample = data[x,0]
        pts = sample['down_sample_point'][0,0]
        edge_labels = np.squeeze(sample['PC_8096_edge_points_label_bin'][0,0])
        corner_labels = np.squeeze(sample['corner_points_label'][0,0])
        assert(pts.shape[0] == edge_labels.shape[0] == corner_labels.shape[0])
        pcd = get_colored_pcd(pts, edge_labels, corner_labels)
        o3d.visualization.draw_geometries([pcd])

def visualize_pienet_npz(inpath):
    start = time.time()
    data_labelled = np.load(inpath)
    data = data_labelled['pcl_data'][:]
    edges = data_labelled['edges'][:]
    corners = data_labelled['corners'][:]
    logger.info(data.shape)
    load_data_duration = time.time() - start
    logger.info(f'{inpath} load time: {load_data_duration}')
    nsamples = data.shape[0]
    
    for x in range(nsamples):
        pts = data[x]
        edge_labels = edges[x]
        corner_labels = corners[x]
        assert(pts.shape[0] == edge_labels.shape[0] == corner_labels.shape[0])
        pcd = get_colored_pcd(pts, edge_labels, corner_labels)
        outpath = inpath + '-' + str(x) +'.ply'
        o3d.io.write_point_cloud(outpath, pcd)
        # o3d.visualization.draw_geometries([pcd])

def soft_labels(labelvals, confidence_threshold):
    labels = np.exp(labelvals)
    labels_sum = np.sum(labels, axis=1)
    labels = labels / labels_sum[:, np.newaxis]
    labels_idxes = np.argwhere(labels[:, 1] >= confidence_threshold)

    return labels, labels_idxes

def localNMS(points, probs, dist_th=0.06):
    """local non maximum supression"""
    df = [distance_field_point2point(p, points) for p in points]
    df = np.asarray(df)
    dists = np.linalg.norm(df, axis=2)
    dists = np.where(dists <= dist_th, 1, 0)
    pdists = dists * probs[:, np.newaxis] #distance hits scaled by probability
    
    local_max_idxes = [i for i in range(pdists.shape[0]) if np.argmax(pdists[i, :]) == i ]
    return local_max_idxes

def view_s2brep_test_npz(inpath, edge_confidence_threshold, corner_confidence_threshold):
    return None

def view_c2wf_test_npz(inpath, edge_confidence_threshold, corner_confidence_threshold):
    return None
    
def view_pienet_test_npz(inpath, edge_confidence_threshold, corner_confidence_threshold):
    start = time.time()
    data_labelled = np.load(inpath)
    inpoints = data_labelled['pcl_data'][:]
    edges = data_labelled['edges'][:]
    corners_ = data_labelled['corners'][:]
    offsets = data_labelled['offsets'][:]
    load_data_duration = time.time() - start
    nsamples = inpoints.shape[0]
    npoints = inpoints.shape[1]
    for i in range(nsamples):
        sample = inpoints[i]
        edge_labels_pre = edges[i]
        edge_labels_probs, edge_idxes = soft_labels(edge_labels_pre, edge_confidence_threshold)

        corner_labels_pre = corners_[i]
        corner_labels_probs, corner_idxes = soft_labels(corner_labels_pre, corner_confidence_threshold)
        
        offset = offsets[i] 
        global_corner_idxes = []
        
        if corner_idxes.shape[0]>1:
            corners = np.squeeze(sample[corner_idxes])
            corner_probs = np.squeeze(corner_labels_probs[corner_idxes])[:, 1]
            local_max_corner_idxes = localNMS(corners, corner_probs)
            global_corner_idxes = corner_idxes[local_max_corner_idxes]
            logger.info(f'corners {len(corner_idxes)}, nms corners {len(global_corner_idxes)}')


        edge_labels = np.zeros(npoints)
        edge_labels[edge_idxes] = 1
        corner_labels = np.zeros(npoints)
        corner_labels[global_corner_idxes] = 1
        pcd = get_colored_pcd_offsets(sample,offset, edge_labels, corner_labels)
        outpath = inpath 
        o3d.io.write_point_cloud(outpath + '-offsets' + str(i) +'.ply', pcd)
        pcd = get_colored_pcd(sample, edge_labels, corner_labels)
        outpath = inpath 
        o3d.io.write_point_cloud(outpath + '-' + str(i) +'.ply', pcd)

def view(args):
    if args.mode == 'test':
        view_pienet_test_npz(str(args.inpath),args.edge_confidence_threshold, args.corner_confidence_threshold)
    else:
        visualize_pienet_npz(str(args.inpath)) 

def _parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers()

    parser_view = subparsers.add_parser("view",
        help="Visualize pienet .mat with labeling of edges and corner points",
    )
    parser_view.add_argument("inpath", type=Path, help="Path to the input file")
    parser_view.add_argument("--infmt",  required=False, choices=(
        ['.mat', '.npz']), default='.npz', help="Format of meshes to look for'")
    parser_view.add_argument("--mode",  required=False, choices=(['test', 'gt', 'val', 'compare']), 
                             default='test', help="visualize predicted test samples or gt samples")
    parser_view.add_argument('--selection', required=False,
                             help="Path to the .txt (selected files) with additional information")
    parser_view.add_argument("--edge_confidence_threshold", default=0.7, type=float, help="edge point confidence threshold'")
    parser_view.add_argument("--corner_confidence_threshold", default=.95, type=float, help="corner point confidence threshold'")
    parser_view.set_defaults(func=view)
    args = parser.parse_args()

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

