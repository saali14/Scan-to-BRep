"""
Extract feature data from a step file using Open Cascade
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import gc
import json
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend import TopologyUtils
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_FORWARD, TopAbs_REVERSED
from OCC.Core.TopAbs import (
    TopAbs_VERTEX,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_WIRE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
)
from OCC.Core.TopExp import topexp
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OtherSurface,
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Hyperbola,
    GeomAbs_Parabola,
    GeomAbs_BezierCurve,
    GeomAbs_BSplineCurve,
    GeomAbs_OffsetCurve,
    GeomAbs_OtherCurve,
)
from OCC.Core.BRepTools import breptools_UVBounds
from OCC.Core.BRepGProp import brepgprop_LinearProperties, brepgprop_SurfaceProperties
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopoDS import TopoDS_Builder

# occwl
from occwl.edge_data_extractor import EdgeDataExtractor, EdgeConvexity
from occwl.edge import Edge
from occwl.face import Face
from occwl.solid import Solid
from occwl.uvgrid import uvgrid

# BRepNet
from .entity_mapper import EntityMapper
from .face_index_validator import FaceIndexValidator
from .segmentation_file_crosschecker import SegmentationFileCrosschecker

#import utils.scale_utils as scale_utils
import traceback


class BRepExtractor:
    def __init__(self, step_file, output_dir, feature_schema, scale_body=True):
        self.step_file = step_file
        self.output_dir = output_dir
        self.feature_schema = feature_schema
        self.scale_body = scale_body

    def scale_solid_to_unit_box(self,solid):
        is_occwl = False
        if isinstance(solid, Solid):
            is_occwl = True
            topods_solid = solid.topods_solid()
        else:
            topods_solid = solid
        bbox = find_box(topods_solid)
        xmin = 0.0
        xmax = 0.0
        ymin = 0.0
        ymax = 0.0
        zmin = 0.0
        zmax = 0.0
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin
        longest_length = dx
        if longest_length < dy:
            longest_length = dy
        if longest_length < dz:
            longest_length = dz

        orig = gp_Pnt(0.0, 0.0, 0.0)
        center_array = [
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
        ]
        scale = 2.0 / longest_length
        center = gp_Pnt(
            center_array[0],
            center_array[1],
            center_array[2],
        )
        vec_center_to_orig = gp_Vec(center, orig)
        move_to_center = gp_Trsf()
        move_to_center.SetTranslation(vec_center_to_orig)

        scale_trsf = gp_Trsf()
        scale_trsf.SetScale(orig, scale)
        trsf_to_apply = scale_trsf.Multiplied(move_to_center)

        apply_transform = BRepBuilderAPI_Transform(trsf_to_apply)
        apply_transform.Perform(topods_solid)
        transformed_solid = apply_transform.ModifiedShape(topods_solid)

        if is_occwl:
            print("Switch back to occwl solid")
            return Solid(transformed_solid), center_array, scale
        
        return transformed_solid, center_array, scale

    """
    def process(self):
        #
        #Process the file and extract the derivative datafrom
        #
        # Load the body from the STEP file
        body = self.load_body_from_step()

        # We want to apply a transform so that the solid
        # is centered on the origin and scaled so it just fits
        # into a box [-1, 1]^3
        if self.scale_body:
            body = scale_solid_to_unit_box(body)

        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)

        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return
        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return
        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is used in multiple loops are not supported")
            return

        entity_mapper = EntityMapper(body)

        face_features = self.extract_face_features_from_body(body, entity_mapper)
        edge_features = self.extract_edge_features_from_body(body, entity_mapper)
        coedge_features = self.extract_coedge_features_from_body(body, entity_mapper)

        face_point_grids = self.extract_face_point_grids(body, entity_mapper)
        assert face_point_grids.shape[1] == 7
        coedge_point_grids = self.extract_coedge_point_grids(body, entity_mapper)
        assert coedge_point_grids.shape[1] == 12

        coedge_lcs = self.extract_coedge_local_coordinate_systems(body, entity_mapper)
        coedge_reverse_flags = self.extract_coedge_reverse_flags(body, entity_mapper)

        next, mate, face, edge = self.build_incidence_arrays(body, entity_mapper)

        coedge_scale_factors = self.extract_scale_factors(
            next, mate, face, face_point_grids, coedge_point_grids
        )

        output_pathname = self.output_dir / f"{self.step_file.stem}.npz"
        np.savez(
            output_pathname,
            face_features=face_features,
            face_point_grids=face_point_grids,
            edge_features=edge_features,
            coedge_point_grids=coedge_point_grids,
            coedge_features=coedge_features,
            coedge_lcs=coedge_lcs,
            coedge_scale_factors=coedge_scale_factors,
            coedge_reverse_flags=coedge_reverse_flags,
            next=next,
            mate=mate,
            face=face,
            edge=edge,
            savez_compressed=True,
        )
    """

    def load_body_from_step(self):
        """
        Load the body from the step file.
        We expect only one body in each file
        """
        step_filename_str = str(self.step_file)
        reader = STEPControl_Reader()
        reader.ReadFile(step_filename_str)
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape

    def body_properties(self):
        """
        Returns a dictionary with the properties of a body
        """
        # Load the body from the STEP fileextract_face_point_grids
        body = self.load_body_from_step()
        # We want to apply a transform so that the solid
        # is centered on the origin and scaled so it just fits
        # into a box [-1, 1]^3
        if self.scale_body:
            body,_,_ = self.scale_solid_to_unit_box(body)

        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)

        if not self.check_manifold(top_exp):
            print("Non-manifold bodies are not supported")
            return

        if not self.check_closed(body):
            print("Bodies which are not closed are not supported")
            return

        if not self.check_unique_coedges(top_exp):
            print("Bodies where the same coedge is uses in multiple loops are not supported")
            return

        properties = {}
        properties["filename"] = os.path.basename(self.step_file)
        properties["face_count"] = self.face_count(top_exp)
        properties["edge_count"] = self.edge_count(top_exp)
        properties["face_type_counts"] = self.face_type_counts(top_exp)
        properties["face_type_counts"] = self.face_type_counts(top_exp)
        per_face_prop = [self.face_properties(face) for face in top_exp.faces()]
        properties["face_properties"] = per_face_prop
        return properties

    def face_count(self, top_exp):
        """
        Returns the number of faces in a body
        """
        return sum(1 for e in top_exp.faces())

    def edge_count(self, top_exp):
        """
        Returns the number of faces in a body
        """
        return sum(1 for e in top_exp.edges())

    def face_type(self, face):
        """
        Extract the face type, possible face types are: Plane, Cylinder, Cone, Sphere,
        Torus, BezierSurface, BSplineSurface, SurfaceOfRevolution, SurfaceOfExtrusion,
        OtherSurface.
        """
        surface = BRepAdaptor_Surface(face)
        surface_type = surface.GetType()
        if surface_type == GeomAbs_Plane:
            return "Plane"
        elif surface_type == GeomAbs_Cylinder:
            return "Cylinder"
        elif surface_type == GeomAbs_Cone:
            return "Cone"
        elif surface_type == GeomAbs_Sphere:
            return "Sphere"
        elif surface_type == GeomAbs_Torus:
            return "Torus"
        elif surface_type == GeomAbs_BezierSurface:
            return "Bezier_surface"
        elif surface_type == GeomAbs_BSplineSurface:
            return "BSplineSurface"
        elif surface_type == GeomAbs_SurfaceOfRevolution:
            return "SurfaceOfRevolution"
        elif surface_type == GeomAbs_SurfaceOfExtrusion:
            return "SurfaceOfExtrusion"
        elif surface_type == GeomAbs_OtherSurface:
            return "OtherSurface"
        return "unsupported face type"

    def face_type_counts(self, top_exp):
        face_types = [self.face_type(face) for face in top_exp.faces()]
        return {type: face_types.count(type) for type in face_types}

    def face_properties(self, face):
        properties = {}
        surface = BRepAdaptor_Surface(face)
        surface_type = surface.GetType()
        if surface_type == GeomAbs_Plane:
            gp_pln = surface.Plane()
            properties = self.plane_features(gp_pln, face)
        elif surface_type == GeomAbs_Cylinder:
            gp_cylinder = surface.Cylinder()
            properties = self.cylinder_features(gp_cylinder, face)
        elif surface_type == GeomAbs_Cone:
            gp_cone = surface.Cone()
            properties = self.cone_features(gp_cone, face)
        elif surface_type == GeomAbs_Sphere:
            gp_sphere = surface.Sphere()
            properties = self.sphere_features(gp_sphere, face)
        elif surface_type == GeomAbs_Torus:
            gp_torus = surface.Torus()
            properties = self.torus_features(gp_torus, face)
        elif surface_type == GeomAbs_BezierSurface:
            gp_BezierSurface = surface.Bezier()
            properties = self.BezierSurface_features(gp_BezierSurface, face)
        elif surface_type == GeomAbs_BSplineSurface:
            gp_BSplineSurface = surface.BSpline()
            properties = self.BSplineSurface_features(gp_BSplineSurface, face)
        elif surface_type == GeomAbs_SurfaceOfRevolution:
            properties["type"] = "SurfaceOfRevolution"
            properties["more"] = "not implemented"
        elif surface_type == GeomAbs_SurfaceOfExtrusion:
            properties["type"] = "SurfaceOfExtrusion"
            properties["more"] = "not implemented"
        elif surface_type == GeomAbs_OtherSurface:
            properties["type"] = "OtherSurface"
            properties["more"] = "not implemented"
        geometry_properties = GProp_GProps()
        brepgprop_SurfaceProperties(face, geometry_properties)
        area = geometry_properties.Mass()
        properties["area"] = area
        return properties

    def edge_properties(self, edge):
        properties = {}
        try:
            curve = BRepAdaptor_Curve(edge)
            curve_type = curve.GetType()
            if curve_type == GeomAbs_Line:
                gp_line = curve.Line()
                properties = self.straight_edge_properties(gp_line, edge)
            elif curve_type == GeomAbs_Circle: 
                gp_circle = curve.Circle()
                properties = self.circular_edge_properties(gp_circle, edge)
            elif curve_type == GeomAbs_Ellipse:
                gp_ellipse = curve.Ellipse()
                properties = self.elliptical_edge_properties(gp_ellipse, edge)
            elif curve_type == GeomAbs_Hyperbola:
                gp_hyperbola = curve.Hyperbola()
                properties = self.hyperbolic_edge_properties(gp_hyperbola, edge)
            elif curve_type == GeomAbs_Parabola:
                gp_parabola = curve.Parabola()
                properties = self.parabolic_edge_properties(gp_parabola, edge)
            elif curve_type == GeomAbs_BezierCurve:
                gp_bezier = curve.Bezier()
                properties = self.bezier_edge_properties(gp_bezier, edge)
            elif curve_type == GeomAbs_BSplineCurve:
                gp_bspline = curve.BSpline()
                properties = self.rational_bspline_edge_properties(gp_bspline, edge)
            elif curve_type == GeomAbs_OffsetCurve:
                gp_offset = curve.OffsetCurve ()
                properties = self.offSet_edge_properties(gp_offset, edge)
            else: 
                print("Nothing is Implemented for Other types of curve!")
            
            return properties
        
        except RuntimeError:    
            return {}
    
    def BezierSurface_features(self, BezierSurface, face):
        properties = {}
        properties["type"] = "BezierSurface"
        """
        TODO
        """

        properties["uv_bounds"] = breptools_UVBounds(face)
        properties["bounding_box"] = self.get_boundingbox(face)

        return properties
    
    def BSplineSurface_features(self, BSplineSurface, face):
        properties = {}
        properties["type"] = "BSplineSurface"
        """
        TODO
        """
        properties["uv_bounds"] = breptools_UVBounds(face)
        properties["bounding_box"] = self.get_boundingbox(face)

        return properties

    def torus_features(self, torus, face):
        properties = {}
        properties["type"] = "Torus"
        """
        TOVERIFY
        """
        #properties["implicit_eq_coefficient"] = torus.Coefficients()
        properties["is_right_handed"] = torus.Direct()
        properties["main_axis"] = [
            torus.Axis().Direction().X(),
            torus.Axis().Direction().Y(),
            torus.Axis().Direction().Z(),
        ]
        properties["major_radius"] = torus.MajorRadius()
        properties["minor_radius"] = torus.MinorRadius()
        properties["origin"] = [
            torus.Location().X(),
            torus.Location().Y(),
            torus.Location().Z(),
        ]
        properties["uv_bounds"] = breptools_UVBounds(face)
        properties["bounding_box"] = self.get_boundingbox(face)

        return properties
    
    def cone_features(self, cone, face):
        properties = {}
        properties["type"] = "Cone"
        """
        TOVERIFY
        """
        properties["implicit_eq_coefficient"] = cone.Coefficients()
        properties["is_right_handed"] = cone.Direct()
        properties["directions"] = [
            cone.Axis().Direction().X(),
            cone.Axis().Direction().Y(),
            cone.Axis().Direction().Z(),
        ]
        properties["angle"] = cone.SemiAngle()
        properties["radius"] = cone.RefRadius()
        properties["origin"] = [
            cone.Location().X(),
            cone.Location().Y(),
            cone.Location().Z(),
        ]
        properties["uv_bounds"] = breptools_UVBounds(face)
        properties["bounding_box"] = self.get_boundingbox(face)

        return properties

    def sphere_features(self, sphere, face):
        properties = {}
        properties["type"] = "Sphere"
        """
        TOVERIFY
        """
        properties["implicit_eq_coefficient"] = sphere.Coefficients()
        properties["is_right_handed"] = sphere.Direct()
        properties["radius"] = sphere.Radius()
        properties["center"] = [
            sphere.Location().X(),
            sphere.Location().Y(),
            sphere.Location().Z(),
        ]
        properties["uv_bounds"] = breptools_UVBounds(face)
        properties["bounding_box"] = self.get_boundingbox(face)

        return properties

    def plane_features(self, plane, face):
        properties = {}
        properties["type"] = "Plane"
        properties["cartesian_coefficient"] = plane.Coefficients()
        properties["is_right_handed"] = plane.Direct()
        properties["normal_axis"] = [
            plane.Axis().Direction().X(),
            plane.Axis().Direction().Y(),
            plane.Axis().Direction().Z(),
        ]
        properties["point_on_plane"] = [
            plane.Location().X(),
            plane.Location().Y(),
            plane.Location().Z(),
        ]
        properties["uv_bounds"] = breptools_UVBounds(face)
        properties["bounding_box"] = self.get_boundingbox(face)

        return properties

    def cylinder_features(self, cylinder, face):
        properties = {}
        properties["type"] = "Cylinder"
        # A1.X**2 + A2.Y**2 + A3.Z**2 + 2.(B1.X.Y + B2.X.Z + B3.Y.Z) + 2.(C1.X + C2.Y + C3.Z) + D = 0.0
        properties["implicit_eq_coefficient"] = cylinder.Coefficients()
        properties["is_right_handed"] = cylinder.Direct()
        properties["symmetry_axis"] = [
            cylinder.Axis().Direction().X(),
            cylinder.Axis().Direction().Y(),
            cylinder.Axis().Direction().Z(),
        ]
        properties["radius"] = cylinder.Radius()
        properties["location_point"] = [
            cylinder.Location().X(),
            cylinder.Location().Y(),
            cylinder.Location().Z(),
        ]
        properties["uv_bounds"] = breptools_UVBounds(face)
        properties["bounding_box"] = self.get_boundingbox(face)
        return properties

    def get_boundingbox(self, shape, tol=1e-8, use_mesh=True):
        """return the bounding box of the TopoDS_Shape `shape`
        Parameters
        ----------
        shape : TopoDS_Shape or a subclass such as TopoDS_Face
            the shape to compute the bounding box from
        tol: float
            tolerance of the computed boundingbox
        use_mesh : bool
            a flag that tells whether or not the shape has first to be meshed before the bbox
            computation. This produces more accurate results
        """
        bbox = Bnd_Box()
        bbox.SetGap(tol)
        if use_mesh:
            mesh = BRepMesh_IncrementalMesh()
            mesh.SetParallelDefault(True)
            mesh.SetShape(shape)
            mesh.Perform()
            if not mesh.IsDone():
                raise AssertionError("Mesh not done.")
        brepbndlib_Add(shape, bbox, use_mesh)

        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return xmin, ymin, zmin, xmax, ymax, zmax, xmax - xmin, ymax - ymin, zmax - zmin

    def extract_face_features_from_body(self, body, entity_mapper):
        """
        Extract the face features from each face of the body
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        face_features = []
        for face in top_exp.faces():
            assert len(face_features) == entity_mapper.face_index(face)
            face_features.append(self.extract_features_from_face(face))
        return np.stack(face_features)
    
    def extract_edge_info_from_body(self, body, entity_mapper):
        """
        Extract the face features from each face of the body
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        edge_point_grids = []
        edge_features = []
        
        for edge in top_exp.edges():
            assert len(edge_features) == entity_mapper.edge_index(edge)
            edge_features.append(self.edge_properties(edge))
        
        return np.stack(edge_features)
        
    def extract_face_info_from_body(self, body, entity_mapper):
        """
        Extract the face features from each face of the body
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        face_point_grids = []
        face_features = []
        face_ID = []
        for face in top_exp.faces():
            assert len(face_features) == entity_mapper.face_index(face)
            face_ID.append(entity_mapper.face_index(face))
            face_features.append(self.face_properties(face))

        try:
            solid = Solid(body) # TODO @Anis Deal with Assertion Error like the problem of coedges --> Discard the examples where we fail to convert into solid such as in extract_coedge_point_grids
            for face in solid.faces():
                assert len(face_point_grids) == entity_mapper.face_index(face.topods_shape())
                face_point_grids.append(self.extract_face_point_grid(face))

            assert len(face_features) == len(face_point_grids) #NOTE Make sure that the face_features and face_point_grids have the same number of faces
            """
            TOCHECK: @Anis face_features and face_point_grids have the same IDs (concide) --> Not sure 
            """
            return np.stack(face_ID), np.stack(face_features), face_point_grids
        except Exception:
            print("An unknown exception has occurred ... passing to next sample")
            pass

    def extract_edge_features_from_body(self, body, entity_mapper):
        """
        Extract the edge features from each edge of the body
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        edge_features = []
        for edge in top_exp.edges():
            assert len(edge_features) == entity_mapper.edge_index(edge)
            faces_of_edge = [Face(f) for f in top_exp.faces_from_edge(edge)]
            edge_features.append(self.extract_features_from_edge(edge, faces_of_edge))
        return np.stack(edge_features)

    def extract_coedge_features_from_body(self, body, entity_mapper):
        """
        Extract the coedge features from each face of the body
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        coedge_features = []
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(coedge_features) == entity_mapper.halfedge_index(coedge)
                coedge_features.append(self.extract_features_from_coedge(coedge))

        return np.stack(coedge_features)

    def extract_features_from_face(self, face):
        face_features = []
        for feature in self.feature_schema["face_features"]:
            if feature == "Plane":
                face_features.append(self.plane_feature(face))
            elif feature == "Cylinder":
                face_features.append(self.cylinder_feature(face))
            elif feature == "Cone":
                face_features.append(self.cone_feature(face))
            elif feature == "SphereFaceFeature":
                face_features.append(self.sphere_feature(face))
            elif feature == "TorusFaceFeature":
                face_features.append(self.torus_feature(face))
            elif feature == "FaceAreaFeature":
                face_features.append(self.area_feature(face))
            elif feature == "RationalNurbsFaceFeature":
                face_features.append(self.rational_nurbs_feature(face))
            else:
                assert False, "Unknown face feature"
        return np.array(face_features)

    def extract_features_from_edge(self, edge, faces):
        feature_list = self.feature_schema["edge_features"]
        if (
            "Concave edge" in feature_list
            or "Convex edge" in feature_list
            or "Smooth" in feature_list
        ):
            convexity = self.find_edge_convexity(edge, faces)
        edge_features = []
        for feature in feature_list:
            if feature == "Concave edge":
                edge_features.append(self.convexity_feature(convexity, feature))
            elif feature == "Convex edge":
                edge_features.append(self.convexity_feature(convexity, feature))
            elif feature == "Smooth":
                edge_features.append(self.convexity_feature(convexity, feature))
            elif feature == "EdgeLengthFeature":
                edge_features.append(self.edge_length_feature(edge))
            elif feature == "CircularEdgeFeature":
                edge_features.append(self.circular_edge_feature(edge))
            elif feature == "ClosedEdgeFeature":
                edge_features.append(self.closed_edge_feature(edge))
            elif feature == "EllipticalEdgeFeature":
                edge_features.append(self.elliptical_edge_feature(edge))
            elif feature == "HelicalEdgeFeature":
                edge_features.append(self.helical_edge_feature(edge))
            elif feature == "IntcurveEdgeFeature":
                edge_features.append(self.int_curve_edge_feature(edge))
            elif feature == "StraightEdgeFeature":
                edge_features.append(self.straight_edge_feature(edge))
            elif feature == "HyperbolicEdgeFeature":
                edge_features.append(self.hyperbolic_edge_feature(edge))
            elif feature == "ParabolicEdgeFeature":
                edge_features.append(self.parabolic_edge_feature(edge))
            elif feature == "BezierEdgeFeature":
                edge_features.append(self.bezier_edge_feature(edge))
            elif feature == "NonRationalBSplineEdgeFeature":
                edge_features.append(self.non_rational_bspline_edge_feature(edge))
            elif feature == "RationalBSplineEdgeFeature":
                edge_features.append(self.rational_bspline_edge_feature(edge))
            elif feature == "OffsetEdgeFeature":
                edge_features.append(self.offset_edge_feature(edge))
            else:
                assert False, "Unknown face feature"
        return np.array(edge_features)
    
    #@Ali: NOTE -- New properties for different types of curves 
    #              are added to be zipped into the *.npz files
    def hyperbolic_edge_properties(self, HyperbolicCurve, edge):
        properties = {}
        properties["type"] = "hyperbola"
        
        # Returns the major radius of the hyperbola.
        # It is the radius on the "XAxis" of the hyperbola.
        properties["major_radius"] = HyperbolicCurve.MajorRadius()
        
        # Returns the minor radius of the hyperbola.
        # It is the radius on the "YAxis" of the hyperbola.
        properties["minor_radius"] = HyperbolicCurve.MinorRadius()
        
        #Returns the FIRST focus of the hyperbola. 
        #This focus is on the +POSITIVE side of the "XAxis" of the hyperbola. 
        properties["focus1"] = [ HyperbolicCurve.Focus1().X(), 
                                  HyperbolicCurve.Focus1().Y(),
                                  HyperbolicCurve.Focus1().Z(),]
        
        #Returns the SECOND focus of the hyperbola. 
        #This focus is on the -NEGATIVE side of the "XAxis" of the hyperbola. 
        properties["focus2"] = [ HyperbolicCurve.Focus1().X(), 
                                  HyperbolicCurve.Focus1().Y(),
                                  HyperbolicCurve.Focus1().Z(),]
        
        # Returns the axis passing through the center, and normal to the plane of this hyperbola.
        properties["axis"] = [ HyperbolicCurve.Axis.Location().X(),
                              HyperbolicCurve.Axis.Location().Y(),
                              HyperbolicCurve.Axis.Location().Z(),
                              HyperbolicCurve.Axis.Direction().X(),
                              HyperbolicCurve.Axis.Direction().Y(),
                              HyperbolicCurve.Axis.Direction().Z(),]
        
        #Returns the location point of the hyperbola. 
        #It is the intersection point between the "XAxis" and the "YAxis".
        properties["location"] = [HyperbolicCurve.Location().X(), 
                                   HyperbolicCurve.Location().Y(), 
                                   HyperbolicCurve.Location().Z()]
        
        properties["u_bounds"] = Edge(edge).u_bounds()
        return properties
        
    def parabolic_edge_properties(self, ParabolicCurve, edge):
        # NOTE --> follow 
        # TODO https://math.stackexchange.com/questions/2770840/general-equation-for-parabola-in-3d-space 
        # and 
        # TODO https://math.stackexchange.com/questions/1136516/equation-of-a-parabola-in-3d-space
        # to reconstruct and verify the parametric reconstruction of parabola
        properties = {}
        properties["type"] = "parabola"
        
        #Returns the FIRST focus of the hyperbola. 
        #This focus is on the +POSITIVE side of the "XAxis" of the hyperbola. 
        properties["focus"] = [ ParabolicCurve.Focus().X(), 
                                ParabolicCurve.Focus().Y(), 
                                ParabolicCurve.Focus().Z(),]
        
        # Returns the axis passing through the center, and normal to the plane of this hyperbola.
        properties["axis"] = [ ParabolicCurve.Axis.Location().X(),
                              ParabolicCurve.Axis.Location().Y(),
                              ParabolicCurve.Axis.Location().Z(),
                              ParabolicCurve.Axis.Direction().X(),
                              ParabolicCurve.Axis.Direction().Y(),
                              ParabolicCurve.Axis.Direction().Z(),]
        
        # Returns the axis passing through the center, and normal to the plane of this hyperbola.
        properties["directrix"] = [ ParabolicCurve.Directrix.Location().X(),
                                    ParabolicCurve.Directrix.Location().Y(),
                                    ParabolicCurve.Directrix.Location().Z(),
                                    ParabolicCurve.Directrix.Direction().X(),
                                    ParabolicCurve.Directrix.Direction().Y(),
                                    ParabolicCurve.Directrix.Direction().Z(),]
        
        # Returns the parameter of the parabola. 
        # It is the distance between the focus and the 
        # directrix of the parabola. This distance is twice the focal length. 
        properties["parameter"] = ParabolicCurve.Parameter()
        properties["u_bounds"] = Edge(edge).u_bounds()
        return properties
    
    def elliptical_edge_properties(self, EllipticalCurve, edge):
        properties = {}
        properties["type"] = "ellipse"
        
        # Returns the center of the ellipse. 
        # It is the "Location" point of the coordinate system of the ellipse.
        properties["center"] = [EllipticalCurve.Axis().Location().X(), 
                                EllipticalCurve.Axis().Location().Y(), 
                                EllipticalCurve.Axis().Location().Z()]
        
        # Returns the major and minor radius (i.e., a, b) of the ellipse. 
        properties["major_radius"] = EllipticalCurve.MajorRadius()
        properties["major_radius"] = EllipticalCurve.MinorRadius()
        
        #NOTE: Returns the main axis of the circle. It is the axis perpendicular to the 
        #      plane of the circle, passing through the "Location" point (center) of the circle.
        properties["axis"] = [EllipticalCurve.Axis().Direction().X(),
                              EllipticalCurve.Axis().Direction().Y(),
                              EllipticalCurve.Axis().Direction().Z(),
                              EllipticalCurve.Axis().Location().X(),
                              EllipticalCurve.Axis().Location().Y(),
                              EllipticalCurve.Axis().Location().Z(),]
        
        #Returns the FIRST focus of the Ellipse. 
        #This focus is on the +POSITIVE side of the "XAxis" of the Ellipse. 
        properties["focus1"] = [ EllipticalCurve.Focus1().X(), 
                                  EllipticalCurve.Focus1().Y(),
                                  EllipticalCurve.Focus1().Z(),]
        
        #Returns the SECOND focus of the Ellipse. 
        #This focus is on the -NEGATIVE side of the "XAxis" of the Ellipse. 
        properties["focus2"] = [ EllipticalCurve.Focus1().X(), 
                                  EllipticalCurve.Focus1().Y(),
                                  EllipticalCurve.Focus1().Z(),]
        
        #properties["perimeter"] = 2 * math.pi * pow(0.5 * (EllipticalCurve.MajorRadius()**2 + EllipticalCurve.MinorRadius()**2), 0.5)
        
        properties["u_bounds"] = Edge(edge).u_bounds()
        
        return properties 
    
    def circular_edge_properties(self, CircularCurve, edge):
        properties = {}
        properties["type"] = "circle"
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
        
        properties["u_bounds"] = Edge(edge).u_bounds()
        
        return properties
        
    def int_curve_edge_properties(self, arc, edge):
        assert False, "Not implemented for the OCC pipeline"
        return 
    
    def straight_edge_properties(self, StraightLineCurve, edge):
        properties = {}
        properties["type"] = "line"
        properties["location"] = [StraightLineCurve.Location().X(),
                                  StraightLineCurve.Location().Y(),
                                  StraightLineCurve.Location().Z(),]
        
        properties["direction"] = [StraightLineCurve.Direction().X(), 
                                   StraightLineCurve.Direction().Y(), 
                                   StraightLineCurve.Direction().Z(),]
        
        properties["u_bounds"] = Edge(edge).u_bounds()
        
        
        # NOTE : very important: When we want to obtain the type and 
        #        parameters of the closed loop that confine a given BRep-Face
        #        i.e., winged face, then the following function provides the facility 
        #        to access them
        # crv, umin, umax = BRep_Tool().CurveOnSurface(edge.topods_shape(), self.topods_shape())
        
        return properties
    
    def bezier_edge_properties(self, BezierCurve, edge):
        properties = {}
        properties["type"] = "bezier"
        
        properties["start"] = [BezierCurve.StartPoint().X(), 
                               BezierCurve.StartPoint().Y(), 
                               BezierCurve.StartPoint().Z(),]
        
        properties["end"] = [BezierCurve.EndPoint().X(), 
                             BezierCurve.EndPoint().Y(),
                             BezierCurve.EndPoint().Z(),]
        
        properties["degree"] = BezierCurve.Degree()
        
        #TODO : ??? 
        
        properties["u_bounds"] = Edge(edge).u_bounds()
        
        return properties
    
    def offSet_edge_properties(self, OffsetCurve, edge):
        # NOTE: The nature of Offset Curve is that it first takes the parameters 
        #       of the Basis Curve
        #       Value (U) = BasisCurve.Value(U) + (Offset * (T ^ V)) / ||T ^ V||
        
        properties = {}
        properties["type"] = "offset"
        
        properties["curve_firstParam"] = OffsetCurve.FirstParameter()
        properties["curve_lastParam"] = OffsetCurve.LastParameter()
        
        properties["baseCurve_firstParam"] = OffsetCurve.BasisCurve().FirstParameter()
        properties["baseCurve_lastParam"] = OffsetCurve.BasisCurve().LastParameter()
        
        properties["baseCurve_isClosed"] = OffsetCurve.BasisCurve().IsClosed()
        
        properties["Direction"] = [OffsetCurve.Direction().X(), 
                                   OffsetCurve.Direction().Y(), 
                                   OffsetCurve.Direction().Z(),]
        
        #TODO : ??? 
        properties["u_bounds"] = Edge(edge).u_bounds()
        return properties
    
    def non_rational_bspline_edge_properties(self, NURBSCurve, edge):
        return 
    
    def rational_bspline_edge_properties(self, BSplineCurve, edge):
        properties = {}
        properties["type"] = "bspline"
        
        properties["start"] = [BSplineCurve.StartPoint().X(), 
                               BSplineCurve.StartPoint().Y(), 
                               BSplineCurve.StartPoint().Z(),]
        
        properties["end"] = [BSplineCurve.EndPoint().X(), 
                             BSplineCurve.EndPoint().Y(),
                             BSplineCurve.EndPoint().Z(),]
        
        properties["degree"] = BSplineCurve.Degree()
        
        properties["knots"] = [i for i in BSplineCurve.KnotSequence()]
        
        properties["u_bounds"] = Edge(edge).u_bounds()
        return properties
    
    def plane_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Plane:
            return 1.0
        return 0.0

    def cylinder_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Cylinder:
            return 1.0
        return 0.0

    def cone_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Cone:
            return 1.0
        return 0.0

    def sphere_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Sphere:
            return 1.0
        return 0.0

    def torus_feature(self, face):
        surf_type = BRepAdaptor_Surface(face).GetType()
        if surf_type == GeomAbs_Torus:
            return 1.0
        return 0.0

    def area_feature(self, face):
        geometry_properties = GProp_GProps()
        brepgprop_SurfaceProperties(face, geometry_properties)
        return geometry_properties.Mass()

    def rational_nurbs_feature(self, face):
        surf = BRepAdaptor_Surface(face)
        if surf.GetType() == GeomAbs_BSplineSurface:
            bspline = surf.BSpline()
        elif surf.GetType() == GeomAbs_BezierSurface:
            bspline = surf.Bezier()
        else:
            bspline = None

        if bspline is not None:
            if bspline.IsURational() or bspline.IsVRational():
                return 1.0
        return 0.0

    def find_edge_convexity(self, edge, faces):
        edge_data = EdgeDataExtractor(Edge(edge), faces, use_arclength_params=False)
        if not edge_data.good:
            # This is the case where the edge is a pole of a sphere
            return 0.0
        angle_tol_rads = 0.0872664626  # 5 degrees
        convexity = edge_data.edge_convexity(angle_tol_rads)
        return convexity

    def convexity_feature(self, convexity, feature):
        if feature == "Convex edge":
            return convexity == EdgeConvexity.CONVEX
        if feature == "Concave edge":
            return convexity == EdgeConvexity.CONCAVE
        if feature == "Smooth":
            return convexity == EdgeConvexity.SMOOTH
        assert False, "Unknown convexity"
        return 0.0

    def edge_length_feature(self, edge):
        geometry_properties = GProp_GProps()
        brepgprop_LinearProperties(edge, geometry_properties)
        return geometry_properties.Mass()

    def circular_edge_feature(self, edge):
        brep_adaptor_curve = BRepAdaptor_Curve(edge)
        curv_type = brep_adaptor_curve.GetType()
        if curv_type == GeomAbs_Circle:
            return 1.0
        return 0.0

    def closed_edge_feature(self, edge):
        if BRep_Tool().IsClosed(edge):
            return 1.0
        return 0.0

    def elliptical_edge_feature(self, edge):
        brep_adaptor_curve = BRepAdaptor_Curve(edge)
        curv_type = brep_adaptor_curve.GetType()
        if curv_type == GeomAbs_Ellipse:
            return 1.0
        return 0.0

    def helical_edge_feature(self, edge):
        # We don't have this feature in Open Cascade
        assert False, "Not implemented for the OCC pipeline"
        return 0.0

    def int_curve_edge_feature(self, edge):
        # We don't have this feature in Open Cascade
        assert False, "Not implemented for the OCC pipeline"
        return 0.0

    def straight_edge_feature(self, edge):
        brep_adaptor_curve = BRepAdaptor_Curve(edge)
        curv_type = brep_adaptor_curve.GetType()
        if curv_type == GeomAbs_Line:
            return 1.0
        return 0.0

    def hyperbolic_edge_feature(self, edge):
        if Edge(edge).curve_type() == "hyperbola":
            return 1.0
        return 0.0

    def parabolic_edge_feature(self, edge):
        if Edge(edge).curve_type() == "parabola":
            return 1.0
        return 0.0

    def bezier_edge_feature(self, edge):
        if Edge(edge).curve_type() == "bezier":
            return 1.0
        return 0.0

    def non_rational_bspline_edge_feature(self, edge):
        occwl_edge = Edge(edge)
        if occwl_edge.curve_type() == "bspline" and not occwl_edge.rational():
            return 1.0
        return 0.0

    def rational_bspline_edge_feature(self, edge):
        occwl_edge = Edge(edge)
        if occwl_edge.curve_type() == "bspline" and occwl_edge.rational():
            return 1.0
        return 0.0

    def offset_edge_feature(self, edge):
        if Edge(edge).curve_type() == "offset":
            return 1.0
        return 0.0

    def extract_features_from_coedge(self, coedge):
        coedge_features = []
        for feature in self.feature_schema["coedge_features"]:
            if feature == "ReversedCoEdgeFeature":
                coedge_features.append(self.reversed_edge_feature(coedge))
            else:
                assert False, "Unknown coedge feature"
        return np.array(coedge_features)

    def reversed_edge_feature(self, edge):
        if edge.Orientation() == TopAbs_REVERSED:
            return 1.0
        return 0.0

    def extract_face_point_grids(self, body, entity_mapper):
        """
        Extract a UV-Net point grid for each face.

        Returns a tensor [ num_faces x 7 x num_pts_u x num_pts_v ]

        For each point the values are

            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)
            - Trimming mask

        """
        face_grids = []
        solid = Solid(body)
        for face in solid.faces():
            assert len(face_grids) == entity_mapper.face_index(face.topods_shape())
            face_grids.append(self.extract_face_point_grid(face))
        return (face_grids)

    def extract_face_point_grid(self, face):
        """
        Extract a UV-Net point grid from the given face.

        Returns a tensor [ 7 x num_pts_u x num_pts_v ]

        For each point the values are
            - x, y, z (point coords)
            - i, j, k (normal vector coordinates)
            - Trimming mast
        """

        # Compute face areas to be used to select the number of pts to be sampled per face 
        # Larger faces should have greater number of pts @Anis
        face_area = self.face_properties(face.topods_shape())["area"]
        
        # minimum nb is 20
        num_u = 20 + int(face_area * 100)
        num_v = 20 + int(face_area * 100)


        points = uvgrid(face, num_u, num_v, method="point")
        normals = uvgrid(face, num_u, num_v, method="normal")
        mask = uvgrid(face, num_u, num_v, method="inside")

        # This has shape [ num_pts_u x num_pts_v x 7 ]
        single_grid = np.concatenate([points, normals, mask], axis=2)

        return np.transpose(single_grid, (2, 0, 1))

    def extract_coedge_point_grids(self, body, entity_mapper):
        """
        Extract coedge grids (aligned with the coedge direction).

        The coedge grids will be of size

            [ num_coedges x 12 x num_u ]

        The values are

            - x, y, z    (coordinates of the points)
            - tx, ty, tz (tangent of the curve, oriented to match the coedge)
            - Lx, Ly, Lz (Normal for the left face)
            - Rx, Ry, Rz (Normal for the right face)
        """
        coedge_grids = []
        solid = None
        try: 
            solid = Solid(body)
            top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
            for wire in top_exp.wires():
                wire_exp = TopologyUtils.WireExplorer(wire)
                for coedge in wire_exp.ordered_edges():
                    assert len(coedge_grids) == entity_mapper.halfedge_index(coedge)
                    occwl_oriented_edge = Edge(coedge)
                    faces = [f for f in solid.faces_from_edge(occwl_oriented_edge)]
                    coedge_grids.append(
                        self.extract_coedge_point_grid(occwl_oriented_edge, faces)
                    )
            return coedge_grids # np.stack(coedge_grids)
        except AssertionError:
            if solid is not None: 
                top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
                for wire in top_exp.wires():
                    wire_exp = TopologyUtils.WireExplorer(wire)
                    for coedge in wire_exp.ordered_edges():
                        #assert len(coedge_grids) == entity_mapper.halfedge_index(coedge)
                        occwl_oriented_edge = Edge(coedge)
                        faces = [f for f in solid.faces_from_edge(occwl_oriented_edge)]
                        try:
                            coedge_grids.append(
                                self.extract_coedge_point_grid(occwl_oriented_edge, faces)
                            )
                        except AssertionError:
                            coedge_grids.append(np.zeros((12, 4)))
                return coedge_grids #np.stack(coedge_grids)
            else:
                print("skipped sample")
                return None

    def extract_coedge_point_grid(self, coedge, faces):
        """
        Extract a coedge grid (aligned with the coedge direction).

        The coedge grids will be of size

            [ 12 x num_u ]

        The values are

            - x, y, z    (coordinates of the points)
            - tx, ty, tz (tangent of the curve, oriented to match the coedge)
            - Lx, Ly, Lz (Normal for the left face)
            - Rx, Ry, Rz (Normal for the right face)
        """
        if math.isnan(self.edge_length_feature(coedge.topods_shape())):
            return np.zeros((12, 20))
        else:
            num_u = 50 + math.ceil(200 * self.edge_length_feature(coedge.topods_shape()))
        #num_u = 50 + math.ceil(5 * self.edge_length_feature(coedge.topods_shape()))
        try:
            coedge_data = EdgeDataExtractor(coedge, faces, num_samples=num_u, use_arclength_params=True)
        except RuntimeError:
            return np.zeros((12, num_u))
        
        if not coedge_data.good:
            # We hit a problem evaluating the edge data.  This may happen if we have
            # an edge with not geometry (like the pole of a sphere).
            # In this case we return zeros
            return np.zeros((12, num_u))

        single_grid = np.concatenate(
            [
                coedge_data.points,
                coedge_data.tangents,
                coedge_data.left_normals,
                coedge_data.right_normals,
            ],
            axis=1,
        )
        return single_grid #np.transpose(single_grid, (1, 0))

    def extract_coedge_local_coordinate_systems(self, body, entity_mapper):
        """
        The coedge LCS is a special coordinate system which aligns with the B-Rep
        geometry.

            - The origin will be at the midpoint of the edge.
            - The w_vec will be the normal vector of the left face.
            - The u_ref will be the coedge tangent at the midpoint.  We get the u_vec by projecting this normal
              to the w_vec
            - The v_vec is computed from the cross product
            - The scale factor will be 1.0.  We keep track of some scale factors in another tensor

        Returns a tensor of size [ num_coedges x 4 x 4]

        This is a homogeneous transform matrix from local to global coordinates
        """
        solid = Solid(body)
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        coedge_lcs = []
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(coedge_lcs) == entity_mapper.halfedge_index(coedge)
                occwl_oriented_edge = Edge(coedge)
                faces = [f for f in solid.faces_from_edge(occwl_oriented_edge)]
                coedge_lcs.append(
                    self.extract_coedge_local_coordinate_system(
                        occwl_oriented_edge, faces
                    )
                )

        return np.stack(coedge_lcs)

    def extract_coedge_local_coordinate_system(self, oriented_edge, faces):
        """
        The coedge LCS is a special coordinate system which aligns with the B-Rep
        geometry.

            - The origin will be at the midpoint of the edge.
            - The w_vec will be the normal vector of the left face.
            - The u_ref will be the coedge tangent at the midpoint.  We get the u_vec by projecting this normal
              to the w_vec
            - The v_vec is computed from the cross product
            - The scale factor will be 1.0.  We keep track of some scale factors in another tensor

        Returns a tensor of size [ 4 x 4]

        This is a homogeneous transform matrix from local to global coordinates
            [[ u_vec.x  v_vec.x  v_vec.x  orig.x]
             [ u_vec.y  v_vec.y  v_vec.y  orig.y]
             [ u_vec.z  v_vec.z  v_vec.z  orig.z]
             [ 0        0        0        1     ]]
        """
        num_u = 3
        edge_data = EdgeDataExtractor(
            oriented_edge, faces, num_samples=num_u, use_arclength_params=True
        )
        if not edge_data.good:
            # We hit a problem evaluating the edge data.  This may happen if we have
            # an edge with not geometry (like the pole of a sphere).
            # We want to return zeros in this case
            return np.zeros((4, 4))
        origin = edge_data.points[1]
        w_vec = edge_data.left_normals[1]

        # Make sure w_vec is a unit vector
        w_vec = w_vec / np.linalg.norm(w_vec)

        # We need to project v_ref normal to w_vec
        v_ref = edge_data.tangents[1]
        v_vec = self.try_to_project_normal(w_vec, v_ref)
        if v_vec is None:
            # This happens when v_ref is parallel to w_vec.
            # In this case we just pick any v_vec at random
            v_vec = self.any_orthogonal(v_vec)

        u_vec = np.cross(v_vec, w_vec)

        # The upper part of the matric should look like this
        # [[ u_vec.x  v_vec.x  v_vec.x  orig.x]
        #  [ u_vec.y  v_vec.y  v_vec.y  orig.y]
        #  [ u_vec.z  v_vec.z  v_vec.z  orig.z]]
        mat_upper = np.transpose(np.stack([u_vec, v_vec, w_vec, origin]))

        mat_lower = np.expand_dims(np.array([0, 0, 0, 1]), axis=0)
        mat = np.concatenate([mat_upper, mat_lower], axis=0)

        return mat

    def try_to_project_normal(self, vec, ref):
        """
        Try to project the vector `ref` normal to vec
        """
        dp = np.dot(vec, ref)
        delta = dp * vec
        normal_dir = ref - delta
        length = np.linalg.norm(normal_dir)
        eps = 1e-7
        if length < eps:
            # Failed to project this vector normal
            return None

        # Return a unit vector
        return normal_dir / length

    def any_orthogonal(self, vec):
        """
        Find any random vector orthogonal to the given vector
        """
        nx = self.try_to_project_normal(vec, np.array([1, 0, 0]))
        if nx is not None:
            return nx

        ny = self.try_to_project_normal(vec, np.array([0, 1, 0]))
        if ny is not None:
            return ny

        nz = self.try_to_project_normal(vec, np.array([0, 0, 1]))
        assert (
            nz is not None
        ), f"Something is wrong with vec {vec}.  No orthogonal vector found"
        return nz

    def bounding_box_point_cloud(self, pts):
        assert pts.shape[1] == 3
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
        return np.array(box)

    def scale_from_point_grids(self, grids):
        assert grids.shape[1] == 7
        face_pts = np.transpose(grids[:, :3, :, :].reshape((3, -1)))
        return self.scale_from_point_cloud(face_pts)

    def scale_from_point_cloud(self, pts):
        assert pts.shape[1] == 3
        bbox = self.bounding_box_point_cloud(pts)
        diag = bbox[1] - bbox[0]
        scale = 2.0 / max(diag[0], diag[1], diag[2])
        return scale

    def extract_scale_factors(self, next, mate, face, face_point_grids, coedge_point_grids):
        """
        The scale factors which need to be applied to the LCS for scale
        invariance
        """

        identity = np.arange(next.size, dtype=next.dtype)
        prev = np.zeros(next.size, dtype=next.dtype)
        prev[next] = identity

        num_coedges = mate.size

        scales = []
        scale_from_faces = False

        if scale_from_faces:
            # Probably very slow
            for i in range(num_coedges):
                left_index = face[i]
                right_index = face[mate[i]]
                left = face_point_grids[left_index]
                right = face_point_grids[right_index]
                scale = self.scale_from_point_grids(np.stack([left, right]))
                scales.append(scale)
        else:
            for i in range(num_coedges):
                # This is a bit like a brepnet kernel.
                # We use the walks
                # c
                # c->next
                # c->prev
                # c->mate->next
                # c->mate->prev

                coedges = []
                coedges.append(i)
                coedges.append(next[i])
                coedges.append(prev[i])
                coedges.append(next[mate[i]])
                coedges.append(prev[mate[i]])
                points_from_coedges = []

                for coedge_index in coedges:
                    points = coedge_point_grids[coedge_index, :3]
                    num_u = 10
                    assert points_from_coedges.shape[0] == 3
                    assert points_from_coedges.shape[1] == num_u
                    points_from_coedges.append(points)
                points_from_coedges = np.concatenate(points_from_coedges, axis=1)
                points_from_coedges = points_from_coedges.transpose(points_from_coedges)
                scale = self.scale_from_point_cloud(points_from_coedges)
                scales.append(scale)
        return np.array(scales)

    def extract_coedge_reverse_flags(self, body, entity_mapper):
        """
        The flags for each coedge telling us if it is reversed wrt
        its parent edge.   Notice that when coedge features are
        created, we need to reverse point ordering, flip tangent directions
        and swap left and right faces based on this flag.
        """
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        reverse_flags = []
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for coedge in wire_exp.ordered_edges():
                assert len(reverse_flags) == entity_mapper.halfedge_index(coedge)
                reverse_flags.append(self.reversed_edge_feature(coedge))
        return np.stack(reverse_flags)

    def build_incidence_arrays(self, body, entity_mapper):
        oriented_top_exp = TopologyUtils.TopologyExplorer(
            body, ignore_orientation=False
        )
        num_coedges = len(entity_mapper.halfedge_map)

        next = np.zeros(num_coedges, dtype=np.uint32)
        mate = np.zeros(num_coedges, dtype=np.uint32)

        # Create the next, pervious and mate permutations
        for loop in oriented_top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            first_coedge_index = None
            previous_coedge_index = None
            for coedge in wire_exp.ordered_edges():
                coedge_index = entity_mapper.halfedge_index(coedge)

                # Set up the mating co-edge
                mating_coedge = coedge.Reversed()
                if entity_mapper.halfedge_exists(mating_coedge):
                    mating_coedge_index = entity_mapper.halfedge_index(mating_coedge)
                else:
                    # If a coedge has no mate then we mate it to
                    # itself.  This typically happens at the poles
                    # of sphere
                    mating_coedge_index = coedge_index
                mate[coedge_index] = mating_coedge_index

                # Set up the next coedge
                if first_coedge_index == None:
                    first_coedge_index = coedge_index
                else:
                    next[previous_coedge_index] = coedge_index
                previous_coedge_index = coedge_index

            # Close the loop
            next[previous_coedge_index] = first_coedge_index

        # Create the arrays from coedge to face
        coedge_to_edge = np.zeros(num_coedges, dtype=np.uint32)
        coedge_to_loop = np.zeros(num_coedges, dtype=np.uint32)
        for loop in oriented_top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                coedge_index = entity_mapper.halfedge_index(coedge)
                edge_index = entity_mapper.edge_index(coedge)
                mating_coedge = coedge.Reversed()
                if entity_mapper.halfedge_exists(mating_coedge):
                    mating_coedge_index = entity_mapper.halfedge_index(mating_coedge)
                else:
                    # If a coedge has no mate then we mate it to
                    # itself. This typically happens at the poles
                    # of sphere
                    mating_coedge_index = coedge_index
                coedge_to_edge[coedge_index] = edge_index
                coedge_to_edge[mating_coedge_index] = edge_index
                
                loop_index = entity_mapper.loop_index(loop)
                coedge_to_loop[coedge_index] = loop_index
                

        # Loop over the faces and make the back
        # pointers back to the edges
        coedge_to_face = np.zeros(num_coedges, dtype=np.uint32)
        unoriented_top_exp = TopologyUtils.TopologyExplorer(
            body, ignore_orientation=True
        )
        for face in unoriented_top_exp.faces():
            face_index = entity_mapper.face_index(face)
            for loop in unoriented_top_exp.wires_from_face(face):
                wire_exp = TopologyUtils.WireExplorer(loop)
                for coedge in wire_exp.ordered_edges():
                    coedge_index = entity_mapper.halfedge_index(coedge)
                    coedge_to_face[coedge_index] = face_index
        
        
        return next, mate, coedge_to_face, coedge_to_edge, coedge_to_loop

    def check_unique_coedges(self, top_exp):
        coedge_set = set()
        for loop in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(loop)
            for coedge in wire_exp.ordered_edges():
                orientation = coedge.Orientation()
                tup = (coedge, orientation)

                # We want to detect the case where the coedges
                # are not unique
                if tup in coedge_set:
                    return False

                coedge_set.add(tup)

        return True

    def check_closed(self, body):
        # In Open Cascade, unlinked (open) edges can be identified
        # as they appear in the edges iterator when ignore_orientation=False
        # but are not present in any wire
        top_exp = TopologyUtils.TopologyExplorer(body, ignore_orientation=False)
        edges_from_wires = self.find_edges_from_wires(top_exp)
        edges_from_top_exp = self.find_edges_from_top_exp(top_exp)
        missing_edges = edges_from_top_exp - edges_from_wires
        return len(missing_edges) == 0

    def find_edges_from_wires(self, top_exp):
        edge_set = set()
        for wire in top_exp.wires():
            wire_exp = TopologyUtils.WireExplorer(wire)
            for edge in wire_exp.ordered_edges():
                edge_set.add(edge)
        return edge_set

    def find_edges_from_top_exp(self, top_exp):
        edge_set = set(top_exp.edges())
        return edge_set

    def check_manifold(self, top_exp):
        faces = set()
        for shell in top_exp.shells():
            for face in top_exp._loop_topo(TopAbs_FACE, shell):
                if face in faces:
                    return False
                faces.add(face)
        return True


def load_json(pathname):
    with open(pathname, "r") as fp:
        return json.load(fp)


def check_face_indices(step_file, mesh_dir):
    if mesh_dir is None:
        # Nothing to check
        return True
    # Check against the given meshes and Fusion labels
    validator = FaceIndexValidator(step_file, mesh_dir)
    return validator.validate()


def crosscheck_faces_and_seg_file(infile, seg_dir):
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


def extract_brepnet_features(infile, output_path, feature_schema, mesh_dir, seg_dir):
    if not check_face_indices(infile, mesh_dir):
        return
    if not crosscheck_faces_and_seg_file(infile, seg_dir):
        return
    extractor = BRepExtractor(infile, output_path, feature_schema)
    extractor.process()


def run_worker(worker_args):
    infile = worker_args[0]
    output_path = worker_args[1]
    feature_schema = worker_args[2]
    mesh_dir = worker_args[3]
    seg_dir = worker_args[4]
    extract_brepnet_features(infile, output_path, feature_schema, mesh_dir, seg_dir)


def filter_out_files_which_are_already_converted(files, output_path):
    files_to_convert = []
    for file in files:
        output_file = output_path / (file.stem + ".npz")
        if not output_file.exists():
            files_to_convert.append(file)
    return files_to_convert


def extract_brepnet_data_from_step(step_path, output_path, mesh_dir=None, seg_dir=None, feature_list_path=None, force_regeneration=True, num_workers=1,):
    parent_folder = Path(__file__).parent.parent
    if feature_list_path is None:
        feature_list_path = parent_folder / "feature_lists/all.json"
    feature_schema = load_json(feature_list_path)
    files = [f for f in step_path.glob("**/*.stp")]
    step_files = [f for f in step_path.glob("**/*.step")]
    files.extend(step_files)

    if not force_regeneration:
        files = filter_out_files_which_are_already_converted(files, output_path)

    use_many_threads = num_workers > 1
    if use_many_threads:
        worker_args = [
            (f, output_path, feature_schema, mesh_dir, seg_dir) for f in files
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(run_worker, worker_args), total=len(worker_args)))
    else:
        for file in tqdm(files):
            extract_brepnet_features(file, output_path, feature_schema, mesh_dir, seg_dir)

    gc.collect()
    print("Completed pipeline/extract_feature_data_from_step.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step_path", type=str, required=True, help="Path to load the step files from"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the save intermediate brep data",
    )
    parser.add_argument(
        "--feature_list",
        type=str,
        required=False,
        help="Optional path to the feature lists",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of worker threads"
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        help="Optionally cross check with Fusion Gallery mesh files to check the segmentation labels",
    )
    parser.add_argument(
        "--seg_dir",
        type=str,
        help="Optionally provide a directory containing segmentation labels seg files.",
    )
    args = parser.parse_args()

    step_path = Path(args.step_path)
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir()

    mesh_dir = None
    if args.mesh_dir is not None:
        mesh_dir = Path(args.mesh_dir)

    seg_dir = None
    if args.seg_dir is not None:
        seg_dir = Path(args.seg_dir)

    feature_list_path = None
    if args.feature_list is not None:
        feature_list_path = Path(args.feature_list)

    extract_brepnet_data_from_step(
        step_path, output_path, mesh_dir, seg_dir, feature_list_path, args.num_workers
    )
