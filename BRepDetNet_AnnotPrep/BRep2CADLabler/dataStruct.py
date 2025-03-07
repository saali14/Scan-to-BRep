import numpy as np 

class LabelBoundaryDataStruct:
    def __init__(self, dataBoundary=None, dataJunction=None):
        if dataBoundary is None:
            self.dataBndr = {"v_vertex_Ids": [], 
                            "v_close_loop_Id":[],
                            "v_close_mate_loop_Id":[],
                            "v_parentedge_Id": [], 
                            "v_memberedge_type": [],
                            "v_proximity_weigts": [], 
                            "v_mateface_Id1": [],
                            "v_mateface_Id2": [], 
                            "v_edgefeats": [], 
                            }
        if dataJunction is None:
            self.dataJunc = {"j_vertex_Ids": [], 
                            "j_close_loop_Ids":[],
                            "j_close_mate_loop_Ids":[],
                            "j_parentedge_Ids": [], 
                            "j_nextedge_Ids": [], 
                            "j_mateface_Id1s": [],
                            "j_mateface_Id2s": []
                            }
            
    def load_npz_BoundaryVertsLabel(self, ):
        return
    
    def load_npz_JunctionVertsLabel(self, ):
        return 
    
    def save_npz_data_BRep2CAD_BoundaryVertsLabel(self, output_pathname, data):
        #NOTE --> annotations related to Closed-Edges/Loops
        num_vertex_Ids = np.asarray(data["v_vertex_Ids"]).shape[0]
        num_closeloop_Id = np.asarray(data["v_close_loop_Id"]).shape[0]
        num_closemateloop_Id = np.asarray(data["v_close_mate_loop_Id"]).shape[0]
        num_parentedge_Id = np.asarray(data["v_parentedge_Id"]).shape[0]
        num_memberedge_type = np.asarray(data["v_memberedge_type"]).shape[0] # TODO
        num_proximity_weigts = np.asarray(data["v_proximity_weigts"]).shape[0] # TODO
        num_edgeFeats = np.asarray(data["v_edgefeats"]).shape[0]
        
        #NOTE --> annotations related-to Closed-Faces 
        num_mateface_Id1 = np.asarray(data["v_mateface_Id1"]).shape[0]
        num_mateface_Id2 = np.asarray(data["v_mateface_Id2"]).shape[0]
        
        # make a check if the lengths of all arrays are same or not!
        #TODO::
        
        np.savez(
            output_pathname,
            # NOTE Edge-Related Attributes 
            v_vertex_Ids = data["v_vertex_Ids"],
            v_close_loop_Id = data["v_close_loop_Id"],
            v_close_mate_loop_Id = data["v_close_mate_loop_Id"],
            v_parentedge_Id = data["v_parentedge_Id"],
            v_memberedge_type = data["v_memberedge_type"], # TODO
            v_proximity_weigts = data["v_proximity_weigts"], # TODO
            v_edgefeats = data["v_edgefeats"],
                                
            # NOTE Face-Related Attributes 
            v_mateface_Id1 = data["v_mateface_Id1"],
            v_mateface_Id2 = data["v_mateface_Id2"]
        )
    
    
    #NOTE --> This is incomplete because we need to know what is maximum 
    #         number of Closed-Faces and Closed-Loops to set for a corner....
    #     --> need to decide after annotating boundary vertices..
    def save_npz_data_BRep2CAD_JunctionVertsLabel(self, output_pathname, data):
        #NOTE --> annotations related to Closed-Edges/Loops
        num_vertex_Ids = np.asarray(data["j_vertex_Ids"]).shape[0]
        num_closeloop_Id = np.asarray(data["j_close_loop_Ids"]).shape[0]
        num_parentedge_IDs = np.asarray(data["j_parentedge_Ids"]).shape[0]
        #num_memberedge_type = np.asarray(data["j_memberedge_types"]).shape[0]
        #num_proximity_weigts = np.asarray(data["j_proximity_weigts"]).shape[0]
        
        #NOTE --> annotations related-to Closed-Faces 
        num_mateface_Id1 = np.asarray(data["j_mateface_Id1s"]).shape[0]
        num_mateface_Id2 = np.asarray(data["j_mateface_Id2s"]).shape[0]

        np.savez(
            output_pathname,
            # NOTE Edge-Related Attributes 
            j_vertex_Ids = data["j_vertex_Ids"],
            j_close_loop_Ids = data["j_close_loop_Ids"],            
            j_parentedge_Ids = data["j_parentedge_Ids"],            
            
            # NOTE Face-Related Attributes 
            j_mateface_Id1s = data["j_mateface_Id1s"],
            j_mateface_Id2s = data["j_mateface_Id2s"]
        )

class LabelFaceDataStruct:
    def __init__(self, data=None):
        if data == None:
            self.data = {"v_face_Ids": [], 
                        "v_face_type": [],
                        "v_face_feat": [],
            }
        else:
            return
    
    def load_npz_FaceVertsLabel(self, ):
        return
    
    def save_npz_data_BRep2CAD_FaceVertsLabel(self, output_pathname, data):
        num_face_Ids = np.asarray(data["v_face_Ids"]).shape[0]
        num_face_type = np.asarray(data["v_face_type"]).shape[0]
        num_face_feat = np.asarray(data["v_face_feat"]).shape[0]
                
        # make a check if the lengths of all arrays are same or not!
        #TODO::
        
        np.savez(
            output_pathname,
            # NOTE Edge-Related Attributes 
            v_face_Ids = data["v_face_Ids"],
            v_face_type = data["v_face_type"],
            v_face_feat  = data["v_face_feat"],
            )
