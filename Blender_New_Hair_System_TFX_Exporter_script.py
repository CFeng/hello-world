# =============================================================Description================================================================
# This exporter is designed to add the functionality of exporting hair curves (the Blender new hair system) as TressFX (TFX) files to 
# Blender. This is done by retrieving the information required by TFX asset loader from the Blender model using Blender Python API (Bpy).
# The exporter extracts the control points of each strand and the value of additional subdivision specified by the user in Blender UI. 
# Based on these two values, the number of vertices in each strand after performing subdivision is calculated. This number might 
# not meet the conditions set by the TFX asset loader. In that case, the algorithm finds the closest larger number of vertices 
# acceptable by the asset loader and uses interpolation to add those vertices. If the number of vertices are higher than the maximum
# value expected by the asset loader, it first tries to decrease the number of vertices by applying fewer rounds of subdivision and
# if not possible, it applies vertex subsampling by removing some vertices. Every change made by the algorithm to the final exported
# hair strands are reported to the user with warning pop-up boxes.
# =========================================================================================================================================

# =============================================================Future Improvements=========================================================
# Improvement #1: Pop up window when file is being overwritten instead of red bar checkout the control points and vertices

# Improvement #2: ExportTFX can be improved by modifying the invoke method of ExportHelper so the file explorer is not opened 
# to just open it inside the execute function

# Improvement #3: Use the same naming policy for the whole code, either camelCase or underlines

# Improvement #4: Add ok buton to let the user choose how to proceed when changing the hair vertices

# Improvement #5: (High Priority): This change will make the code more readible and the algorithm more robust.
# The subdivide algorithm should change. Currently the calculations for numVerticesPerStrand, points_added, addVertFix and extraVert
# are based on the number of vertices in the strand but they should be based on the number of segments (number_of_vertices - 1). 
# The new algorithm should turn every 4 points which is a segment into n points (where n-4 would be points_added). addVertFix will be 
# extraVert//n_segments and extraVert will be extraVert%n_segments. Keep in mind that the first and last vertices are always part of 
# the strand.

# Improvement #6: Divide the code into modules in different files
# =========================================================================================================================================

# =============================================================Known Bugs==================================================================
# Error #1: # unregister() function is not working, also when doing unregister with the current method, blender gets slow and needs 
# to be restarted.
# =========================================================================================================================================

import bpy, ctypes, bpy_extras, os, math, mathutils, bmesh
import numpy as np
from collections import namedtuple

from bpy.props import (
    BoolProperty,
    EnumProperty,
    StringProperty,
)

# Constants
TRESSFX_SIM_THREAD_GROUP_SIZE = 64
TRESSFX_MAX_INFLUENTIAL_BONE_COUNT = 4


bl_info = {
    # required
    'name': 'Blender TFX Exporter',
    'blender': (3, 3, 0),
    'category': 'Object',
    # optional
    'version': (1, 0, 0),
    'author': 'Abtin Riasatian',
    'description': '',
}

class TressFXTFXFileHeader(ctypes.Structure):
    _fields_ = [('version', ctypes.c_float),
                ('numHairStrands', ctypes.c_uint),
                ('numVerticesPerStrand', ctypes.c_uint),
                ('offsetVertexPosition', ctypes.c_uint),
                ('offsetStrandUV', ctypes.c_uint),
                ('offsetVertexUV', ctypes.c_uint),
                ('offsetStrandThickness', ctypes.c_uint),
                ('offsetVertexColor', ctypes.c_uint),
                ('reserved', ctypes.c_uint * 32)]

class tressfx_float4(ctypes.Structure):
    _fields_ = [('x', ctypes.c_float),
                ('y', ctypes.c_float),
                ('z', ctypes.c_float),
                ('w', ctypes.c_float)]

class tressfx_float2(ctypes.Structure):
    _fields_ = [('x', ctypes.c_float),
                ('y', ctypes.c_float)]


# used the Blender source code: BKE_curves.hh
def calculate_basis(parameter):
  t = parameter
  s = 1.0 - parameter
  weights = []
  w_a = -t * s * s
  w_b = 2.0 + t * t * (3.0 * t - 5.0)
  w_c = 2.0 + s * s * (3.0 * s - 5.0)
  w_d = -s * t * t
  weights = [w_a, w_b, w_c, w_d]
  return weights

# used the Blender source code: BKE_curves.hh
def interpolate(four_points, parameter):
    weights = calculate_basis(parameter)
    weights_np = np.array(weights).reshape(1, 4)
    four_points_np = np.array(four_points)
    res = 0.5 * np.matmul(weights_np, four_points_np)
    return res[0]

# used the Blender source code: curve_catmoll_rom.cc
def evaluate_segment(four_points, res_size):
    if res_size==0:
        return []
    
    step = 1.0 / (res_size)
    res = []
    res.append(four_points[1])
    for i in range(1, res_size):
        res.append(interpolate(four_points, i*step))
        
    return res

# interpolate points evenly on a straight line between two points
def linear_interpolation(coords, total_points):
    left_point = mathutils.Vector(coords[0])
    right_point = mathutils.Vector(coords[1])
    res = []
    
    for i in range(total_points):
        t = i / (total_points-1)
        interpolated_point = left_point.lerp(right_point, t)
        res.append(interpolated_point.to_tuple())
        
    return res

def subdivide(coords, points_added, extraVert):
    # In Blender, each strand can have a minimum of 2 vertices so cases with fewer vertices are not handled
        
    # Subdividing the vertices into 4 when we have 2 vertices per strand since AtomTressFX expects the number of vertices per strands to be a power of 2, greater than 2 and less than or equal to 64.
    # Since with 2 vertices we don't have a curve shape, we use linear interpolation.
    if len(coords) == 2:
        res = linear_interpolation(coords, 4)
    
    else:
        addVertFix = extraVert // len(coords)
        extraVert = extraVert % len(coords) + addVertFix
        
        addVert = 1 if extraVert>0 else 0
        
        res_beg = evaluate_segment([coords[0], coords[0], coords[1], coords[2]], points_added + addVertFix + addVert + 1)
        extraVert = max(extraVert-1, 0)
        addVert = 1 if extraVert>0 else 0
        
        res_end = evaluate_segment([coords[-3], coords[-2], coords[-1], coords[-1]], points_added + addVertFix + addVert)
        extraVert = max(extraVert-1, 0)
        addVert = 1 if extraVert>0 else 0
        
        res = []
        res += res_beg
    
        for st_ind in range(1, len(coords)-2):
            res_tmp = evaluate_segment([coords[st_ind-1], coords[st_ind], coords[st_ind+1], coords[st_ind+2]], points_added + addVertFix + addVert)
            res += res_tmp
            extraVert = max(extraVert-1, 0)
            addVert = 1 if extraVert>0 else 0
        
        res += res_end
        res += [coords[-1]]
           
    res = [tuple(x) for x in res]
    
    return res

def subdivide_strand(strand_coords, hairSubdiv, pointsAdded, extraVerticesToAdd):
    if hairSubdiv == 0 and extraVerticesToAdd>0:
        strand_coords = subdivide(strand_coords, 1, extraVerticesToAdd-1)
  
    elif hairSubdiv > 0:
        for i in range(hairSubdiv):
            if i == hairSubdiv-1:
                strand_coords = subdivide(strand_coords, pointsAdded, extraVerticesToAdd)
            else:
                strand_coords = subdivide(strand_coords, pointsAdded, 0)
        
    return strand_coords
    
def subsample_strand(strand_coords):
    res = strand_coords[:]
    interval_to_remove = len(res) // TRESSFX_SIM_THREAD_GROUP_SIZE
       
    if interval_to_remove>1:
        for i in range(len(res)-1, -1, -(interval_to_remove-1)):
            del res[i]
         
    remaining_vertices = len(res) % TRESSFX_SIM_THREAD_GROUP_SIZE
    interval_to_remove = len(res) // remaining_vertices
        
    for i in range(len(res)-1, len(res)-interval_to_remove*remaining_vertices-1, -interval_to_remove):
        del res[i]
            
    return res

def calc_subdivs_to_drop(x, pointsAdded):
    tmp = x/TRESSFX_SIM_THREAD_GROUP_SIZE
    tmp = math.log(tmp, pointsAdded)
    tmp = math.ceil(tmp)
    tmp = max(0, tmp)
    return tmp

def calc_target_num_vertices_per_strand(x):
    range_end = int(math.log(TRESSFX_SIM_THREAD_GROUP_SIZE, 2)) + 1
    
    for p in range(2, range_end):
        res = 2**p
        if (res == TRESSFX_SIM_THREAD_GROUP_SIZE) or (res >= x):
            return res

def find_closest_vert_ind(location, face_verts):
    dist_list = [math.dist(location, vert.co) for vert in face_verts]
    closest_vert_index = dist_list.index(min(dist_list))
    closest_vert = face_verts[closest_vert_index]
    return closest_vert

def getInfluentialBones(base_mesh, bone_list, root_point):
    "Gets the 4 most influential bones for the strand associated with the root point"
    
    weight_boneIndex_pair = namedtuple('weight_boneIndex_pair', 'bone_index weight')
    weight_boneIndex_pair_list = []

    # find the closest location on mesh to the root point
    is_found, location, normal, face_ind = base_mesh.closest_point_on_mesh(root_point)
    
    # get the face object in which the closest location is located
    face_object = base_mesh.data.polygons[face_ind]
    # get the vertices of the face object
    face_verts = [base_mesh.data.vertices[vert_ind] for vert_ind in face_object.vertices]
    # get the closest vertex to the closest location
    closest_vert = find_closest_vert_ind(location, face_verts)
        
    # find bones that influence the vertex and add pairs of (bone index, influence weight) to weight_boneIndex_pair_list
    for bone_ind, bone in enumerate(bone_list):
        try:
            weight = base_mesh.vertex_groups[bone.name].weight(closest_vert.index)
            pair = weight_boneIndex_pair(bone_ind, weight)
            weight_boneIndex_pair_list.append(pair)
        except:
            pass
            
    # sort the list in descending order based on the weight
    weight_boneIndex_pair_list.sort(key=lambda x: x.weight, reverse=True)
    
    # get the first 4 bones (4 most influential bones) if we have more than 4
    if len(weight_boneIndex_pair_list) > TRESSFX_MAX_INFLUENTIAL_BONE_COUNT:
        weight_boneIndex_pair_list = weight_boneIndex_pair_list[:4]
        
    # add dummy bones to the list in case we have fewer than 4 bones
    if len(weight_boneIndex_pair_list) < TRESSFX_MAX_INFLUENTIAL_BONE_COUNT:
        for i in range(TRESSFX_MAX_INFLUENTIAL_BONE_COUNT - len(weight_boneIndex_pair_list)):
            weight_boneIndex_pair_list.append(weight_boneIndex_pair(-1, 0))
            
    return weight_boneIndex_pair_list
        

def save_tfxbone_binary_file(saveFilePath, base_mesh, bone_list, root_point_list):
    
    '''
    Writes and saves the tfxbone file
    '''
    
    if os.path.exists(saveFilePath):
        os.remove(saveFilePath)

    f = open(saveFilePath, "wb")
    
    # save number of bones
    f.write(ctypes.c_int(len(bone_list)))

    # Write all bone names
    for i in range(len(bone_list)):
        bone = bone_list[i]
        bone_name = bone.name
        
        # Write the bone index
        f.write(ctypes.c_int(i))
        # Write size of the string, add 1 to leave room for the null terminate
        f.write(ctypes.c_int(len(bone_name) + 1))
        # Write the characters of the string 1 by 1.
        for j in range(len(bone_name)):
            f.write(ctypes.c_byte(ord(bone_name[j])))
        # Add a zero to null terminate the string
        f.write(ctypes.c_byte(0))

    # number of strands which don't have any influential bones
    strands_with_no_bone_count = 0
    # Write the number of strands
    f.write(ctypes.c_int(len(root_point_list)))

    for i in range(len(root_point_list)):

        # Write the straind index
        f.write(ctypes.c_int(i))
        
        # get the 4 most influential bones for the strand (dummy bones are addded if fewer than 4 bones are available)
        weight_boneIndex_pair_list = getInfluentialBones(base_mesh, bone_list, root_point_list[i])
        
        # if the bone index of the first bone is -1, it means that no influential bones existed
        print(weight_boneIndex_pair_list[0][0])
        if weight_boneIndex_pair_list[0][0] == -1:
            strands_with_no_bone_count += 1
            
        weight_sum = float(sum([x.weight for x in weight_boneIndex_pair_list]))
        
        for pair in weight_boneIndex_pair_list:
            # write the bone index
            f.write(ctypes.c_int(pair.bone_index))
            
            # write the normalized bone weight
            if weight_boneIndex_pair_list[0][0] == -1:
                f.write(ctypes.c_float(0))
            else:
                f.write(ctypes.c_float(pair.weight/weight_sum))
    
    f.close()
    
    if strands_with_no_bone_count > 0:
        ShowMessageBox("No bones available for {} strands".format(strands_with_no_bone_count), "Missing Bones", 'INFO')

    return {'FINISHED'}


def save_tfxmesh_file(saveFilePath, curve_obj, decimate_ratio):
   
    '''
    This function is responsible for exporting the collision mesh of the passed hair object, called curve_obj, which is saved into a file 
    with .tfxmesh extension.
    
    It does so by creating a copy of the mesh and applying the Blender collapse decimation to it. This algorithm merges the mesh vertices 
    together progressively, taking the shape of the mesh into account. The input decimate_ratio, specifies the ratio of faces
    to keep after decimation.
    
    The .tfxmesh file is saved as plain text, unlike the .tfx and .tfxbone files which are saved in binary. In short, the data in the file
    includes vertex and triangles information for the collision mesh in addition to 4 most influential bones and their weights for each
    vertex which are required for skinning the collision mesh in O3DE.     
    ''' 
   
    org_mesh = curve_obj.parent
    
    # create a copy of the mesh
    mesh = org_mesh.copy()
    mesh.data = org_mesh.data.copy()
    # select the mesh
    bpy.context.collection.objects.link(mesh)
    bpy.context.view_layer.objects.active = mesh
    # create a decimate modifier
    modifier = mesh.modifiers.new('decimate_modifier','DECIMATE')
    modifier.ratio = decimate_ratio
    modifier.use_collapse_triangulate = True
    # apply decimation on the mesh
    bpy.ops.object.modifier_apply(modifier='decimate_modifier')
    
    bone_count = len(mesh.vertex_groups)
    vertex_count = len(mesh.data.vertices)
    
    print(saveFilePath)
    
    with open(saveFilePath, "w") as f:
     
        f.write("# TressFX collision mesh exported by TressFX Exporter in Blender\n")
        
        # Write number of bones
        f.write("numOfBones %d\n" % bone_count)
        # Write bone names
        f.write("# bone index, bone name\n")
        for i, vg in enumerate(mesh.vertex_groups):
            f.write("%d %s\n" % (i, vg.name))
            
        # Write vertices
        f.write("numOfVertices %d\n" % vertex_count)
        f.write("# vertex index, vertex position x, y, z, normal x, y, z, joint index 0, joint index 1, joint index 2, joint index 3, weight 0, weight 1, weight 2, weight 3\n")
        for i, v in enumerate(mesh.data.vertices):
            co = v.co
            normal = v.normal
            f.write("%d %f %f %f %f %f %f " % (i, co.x, co.y, co.z, normal.x, normal.y, normal.z))

            # Get bone weights for the vertex
            weights = []
            for g in v.groups:
                weights.append((g.group, g.weight))
            # sort the list in descending order based on the weight
            weights.sort(key=lambda x: x[1], reverse=True)
            # take the most influential weights in case there are more than 4
            if len(weights)>4:
                weights = weights[:4]
            # add dummy weights in case there are fewer than 4 weights
            else:
                for i in range(4-len(weights)):
                    weights.append((-1, 0))
            
            # if all weights are zero don't normalize the weights
            weight_sum = sum([x[1] for x in weights])
            if weight_sum == 0:
                normalized_weights = weights
            else:
                normalized_weights = [(g, w/weight_sum) for (g, w) in weights]
                
            index_string_list = [str(g) for (g, w) in normalized_weights]
            index_string = ' '.join(index_string_list) + ' ' 
            f.write(index_string)          
        
            normalized_weights_string_list = [str(w) for (g, w) in normalized_weights]
            normalized_weights_string = ' '.join(normalized_weights_string_list)
            f.write(normalized_weights_string)
                
            f.write("\n")
    
        triangle_count = len(mesh.data.polygons)
    
        # Write triangles
        f.write("numOfTriangles %d\n" % triangle_count)
        f.write("# triangle index, vertex index 0, vertex index 1, vertex index 2\n")
        for i, face in enumerate(mesh.data.polygons):
            f.write("%d %d %d %d\n" % (i, face.vertices[0], face.vertices[1],face.vertices[2]))
        
        # remove the created copy mesh
        bpy.data.meshes.remove(mesh.data)
        # select back the initially selected hair curve
        bpy.context.view_layer.objects.active = curve_obj
        
    f.close()

# getting a curve object as input, this function extracts the required data
# and saves it as a .tfx file
def export_hair_files(saveFilePath, settings_dict, curves_object):
    curves = curves_object.data.curves
    # number of curves
    numCurves = len(curves)
    # number of control points per strand
    ctrlPointCount = curves[0].points_length    
    # number of addtional subdivisions
    hairSubdiv = bpy.context.scene.render.hair_subdiv
    # TODO get from user
    pointsAdded = 2
    # number of vertices per strand
    numVerticesPerStrand = ctrlPointCount * pointsAdded**hairSubdiv
    # if sampling at a lower amount than the full curve set, div by sample over entire set, then mult by sample to jump
    # if offseting, subtract offset amount from full range of loop, so when it's added to the index
    # it won't overshoot the actual number of curves range    
    curveSample = 1
    curveIndex_Offset = 0
    adjustedCurveRange = numCurves//curveSample - curveIndex_Offset
    vertexCountCondition = (2 < numVerticesPerStrand) and (numVerticesPerStrand <= TRESSFX_SIM_THREAD_GROUP_SIZE) and (TRESSFX_SIM_THREAD_GROUP_SIZE%numVerticesPerStrand == 0)
    # if the vertex count does not satisfy the condition, open an error pop up box
    extraVerticesToAdd = 0
    targetNumVerticesPerStrand = numVerticesPerStrand
    do_subsample = False
    
    if not vertexCountCondition:
        targetNumVerticesPerStrand = calc_target_num_vertices_per_strand(numVerticesPerStrand)
        # it's assumed that each subdivision muliplies the number of vertices by pointsAdded
        subdivsToDrop = calc_subdivs_to_drop(numVerticesPerStrand, pointsAdded)
        
        if subdivsToDrop>0:
            if hairSubdiv >= subdivsToDrop:
                hairSubdiv = hairSubdiv - subdivsToDrop
                ShowMessageBox("The number of vertices per strand is {} but should be a power of 2, greater than 2 and less than or equal to 64. So the number of performed additional subdivisions is reduced to {} and some vertices are added by interpolation. The number of vertices in the exported file is {}.".format(numVerticesPerStrand, hairSubdiv, targetNumVerticesPerStrand), "Strand Vertex Count Warning", 'INFO')
                numVerticesPerStrand = numVerticesPerStrand//(pointsAdded**subdivsToDrop)
                
            else:
                ShowMessageBox("The number of vertices per strand is {} but should be a power of 2, greater than 2 and less than or equal to 64. Vertices are subsampled and 64 vertices are saved per strand.".format(numVerticesPerStrand), "Strand Vertex Count Warning", 'INFO')
                targetNumVerticesPerStrand = TRESSFX_SIM_THREAD_GROUP_SIZE
                do_subsample = True
                
        if subdivsToDrop==0:
            ShowMessageBox("The number of vertices per strand is {} but should be a power of 2, greater than 2 and less than or equal to 64. So some vertices are added by interpolation. The number of vertices in the exported file is {}.".format(numVerticesPerStrand, targetNumVerticesPerStrand), "Strand Vertex Count Warning", 'INFO')
        
        extraVerticesToAdd = max(0, targetNumVerticesPerStrand - numVerticesPerStrand)

    # writing the header files
    tfxHeader = TressFXTFXFileHeader()
    tfxHeader.version = 4.0
    tfxHeader.numHairStrands = int(numCurves//curveSample)
    tfxHeader.numVerticesPerStrand = targetNumVerticesPerStrand
    tfxHeader.offsetVertexPosition = ctypes.sizeof(TressFXTFXFileHeader)
    tfxHeader.offsetStrandUV = tfxHeader.offsetVertexPosition + adjustedCurveRange * tfxHeader.numVerticesPerStrand * ctypes.sizeof(tressfx_float4)
    tfxHeader.offsetVertexUV = 0
    tfxHeader.offsetStrandThickness = 0
    tfxHeader.offsetVertexColor = 0   

    current_root_point_list = []
    if os.path.exists(saveFilePath):
        os.remove(saveFilePath)
    f = open(saveFilePath, "wb")
    f.write(tfxHeader)
    
    # adjusted curve range to accomodate sampling and offsetting into the curve set
    for i in range(adjustedCurveRange):
        strand = curves[(i*curveSample) + curveIndex_Offset] 
        strand_coords = [tuple(p.position) for p in strand.points.values()]
        
        if do_subsample:
            strand_vertex_list = subsample_strand(strand_coords)

        else:
            strand_vertex_list = subdivide_strand(strand_coords, hairSubdiv, pointsAdded, extraVerticesToAdd)

        current_root_point_list.append(strand_vertex_list[0])     

        for j in range(0, targetNumVerticesPerStrand):
            pos = strand_vertex_list[j]

            p = tressfx_float4()
            p.x = pos[0]
            p.y = pos[1]
            p.z = pos[2]

            # w component is an inverse mass
            # the first two vertices are immovable
            if j < 2:  
                p.w = 0
            else:
                p.w = 1.0 
            
            # fix the last vertice of strand if bothEndsImmovable is True
            if j == (targetNumVerticesPerStrand-1) and settings_dict['bothEndsImmovable']: 
                p.w = 0
            
            p.x = p.x * curves_object.scale.x
            p.y = p.y * curves_object.scale.y
            p.z = p.z * curves_object.scale.z
            
            # take the rotation value for the hair curve object
            rotation = curves_object.rotation_euler
            # creating a vector from vertex positions to apply the rotation
            vertex_vector = mathutils.Vector((p.x, p.y, p.z))
            # the rotation is applied in place
            vertex_vector.rotate(rotation)
            p.x, p.y, p.z = vertex_vector
            
            f.write(p)
            
    # writing uv strands
    uv_pair_list = curves_object.data.attributes['surface_uv_coordinate'].data.values()
    for i in range(adjustedCurveRange):
        uv_pair = uv_pair_list[(i*curveSample) + curveIndex_Offset] 
        uv_coord = tressfx_float2()
        uv_coord.x, uv_coord.y = uv_pair.vector
        f.write(uv_coord)    

    f.close()
    
    if settings_dict['exportBones']:
        base_mesh = curves_object.parent
        armature = base_mesh.parent
        
        if armature is None:
            ShowMessageBox("No bones available for this model. Only the tfx file will be saved.", "Warning: No Bones Available", 'INFO')
        
        else:
            bone_list = [bn for bn in armature.data.bones]
            tfxbone_saveFilePath = saveFilePath + 'bone'
            save_tfxbone_binary_file(tfxbone_saveFilePath, base_mesh, bone_list, current_root_point_list)
        
    
    if settings_dict['exportCollisionMesh']:
        
        # If the exportCollisionMesh is checked, the exportBones is also checked so armature is defined. If armature is None it means we have no bones available.
        if armature is None:
            ShowMessageBox("No bones available for this model. Only the tfx file will be saved.", "Warning: No Bones Available", 'INFO')
        
        else:
            base_mesh = curves_object.parent
            tfxmesh_saveFilePath = saveFilePath + 'mesh'
            save_tfxmesh_file(tfxmesh_saveFilePath, curves_object, settings_dict['collisionMeshDecimateRatio'])
            
    
    return {'FINISHED'}

def ShowMessageBox(message = "", title = "Message Box", icon = 'INFO'):
    def draw(self, context):
        self.layout.label(text=message)
    bpy.context.window_manager.popup_menu(draw, title = title, icon = icon)

def axis_conversion_ensure(operator, forward_attr, up_attr):
    """
    Function to ensure an operator has valid axis conversion settings, intended
    to be used from :class:`bpy.types.Operator.check`.

    :arg operator: the operator to access axis attributes from.
    :type operator: :class:`bpy.types.Operator`
    :arg forward_attr: attribute storing the forward axis
    :type forward_attr: string
    :arg up_attr: attribute storing the up axis
    :type up_attr: string
    :return: True if the value was modified.
    :rtype: boolean
    """
    def validate(axis_forward, axis_up):
        if axis_forward[-1] == axis_up[-1]:
            axis_up = axis_up[0:-1] + 'XYZ'[('XYZ'.index(axis_up[-1]) + 1) % 3]

        return axis_forward, axis_up

    axis = getattr(operator, forward_attr), getattr(operator, up_attr)
    axis_new = validate(*axis)

    if axis != axis_new:
        setattr(operator, forward_attr, axis_new[0])
        setattr(operator, up_attr, axis_new[1])

        return True
    else:
        return False
        
def _check_axis_conversion(op):
    if hasattr(op, "axis_forward") and hasattr(op, "axis_up"):
        return axis_conversion_ensure(
            op,
            "axis_forward",
            "axis_up",
        )
    return False

def update_save_filepath(hair_curve_obj_name):
    blender_filepath = bpy.context.blend_data.filepath
    file_save_dir = os.path.dirname(blender_filepath)
    file_save_name = os.path.basename(blender_filepath)
    file_save_name = os.path.splitext(file_save_name)[0]
    final_file_save_path = '{}/{}_{}.tfx'.format(file_save_dir, file_save_name, hair_curve_obj_name)
    return final_file_save_path 

# ExportHelper is a helper class, defines filename and invoke() function which calls the file selector
class ExportTFX(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):    
    # important since its how bpy.ops.import_hair.tfx is constructed
    bl_idname = "export_hair.tfx"  
    bl_label = "Export TFX!"
    # ExportHelper mixin class uses this
    filename_ext = ".tfx"

    filter_glob: bpy.props.StringProperty(
        default="*.tfx",
        options={'HIDDEN'},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )
    
    bothEndsImmovable: bpy.props.BoolProperty(
        name="Both ends immovable",
        description="Sets both ends of the strand (the endpoint vertices) to use zero inverse mass",
        default=False,
    )
    
    exportBones: bpy.props.BoolProperty(
        name="Export Bone Weights",
        description="Bone weights will be exported (generates a tfxbone file in addtion to the tfx file) if checked.",
        default=True,
    )
    
    exportCollisionMesh: bpy.props.BoolProperty(
        name="Export Collision Mesh",
        description="Collision mesh will be exported (generates a tfxmesh file in addtion to the tfx file) if checked.",
        default=True,
    ) 

    collisionMeshDecimateRatio: bpy.props.FloatProperty(
        name = "Collision Mesh Decimate Ratio",
        description = "The ratio of faces to keep after decimation of the collision mesh.",
        default = 0.5,
        min = 0.0,
        max = 1.0
    )
 
    # This part is copied from ExportHelper defined in bpy_extras.io_utils =================================
    filepath: StringProperty(
        name="File Path",
        description="Filepath used for exporting the file",
        maxlen=1024,
        subtype='FILE_PATH',
    )
    
    check_existing: BoolProperty(
        name="Check Existing",
        description="Check and warn on overwriting existing files",
        default=True,
        options={'HIDDEN'},
    )

    # subclasses can override with decorator
    # True == use ext, False == no ext, None == do nothing.
    check_extension = True
    hair_curve_ind = 0
    hair_curve_list_org = []
    hair_curve_list = []
    settings_dict = {}
    current_hair_curve_obj = None

    def invoke(self, context, _event):
        if not self.filepath:
            blend_filepath = bpy.context.blend_data.filepath
            if not blend_filepath:
                blend_filepath = "untitled"
            else:
                blend_filepath = os.path.splitext(blend_filepath)[0]

            self.filepath = blend_filepath + self.filename_ext
            
        ## Initialization ---
        # iterating over the selected objects and filtering out any non-hair (non-CURVE) objects
        # make sure to perform only when the hair_curve_list_org is empty which means this part is run only once, 
        # if we do this several times the converted hair objects from the last function call will be added to the list
        self.hair_curve_list_org = [hair_curve for hair_curve in bpy.context.selected_objects if hair_curve.type=='CURVES']
                
        if len(self.hair_curve_list_org)==0:
            ShowMessageBox("No hair object is selected", "No Object Selected Error", 'ERROR')
            return {'FINISHED'}
        
        # Convert every hair model to CURVES and keep the original hair models, 
        # this will select all the generated converted copies and unselects the original objects
        # TODO: this is a quick fix, although the original objects are of the CURVES type but only a few of the strands are retrieved, 
        # this bug was introduced after the exporter was used on Blender 3.5. The exporter was originally developed on Blender 3.3.
        bpy.ops.object.convert(target='CURVES', keep_original=True)
        
        # Populate the hair_curve_list with the converted hair objects
        self.hair_curve_list = [hair_curve for hair_curve in bpy.context.selected_objects if hair_curve.type=='CURVES']
        
        ## ---
        
        ## Exporting Prep ---
        # Here it's assumed that the converted curves, which are stored in hair_curve_list, 
        # have the same order as hair objects stored in hair_curve_list_org
        self.current_hair_curve_obj = self.hair_curve_list[self.hair_curve_ind]
        current_hair_curve_name = self.hair_curve_list_org[self.hair_curve_ind].name
        self.filepath = update_save_filepath(current_hair_curve_name)
        bpy.context.window_manager.fileselect_add(self)     
        ## --- 
           
        return {'RUNNING_MODAL'}

    def check(self, context):
        import os
        change_ext = False
        change_axis = _check_axis_conversion(self)

        check_extension = self.check_extension

        if check_extension is not None:
            filepath = self.filepath
            if os.path.basename(filepath):
                if check_extension:
                    filepath = bpy.path.ensure_ext(
                        os.path.splitext(filepath)[0],
                        self.filename_ext,
                    )
                if filepath != self.filepath:
                    self.filepath = filepath
                    change_ext = True

        return (change_ext or change_axis)
        
    #============================================================================
    
    def reset_variables(self):
        """
        Turns the Blender scene back to the state it was before starting the export process which includes removing the created hair copies,
        setting the original hair objects as selected and resetting the current hair curve index to 0 for the next time the exporter is called.
        """
        
        self.hair_curve_ind = 0
        
        # remove the created hair copies (by unlinking them from collections and removing the objects) and select the original hair objects
        # The following seems to be a better way to remove an object but I could not find the similar functionalities for Blender 3.5
        # https://blender.stackexchange.com/questions/32349/python-scripting-remove-curve-object
        for hair_obj in self.hair_curve_list:
            bpy.context.collection.objects.unlink(hair_obj)
            bpy.data.objects.remove(hair_obj)

        # Select back the original hair objects
        for hair_obj in self.hair_curve_list_org:
            bpy.context.view_layer.objects.active = hair_obj    
            hair_obj.select_set(True)        
    
    
    # since the user might have selected multiple objects, we will need to open the file explorer window multiple times; that's why
    # RUNNING_MODAL  is used which causes the execute function to be called multiple times. Because of this hair_curve_ind and 
    # hair_curve_list are defined as class properties.
    
    # This function is executed when the ExportTFX! button is hit
    def execute(self, context):        
        
        ## Updating the settings for export ---
        self.settings_dict['bothEndsImmovable'] = self.bothEndsImmovable
        self.settings_dict['exportBones'] = self.exportBones
        self.settings_dict['exportCollisionMesh'] = self.exportCollisionMesh
        self.settings_dict['collisionMeshDecimateRatio'] = self.collisionMeshDecimateRatio
        ## ---
        
        ## Doing the export ---
        export_hair_files(self.filepath, self.settings_dict, self.current_hair_curve_obj) 
        ## ---
        
        # Check if this round of exports are done
        if self.hair_curve_ind == len(self.hair_curve_list) - 1:
            # setting the hair_curve_ind to 0 for the next export round

            self.reset_variables()
            
            return {'FINISHED'}
        
        ## Updating the filepath ---
        self.hair_curve_ind += 1
        self.current_hair_curve_obj = self.hair_curve_list[self.hair_curve_ind]
        current_hair_curve_name = self.hair_curve_list_org[self.hair_curve_ind].name
        self.filepath = update_save_filepath(current_hair_curve_name)
        ## ---
        
        # opening the file explorer to save the object
        context.window_manager.fileselect_add(self)

        return {'RUNNING_MODAL'}
        
    def draw(self, context):
        layout = self.layout
    
        layout.prop(self, "bothEndsImmovable")
        layout.prop(self, "exportBones")
        
        if self.exportBones:
            layout.prop(self, "exportCollisionMesh")
        
        else:
            self.exportCollisionMesh = False
    
        if self.exportBones and self.exportCollisionMesh:
            layout.prop(self, "collisionMeshDecimateRatio")
                
            
    def cancel(self, context):
        self.reset_variables()

# this class is created to separate the menu item from the function side of the exporter (handled by ExportTFX)
class TOPBAR_MT_ExportTFX_menu(bpy.types.Menu):
    """Export the selected hair files as TressFX (.tfx)"""
    
    bl_idname = "TOPBAR_MT_ExportTFX_menu_id"
    bl_label = "TOPBAR_MT_ExportTFX_menu_label"

    def draw(self, context):
        # text is the string that is shown for the menu item
        self.layout.operator(ExportTFX.bl_idname, text="TressFX (.tfx)")
        
# classes to register
classes = (TOPBAR_MT_ExportTFX_menu, ExportTFX)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # adding the ExportTFX menu item to the File->Export menu
    bpy.types.TOPBAR_MT_file_export.append(TOPBAR_MT_ExportTFX_menu.draw)

def unregister():
    bpy.ops.script.reload()
#    bpy.types.TOPBAR_MT_editor_menus.remove(TOPBAR_MT_TFX_menu.menu_draw)
#    for cls in classes:
#        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()