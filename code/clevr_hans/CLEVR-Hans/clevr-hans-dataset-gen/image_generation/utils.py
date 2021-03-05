# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import sys
import random
import os
import bpy
import bpy_extras
import math
import numpy as np
import json

"""
Some utility functions for interacting with Blender
"""


def extract_args(input_argv=None):
    """
    Pull out command-line arguments after "--". Blender ignores command-line flags
    after --, so this lets us forward command line arguments from the blender
    invocation to our own script.
    """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if '--' in input_argv:
        idx = input_argv.index('--')
        output_argv = input_argv[(idx + 1):]
    return output_argv


def parse_args(parser, argv=None):
    return parser.parse_args(extract_args(argv))


# I wonder if there's a better way to do this?
def delete_object(obj):
    """ Delete a specified blender object """
    for o in bpy.data.objects:
        o.select = False
    obj.select = True
    bpy.ops.object.delete()


def get_camera_coords(cam, pos):
    """
    For a specified point, get both the 3D coordinates and 2D pixel-space
    coordinates of the point from the perspective of the camera.

    Inputs:
    - cam: Camera object
    - pos: Vector giving 3D world-space position

    Returns a tuple of:
    - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
      in the range [-1, 1]
    """
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = int(scale * scene.render.resolution_x)
    h = int(scale * scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return px, py, z


def set_layer(obj, layer_idx):
    """ Move an object to a particular layer """
    # Set the target layer to True first because an object must always be on
    # at least one layer.
    obj.layers[layer_idx] = True
    for i in range(len(obj.layers)):
        obj.layers[i] = (i == layer_idx)


def add_object(object_dir, name, scale, loc, theta=0):
    """
    Load an object from a file. We assume that in the directory object_dir, there
    is a file named "$name.blend" which contains a single object named "$name"
    that has unit size and is centered at the origin.

    - scale: scalar giving the size that the object should be in the scene
    - loc: tuple (x, y) giving the coordinates on the ground plane where the
      object should be placed.
    """
    # First figure out how many of this object are already in the scene so we can
    # give the new object a unique name
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
    bpy.ops.wm.append(filename=filename)

    # Give it a new name to avoid conflicts
    new_name = '%s_%d' % (name, count)
    bpy.data.objects[name].name = new_name

    # Set the new object as active, then rotate, scale, and translate it
    x, y = loc
    bpy.context.scene.objects.active = bpy.data.objects[new_name]
    bpy.context.object.rotation_euler[2] = theta
    bpy.ops.transform.resize(value=(scale, scale, scale))
    bpy.ops.transform.translate(value=(x, y, scale))


def load_materials(material_dir):
    """
    Load materials from a directory. We assume that the directory contains .blend
    files with one material each. The file X.blend has a single NodeTree item named
    X; this NodeTree item must have a "Color" input that accepts an RGBA value.
    """
    for fn in os.listdir(material_dir):
        if not fn.endswith('.blend'):
            continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, 'NodeTree', name)
        bpy.ops.wm.append(filename=filepath)


def add_material(name, **properties):
    """
    Create a new material and assign it to the active object. "name" should be the
    name of a material that has been previously loaded using load_materials.
    """
    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)

    # Create a new material; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    # then the next time we make a new material it will still be called
    # "Material" and we will still be able to look it up by name
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % mat_count

    # Attach the new material to the active object
    # Make sure it doesn't already have materials
    obj = bpy.context.active_object
    assert len(obj.data.materials) == 0
    obj.data.materials.append(mat)

    # Find the output node of the new material
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == 'Material Output':
            output_node = n
            break

    # Add a new GroupNode to the node tree of the active material,
    # and copy the node tree from the preloaded node group to the
    # new group node. This copying seems to happen by-value, so
    # we can create multiple materials of the same type without them
    # clobbering each other
    group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
    group_node.node_tree = bpy.data.node_groups[name]

    # Find and set the "Color" input of the new group node
    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Wire the output of the new group node to the input of
    # the MaterialOutput node
    mat.node_tree.links.new(
        group_node.outputs['Shader'],
        output_node.inputs['Surface'],
    )


"""
Some functions for handling constraints to the clevr hans data set.
"""
class RetrySceneException(Exception):
    def __init__(self, message='Must retry scene form scratch'):
        super(RetrySceneException, self).__init__(message)


def convert_class_str_keys_to_int(dictionary):
    """
    Convert the highest level keys of a dictionary from strings to integers.
    :param dictionary:
    :return:
    """
    converted_dict = {}
    for key in dictionary.keys():
        assert isinstance(key, str)
        converted_dict[int(key)] = dictionary[key]
    return converted_dict


def repair_obj_dict(d):
    """
    If one of the attributes from (shape, size, material, color, general_position, relationships) is missing,
    add empty strings.
    :param d:
    :return:
    """
    relation_constraints = None
    key_list = list(d.keys())
    if "shape" not in key_list:
        d["shape"] = ""
    if "size" not in key_list:
        d["size"] = ""
    if "material" not in key_list:
        d["material"] = ""
    if "color" not in key_list:
        d["color"] = ""
    if "general_position" not in key_list:
        d["general_position"] = ""
    if "relationships" not in key_list:
        d["relationships"] = {"behind": "", "front": "", "right": "", "left": ""}
    if "relationships" in key_list:
        relationships_key_list = list(d["relationships"].keys())
        if "behind" not in relationships_key_list:
            d["relationships"]["behind"] = ""
        if "front" not in relationships_key_list:
            d["relationships"]["front"] = ""
        if "right" not in relationships_key_list:
            d["relationships"]["right"] = ""
        if "left" not in relationships_key_list:
            d["relationships"]["left"] = ""


def check_empty_dict(d):
    empty = True
    for key in list(d.keys()):
        if d[key]:
            empty = False
            break
    return empty


def unrole_variations_to_list(dictionary):
    """

    :param dictionary: e.g.
   {
   'b': {'0.6': [{'shape': 'sphere', 'general_position': 'left'},
   {'shape': 'sphere', 'general_position': 'left'},
   {'shape': 'sphere', 'general_position': 'left'},
   {'shape': 'cylinder', 'general_position': 'right', 'material': 'metal'},
   {'shape': 'cylinder', 'general_position': 'right', 'material': 'metal'},
   {'shape': 'cylinder', 'general_position': 'right', 'material': 'metal'}]},
   'a': {'0.4': [{'shape': 'sphere', 'general_position': 'left'},
   {'shape': 'sphere', 'general_position': 'left'},
   {'shape': 'sphere', 'general_position': 'left'}]}
   }
    :return:
    """
    return [list(list(dictionary.items())[i][1].items())[0][1] for i in range(len(dictionary.items()))]


def variations_subset_of_another(class_set):
    """
    If a class is defined by several variations, one of these variations must be a subset of all other. In other words
    it contains the minimal class rule which also occurs in the other variations.
    :param class_set:
    :return:
    """
    # get tha variations into one list
    variations = unrole_variations_to_list(class_set)
    smallest_subset_idx = np.argmin([len(variation) for variation in variations])
    # check that all variations contain the shortest variation
    correct_subset = 0
    for variation in variations:
        correct_subset += check_list_subset(variation, variations[smallest_subset_idx])
    return correct_subset == len(variations)


def get_relationship_constraints(obj_list, class_id):
    """
    Iterate over list of objects making class rules and extract if an object has any relationship constraints. Returns
    a list of dictionaries. Each dictionary represents an object with relevant attributes and the relationhsips to
    other objects, which are also represented as dictionaries of relevant attributes.
    E.g.: [{'color': 'cyan', 'relationships': {'behind': [{'color': 'red'}, {'color': 'red'}]}},
           {'color': 'red', 'relationships': {'front': [{'color': 'cyan'}]}},
           {'color': 'red', 'relationships': {'front': [{'color': 'cyan'}]}}]
    :param obj_list:
    :return:
    """
    relationship_constraints = []
    for obj_dict in obj_list:
        if not check_empty_dict(obj_dict["relationships"]):
            relationship_constraints.append(obj_dict.copy())
            relationship_constraints[-1]["class_id"] = int(class_id)
            relationship_constraints[-1]["relationships"] = {}
            for direction_key in list(obj_dict["relationships"].keys()):
                relationship_constraints[-1]["relationships"][direction_key] = []
                for related_obj_id in obj_dict["relationships"][direction_key]:
                    relationship_constraints[-1]["relationships"][direction_key].append(obj_list[related_obj_id].copy())
                    relationship_constraints[-1]["relationships"][direction_key][-1].pop("relationships", None)
    return relationship_constraints


def convert_json_dict(json_file_handle):
    """
    Given a json file handle go through all class rules. If the dictionary of the individual objects of a class rule
    does not contain the attributes color, shape, etc, add these as en empty string.
    :param json_file_handle:
    :return:
    """
    json_dict = json.load(json_file_handle)
    relationship_constraints = []
    for class_key in json_dict.keys():
        class_set = json_dict[class_key]
        # check that all attributes are added to all obj dicts, if only as an empty string value
        if isinstance(class_set, list):
            for d in class_set:
                repair_obj_dict(d)
            relationship_constraints += get_relationship_constraints(class_set, class_key)

        # this is the case if one class has variations with specific probabilities
        elif isinstance(class_set, dict):
            # check that shorter variation is contained in larger variation
            assert variations_subset_of_another(class_set)

            variation_key_list = list(class_set.keys())
            for variation_key in variation_key_list:
                prob_key_list = list(class_set[variation_key].keys())
                assert len(prob_key_list) == 1
                prob_key = prob_key_list[0]
                for d in class_set[variation_key][prob_key]:
                    repair_obj_dict(d)
                relationship_constraints += get_relationship_constraints(class_set[variation_key][prob_key], class_key)

    # finally convert string class ids to ints
    json_dict = convert_class_str_keys_to_int(json_dict)
    return json_dict, relationship_constraints


def check_list_subset(list_large, list_small):
    """
    Checks if one list of dictionaries is a subset of another.
    :param list_large:
    :param list_small:
    :return:
    """
    list_large_copy = list_large.copy()
    sim_count = 0
    for elem in list_small:
        for elem2 in list_large_copy:
            assert isinstance(elem, dict) and isinstance(elem2, dict)
            if elem == elem2:
                sim_count += 1
                list_large_copy.remove(elem2)
                break
    if sim_count == len(list_small):
        return True
    else:
        return False


def compare_objs(cur_obj, must_obj):
    """
    Compare the attributes between two objects, where must_obj is the more general object, i.e. if an attribute in
    must_obj is an empty string, the two objects are considered to have the same attribute here. Very IMPORTANTE!!!!!!
    :param obj:
    :param must_obj:
    :return:
    """
    # if a specific attribute is not an empty string of the must attributes, i.e. there is a specification
    # the comparison is automatically set to True
    same_shape = (cur_obj["shape"] == must_obj["shape"]) if must_obj["shape"] else True
    same_size = (cur_obj["size"] == must_obj["size"]) if must_obj["size"] else True
    same_material = (cur_obj["material"] == must_obj["material"]) if must_obj["material"] else True
    same_color = (cur_obj["color"] == must_obj["color"]) if must_obj["color"] else True
    same_general_pos = (must_obj["general_position"] in cur_obj["general_position"]) \
        if must_obj["general_position"] else True

    same_attrs = False
    if same_shape and same_size and same_color and same_material and same_general_pos:
        same_attrs = True
    return same_attrs


def check_conflicts_all_classes(all_class_attr_combos_dict, cur_objs_attr_dict, cur_class_id):
    """
    Iterate over all classes and check if there are too many conflicts between the current objects to be placed and
    those that must be placed within the other classes. Also make sure that the must be placed objects of own current
    class are not repeated.

    :param all_class_attr_combos_dict:
    :param cur_objs_attr_dict:
    :param cur_class_id:
    :return:
    """

    conflict = False
    for class_id in all_class_attr_combos_dict.keys():
        n_conflicts_per_class = 0
        must_objs_attr_list = all_class_attr_combos_dict[class_id].copy()

        # if it happens that a class rule list of one class id is contained within the list of another class id,
        # we don't check for conflicts
        if check_list_subset(all_class_attr_combos_dict[cur_class_id], all_class_attr_combos_dict[class_id])\
                and cur_class_id != class_id:
            continue

        for obj_id in cur_objs_attr_dict.keys():
            obj = cur_objs_attr_dict[obj_id]
            for must_obj in must_objs_attr_list:
                same_attrs = compare_objs(obj, must_obj)

                if same_attrs:
                    n_conflicts_per_class += 1
                    # remove the object that caused the conflicts, so that it is not counted twice, e.g. cylinder twice
                    # in must_obj_dict and 1 cylinder is in cur_objs_attr_dict
                    must_objs_attr_list.remove(must_obj)
                    break
            # if there are more conflcits then must be in the current class
            if (n_conflicts_per_class > len(all_class_attr_combos_dict[class_id])) and (class_id == cur_class_id):
                conflict = True
                break
            # if there are as many conflicts in another class as must be placed objects
            elif (n_conflicts_per_class == len(all_class_attr_combos_dict[class_id])) \
                    and (class_id != cur_class_id):
                conflict = True
                break
    return conflict


def compute_corrected_x_y(x, y, r, scene_struct):
    """
    Correct x, y position based on camera position.
    :param x:
    :param y:
    :param r:
    :param scene_struct:
    :return:
    """
    coords = [x, y, r]
    # correct for camera position to get true position
    x_corrected = np.dot(coords, scene_struct['directions']['right'])
    y_corrected = np.dot(coords, scene_struct['directions']['front'])
    return x_corrected, y_corrected


def sample_obj_place_with_general_constraint(constraint, r, scene_struct):
    """
    Get the object positions depending on whether there are general palcement constraints, e.g. object should be in
    right half of image. If no constraint, just sample randomly.
    :param constraint:
    :param r:
    :param scene_struct:
    :return:
    """

    if constraint:
        correct_placement = False
        while not correct_placement:
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)

            x_corrected, y_corrected = compute_corrected_x_y(x, y, r, scene_struct)

            if 'right' in constraint and x_corrected > 0:
                correct_placement = True
            elif 'left' in constraint and x_corrected < 0:
                correct_placement = True
            elif 'top' in constraint and y_corrected < 0:
                correct_placement = True
            elif 'bottom' in constraint and y_corrected > 0:
                correct_placement = True
    else:
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
    return x, y


def get_general_obj_placement_name(x, y, r, scene_struct):
    """
    Given the x, y positions, correct based on camera position and get general placement in the scene in terms of
    top, bottom, right, left.
    :param x:
    :param y:
    :param r:
    :param scene_struct:
    :return:
    """
    x_corrected, y_corrected = compute_corrected_x_y(x, y, r, scene_struct)

    if x_corrected > 0:
        x_placement = 'right'
    elif x_corrected < 0:
        x_placement = 'left'
    if y_corrected < 0:
        y_placement = 'top'
    elif y_corrected > 0:
        y_placement = 'bottom'
    return y_placement + ' ' + x_placement


def check_relationship_constraints(x, y, r, scene_struct, cand_obj, cur_objs_attr_dict,
                                   relationship_constraints, cur_class_id):
    """
    This is a monster, beware the beast!!!!!!
    :param x:
    :param y:
    :param r:
    :param scene_struct:
    :param cand_obj:
    :param cur_objs_attr_dict:
    :param relationship_constraints:
    :param cur_class_id:
    :return:
    """
    x_corrected, y_corrected = compute_corrected_x_y(x, y, r, scene_struct)

    # set dummy value
    relationship_constraints_hold = True
    relationship_must_hold = True
    for idx, must_obj in enumerate(relationship_constraints):
        # if the current class id is the same as that of the relationship_constraint list
        # depending on relationship_must_hold we must evaluate differently when a relationship holds
        relationship_must_hold = True if must_obj["class_id"] == cur_class_id else False
        relationship_constraints_hold = relationship_must_hold # this ensures the first if cases to be passed

        # check if candidate obj attributes one with potential relationship constraints
        if compare_objs(cand_obj, must_obj):
            for relationship_key, relationship_list in list(must_obj["relationships"].items()):
                # find a relationship direction with constraints
                if relationship_list:
                    # for each object within this constraint list check if a corresponding object has already been
                    # placed in the scene
                    for constraint_obj in relationship_list:

                        # if a constraint with one object is already broken, but is supposed to hold
                        # or a constraint holds when it should not
                        if not(xnor(relationship_constraints_hold, relationship_must_hold)):
                            break
                        # otherwise check next constraint obj with all objects that have been placed in scene
                        for _, obj in cur_objs_attr_dict.items():
                            # if a constraint with one object is already broken, but is supposed to hold
                            # or a constraint holds when it should not
                            if not(xnor(relationship_constraints_hold, relationship_must_hold)):
                                break

                            # if the cand_obj has relationship constraints and another object in the scene stands in
                            # relationship with cand_obj, check if the relationship holds
                            if compare_objs(obj, constraint_obj):
                                relationship_constraints_hold = relate_objs([x_corrected, y_corrected],
                                                                            obj["corr_position"], relationship_key)

    # if all constraints hold and are supposed to or do not hold and are not supposed to, keep cand_obj x, y positions
    # otherwise resample until these conditions hold
    relationship_conflict = xnor(relationship_constraints_hold, relationship_must_hold)
    return relationship_conflict


# def place_obj(scene_struct, num_objects, args, camera,
#               cand_obj, cur_objs_attr_dict, blender_objects, positions,
#               relationship_constraints, general_position, r):
#     """
#     Place a candidate object, making sure it is far enough from the borders and other objects.
#     :param scene_struct:
#     :param blender_objects:
#     :param positions:
#     :param general_position:
#     :param r:
#     :param args:
#     :return:
#     """
#     # Try to place the object, ensuring that we don't intersect any existing
#     # objects and that we are more than the desired margin away from all existing
#     # objects along all cardinal directions.
#     num_tries = 0
#     while True:
#         # If we try and fail to place an object too many times, then delete all
#         # the objects in the scene and send message to wrapper.
#         num_tries += 1
#         if num_tries > args.max_retries:
#             for obj in blender_objects:
#                 delete_object(obj)
#             raise RetrySceneException()
#             # print("\nRetrying on new image!!\n")
#             # return add_random_objects(scene_struct, num_objects, args, camera)
#
#         # first make sure higher level constraints hold
#         relationship_constraint_hold = False
#         while not relationship_constraint_hold:
#             # first catch general placement constraints
#             if general_position:
#                 x, y = sample_obj_place_with_general_constraint(general_position, r, scene_struct)
#             else:
#                 x, y = sample_obj_place_with_general_constraint(None, r, scene_struct)
#             # next check if position holds for any given relationship constraints
#             relationship_constraint_hold = check_relationship_constraints(x, y, r, scene_struct, cand_obj,
#                                                                           cur_objs_attr_dict, relationship_constraints,
#                                                                           cur_class_id=args.img_class_id)
#
#         print("\nDirection constraints all hold!\n")
#         # Check to make sure the new object is further than min_dist from all
#         # other objects, and further than margin along the four cardinal directions
#         dists_good = True
#         margins_good = True
#         for (xx, yy, rr) in positions:
#             dx, dy = x - xx, y - yy
#             dist = math.sqrt(dx * dx + dy * dy)
#             if dist - r - rr < args.min_dist:
#                 dists_good = False
#                 break
#             for direction_name in ['left', 'right', 'front', 'behind']:
#                 direction_vec = scene_struct['directions'][direction_name]
#                 assert direction_vec[2] == 0
#                 margin = dx * direction_vec[0] + dy * direction_vec[1]
#                 if 0 < margin < args.margin:
#                     print(margin, args.margin, direction_name)
#                     print('BROKEN MARGIN!')
#                     margins_good = False
#                     break
#             if not margins_good:
#                 break
#
#         if dists_good and margins_good:
#             break
#
#     general_position_name = get_general_obj_placement_name(x, y, r, scene_struct)
#     x_corrected, y_corrected = compute_corrected_x_y(x, y, r, scene_struct)
#
#     return x, y, general_position_name, x_corrected, y_corrected


def check_for_classes_with_probabilities(conf_combos_all_classes, gt_combos_all_classes, img_class_id):
    """
    Given a class rule set with several variations with specific probabilities, choose one of these variations based on
    the specified probabilities and update the global rule lists to only contain that variation rule that was chosen.
    :param conf_combos_all_classes:
    :param gt_combos_all_classes:
    :param img_class_id:
    :return:
    """
    # if the class rule list which has variations does not correspond to img_class_id then make the smallest subset
    # variation the standard rule list, otherwise sample according to the probabilities
    if isinstance(conf_combos_all_classes[img_class_id], dict):
        must_attr_combos = conf_combos_all_classes[img_class_id]
        variation_keys = list(must_attr_combos.keys())

        # get the probabilities of every sub rule list
        probabilities = [float(prob_key) for key in must_attr_combos.keys()
                         for prob_key in must_attr_combos[key].keys()]
        assert np.sum(probabilities) == 1.

        # choose one of the rule lists based on the given probabilities
        idx = np.random.choice(np.arange(0, len(must_attr_combos)), p=probabilities)
        chosen_variation_key = variation_keys[idx]

        # get the prob key
        conf_chosen_prob_key = list(conf_combos_all_classes[img_class_id][chosen_variation_key].keys())[0]
        gt_chosen_prob_key = list(gt_combos_all_classes[img_class_id][chosen_variation_key].keys())[0]

        # remove those options that were not chosen
        conf_combos_all_classes[img_class_id] = conf_combos_all_classes[img_class_id][chosen_variation_key][conf_chosen_prob_key]
        gt_combos_all_classes[img_class_id] = gt_combos_all_classes[img_class_id][chosen_variation_key][gt_chosen_prob_key]

    else:
        for class_id, class_set in conf_combos_all_classes.items():
            if isinstance(class_set, dict):
                variations = unrole_variations_to_list(class_set)
                smallest_subset_idx = np.argmin([len(variation) for variation in variations])

                variation_key_shortest_subset = list(class_set.keys())[smallest_subset_idx]

                conf_smallest_subset_prob_key = list(conf_combos_all_classes[class_id][variation_key_shortest_subset].keys())[0]
                gt_smallest_subset_prob_key = list(gt_combos_all_classes[class_id][variation_key_shortest_subset].keys())[0]

                # keep the smallest subset variation and remove the rest
                conf_combos_all_classes[class_id] = conf_combos_all_classes[class_id][variation_key_shortest_subset][
                    conf_smallest_subset_prob_key]
                gt_combos_all_classes[class_id] = gt_combos_all_classes[class_id][variation_key_shortest_subset][
                    gt_smallest_subset_prob_key]

    return conf_combos_all_classes, gt_combos_all_classes


def relate_objs(obj1_coords, obj2_coords, relation):
    """
    Relates obj2 in relation to obj1, e.g. whether obj2 is behind obj1 or whether obj2 is right of obj1.
    :param obj1_coords:
    :param obj2_coords:
    :param relation:
    :return:
    """
    relation_holds = False
    if relation == "behind" and obj1_coords[1] > obj2_coords[1]:
        relation_holds = True
    elif relation == "front" and obj1_coords[1] < obj2_coords[1]:
        relation_holds = True
    elif relation == "right" and obj1_coords[0] < obj2_coords[0]:
        relation_holds = True
    elif relation == "left" and obj1_coords[0] > obj2_coords[0]:
        relation_holds = True
    return relation_holds


def xnor(x, y):
    return not (x or y) or (x and y)
