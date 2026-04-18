"""
Gmsh Visualization Utilities for FEM Results
"""

import numpy as np
import gmsh


def add_element_data_view(viewname, 
                          element_tags, 
                          data, 
                          visible=False):
    viewnum = gmsh.view.add(viewname)
    gmsh.view.addHomogeneousModelData(
        tag=viewnum,
        step=0,
        time=0,
        modelName=gmsh.model.getCurrent(),
        dataType="ElementData",
        numComponents=-1,
        tags=element_tags,
        data=data.reshape((-1))
    )
    gmsh.view.option.setNumber(viewnum, "Visible", 1 if visible else 0)

    return viewnum


def add_node_data_view(viewname,
                       node_tags,
                       data,
                       visible=False,
                       vector_type=None,
                       factor=None,
                       arrow_size_max=None,
                       arrow_size_min=None,
                       deformed_view=None):  
    """
    Add a nodal data view to the current Gmsh model.

    Args:
        viewname       (str)        : Label shown in the Gmsh view panel.
        node_tags      (np.ndarray) : Array of node tags matching the model.
        data           (np.ndarray) : Data array, shape (n_nodes, n_components).
                                      1 component → scalar, 3 components → vector.
        visible        (bool)       : Whether the view is visible on creation.
                                      Default is False.
        vector_type    (int|None)   : Gmsh VectorType option.
                                        3 → displacement (deformed shape)
                                        4 → arrows at nodes
                                        5 → arrows + displacement
                                      Default is None (not set).
        factor         (float|None) : DisplacementFactor — scales the vector
                                      for deformed-shape views (vector_type=5).
                                      Default is None (not set).
        arrow_size_max (int|None)   : Maximum arrow size in pixels (ArrowSizeMax).
                                      Useful when force magnitudes span many orders
                                      of magnitude and small arrows are invisible.
                                      Default is None (not set → Gmsh default).
        arrow_size_min (int|None)   : Minimum arrow size in pixels (ArrowSizeMin).
                                      Default is None (not set → Gmsh default).

    Returns:
        viewnum (int): Tag of the created Gmsh view.
    """
    viewnum = gmsh.view.add(viewname)
    gmsh.view.addHomogeneousModelData(
        tag=viewnum,
        step=0,
        time=0,
        modelName=gmsh.model.getCurrent(),
        dataType="NodeData",
        numComponents=-1,
        tags=node_tags,
        data=data.reshape((-1))
    )

    gmsh.view.option.setNumber(viewnum, "Visible",       1 if visible else 0)
    gmsh.view.option.setNumber(viewnum, "GlyphLocation", 2)

    if vector_type is not None:
        gmsh.view.option.setNumber(viewnum, "VectorType",         vector_type)

    if factor is not None:
        gmsh.view.option.setNumber(viewnum, "DisplacementFactor", factor)

    if arrow_size_max is not None:
        gmsh.view.option.setNumber(viewnum, "ArrowSizeMax",       arrow_size_max)

    if arrow_size_min is not None:
        gmsh.view.option.setNumber(viewnum, "ArrowSizeMin",       arrow_size_min)

    if deformed_view is not None and factor is not None:
        gmsh.plugin.setNumber("Warp", "View",      viewnum)
        gmsh.plugin.setNumber("Warp", "OtherView", deformed_view)
        gmsh.plugin.setNumber("Warp", "Factor",    factor)
        gmsh.plugin.run("Warp")

    return viewnum


def compute_nodal_average(mesh,
                          element_tags_list,
                          element_data):
    """
    Compute nodal average of element data using raw mesh connectivity.

    Works for both FEM own solver and OpenSees flows — only needs the mesh.

    Parameters
    ----------
    mesh              : GMSHtools   mesh with node_map and elements
    element_tags_list : list[int]   element tags matching element_data order
    element_data      : np.ndarray  (n_elements,)  scalar value per element

    Returns
    -------
    np.ndarray  (n_nodes,)  averaged nodal values in mesh.nodes order
    """
    # build {element_tag: connectivity} across all groups
    tag_to_conn = {}
    for group in mesh.elements.values():
        for tag, conn in zip(group['element_tags'], group['connectivity']):
            tag_to_conn[tag] = conn

    # node tag → position index in mesh.nodes order
    all_node_tags = list(mesh.nodes.keys())
    node_idx      = {tag: i for i, tag in enumerate(all_node_tags)}
    nNodes        = len(all_node_tags)

    nodal_values = np.zeros(nNodes)
    count        = np.zeros(nNodes)

    for elem_tag, value in zip(element_tags_list, element_data):
        if elem_tag not in tag_to_conn:
            continue
        for tag in tag_to_conn[elem_tag]:
            if tag in node_idx:
                idx               = node_idx[tag]
                nodal_values[idx] += value
                count[idx]        += 1

    count[count == 0] = 1
    nodal_values /= count

    return nodal_values