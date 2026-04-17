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
                       arrow_size_min=None):
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

    return viewnum


def compute_nodal_average(elements,
                          element_data,
                          nodes):

    node_map = {node.name: idx for idx, node in enumerate(nodes)}
    nNodes   = len(nodes)

    nodal_values = np.zeros(nNodes)
    count        = np.zeros(nNodes)

    for element, value in zip(elements, element_data):
        for node in element.nodes:
            idx = node_map[node.name]
            nodal_values[idx] += value
            count[idx]        += 1

    count[count == 0] = 1
    nodal_values /= count

    return nodal_values