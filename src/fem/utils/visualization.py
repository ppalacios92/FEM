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
                       displacement_factor=None):

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
    gmsh.view.option.setNumber(viewnum, "Visible", 1 if visible else 0)
    gmsh.view.option.setNumber(viewnum, "GlyphLocation", 2)
    
    if vector_type is not None:
        gmsh.view.option.setNumber(viewnum, "VectorType", vector_type)
    
    if displacement_factor is not None:
        gmsh.view.option.setNumber(viewnum, "DisplacementFactor", displacement_factor)
    
    return viewnum


def compute_nodal_average(elements, 
                          element_data, 
                          nodes):

    node_map = {node.name: idx for idx, node in enumerate(nodes)}
    nNodes = len(nodes)
    
    nodal_values = np.zeros(nNodes)
    count = np.zeros(nNodes)
    
    # Accumulate values from elements
    for element, value in zip(elements, element_data):
        for node in element.nodes:
            idx = node_map[node.name]
            nodal_values[idx] += value
            count[idx] += 1
    
    # Compute average (avoid division by zero)
    count[count == 0] = 1
    nodal_values /= count
    
    return nodal_values
