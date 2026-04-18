# =============================================================================
# Based on the course by José Antonio Abell, Universidad de los Andes, Chile
# =============================================================================

import inspect
import numpy as np
from fem.core.parameters import globalParameters
from fem.core.Node import Node


# -- Matrix utilities ----------------------------------------------------------

def matrix_extract(matrix: np.ndarray,
                   row_indices: np.ndarray,
                   col_indices: np.ndarray) -> np.ndarray:
    """
    Extract a submatrix from a matrix at the given row and column indices.

    Parameters
    ----------
    matrix      : np.ndarray  Source matrix
    row_indices : np.ndarray  Row indices to extract
    col_indices : np.ndarray  Column indices to extract

    Returns
    -------
    np.ndarray  Submatrix of shape (len(row_indices), len(col_indices))
    """
    return matrix[np.ix_(row_indices, col_indices)]


def matrix_replace(matrix: np.ndarray,
                   matrix_add: np.ndarray,
                   row_indices: np.ndarray,
                   col_indices: np.ndarray) -> np.ndarray:
    """
    Add a submatrix into a matrix at the given row and column indices.

    Parameters
    ----------
    matrix      : np.ndarray  Source matrix (not modified in place)
    matrix_add  : np.ndarray  Submatrix to add
    row_indices : np.ndarray  Row indices where submatrix is added
    col_indices : np.ndarray  Column indices where submatrix is added

    Returns
    -------
    np.ndarray  Updated matrix with submatrix added at specified indices
    """
    updated_matrix = matrix.copy()
    updated_matrix[np.ix_(row_indices, col_indices)] += matrix_add
    return updated_matrix


# -- DOF numbering and boundary conditions -------------------------------------

def plan(mesh, restrain_dictionary: dict = None) -> tuple:
    """
    Assign consecutive DOF indices to all nodes in the mesh and
    optionally apply boundary conditions.

    This is required before assembling the global stiffness matrix because
    the matrix size is system_nDof x system_nDof and DOFs must be contiguous.

    The number of DOFs per node is read from globalParameters['nDoF'].
    The spatial dimension is read from globalParameters['nDIM'].

    Parameters
    ----------
    mesh                : GMSHtools   Mesh object from gmshtools.GMSHtools.
    restrain_dictionary : dict        {phys_id: ['r'/'f', ...]}  (optional)
                                      Applies boundary conditions to nodes
                                      belonging to the specified physical groups.

    Returns
    -------
    node_map    : dict   {gmsh_tag: Node}  — Node objects with idx assigned.
    system_nDof : int    Total number of DOFs in the system.

    Examples
    --------
    mesh = GMSHtools('model.msh')
    node_map, system_nDof = plan(mesh, restrain_dictionary)

    # Access nodes
    node = node_map[tag]
    node.idx          # DOF indices in global system
    node.coordinates  # spatial coordinates
    node.restrain     # ['r','r'] or ['f','f']
    """
    nDoF = globalParameters['nDoF']
    nDIM = globalParameters['nDIM']

    all_tags     = sorted(mesh.nodes.keys())
    tag_to_index = {tag: i for i, tag in enumerate(all_tags)}

    node_map = {}
    for tag in all_tags:
        i      = tag_to_index[tag]
        coords = list(mesh.nodes[tag][:nDIM])
        node   = Node(name=int(tag), coordinates=coords)
        node.idx = np.array([nDoF * i + j for j in range(nDoF)])
        node_map[tag] = node

    # mesh.node_map = node_map
    # if restrain_dictionary:
    #     apply_restraints(mesh, restrain_dictionary)

    system_nDof = len(all_tags) * nDoF

    return node_map, system_nDof


# -- FEM element builders ------------------------------------------------------

def _filter_kwargs(element_class: type, kwargs: dict) -> dict:
    """
    Filter a kwargs dictionary to only include parameters accepted by the
    element class __init__ method.

    This allows build_elements to pass a uniform set of parameters without
    raising TypeError on elements that do not support all of them.
    For example, CST does not have sampling_points or eval_points, while
    Quad4, LST and Quad9 do.

    Parameters
    ----------
    element_class : type   The element class to inspect
    kwargs        : dict   Full set of candidate keyword arguments

    Returns
    -------
    dict  Filtered kwargs containing only keys accepted by element_class
    """
    valid_params = inspect.signature(element_class.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in valid_params}


def _build_loaded_edges(mesh, load_dictionary: dict) -> dict:
    """
    Pre-process the load_dictionary to build a lookup of loaded edges.

    Handles both dim=1 (line) and dim=2 (surface) physical groups:
      - dim=1: each connectivity entry is a segment [tag_i, tag_j] — used directly.
      - dim=2: each connectivity entry is a triangle [n0, n1, n2] — all three
               edges are extracted and registered as loaded.

    Parameters
    ----------
    mesh            : GMSHtools
    load_dictionary : dict  {phys_id: {'value': float, 'direction': str or float}}

    Returns
    -------
    loaded_edges : dict  {frozenset({tag_i, tag_j}): [qx, qy]}
    """
    loaded_edges = {}

    for phys_id, load_spec in load_dictionary.items():
        if phys_id not in mesh.elements:
            continue

        group = mesh.elements[phys_id]
        dim   = group['dim']

        if dim not in (1, 2):
            continue

        load_value = load_spec['value']
        from fem.utils.gmshtools import GMSHtools
        nDoF = globalParameters['nDoF']
        d = GMSHtools._direction_to_vector(load_spec['direction'], nDoF)
        q          = load_value * d

        if dim == 1:
            for connectivity in group['connectivity']:
                tag_i = connectivity[0]
                tag_j = connectivity[1]
                key   = frozenset([tag_i, tag_j])
                loaded_edges[key] = q.tolist()

        elif dim == 2:
            n_corners = _ELEMENT_CORNER_COUNT.get(group['n_nodes'], group['n_nodes'])
            for connectivity in group['connectivity']:
                for i in range(n_corners):
                    j     = (i + 1) % n_corners
                    key   = frozenset([connectivity[i], connectivity[j]])
                    loaded_edges[key] = q.tolist()

    return loaded_edges


# Number of corner nodes per element type.
# Higher-order elements (LST=6, Quad9=9) have mid-side/centre nodes
# appended after the corners. Edges are defined only by corner pairs.
_ELEMENT_CORNER_COUNT = {   3: 3, 
                            4: 4, 
                            6: 3, 
                            9: 4}


def _get_element_surface_loads(node_list: list, loaded_edges: dict) -> list:
    """
    Detect which edges of a 2D element are loaded and return the
    corresponding surface_loads list.

    Parameters
    ----------
    node_list    : list  Ordered list of Node objects for the element
    loaded_edges : dict  {frozenset({tag_i, tag_j}): [qx, qy]}

    Returns
    -------
    surface_loads : list  List of dicts with keys:
                            'node_indices' : tuple (i, j)
                            'value'        : [qx, qy]
    """
    surface_loads = []
    n         = len(node_list)
    n_corners = _ELEMENT_CORNER_COUNT.get(n, n)

    for i in range(n_corners):
        j     = (i + 1) % n_corners
        tag_i = node_list[i].name
        tag_j = node_list[j].name
        key   = frozenset([tag_i, tag_j])

        if key in loaded_edges:
            surface_loads.append({
                'node_indices': (i, j),
                'value':        loaded_edges[key]
            })

    return surface_loads


def build_elements(mesh,
                   node_map: dict,
                   section_dictionary: dict,
                   element_class_map: dict,
                   load_dictionary: dict = None,
                   load_direction: list = None,
                   type: str = 'planeStress',
                   sampling_points: int = None,
                   eval_points: list = None,
                   print_summary: bool = False) -> np.ndarray:
    """
    Instantiate FEM element objects from mesh data.

    Handles both body forces (self-weight) and surface forces (line loads)
    entirely within each element:

    - Body forces are computed from load_direction and material.rho.
      If load_direction is None, each element reads gravity from
      globalParameters['gravity'] automatically.

    - Surface forces are detected from load_dictionary (dim=1 groups).
      For each element, any edge whose two endpoint node tags match a
      loaded segment in load_dictionary receives a surface_load entry.

    Parameters
    ----------
    mesh               : GMSHtools    Mesh object from gmshtools.
    node_map           : dict         {gmsh_tag: Node}  from plan().
    section_dictionary : dict         {phys_id: section object}
    element_class_map  : dict         {n_nodes: ElementClass}
                                      e.g. {3: CST, 4: Quad4, 6: LST, 9: Quad9}
    load_dictionary    : dict         {phys_id: {'value': float, 'direction': str}}
    load_direction     : list         [Cx, Cy] body force direction cosines.
    type               : str          'planeStress' or 'planeStrain'.
    sampling_points    : int          Gauss integration points per direction.
    eval_points        : list         [zeta, eta] for stress/strain recovery.
    print_summary      : bool         Print element summary on creation.

    Returns
    -------
    np.ndarray of element objects

    Examples
    --------
    node_map, system_nDof = plan(mesh, restrain_dictionary)

    elements = build_elements(
        mesh               = mesh,
        node_map           = node_map,
        section_dictionary = section_dictionary,
        element_class_map  = {3: CST, 4: Quad4, 6: LST, 9: Quad9},
        load_dictionary    = load_dictionary,
        type               = 'planeStrain',
        sampling_points    = 3,
        eval_points        = [0, 0],
    )
    """
    elements     = []
    loaded_edges = _build_loaded_edges(mesh, load_dictionary) \
                   if load_dictionary is not None else {}

    all_kwargs = {
        'load_direction': load_direction,
        'type':           type,
        'print_summary':  print_summary,
    }
    if sampling_points is not None:
        all_kwargs['sampling_points'] = sampling_points
    if eval_points is not None:
        all_kwargs['eval_points'] = eval_points

    for phys_id, section in section_dictionary.items():
        if phys_id not in mesh.elements:
            continue

        group   = mesh.elements[phys_id]
        n_nodes = group['n_nodes']

        if n_nodes not in element_class_map:
            continue

        ElementClass = element_class_map[n_nodes]
        kwargs       = _filter_kwargs(ElementClass, all_kwargs)

        for elem_tag, connectivity in zip(group['element_tags'], group['connectivity']):
            node_list = [node_map[tag] for tag in connectivity]

            if 'surface_loads' in inspect.signature(ElementClass.__init__).parameters:
                kwargs['surface_loads'] = _get_element_surface_loads(
                    node_list, loaded_edges
                )

            elem = ElementClass(
                element_tag = elem_tag,
                node_list   = node_list,
                section     = section,
                **kwargs
            )
            elements.append(elem)

    return np.array(elements, dtype=object)


# def build_plot_elements(mesh, node_map: dict, phys_ids: list) -> np.ndarray:
#     """
#     Build lightweight element objects for plotting only.
#     Each element has only element_tag and nodes attributes.

#     Parameters
#     ----------
#     mesh     : GMSHtools or dict
#     node_map : dict       {gmsh_tag: Node} from plan()
#     phys_ids : list       Physical group IDs to include

#     Returns
#     -------
#     np.ndarray of SimpleNamespace objects with .element_tag and .nodes
#     """
#     from types import SimpleNamespace

#     elements = []
#     for phys_id in phys_ids:
#         if phys_id not in mesh.elements:
#             continue
#         group = mesh.elements[phys_id]
#         for elem_tag, conn in zip(group['element_tags'], group['connectivity']):
#             elements.append(SimpleNamespace(
#                 element_tag = elem_tag,
#                 nodes       = [node_map[tag] for tag in conn]
#             ))
#     return np.array(elements, dtype=object)


# -- Load vector builders ------------------------------------------------------


def get_line_load_global_vector(node_start, node_end, wj, wk, alpha_degree=None):
    """
    Compute the global equivalent nodal force vector for a trapezoidal line load
    on a 2-node frame element.

    Parameters
    ----------
    node_start   : Node   Start node
    node_end     : Node   End node
    wj           : float  Load magnitude at start node [force/length]
    wk           : float  Load magnitude at end node [force/length]
    alpha_degree : float or str or None
        Load angle relative to element axis.

    Returns
    -------
    np.ndarray  (4,)  Global equivalent nodal forces
    """
    delta = node_end.coordinates - node_start.coordinates
    L     = np.linalg.norm(delta)
    theta = np.arctan2(delta[1], delta[0])

    if alpha_degree is None:
        alpha = np.radians(90) - theta
    elif alpha_degree == 'x':
        alpha = -theta
    elif alpha_degree == 'y':
        alpha = np.radians(90) - theta
    else:
        alpha = np.radians(alpha_degree)

    w1 = wk
    w2 = wj - wk

    wjx = -w1 * np.cos(alpha) * L / 2 - w2 * np.cos(alpha) * L / 3
    wkx = -w1 * np.cos(alpha) * L / 2 - w2 * np.cos(alpha) * L / 6
    wjy = -w1 * np.sin(alpha) * L / 2 - w2 * np.sin(alpha) * L / 3
    wky = -w1 * np.sin(alpha) * L / 2 - w2 * np.sin(alpha) * L / 6

    F_local = -np.array([wjx, wjy, wkx, wky])
    c       = np.cos(theta)
    s       = np.sin(theta)
    Tlg     = np.array([
        [ c,  s, 0, 0],
        [-s,  c, 0, 0],
        [ 0,  0, c, s],
        [ 0,  0,-s, c]
    ])
    return Tlg.T @ F_local