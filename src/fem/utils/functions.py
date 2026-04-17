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

    Gmsh tags are arbitrary integers (1, 5, 13, 274...). This function
    remaps them to consecutive zero-based indices so that:
        node.idx = [nDoF*i, nDoF*i+1, ..., nDoF*i + nDoF-1]

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

    if restrain_dictionary:
        _apply_restraints(node_map, mesh.elements, restrain_dictionary)

    system_nDof = len(all_tags) * nDoF

    return node_map, system_nDof


def _apply_restraints(node_map: dict, elements: dict, restrain_dictionary: dict):
    """
    Apply boundary conditions to nodes belonging to physical groups.

    Parameters
    ----------
    node_map            : dict  {gmsh_tag: Node}
    elements            : dict  mesh.elements from GMSHtools
    restrain_dictionary : dict  {phys_id: ['r'/'f', ...]}
    """
    for phys_id, condition in restrain_dictionary.items():
        if phys_id not in elements:
            continue
        for node_tags in elements[phys_id]['connectivity']:
            for tag in node_tags:
                if tag in node_map:
                    node_map[tag].set_restrain(condition)


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

    Iterates over all line groups (dim=1) in load_dictionary and collects
    every segment as a frozenset of two node tags mapped to its load vector.
    Using frozenset as key makes the lookup order-independent.

    Parameters
    ----------
    mesh            : GMSHtools or dict
    load_dictionary : dict  {phys_id: {'value': float, 'direction': str or float}}

    Returns
    -------
    loaded_edges : dict  {frozenset({tag_i, tag_j}): [qx, qy]}
    """
    from fem.utils.gmshtools import GMSHtools
    elements = mesh.elements if isinstance(mesh, GMSHtools) else mesh['elements']

    loaded_edges = {}

    for phys_id, load_spec in load_dictionary.items():
        if phys_id not in elements:
            continue

        group = elements[phys_id]

        if group['dim'] != 1:
            continue

        load_value = load_spec['value']
        d          = _direction_to_vector(load_spec['direction'])
        q          = load_value * d

        for connectivity in group['connectivity']:
            tag_i = connectivity[0]
            tag_j = connectivity[1]
            key   = frozenset([tag_i, tag_j])
            loaded_edges[key] = q.tolist()

    return loaded_edges


# Number of corner nodes per element type.
# Higher-order elements (LST=6, Quad9=9) have mid-side/centre nodes
# appended after the corners. Edges are defined only by corner pairs.
_ELEMENT_CORNER_COUNT = {3: 3, 4: 4, 6: 3, 9: 4}


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
    from fem.utils.gmshtools import GMSHtools
    elements_raw = mesh.elements if isinstance(mesh, GMSHtools) else mesh['elements']

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
        if phys_id not in elements_raw:
            continue

        group   = elements_raw[phys_id]
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


def build_plot_elements(mesh, node_map: dict, phys_ids: list) -> np.ndarray:
    """
    Build lightweight element objects for plotting only.
    Each element has only element_tag and nodes attributes.

    Parameters
    ----------
    mesh     : GMSHtools or dict
    node_map : dict       {gmsh_tag: Node} from plan()
    phys_ids : list       Physical group IDs to include

    Returns
    -------
    np.ndarray of SimpleNamespace objects with .element_tag and .nodes
    """
    from types import SimpleNamespace
    from fem.utils.gmshtools import GMSHtools
    elements_raw = mesh.elements if isinstance(mesh, GMSHtools) else mesh['elements']

    elements = []
    for phys_id in phys_ids:
        if phys_id not in elements_raw:
            continue
        group = elements_raw[phys_id]
        for elem_tag, conn in zip(group['element_tags'], group['connectivity']):
            elements.append(SimpleNamespace(
                element_tag = elem_tag,
                nodes       = [node_map[tag] for tag in conn]
            ))
    return np.array(elements, dtype=object)


# -- Load vector builders ------------------------------------------------------

def _direction_to_vector(direction) -> np.ndarray:
    """
    Convert a direction specification to a unit vector.
    Automatically returns [cx, cy] or [cx, cy, cz] based on globalParameters['nDoF'].

    Parameters
    ----------
    direction : str or float
        'x'   ->  positive X
        '-x'  ->  negative X
        'y'   ->  positive Y
        '-y'  ->  negative Y
        'z'   ->  positive Z (3D only)
        '-z'  ->  negative Z (3D only)
        float ->  angle in degrees from positive X axis, counterclockwise (2D only)

    Returns
    -------
    np.ndarray  [cx, cy] or [cx, cy, cz]
    """
    nDoF = globalParameters['nDoF']

    if nDoF >= 3:
        if direction == 'x':
            return np.array([1.0,  0.0,  0.0])
        elif direction == '-x':
            return np.array([-1.0, 0.0,  0.0])
        elif direction == 'y':
            return np.array([0.0,  1.0,  0.0])
        elif direction == '-y':
            return np.array([0.0, -1.0,  0.0])
        elif direction == 'z':
            return np.array([0.0,  0.0,  1.0])
        elif direction == '-z':
            return np.array([0.0,  0.0, -1.0])
        else:
            rad = np.radians(float(direction))
            return np.array([np.cos(rad), np.sin(rad), 0.0])
    else:
        if direction == 'x':
            angle = 0.0
        elif direction == '-x':
            angle = 180.0
        elif direction == 'y':
            angle = 90.0
        elif direction == '-y':
            angle = 270.0
        else:
            angle = float(direction)
        rad = np.radians(angle)
        return np.array([np.cos(rad), np.sin(rad)])


def build_load_vector(mesh,
                      node_map: dict,
                      load_dictionary: dict,
                      system_nDof: int) -> np.ndarray:
    """
    Build the global nodal force vector from point loads (dim=0) and
    line loads (dim=1) defined on physical groups.

    Detects the physical group dimension automatically:
      - dim=0  ->  point load applied directly to each node in the group.
      - dim=1  ->  consistent line load integrated along each segment using
                   Lagrange shape functions (linear or quadratic).

    Line load integration follows the element type reported by Gmsh:
      - gmsh_type=1  (2-node linear segment)    ->  [1/2, 1/2] x q x L
      - gmsh_type=8  (3-node quadratic segment) ->  [1/6, 1/6, 4/6] x q x L

    Parameters
    ----------
    mesh            : GMSHtools or dict
    node_map        : dict   {gmsh_tag: Node}  from plan().
    load_dictionary : dict   {phys_id: {'value': float, 'direction': str or float}}
    system_nDof     : int    Total number of DOFs in the system.

    Returns
    -------
    F : np.ndarray  (system_nDof,)  global nodal force vector.

    Examples
    --------
    load_dictionary = {
        10: {'value': 5000.0, 'direction': '-y'},   # point load  [N]
        20: {'value':   10.0, 'direction':  'x'},   # line load   [N/mm]
    }
    F = build_load_vector(mesh, node_map, load_dictionary, system_nDof)
    """
    from fem.utils.gmshtools import GMSHtools
    elements = mesh.elements if isinstance(mesh, GMSHtools) else mesh['elements']

    F = np.zeros(system_nDof)

    for phys_id, load_spec in load_dictionary.items():
        if phys_id not in elements:
            continue

        group      = elements[phys_id]
        dim        = group['dim']
        load_value = load_spec['value']
        d          = _direction_to_vector(load_spec['direction'])

        if dim == 0:
            # Point load — applied directly to each node in the group
            for connectivity in group['connectivity']:
                for tag in connectivity:
                    if tag in node_map:
                        F[node_map[tag].idx] += load_value * d

        elif dim == 1:
            # Consistent line load — accumulate by node tag to avoid
            # double-counting nodes shared between adjacent segments
            gmsh_type    = group['gmsh_type']
            nodal_forces = {}

            for connectivity in group['connectivity']:
                tags      = [t for t in connectivity if t in node_map]
                node_list = [node_map[t] for t in tags]
                if len(node_list) < 2:
                    continue

                f_scalar = _consistent_line_load(node_list, gmsh_type, load_value)

                for tag, f in zip(tags, f_scalar):
                    nodal_forces[tag] = nodal_forces.get(tag, 0.0) + f

            for tag, f in nodal_forces.items():
                F[node_map[tag].idx] += f * d

        elif dim == 2:
    
            unique_tags = set()
            for conn in group['connectivity']:
                unique_tags.update(conn)
            unique_tags = [t for t in unique_tags if t in node_map]
            n = len(unique_tags)
            if n > 0:
                for tag in unique_tags:
                    # F[node_map[tag].idx] += (load_value / n) * d
                    F[node_map[tag].idx[:len(d)]] += (load_value / n) * d


    return F


def _consistent_line_load(node_list: list,
                           gmsh_type: int,
                           load_value: float) -> np.ndarray:
    """
    Compute consistent nodal scalar forces for a uniform line load on a
    single line segment using Lagrange shape functions.

    Node ordering follows gmsh conventions:
      - gmsh_type=1  (2-node linear):    [n_start, n_end]
      - gmsh_type=8  (3-node quadratic): [n_start, n_end, n_mid]

    Parameters
    ----------
    node_list  : list of Node   Nodes in gmsh connectivity order.
    gmsh_type  : int            Gmsh element type (1=linear, 8=quadratic).
    load_value : float          Load magnitude [force / length].

    Returns
    -------
    f_scalar : np.ndarray   Scalar force at each node in gmsh connectivity order.
    """
    coords = np.array([n.coordinates for n in node_list])

    if gmsh_type == 1:
        L        = np.linalg.norm(coords[1] - coords[0])
        f_scalar = load_value * L / 2.0 * np.array([1.0, 1.0])

    elif gmsh_type == 8:
        L        = np.linalg.norm(coords[1] - coords[0])
        f_scalar = load_value * L / 6.0 * np.array([1.0, 1.0, 4.0])

    else:
        raise NotImplementedError(
            f"Line gmsh_type={gmsh_type} is not supported by build_load_vector. "
            f"Supported: 1 (2-node linear), 8 (3-node quadratic)."
        )

    return f_scalar


# -- Legacy utilities ----------------------------------------------------------

def build_nodes(mesh, restrain_dictionary: dict = None):
    """
    Backward-compatible wrapper around plan().

    New code should use plan() directly.

    Returns
    -------
    node_map    : dict
    nodes       : np.ndarray  Node objects ordered by gmsh tag
    system_nDof : int
    """
    from fem.utils.gmshtools import GMSHtools
    if isinstance(mesh, GMSHtools):
        node_map, system_nDof = plan(mesh, restrain_dictionary)
    else:
        # Plain dict — wrap temporarily
        class _DictMesh:
            def __init__(self, d):
                self.nodes    = d['nodes']
                self.elements = d['elements']
        node_map, system_nDof = plan(_DictMesh(mesh), restrain_dictionary)

    all_tags = sorted(node_map.keys())
    nodes    = np.array([node_map[t] for t in all_tags], dtype=object)
    return node_map, nodes, system_nDof


def get_nodes_from_physical_id(mesh, target_id: int, nodes: list):
    """
    Deprecated: use plan() with gmshtools.GMSHtools instead.
    """
    found_nodes = []
    for mesh_geo, phys_ids in zip(mesh.cells, mesh.cell_data['gmsh:physical']):
        for conn, phys_id in zip(mesh_geo.data, phys_ids):
            if phys_id == target_id:
                found_nodes.extend(conn)
    found_nodes = np.unique(found_nodes)
    return nodes[found_nodes]


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