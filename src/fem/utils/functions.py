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


# -- FEM node builders ---------------------------------------------------------

def build_nodes(mesh: dict, restrain_dictionary: dict = None):
    """
    Instantiate Node objects from raw mesh data.
    Gmsh tags are remapped to consecutive zero-based indices so that
    node.idx = [nDoF*i, ..., nDoF*i + nDoF-1].
    The number of DOFs per node is read from globalParameters['nDoF'].
    The spatial dimension is read from globalParameters['nDIM'].

    Parameters
    ----------
    mesh                : dict  Output of gmshtools.read_mesh
    restrain_dictionary : dict  {phys_id: ['r'/'f', ...]}  (optional)

    Returns
    -------
    node_map    : dict        {gmsh_tag: Node}
    nodes       : np.ndarray  Node objects ordered by gmsh tag
    system_nDof : int         Total number of DOFs in the system
    """
  
    nDoF = globalParameters['nDoF']
    nDIM = globalParameters['nDIM']

    raw      = mesh['nodes']
    all_tags = sorted(raw.keys())

    # Remap gmsh tags to consecutive zero-based indices
    tag_to_index = {tag: i for i, tag in enumerate(all_tags)}

    node_map = {}
    for tag in all_tags:
        i        = tag_to_index[tag]
        coords   = list(raw[tag][:nDIM])          # take first nDIM coordinates
        node     = Node(name=int(tag), coordinates=coords)
        node.idx = np.array([nDoF * i + j for j in range(nDoF)])
        node_map[tag] = node

    if restrain_dictionary:
        _apply_restraints(node_map, mesh, restrain_dictionary)

    nodes       = np.array([node_map[t] for t in all_tags], dtype=object)
    system_nDof = len(nodes) * nDoF

    return node_map, nodes, system_nDof


def _apply_restraints(node_map: dict, mesh: dict, restrain_dictionary: dict):
    """
    Apply boundary conditions to nodes belonging to physical groups.

    Parameters
    ----------
    node_map            : dict  {gmsh_tag: Node}
    mesh                : dict  Output of gmshtools.read_mesh
    restrain_dictionary : dict  {phys_id: ['r'/'f', ...]}
    """
    for phys_id, condition in restrain_dictionary.items():
        if phys_id not in mesh['elements']:
            continue
        for node_tags in mesh['elements'][phys_id]['connectivity']:
            for tag in node_tags:
                if tag in node_map:
                    node_map[tag].set_restrain(condition)


# -- FEM element builders ------------------------------------------------------

def build_elements(mesh: dict, node_map: dict, section_dictionary: dict,
                   element_class_map: dict) -> np.ndarray:
    """
    Instantiate FEM element objects from raw mesh data.

    Parameters
    ----------
    mesh               : dict  Output of gmshtools.read_mesh
    node_map           : dict  {gmsh_tag: Node}  from build_nodes
    section_dictionary : dict  {phys_id: section object}
    element_class_map  : dict  {n_nodes: ElementClass}
                         e.g. {3: CST, 4: Quad4, 6: LST, 9: Quad9}

    Returns
    -------
    np.ndarray of element objects
    """
    elements = []

    for phys_id, section in section_dictionary.items():
        if phys_id not in mesh['elements']:
            continue

        group    = mesh['elements'][phys_id]
        n_nodes  = group['n_nodes']

        if n_nodes not in element_class_map:
            continue

        ElementClass = element_class_map[n_nodes]

        for elem_tag, connectivity in zip(group['element_tags'], group['connectivity']):
            node_list = [node_map[tag] for tag in connectivity]
            elem      = ElementClass(
                element_tag=elem_tag,
                node_list=node_list,
                section=section,
            )
            elements.append(elem)

    return np.array(elements, dtype=object)

def build_plot_elements(mesh: dict, node_map: dict, phys_ids: list) -> np.ndarray:
    """
    Build lightweight element objects for plotting.
    Each element has only element_tag and nodes attributes.

    Parameters
    ----------
    mesh     : dict       Output of read_mesh
    node_map : dict       {gmsh_tag: Node} from build_nodes
    phys_ids : list       Physical group IDs to include

    Returns
    -------
    np.ndarray of SimpleNamespace objects with:
        .element_tag  : int
        .nodes        : list of Node
    """
    from types import SimpleNamespace
    elements = []
    for phys_id in phys_ids:
        if phys_id not in mesh['elements']:
            continue
        group = mesh['elements'][phys_id]
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

    if nDoF == 3:
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


def build_load_vector(mesh: dict, node_map: dict, load_dictionary: dict ) -> np.ndarray:
    """
    Build the global nodal force vector from point, line, surface or volume loads.
    Detects physical group dimension automatically:
    - dim=0 (points)  -> point load applied directly to the node
    - dim=1 (lines)   -> consistent line load distributed along the edge
    - dim=2 (surface) -> total load distributed equally among all nodes
    - dim=3 (volume)  -> total load distributed equally among all nodes

    Parameters
    ----------
    mesh            : dict  Output of gmshtools.read_mesh
    node_map        : dict  {gmsh_tag: Node}  from build_nodes
    load_dictionary : dict  {phys_id: {'value': float, 'direction': str or float}}
                            direction: 'x', '-x', 'y', '-y', 'z', '-z', or angle in degrees
    system_nDof     : int   Total number of DOFs in the system

    Returns
    -------
    F : np.ndarray  (system_nDof,)  global load vector
    """
    nDoF        = globalParameters['nDoF']
    system_nDof = len(node_map) * nDoF
    F = np.zeros(system_nDof)

    for phys_id, load_spec in load_dictionary.items():
        if phys_id not in mesh['elements']:
            continue

        group      = mesh['elements'][phys_id]
        dim        = group['dim']
        load_value = load_spec['value']
        d          = _direction_to_vector(load_spec['direction'])

        if dim == 0:
            # Point load — apply directly to each node in the group
            for connectivity in group['connectivity']:
                for tag in connectivity:
                    if tag in node_map:
                        F[node_map[tag].idx] += load_value * d

        elif dim == 1:
            # Consistent line load — distribute using Lagrange shape functions
            _apply_line_load(F, group, node_map, load_value, d)

        elif dim == 2 or dim == 3:
            # Collect all unique nodes in the group
            unique_nodes = set()
            for connectivity in group['connectivity']:
                for tag in connectivity:
                    if tag in node_map:
                        unique_nodes.add(tag)

            # Distribute total load equally among all nodes
            n = len(unique_nodes)
            if n > 0:
                for tag in unique_nodes:
                    F[node_map[tag].idx] += (load_value / n) * d

    return F



def _apply_line_load(F: np.ndarray, group: dict, node_map: dict,
                     load_value: float, d: np.ndarray):
    """
    Distribute a uniform line load consistently over line elements.

    Accumulates nodal forces by tag before assigning to F to avoid
    double-counting nodes shared between adjacent line elements.

    Parameters
    ----------
    F          : np.ndarray  Global force vector (modified in place)
    group      : dict        Element group from mesh['elements']
    node_map   : dict        {gmsh_tag: Node}
    load_value : float       Load magnitude [force/length]
    d          : np.ndarray  Unit direction vector [cx, cy]
    """
    gmsh_type    = group['gmsh_type']
    nodal_forces = {}

    for connectivity in group['connectivity']:
        tags      = [t for t in connectivity if t in node_map]
        node_list = [node_map[t] for t in tags]
        if len(node_list) < 2:
            continue
        f_scalar = _consistent_line_load(node_list, tags, gmsh_type, load_value)
        for tag, f in zip(tags, f_scalar):
            nodal_forces[tag] = nodal_forces.get(tag, 0.0) + f

    for tag, f in nodal_forces.items():
        F[node_map[tag].idx] += f * d


def _consistent_line_load(node_list: list, tags: list,
                           gmsh_type: int, load_value: float) -> np.ndarray:
    """
    Compute consistent nodal scalar forces for a uniform line load
    using Lagrange shape functions.

    Node ordering follows gmsh conventions per element type:
    - gmsh_type=1 (2-node line)  : [n1, n2]
    - gmsh_type=8 (3-node line)  : [n_start, n_end, n_mid]

    Parameters
    ----------
    node_list  : list of Node  Nodes in gmsh connectivity order
    tags       : list of int   Gmsh node tags in same order
    gmsh_type  : int           Gmsh element type (1=linear, 8=quadratic)
    load_value : float         Load magnitude [force/length]

    Returns
    -------
    f_scalar : np.ndarray  Scalar force at each node in gmsh connectivity order
    """
    coords = np.array([n.coordinates for n in node_list])

    if gmsh_type == 1:
        # Linear line: [n_start, n_end]
        L        = np.linalg.norm(coords[1] - coords[0])
        f_scalar = load_value * L / 2 * np.array([1.0, 1.0])

    elif gmsh_type == 8:
        # Quadratic line: gmsh gives [n_start, n_end, n_mid]
        # L is distance from n_start (index 0) to n_end (index 1)
        L        = np.linalg.norm(coords[1] - coords[0])
        # Consistent forces in gmsh order [start, end, mid]: [1/6, 1/6, 4/6] * L
        f_scalar = load_value * L / 6 * np.array([1.0, 1.0, 4.0])

    else:
        raise NotImplementedError(
            f"Line gmsh_type={gmsh_type} is not supported. "
            f"Supported: 1 (2-node linear), 8 (3-node quadratic)."
        )

    return f_scalar


# -- Legacy utilities ----------------------------------------------------------

def get_nodes_from_physical_id(mesh, target_id: int, nodes: list):
    """
    Return nodes belonging to a physical group using a meshio mesh object.

    Deprecated: use build_nodes with gmshtools.read_mesh instead.
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
        None or 'y' -> perpendicular to element (local y)
        'x'         -> parallel to element (local x)
        float       -> angle in degrees

    Returns
    -------
    np.ndarray  (4,)  Global equivalent nodal forces [Fx_j, Fy_j, Fx_k, Fy_k]
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