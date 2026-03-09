import numpy as np
import gmsh


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


# -- Gmsh node builders --------------------------------------------------------

def _read_nodes(output_file):
    """
    Open a gmsh mesh file and create one Node object per mesh node.

    Gmsh tags are not necessarily consecutive or zero-based. This function
    remaps them to consecutive indices starting from 0 so that
    node.idx = [2*i, 2*i+1] for nDoF=2.

    Returns
    -------
    node_map : dict  {gmsh_tag: Node}
    nodes    : np.ndarray of Node objects, ordered by gmsh tag
    """
    from fem.core.Node import Node

    gmsh.initialize()
    gmsh.open(output_file)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)

    # Map gmsh tag -> consecutive index 0, 1, 2, ...
    all_tags = sorted(node_tags)
    tag_to_index = {tag: i for i, tag in enumerate(all_tags)}

    # Build coord lookup by tag
    coord_map = {tag: (x, y) for tag, (x, y, z) in zip(node_tags, coords)}

    node_map = {}
    for tag in all_tags:
        i = tag_to_index[tag]
        x, y = coord_map[tag]
        node = Node(name=int(tag), coordinates=[x, y])
        node.idx = np.array([2 * i, 2 * i + 1])  # override to ensure 0-based
        node_map[tag] = node

    nodes = np.array([node_map[t] for t in all_tags], dtype=object)

    return node_map, nodes


def _apply_restraints(node_map, restrain_dictionary):
    """
    Apply boundary conditions to nodes belonging to each physical group.

    Parameters
    ----------
    node_map           : dict  {gmsh_tag: Node}
    restrain_dictionary: dict  {phys_id: ['r'/'f', 'r'/'f']}
    """
    for dim, phys_id in gmsh.model.getPhysicalGroups():
        if phys_id not in restrain_dictionary:
            continue
        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_id)
        node_tags_in_group = set()
        for ent_tag in entity_tags:
            _, _, elem_node_tags = gmsh.model.mesh.getElements(dim, ent_tag)
            for arr in elem_node_tags:
                node_tags_in_group.update(arr)
        for tag in node_tags_in_group:
            node_map[tag].set_restrain(restrain_dictionary[phys_id])


def build_nodes_from_gmsh(output_file, restrain_dictionary=None):
    """
    Read a gmsh mesh file and return fully configured Node objects.

    Coordinates and boundary conditions are applied before returning.
    Nodal loads from line groups are handled separately by build_line_load_vector.

    Parameters
    ----------
    output_file         : str   Path to the .msh file
    restrain_dictionary : dict  {phys_id: ['r'/'f', ...]}  (optional)

    Returns
    -------
    node_map : dict         {gmsh_tag: Node}
    nodes    : np.ndarray   Node objects ordered by gmsh tag
    """
    node_map, nodes = _read_nodes(output_file)

    if restrain_dictionary:
        _apply_restraints(node_map, restrain_dictionary)

    gmsh.finalize()
    return node_map, nodes


# -- Gmsh element builders -----------------------------------------------------

def create_elements_from_gmsh(output_file, node_map, section_dictionary,
                               element_class_map):
    """
    Create element objects from a gmsh mesh file using an existing node_map.

    Parameters
    ----------
    output_file       : str   Path to the .msh file
    node_map          : dict  {gmsh_tag: Node}  from build_nodes_from_gmsh
    section_dictionary: dict  {phys_id: section object}
    element_class_map : dict  {num_nodes: ElementClass}
                        e.g. {3: CST, 4: Quad4, 6: LST}

    Returns
    -------
    elements : np.ndarray of element objects
    """
    gmsh.initialize()
    gmsh.open(output_file)

    elements = []
    for dim, phys_id in gmsh.model.getPhysicalGroups():
        if phys_id not in section_dictionary:
            continue
        section_obj = section_dictionary[phys_id]
        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_id)
        for entity_tag in entity_tags:
            elem_types, elem_tags_list, node_tags_list = gmsh.model.mesh.getElements(dim, entity_tag)
            for elem_type, elem_tags, elem_node_tags in zip(elem_types, elem_tags_list, node_tags_list):
                num_nodes = len(elem_node_tags) // len(elem_tags)
                if num_nodes not in element_class_map:
                    continue
                ElementClass = element_class_map[num_nodes]
                for i, gmsh_elem_tag in enumerate(elem_tags):
                    node_tag_subset = elem_node_tags[i * num_nodes : (i + 1) * num_nodes]
                    node_list = [node_map[tag] for tag in node_tag_subset]
                    elem = ElementClass(
                        element_tag=gmsh_elem_tag,
                        node_list=node_list,
                        section=section_obj,
                    )
                    elements.append(elem)

    gmsh.finalize()
    return np.array(elements, dtype=object)



# -- Line load builders --------------------------------------------------------

def _direction_to_vector(direction):
    """
    Convert a direction specification to a unit vector [cx, cy].

    Parameters
    ----------
    direction : str or float
        'x'   ->  0 degrees  (positive right)
        '-x'  -> 180 degrees
        'y'   ->  90 degrees (positive up)
        '-y'  -> 270 degrees
        float -> angle in degrees measured from positive X axis, counterclockwise
    """
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


def _consistent_line_load(node_list, load_value):
    """
    Compute consistent nodal scalar forces for a uniform line load
    using Lagrange shape functions.

    Parameters
    ----------
    node_list  : list of Node  (2 or 3 nodes)
    load_value : float         load magnitude [force/length]

    Returns
    -------
    f_scalar : np.ndarray  scalar force at each node
    """
    coords = np.array([n.coordinates for n in node_list])
    L = np.linalg.norm(coords[-1] - coords[0])
    n = len(node_list)

    if n == 2:
        # Linear Lagrange: integral of N_i over [-1,1] * L/2
        # N1 = (1-xi)/2, N2 = (1+xi)/2  ->  each integrates to 1
        f_scalar = load_value * L / 2 * np.array([1.0, 1.0])

    elif n == 3:
        # Quadratic Lagrange: N1=(xi^2-xi)/2, N2=1-xi^2, N3=(xi^2+xi)/2
        # integrals over [-1,1]: 1/3, 4/3, 1/3  ->  multiply by L/2
        f_scalar = load_value * L / 6 * np.array([1.0, 4.0, 1.0])

    else:
        raise NotImplementedError(f"Line element with {n} nodes is not supported.")

    return f_scalar


def build_load_vector(output_file, node_map, load_dictionary, system_nDof):
    """
    Build the global nodal force vector from point and/or line loads.

    Automatically detects physical group dimension:
    - dim=0 (points) -> point load applied directly to node
    - dim=1 (lines)  -> consistent line load distributed along edge

    Parameters
    ----------
    output_file     : str   Path to the .msh file
    node_map        : dict  {gmsh_tag: Node}
    load_dictionary : dict  {phys_id: {'value': float, 'direction': str or float}}
                            direction: 'x', '-x', 'y', '-y', or angle in degrees
    system_nDof     : int   Total number of DOFs in the system

    Returns
    -------
    F : np.ndarray  (system_nDof,)  global load vector

    Examples
    --------
    load_dictionary = {
        50:  {'value': 1000, 'direction': 'y'},   # point load  [N]
        101: {'value':   10, 'direction': 'x'},   # line load   [N/mm]
    }
    F = build_load_vector(output_file, node_map, load_dictionary, system_nDof)
    """
    F = np.zeros(system_nDof)

    gmsh.initialize()
    gmsh.open(output_file)

    for dim, phys_id in gmsh.model.getPhysicalGroups():
        if phys_id not in load_dictionary:
            continue

        load_value  = load_dictionary[phys_id]['value']
        d           = _direction_to_vector(load_dictionary[phys_id]['direction'])
        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_id)

        if dim == 0:
            # Point load — apply directly to node
            for ent_tag in entity_tags:
                _, _, elem_node_tags = gmsh.model.mesh.getElements(dim, ent_tag)
                for arr in elem_node_tags:
                    for tag in arr:
                        F[node_map[tag].idx] += load_value * d

        elif dim == 1:
            for ent_tag in entity_tags:
                # Get ALL nodes including boundary corners
                node_tags_line, coords, _ = gmsh.model.mesh.getNodes(dim, ent_tag, includeBoundary=True)
                coords = coords.reshape(-1, 3)[:, :2]

                # Sort along the dominant axis
                delta    = coords[-1] - coords[0]
                axis     = 0 if abs(delta[0]) > abs(delta[1]) else 1
                sort_idx = np.argsort(coords[:, axis])

                node_tags_sorted = node_tags_line[sort_idx]
                coords_sorted    = coords[sort_idx]

                # Detect element order
                elem_types, _, _ = gmsh.model.mesh.getElements(dim, ent_tag)
                is_second_order  = any(et in [8, 26] for et in elem_types)

                step = 2 if is_second_order else 1

                for i in range(0, len(node_tags_sorted) - 1, step):
                    tags      = node_tags_sorted[i: i + step + 1]
                    node_list = [node_map[t] for t in tags if t in node_map]
                    if len(node_list) < 2:
                        continue
                    f_scalar = _consistent_line_load(node_list, load_value)
                    for node, f in zip(node_list, f_scalar):
                        F[node.idx] += f * d

    gmsh.finalize()
    return F



def get_nodes_from_physical_id(mesh, target_id: int, nodes: list):
    found_nodes = []
    for mesh_geo, phys_ids in zip(mesh.cells, mesh.cell_data['gmsh:physical']):
        for conn, phys_id in zip(mesh_geo.data, phys_ids):
            if phys_id == target_id:
                found_nodes.extend(conn)
    found_nodes = np.unique(found_nodes)
    return nodes[found_nodes]


def get_line_load_global_vector(node_start, node_end, wj, wk, alpha_degree=None):
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
    c   = np.cos(theta)
    s   = np.sin(theta)
    Tlg = np.array([
        [ c,  s, 0, 0],
        [-s,  c, 0, 0],
        [ 0,  0, c, s],
        [ 0,  0,-s, c]
    ])
    return Tlg.T @ F_local