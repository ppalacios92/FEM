import numpy as np

# We create some wrapper functions to access and replace matrix coefficients using row and column indices

def matrix_extract(matrix: np.ndarray, row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
    """
    Extracts submatrix values from a given matrix using specified row and column indices.

    Args:
        matrix (np.ndarray): The original matrix.
        row_indices (np.ndarray): Indices of the rows to extract.
        col_indices (np.ndarray): Indices of the columns to extract.

    Returns:
        np.ndarray: The extracted submatrix.
    """
    return matrix[np.ix_(row_indices, col_indices)]

def matrix_replace(matrix: np.ndarray, matrix_add: np.ndarray, row_indices: np.ndarray, col_indices: np.ndarray) -> np.ndarray:
    """
    Adds a given submatrix to specific positions in a target matrix.

    Args:
        matrix (np.ndarray): The target matrix.
        matrix_add (np.ndarray): The submatrix to be added.
        row_indices (np.ndarray): Indices of the rows where addition should occur.
        col_indices (np.ndarray): Indices of the columns where addition should occur.

    Returns:
        np.ndarray: A new matrix with the submatrix added to the specified indices.
    """
    # Create a copy to avoid modifying the original matrix
    updated_matrix = matrix.copy()
    
    # Use np.ix_ for better readability and performance
    updated_matrix[np.ix_(row_indices, col_indices)] += matrix_add
    
    return updated_matrix


def get_nodes_from_physical_id(mesh, target_id:int, nodes:list):
    """
    Returns np.array of node objects belonging to a given physical id
    regardless of dimension (line, surface, point).
    """
    found_nodes = []

    for mesh_geo, phys_ids in zip(mesh.cells, mesh.cell_data['gmsh:physical']):
        
        for conn, phys_id in zip(mesh_geo.data, phys_ids):
            
            if phys_id == target_id:
                found_nodes.extend(conn)
    
    found_nodes = np.unique(found_nodes)  # remove duplicates
    return nodes[found_nodes]  # this works because nodes is np.array

def get_line_load_global_vector(node_start, node_end, wj, wk, alpha_degree=None):
    """
    Computes equivalent global force vector for a line load.

    alpha_degree:
        None → vertical in global Y
        'x'  → along global X
        'y'  → along global Y
        float → custom angle in degrees relative to local edge
    """
    delta = node_end.coordenadas - node_start.coordenadas
    L = np.linalg.norm(delta)
    theta = np.arctan2(delta[1], delta[0])  # local x'

    # Determine load angle
    if alpha_degree is None:
        # Projected vertical
        alpha = np.radians(90) - theta
    elif alpha_degree == 'x':
        alpha = -theta  # because local x' vs global x difference is theta
    elif alpha_degree == 'y':
        alpha = np.radians(90) - theta
    else:
        # Custom user angle
        alpha = np.radians(alpha_degree)

    w1 = wk
    w2 = wj - wk

    wjx = -w1 * np.cos(alpha) * L / 2 - w2 * np.cos(alpha) * L / 3
    wkx = -w1 * np.cos(alpha) * L / 2 - w2 * np.cos(alpha) * L / 6
    wjy = -w1 * np.sin(alpha) * L / 2 - w2 * np.sin(alpha) * L / 3
    wky = -w1 * np.sin(alpha) * L / 2 - w2 * np.sin(alpha) * L / 6

    F_local = -np.array([wjx, wjy, wkx, wky])

    c = np.cos(theta)
    s = np.sin(theta)

    Tlg = np.array([
        [c, s, 0, 0],
        [-s, c, 0, 0],
        [0, 0, c, s],
        [0, 0, -s, c]
    ])

    F_global = Tlg.T @ F_local

    return F_global