"""FEM Utilities Module"""

from .functions import (
    matrix_extract,
    matrix_replace,
    get_nodes_from_physical_id,
    get_line_load_global_vector
)

from .visualization import (
    add_element_data_view,
    add_node_data_view,
    compute_nodal_average
)

__all__ = [
    'matrix_extract',
    'matrix_replace', 
    'get_nodes_from_physical_id',
    'get_line_load_global_vector',
    'add_element_data_view',
    'add_node_data_view',
    'compute_nodal_average'
]
