"""FEM Utilities Module"""

from .functions import (
    matrix_extract,
    matrix_replace,
    get_nodes_from_physical_id,
    get_line_load_global_vector
)

__all__ = [
    'matrix_extract',
    'matrix_replace', 
    'get_nodes_from_physical_id',
    'get_line_load_global_vector'
]
