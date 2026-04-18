"""FEM Utilities Module"""
from .functions import (
    matrix_extract,
    matrix_replace,
    get_line_load_global_vector,
    build_elements,
    # build_plot_elements,
    plan,
    # apply_restraints,
    build_load_vector,
)
from .gmshtools import (
    get_element_info,
    GMSHtools,
)
from .visualization import (
    add_element_data_view,
    add_node_data_view,
    compute_nodal_average,
)
from .plotting import (
    plot_mesh,
    plot_field_2d,
    plot_deformed,
    plot_loads_2d,
    plot_gmsh_mesh,
)

__all__ = [
    # Matrix utilities
    'matrix_extract',
    'matrix_replace',
    # FEM builders
    'plan',
    # 'apply_restraints',
    'build_elements',
    'build_plot_elements',
    'build_load_vector',
    # Gmsh tools
    'get_element_info',
    'GMSHtools',
    # Frame2D load utility
    'get_line_load_global_vector',
    # Visualization
    'add_element_data_view',
    'add_node_data_view',
    'compute_nodal_average',
    # Plotting
    'plot_mesh',
    'plot_field_2d',
    'plot_deformed',
    'plot_loads_2d',
    'plot_gmsh_mesh',
]