# from .Node import Node
# from .Material import Material
# from .Membrane import Membrane
# from .CST import CST
# from .LST import LST 
# from .Truss2D import Truss2D
# from .Frame2D import Frame2D
# from .functions import matrix_extract, matrix_replace, get_nodes_from_physical_id, get_line_load_global_vector

"""FEM - Finite Element Analysis Library"""

# Core modules
from .core.Node import Node
from .core.Material import Material

# Sections
from .sections.Membrane import Membrane

# Elements
from .elements.CST import CST
from .elements.LST import LST
from .elements.Truss2D import Truss2D
from .elements.Frame2D import Frame2D
from .elements.Quad4 import Quad4
from .elements.Quad9 import Quad9

# Utilities
from .utils.functions import (
    matrix_extract,
    matrix_replace,
    get_nodes_from_physical_id,
    get_line_load_global_vector,
    build_nodes_from_gmsh,         
    create_elements_from_gmsh,     
    build_load_vector               
)
# Visualization utilities
from .utils.visualization import (
    add_element_data_view,
    add_node_data_view,
    compute_nodal_average
)
# Units                           
from .utils.units import (
    mm, cm, m,
    kgf, N, tf, kN,
    MPa, GPa
)

# Global parameters
from .core.parameters import globalParameters

__version__ = "0.1.0"


print("""
  FEM -- Finite Element Method for Structural Analysis
  Based on the course by Prof. José Abell

  Version 0.1.0                        © 2026 All Rights Reserved

  Repository  :  https://github.com/ppalacios92/FEM
  Web Book    :  https://books.nmorabowen.com/books/fem

  Patricio Palacios B.    |    Nicolas Mora Bowen

  ********* (>'-')> Ladruño4ever  *********
""")