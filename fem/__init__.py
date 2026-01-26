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
from .elements.quad4 import Quad4
from .elements.quad9_1 import Quad9

# Utilities
from .utils.functions import (
    matrix_extract,
    matrix_replace,
    get_nodes_from_physical_id,
    get_line_load_global_vector
)

__version__ = "0.1.0"
