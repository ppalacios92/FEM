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
    get_line_load_global_vector,
    build_elements,
    plan,
)
from .utils.gmshtools import (
    get_element_info,
    GMSHtools,
)
# Visualization
from .utils.visualization import (
    add_element_data_view,
    add_node_data_view,
    compute_nodal_average,
    results2gmsh,
    opensees2gmsh,
)
# Plotting
from .utils.plotting import (
    plot_mesh,
    plot_field_2d,
    plot_deformed,
    plot_loads_2d,
    plot_gmsh_mesh,
)
# Units
from .utils.units import (
    mm, cm, m, km, inches, ft, yard, mile,
    N, kN, kgf, tf, dyne, lbf, kip,
    tonne, kg, mg, lb, oz,
    MPa, GPa, kPa, Pa, kgf_cm2, ksi,
    J, kJ, mJ, cal, kcal, eV, Wh, kWh,
    W, kW, MW, HP,
    s, minutes, h, day, week, month, year,
    g,
    radian, degree,
    K, C, F,
)
# Global parameters
from .core.parameters import globalParameters

__version__ = "1.1.0"

print("""
  FEM -- Finite Element Method for Structural Analysis
  Based on the course by Prof. José Abell

  Version 1.1.0                        © 2026 All Rights Reserved

  Repository  :  https://github.com/ppalacios92/FEM
  Web Book    :  https://books.nmorabowen.com/books/fem

  Patricio Palacios B.    |    Nicolas Mora Bowen

  ********* (>'-')> Ladruño4ever  *********
""")