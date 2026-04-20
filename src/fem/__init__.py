"""FEM - Finite Element Analysis Library"""

# Core
from .core.Node import Node
from .core.Material import Material
from .core.parameters import globalParameters

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

# Visualization — gmsh
from .utils.visualization import (
    add_element_data_view,
    add_node_data_view,
    compute_nodal_average,
    results2gmsh,
    opensees2gmsh,
    animate_nodal_view,
    animate_results,
)

# Plotting — matplotlib
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

# Model
from .model.result import FEMResult
from .model.modal_result import ModalResult
from .model.fem_model import FEMModel

__version__ = "1.2.0"

print("""
  FEM -- Finite Element Method for Structural Analysis
  Based on the course by Prof. Jose Abell

  Version 1.2.0                        (c) 2026 All Rights Reserved

  Repository  :  https://github.com/ppalacios92/FEM
  Web Book    :  https://books.nmorabowen.com/books/fem

  Patricio Palacios B.    |    Nicolas Mora Bowen
  
  ********* (>'-')> Ladruno4ever  *********
""")
