# FEM – Finite Element Analysis

A Python library for structural analysis using the Finite Element Method, developed for educational and research purposes in civil and structural engineering.

> Based on the FEM course by Prof. José Abell — Universidad de los Andes.
---

## Features

- Modular element library covering 1D, 2D, and 3D elements.
- Implements the following element types:
  - **Truss2D** – 2-node axial element (2 DOF/node)
  - **Frame2D** – 2-node Euler-Bernoulli beam-column (3 DOF/node)
  - **CST** – Constant Strain Triangle (3 nodes · 6 DOF)
  - **LST** – Linear Strain Triangle (6 nodes · 12 DOF)
  - **Quad4** – Bilinear Quadrilateral (4 nodes · 8 DOF)
  - **Quad9** – Biquadratic Lagrangian Quadrilateral (9 nodes · 18 DOF)
- Full isoparametric formulation with Gauss-Legendre numerical integration.
- Direct Stiffness Method (DSM) assembly pipeline.
- `FEMModel` — unified model container for mesh, BCs, elements, solver, and results.
- gmsh-based mesh generation and node/element import.
- OpenSeesPy integration for 2D and 3D solid analysis.
- Modal analysis with animated gmsh visualization.
- Interactive Jupyter widgets for visualization of shape functions, Jacobian fields, B-matrix, and stiffness integrand components.
- Rigid body mode verification for membrane elements.

---

## Requirements

- Python 3.8 or higher
- Python libraries:
  - `numpy`
  - `scipy`
  - `sympy`
  - `matplotlib`
  - `gmsh`
  - `openseespy`
  - `ipywidgets`
  - `jupyter`

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ppalacios92/FEM.git
cd FEM
pip install -e .
```

---

## Repository Structure

```bash
FEM/
├── src/
│   └── fem/
│       ├── core/             # Node, Material, parameters
│       ├── elements/         # CST, LST, Quad4, Quad9, Truss2D, Frame2D
│       ├── sections/         # Membrane section
│       ├── model/            # FEMModel, FEMResult, ModalResult, Solver, BCs
│       └── utils/            # functions, gmshtools, plotting, visualization, units
├── examples_1D/              # Frame2D / Truss2D — direct assembly, no gmsh
├── examples_2D/              # Membrane FEM — gmsh + own solver + matplotlib/gmsh plots
├── examples_3D/              # 3D solid — gmsh + OpenSees + gmsh visualization
├── docs/
│   └── images/               # Reference plots and visualization outputs
└── README.md
```

---

## Import Modules

```python
from fem import (
    # Core
    Material, Membrane,
    # Elements
    CST, LST, Truss2D, Frame2D, Quad4, Quad9,
    # Mesh
    GMSHtools,
    # Model
    FEMModel, FEMResult, ModalResult,
    # Units
    mm, cm, m, N, kN, tf, MPa, GPa, kg, g,
    # Parameters
    globalParameters,
)
```

---

## FEMModel — Unified Model Container

`FEMModel` orchestrates the entire FEM pipeline — mesh, boundary conditions, elements, solver, and results — in a single object.

```python
model = FEMModel(
    mesh                = mesh,
    section_dictionary  = section_dictionary,
    restrain_dictionary = restrain_dictionary,
    load_dictionary     = load_dictionary,
    element_class_map   = {3: CST, 6: LST},  # None = OpenSees only
    analysis_type       = 'planeStress',       # 'planeStress', 'planeStrain', '3D'
    consistent_loads    = True,
    sampling_points     = 3,
)
```

### Dictionaries

```python
section_dictionary  = {201: Membrane(material=Steel, thickness=15)}
restrain_dictionary = {101: ['r', 'r'], 102: ['f', 'r']}
load_dictionary     = {50:  {'value': -500.0, 'direction': 'y'}}
```

### Solve — own solver

```python
model.solve_static(n_steps=10, load_factor=1.0)
```

### Solve — OpenSees

```python
import openseespy.opensees as ops

ops.wipe()
ops.model('basicBuilder', '-ndm', 2, '-ndf', 2)

for tag, (x, y, z) in mesh.nodes.items():
    ops.node(tag, x, y)

for tag, condition in model.restrained_nodes.items():
    ops.fix(tag, *[1 if r == 'r' else 0 for r in condition])

ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
for tag, force in model.F_nodal.items():
    ops.load(tag, *force.tolist())

# ... define materials and elements ...

ops.system('SparseSYM')
ops.numberer('RCM')
ops.constraints('Plain')
ops.integrator('LoadControl', 0.1)
ops.algorithm('Newton')
ops.analysis('Static')
ops.analyze(10)

model.set_results_opensees(ops, step=0, time=1.0)
```

### Modal analysis

```python
model.set_modal_results(ops, n_modes=6)
model.plot_modal(modes=[1, 2, 3], n_steps=30, disp_factor=50)
```

### Result queries

```python
model.get_node(tag=6)
model.get_node(x=2500, y=250)
model.get_element(tag=10)

model.node_history(tag=6, component='uy', source='fem')
model.element_history(tag=10, component='sxx', source='opensees')
```

### Mesh diagnostics

```python
model.check_mesh()   # reports orphan nodes, restrained nodes, load summary
                     # automatically removes orphan nodes and rebuilds node_map
```

---

## Plotting

### matplotlib

```python
model.plot(show_element_edges=True, show_supports=True, figsize=(12, 8))

model.plot_loads(show_element_edges=True, figsize=(12, 8))

model.plot_deformed(
    sfac    = 50,
    step    = -1,
    source  = 'fem',          # 'fem', 'opensees', None=last used
    figsize = (12, 8),
)

model.plot_field(
    component          = 'vm',         # 'sxx','syy','sxy','vm','exx','eyy','exy'
    step               = -1,
    source             = 'opensees',   # 'fem', 'opensees', None=last used
    deformed           = True,
    sfac               = 50,
    cmap               = 'turbo',
    figsize            = (12, 8),
)

model.plot_node_history(tags=[6, 10], component='uy', source='fem')
model.plot_element_history(tags=[100], component='sxx', source='fem')
```

> `plot_field` automatically dispatches to `plot_field_2d` (own solver with FEM elements) or `plot_field_3d` (OpenSees/3D scatter) depending on the model type.

### gmsh

```python
model.plot2gmsh(
    step           = -1,
    source         = 'opensees',
    disp_factor    = 50,
    show_disp      = True,
    show_loads     = True,
    show_reactions = True,
    show_stress    = True,
    show_strain    = True,
    show_vm        = True,
    show_averaged  = True,
)

model.plot2gmsh_animate(disp_factor=50, source='fem')
```

---

## Plotting Functions Reference

| Function | Description |
|---|---|
| `model.plot` | Mesh geometry with node/element labels and support symbols |
| `model.plot_loads` | Normalized load arrows over mesh background |
| `model.plot_deformed` | Deformed shape colored by displacement component |
| `model.plot_field` | Stress or strain field — 2D contour or 3D scatter |
| `model.plot_node_history` | Time history of displacement at selected nodes |
| `model.plot_element_history` | Time history of stress/strain at selected elements |
| `model.plot2gmsh` | Full results in gmsh — displacements, stresses, reactions |
| `model.plot2gmsh_animate` | Animated displacement steps in gmsh |
| `model.plot_modal` | Animated modal shapes in gmsh |

---

## gmsh Integration

`GMSHtools` reads a `.msh` file generated by gmsh and instantiates a structured mesh object — nodes, elements, and physical groups are immediately accessible as attributes:

```python
mesh = GMSHtools('mesh.msh')
```

### Accessing mesh data

```python
mesh.nodes                           # {tag: (x, y, z)}
mesh.elements                        # {phys_id: {gmsh_type, connectivity, ...}}
mesh.physical_groups                 # accessible by integer ID or name string
```

Physical groups can be queried by **ID** or by **name**:

```python
group = mesh.physical_groups[201]
group = mesh.physical_groups['Beam']
```

Each `PhysicalGroup` exposes:

| Attribute | Type | Description |
|---|---|---|
| `.id` | `int` | Gmsh physical group ID |
| `.name` | `str` | Name as defined in gmsh |
| `.dim` | `int` | Dimension (0=point, 1=line, 2=surface, 3=volume) |
| `.nodes` | `dict` | `{tag: (x, y, z)}` — nodes belonging to this group |
| `.elements` | `dict` | Raw element data (connectivity, gmsh_type, etc.) |

---

## Workflows

### 1D — Frame / Truss

Direct assembly without gmsh. Nodes and elements defined manually.

```python
globalParameters['nDoF'] = 3
globalParameters['nDIM'] = 2

n0 = Node(0, [0, 0],  restrain=['r', 'r', 'r'])
n1 = Node(1, [5, 0],  restrain=['f', 'f', 'f'], nodal_load=[0, -20, 0])
n2 = Node(2, [10, 0], restrain=['r', 'f', 'r'])

steel = Material(name='Steel', E=29000, nu=0.3, rho=0)
e0 = Frame2D(n0, n1, material=steel, A=3, I=400)
e1 = Frame2D(n1, n2, material=steel, A=3, I=400)
```

### 2D — Membrane (own solver)

```python
globalParameters['nDoF'] = 2
globalParameters['nDIM'] = 2

Steel  = Material(name='Steel', E=200000.0, nu=0.30, rho=0.0)
Plate  = Membrane(name='Plate', thickness=10.0, material=Steel)

section_dictionary  = {201: Plate}
restrain_dictionary = {101: ['r', 'r']}
load_dictionary     = {50:  {'value': -500.0, 'direction': 'y'}}

mesh  = GMSHtools('mesh.msh')
model = FEMModel(
    mesh                = mesh,
    section_dictionary  = section_dictionary,
    restrain_dictionary = restrain_dictionary,
    load_dictionary     = load_dictionary,
    element_class_map   = {3: CST, 6: LST},
    analysis_type       = 'planeStress',
    consistent_loads    = True,
    sampling_points     = 3,
)

model.solve_static(n_steps=1)
model.plot2gmsh(source='fem', disp_factor=50)
```

### 2D — Membrane (OpenSees solver)

```python
globalParameters['nDoF'] = 2
globalParameters['nDIM'] = 2

mesh  = GMSHtools('mesh.msh')
model = FEMModel(
    mesh                = mesh,
    section_dictionary  = {},
    restrain_dictionary = restrain_dictionary,
    load_dictionary     = load_dictionary,
    element_class_map   = None,
    analysis_type       = 'planeStress',
)

model.check_mesh()

import openseespy.opensees as ops
ops.wipe()
ops.model('basicBuilder', '-ndm', 2, '-ndf', 2)

for tag, (x, y, z) in mesh.nodes.items():
    ops.node(tag, x, y)

for tag, condition in model.restrained_nodes.items():
    ops.fix(tag, *[1 if r == 'r' else 0 for r in condition])

ops.nDMaterial('ElasticIsotropic', 1, 200000.0, 0.3)

group = mesh.physical_groups['Beam'].elements
for etag, conn in zip(group['element_tags'], group['connectivity']):
    ops.element('tri31', etag, *conn, 10.0, 'PlaneStress', 1)

ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
for tag, force in model.F_nodal.items():
    ops.load(tag, *force.tolist())

ops.system('SparseSYM')
ops.numberer('RCM')
ops.constraints('Plain')
ops.integrator('LoadControl', 0.1)
ops.algorithm('Newton')
ops.analysis('Static')
ops.analyze(10)

model.set_results_opensees(ops, step=0, time=1.0)
model.plot2gmsh(source='opensees', disp_factor=50)
```

### 3D — Solid (OpenSees solver)

```python
globalParameters['nDoF'] = 3
globalParameters['nDIM'] = 3

mesh  = GMSHtools('mesh.msh')
model = FEMModel(
    mesh                = mesh,
    section_dictionary  = {},
    restrain_dictionary = {'support': ['r', 'r', 'r']},
    load_dictionary     = {},
    element_class_map   = None,
    analysis_type       = '3D',
)

model.check_mesh()

import openseespy.opensees as ops
ops.wipe()
ops.model('basicBuilder', '-ndm', 3, '-ndf', 3)

for tag, (x, y, z) in mesh.nodes.items():
    ops.node(tag, x, y, z)

for tag, condition in model.restrained_nodes.items():
    ops.fix(tag, *[1 if r == 'r' else 0 for r in condition])

ops.nDMaterial('ElasticIsotropic', 1, 210e3, 0.3, 7.85e-9)

group = mesh.physical_groups['solid'].elements
for etag, conn in zip(group['element_tags'], group['connectivity']):
    ops.element('FourNodeTetrahedron', etag, *conn, 1)

ops.system('SparseSYM')
ops.numberer('RCM')
ops.constraints('Plain')
ops.integrator('LoadControl', 0.1)
ops.algorithm('Newton')
ops.analysis('Static')
ops.analyze(10)

model.set_results_opensees(ops, step=0, time=1.0)
model.set_modal_results(ops, n_modes=6)
model.plot_modal(modes=[1, 2, 3], n_steps=30, disp_factor=50)
model.plot2gmsh(source='opensees', disp_factor=50)
```

---

## Save and Load Results

```python
model.save('results.h5')

data = FEMModel.load_results('results.h5')
fem_results      = data['fem']
opensees_results = data['opensees']
```

---

## 🛑 Disclaimer

This library is developed for educational purposes in the context of the Finite Element Method course at Universidad de los Andes. Results should always be validated against reference solutions and established FEM software.

The author assumes no responsibility for incorrect use, misinterpretation of results, or consequences of numerical errors.

---

## Author

Developed by **Patricio Palacios B. - Nicolas Mora Bowen**
GitHub: [@ppalacios92](https://github.com/ppalacios92)
GitHub: [@nmorabowen](https://github.com/nmorabowen)

---

## How to Cite

```bibtex
@misc{palacios2025fem,
  author       = {Patricio Palacios B., Nicolas Mora Bowen},
  title        = {FEM: A Python Library for Finite Element Analysis},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/ppalacios92/FEM}}
}
```

**APA (7th Edition):**
Palacios P. , Mora Bowen N. (2025). *FEM: A Python library for finite element analysis* [Computer software]. GitHub. https://github.com/ppalacios92/FEM

---

## License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## Contributing

Contributions are welcome! Feel free to submit pull requests, report bugs, or suggest new features through the GitHub issues page.

---

## Get Fun with FEM!

Interactive visualizations included in this library — explore shape functions, Jacobian fields, and stiffness integrands live in Jupyter.

| <img src="docs/images/01.png" width="150"/> | <img src="docs/images/02.png" width="150"/> | <img src="docs/images/03.png" width="150"/> | <img src="docs/images/04.png" width="150"/> |
|:---:|:---:|:---:|:---:|
| <img src="docs/images/05.png" width="150"/> | <img src="docs/images/06.png" width="150"/> | <img src="docs/images/07.png" width="150"/> | <img src="docs/images/08.png" width="150"/> |

## Examples

A collection of problems solved with this library.

| <img src="docs/images/100.png" width="150"/> | <img src="docs/images/101.png" width="150"/> | <img src="docs/images/102.png" width="150"/> |
|:---:|:---:|:---:|
| <img src="docs/images/103.png" width="150"/> | <img src="docs/images/104.png" width="150"/> | <img src="docs/images/105.png" width="150"/> |

## Why not?

| <img src="docs/images/00.png" width="500"/> |
|:---:|

## 3D Solid Elements

| <img src="docs/images/200.png" width="300"/> | <img src="docs/images/201.png" width="300"/> |
|:---:|:---:|