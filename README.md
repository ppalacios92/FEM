# FEM – Finite Element Analysis

A Python library for structural analysis using the Finite Element Method, developed for educational and research purposes in civil and structural engineering.

> Based on the FEM course by Prof. José Abell — Universidad de los Andes.

---

## Elements

| Element | Type | Nodes | DOF/node |
|---|---|---|---|
| `Truss2D` | Axial | 2 | 2 |
| `Frame2D` | Euler-Bernoulli beam-column | 2 | 3 |
| `CST` | Constant Strain Triangle | 3 | 2 |
| `LST` | Linear Strain Triangle | 6 | 2 |
| `Quad4` | Bilinear Quadrilateral | 4 | 2 |
| `Quad9` | Biquadratic Lagrangian Quadrilateral | 9 | 2 |

---

## Requirements

```
numpy · scipy · sympy · matplotlib · gmsh · ipywidgets · jupyter
```

---

## Installation

```bash
git clone https://github.com/ppalacios92/FEM.git
cd FEM
pip install -e .
```

---

## Repository Structure

```
FEM/
├── src/
│   └── fem/
│       ├── core/         # Node, Material, parameters
│       ├── elements/     # CST, LST, Quad4, Quad9, Truss2D, Frame2D
│       ├── sections/     # Membrane section
│       └── utils/        # functions, gmshtools, plotting, visualization, units
├── examples_1D/          # Frame2D examples
├── examples_2D/          # Membrane FEM + OpenSees
├── examples_3D/          # 3D solid via GMSHtools + OpenSees
└── README.md
```

---

## Workflows

### 1D — Frame / Truss

Direct assembly without gmsh. Nodes and elements are defined manually.

```python
from fem import Node, Material, Frame2D, globalParameters
from fem.utils.functions import matrix_extract, matrix_replace

globalParameters['nDoF'] = 3
globalParameters['nDIM'] = 2

# Nodes
n0 = Node(0, [0, 0],   restrain=['r', 'r', 'r'])
n1 = Node(1, [5, 0],   restrain=['f', 'f', 'f'])
n2 = Node(2, [10, 0],  restrain=['r', 'f', 'r'])

all_nodes = [n0, n1, n2]

# Material and elements
steel = Material(name='Steel', E=29000, nu=0.3, rho=0)
e0 = Frame2D(n0, n1, material=steel, A=3, I=400)
e1 = Frame2D(n1, n2, material=steel, A=3, I=400)

elements = [e0, e1]

# Assembly
n_dof = max(node.idx.max() for node in all_nodes) + 1
K = np.zeros((n_dof, n_dof))
F = np.zeros(n_dof)

for elem in elements:
    K = matrix_replace(K, elem.kg, elem.idx, elem.idx)
    F[elem.idx] += elem.fe          # self-weight / distributed loads

for node in all_nodes:
    F[node.idx] += node.nodalLoad  # concentrated nodal loads

# DOF classification
dof_flags       = np.concatenate([node.restrain for node in all_nodes])
free_dofs       = np.where(dof_flags == 'f')[0]
restrained_dofs = np.where(dof_flags == 'r')[0]

Kff = matrix_extract(K, free_dofs, free_dofs)
Kfr = matrix_extract(K, free_dofs, restrained_dofs)
Krf = matrix_extract(K, restrained_dofs, free_dofs)
Krr = matrix_extract(K, restrained_dofs, restrained_dofs)

# Solution
ur = np.zeros(len(restrained_dofs))
uf = np.linalg.solve(Kff, F[free_dofs] - Kfr @ ur)

u = np.zeros(n_dof)
u[free_dofs]       = uf
u[restrained_dofs] = ur

# Reactions
Rr = Krf @ uf + Krr @ ur
```

**Post-processing:**
```python
# Element force recovery
for elem in elements:
    res = elem.get_results(u)
    print(res['fe_local'])   # [N_i, V_i, M_i, N_j, V_j, M_j]

# Plots
fig, ax = plt.subplots()
for elem in elements:
    elem.plot_deformed(u, scale=100, ax=ax)
    elem.plot_moment(u, ax=ax, scale=0.1)
    elem.plot_shear(u, ax=ax)
    elem.plot_axial(u, ax=ax)
```

---

### 2D — Membrane (own solver)

Uses gmsh for mesh generation and the built-in FEM solver.

```python
from fem import (
    Material, Membrane, GMSHtools, build_elements,
    CST, LST, Quad4, Quad9,
    add_element_data_view, add_node_data_view, compute_nodal_average,
    plot_mesh, plot_loads_2d, plot_deformed, plot_field_2d,
    globalParameters,
)

globalParameters['nDoF'] = 2
globalParameters['nDIM'] = 2

# --- Step 1: Generate mesh in gmsh ---
# (define points, lines, surfaces, physical groups)
# gmsh.model.mesh.generate()
# gmsh.write(output_file)

# --- Step 2: Read mesh ---
mesh = GMSHtools(output_file)

# --- Step 3: Apply boundary conditions ---
restrain_dictionary = {101: ['r', 'r']}
load_dictionary     = {50:  {'value': 10.0, 'direction': 'x'}}  # [N/mm²] for dim=1
section_dictionary  = {201: SteelPlate}

mesh.apply_boundary_conditions(restrain_dictionary, load_dictionary, section_dictionary)

system_nDof = mesh.system_nDof
node_map    = mesh.node_map

# --- Step 4: Build elements ---
element_map = {3: CST, 4: Quad4, 6: LST, 9: Quad9}

elements = build_elements(
    mesh               = mesh,
    node_map           = node_map,
    section_dictionary = section_dictionary,
    element_class_map  = element_map,
    load_dictionary    = load_dictionary,
    type               = 'planeStress',
    sampling_points    = 3,
    eval_points        = [0, 0],
)

# --- Step 5: Assemble force vector ---
# Consistent loads (recommended for higher-order elements)
F_load = np.zeros(mesh.system_nDof)
for node in node_map.values():
    F_load[node.idx] += node.nodalLoad     # dim=0 point loads
for elem in elements:
    F_load[elem.idx] += elem.F_fe_global   # consistent surface/body loads
F_load[np.abs(F_load) < 1e-4] = 0.0

# --- Step 6: Assemble stiffness and solve ---
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

K = lil_matrix((system_nDof, system_nDof))
for element in elements:
    for i_local, i_global in enumerate(element.idx):
        for j_local, j_global in enumerate(element.idx):
            K[i_global, j_global] += element.kg[i_local, j_local]
K = K.tocsr()

F = F_load.copy()

dof_flags       = np.concatenate([node.restrain for node in node_map.values()])
free_dofs       = np.where(dof_flags == 'f')[0]
restrained_dofs = np.where(dof_flags == 'r')[0]

Kff = K[free_dofs[:, None], free_dofs]
Kfr = K[free_dofs[:, None], restrained_dofs]
Krf = K[restrained_dofs[:, None], free_dofs]
Krr = K[restrained_dofs[:, None], restrained_dofs]

ur = np.zeros(len(restrained_dofs))
uf = spsolve(Kff, F[free_dofs] - Kfr @ ur)

u = np.zeros(system_nDof)
u[free_dofs]       = uf
u[restrained_dofs] = ur

Rr = Krf @ uf + Krr @ ur
R  = np.zeros(system_nDof)
R[restrained_dofs] = Rr
```

**Post-processing — matplotlib:**
```python
plot_mesh(nodes=node_map.values(), elements=elements)
plot_loads_2d(nodes=node_map.values(), elements=elements, F_load=F_load)
plot_deformed(nodes=node_map.values(), elements=elements, u=u, component='umag', sfac=50)
plot_field_2d(nodes=node_map.values(), elements=elements, u=u, component='sxx',
              result_type='nodal_avg', deformed=True, sfac=50)
```

**Post-processing — gmsh:**
```python
# Build result arrays
n_nodes = len(node_map)
u_3d = np.zeros((n_nodes, 3));  F_3d = np.zeros((n_nodes, 3));  R_3d = np.zeros((n_nodes, 3))
for i, node in enumerate(node_map.values()):
    u_3d[i, :2] = u[node.idx];  F_3d[i, :2] = F[node.idx];  R_3d[i, :2] = R[node.idx]

node_tags         = np.array([node.name for node in node_map.values()])
element_tags_list = [elem.element_tag for elem in elements]

sigma_gmsh     = np.array([elem.get_results(u)['stress'].flatten()   for elem in elements])
von_mises_gmsh = np.sqrt(sigma_gmsh[:,0]**2 - sigma_gmsh[:,0]*sigma_gmsh[:,1] + sigma_gmsh[:,1]**2)

# Open gmsh and add views
gmsh.initialize();  gmsh.open(output_file)

add_node_data_view("Displacements", node_tags, u_3d, vector_type=5, factor=50)
add_node_data_view("Applied Loads", node_tags, F_3d, arrow_size_max=60, arrow_size_min=20)
add_node_data_view("Reactions",     node_tags, R_3d)

add_element_data_view("Von Mises",  element_tags_list, von_mises_gmsh)
add_element_data_view("Stress Sxx", element_tags_list, sigma_gmsh[:, 0])

vm_nodal = compute_nodal_average(mesh, element_tags_list, von_mises_gmsh)
add_node_data_view("Von Mises Averaged", node_tags, vm_nodal)

gmsh.fltk.run();  gmsh.finalize()
```

---

### 3D — Solid (OpenSees solver)

Uses gmsh for mesh generation and OpenSees for analysis. The built-in solver is not used for 3D.

```python
from fem import GMSHtools, add_element_data_view, add_node_data_view, compute_nodal_average
import opensees as ops

globalParameters['nDoF'] = 3
globalParameters['nDIM'] = 3

# --- Step 1: Read mesh ---
mesh = GMSHtools(output_file)

# --- Step 2: Build lumped load vector ---
load_dictionary = {257: {'value': 4.0, 'direction': '-y'}}

mesh.section_dictionary = None   # no section needed for 3D
F_nodal = mesh.build_load_vector(load_dictionary)

F_load = np.zeros(mesh.system_nDof)
for tag, f_vec in F_nodal.items():
    if tag in mesh.node_map:
        F_load[mesh.node_map[tag].idx[:len(f_vec)]] += f_vec

# --- Step 3: OpenSees model ---
ops.wipe()
ops.model('basicBuilder', '-ndm', 3, '-ndf', 3)

for tag, (x, y, z) in mesh.nodes.items():
    ops.node(tag, x, y, z)

for tag in mesh.physical_groups['Support'].nodes:
    ops.fix(tag, 1, 1, 1)

ops.nDMaterial('ElasticIsotropic', 1, E, nu)

group = mesh.physical_groups['solido'].elements
for elem_tag, conn in zip(group['element_tags'], group['connectivity']):
    ops.element('FourNodeTetrahedron', elem_tag, *conn, 1)

# --- Step 4: Loads and analysis ---
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
for tag, force in F_nodal.items():
    if np.any(np.abs(force) > 0):
        ops.load(tag, *force.tolist())

ops.system("FullGeneral");  ops.numberer("Plain");  ops.constraints("Plain")
ops.integrator("LoadControl", 0.1);  ops.algorithm("Newton");  ops.analysis("Static")
ops.analyze(10)

# --- Step 5: Extract results and visualize in gmsh ---
n_nodes   = len(mesh.nodes)
u_3d      = np.zeros((n_nodes, 3))
node_tags = np.array(list(mesh.nodes.keys()))

ops.reactions()
for i, tag in enumerate(mesh.nodes):
    u_3d[i] = [ops.nodeDisp(tag, j+1) for j in range(3)]

gmsh.initialize();  gmsh.open(output_file)
add_node_data_view("Displacements", node_tags, u_3d, vector_type=5, factor=2)
gmsh.fltk.run();  gmsh.finalize()
```

---

## GMSHtools Reference

```python
mesh = GMSHtools('mesh.msh')

mesh.nodes                        # {tag: (x, y, z)}
mesh.elements                     # {phys_id: {gmsh_type, connectivity, ...}}
mesh.physical_groups              # accessible by int ID or name string
mesh.physical_groups[201]         # by ID
mesh.physical_groups['Beam']      # by name

mesh.physical_groups[201].dim     # 0=point, 1=line, 2=surface, 3=volume
mesh.physical_groups[201].nodes   # {tag: (x,y,z)}
mesh.physical_groups[201].elements

mesh.apply_boundary_conditions(restrain_dictionary, load_dictionary, section_dictionary)
mesh.build_load_vector(load_dictionary)   # returns {tag: np.array([fx, fy, ...])}
```

**`load_dictionary` format:**

| dim | value units | description |
|---|---|---|
| `dim=0` | `[N]` | Point load — applied directly to each node |
| `dim=1` | `[N/mm²]` (× thickness) | Line load — lumped along segments |
| `dim=2` | `[N/mm²]` | Surface pressure — lumped per element area |
| `dim=3` | `[N/mm³]` | Body force — lumped per element volume |

---

## Units

Base system: **mm · N · MPa**

```python
from fem import mm, cm, m, N, kN, tf, MPa, GPa, kg, g
```

| Symbol | Value | Meaning |
|---|---|---|
| `mm` | 1 | millimeter |
| `cm` | 10 | centimeter |
| `m` | 1000 | meter |
| `N` | 1 | Newton |
| `kN` | 1000 | kilonewton |
| `tf` | 9807 | tonelada-fuerza |
| `MPa` | 1 | megapascal |
| `GPa` | 1000 | gigapascal |
| `kg` | 1e-3 | kilogram (mass) |
| `g` | 9810 mm/s² | gravitational acceleration |

---

## Plotting (matplotlib)

| Function | Description |
|---|---|
| `plot_mesh` | Mesh geometry with node/element labels and supports |
| `plot_loads_2d` | Load arrows over mesh |
| `plot_deformed` | Deformed shape colored by `ux`, `uy`, or `umag` |
| `plot_field_2d` | Stress/strain contour — `sxx`, `syy`, `vmis`, `s1`, `s2`, ... |
| `plot_gmsh_mesh` | Raw gmsh mesh with physical groups colored |

---

## Author

**Patricio Palacios B. · Nicolas Mora Bowen**
GitHub: [@ppalacios92](https://github.com/ppalacios92) · [@nmorabowen](https://github.com/nmorabowen)

---

## License

MIT License — see LICENSE file for details.