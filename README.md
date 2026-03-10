# FEM – Finite Element Analysis

A Python library for structural analysis using the Finite Element Method, developed for educational and research purposes in civil and structural engineering.

---

## ⚙️ Features

- Modular element library covering 1D, 2D, and membrane elements.
- Implements the following element types:
  - **Truss2D** – 2-node axial element (2 DOF/node)
  - **Frame2D** – 2-node Euler-Bernoulli beam-column (3 DOF/node)
  - **CST** – Constant Strain Triangle (3 nodes · 6 DOF)
  - **LST** – Linear Strain Triangle (6 nodes · 12 DOF)
  - **Quad4** – Bilinear Quadrilateral (4 nodes · 8 DOF)
  - **Quad9** – Biquadratic Lagrangian Quadrilateral (9 nodes · 18 DOF)
- Full isoparametric formulation with Gauss-Legendre numerical integration.
- Direct Stiffness Method (DSM) assembly pipeline.
- gmsh-based mesh generation and node/element import.
- Interactive Jupyter widgets for visualization of shape functions, Jacobian fields, B-matrix, and stiffness integrand components.
- Rigid body mode verification for membrane elements.

---

## 📦 Requirements

- Python 3.8 or higher
- Python libraries:
  - `numpy`
  - `scipy`
  - `sympy`
  - `matplotlib`
  - `gmsh`
  - `ipywidgets`
  - `jupyter`

---

## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ppalacios92/FEM.git
cd FEM
pip install -e .
```

---

## 📁 Repository Structure

```bash
FEM/
├── fem/
│   ├── core/             # Node, Material, Section definitions
│   ├── elements/         # CST, LST, Quad4, Quad9, Truss2D, Frame2D
│   ├── sections/         # Membrane section
│   └── functions.py      # Assembly, load vector, gmsh utilities
├── examples/             # Jupyter notebooks with usage examples
├── docs/
│   └── images/           # Reference plots and visualization outputs
└── README.md
```

---

## 🧩 Import Modules

```python
from fem.core import Node, Material
from fem.elements import CST, LST, Truss2D, Frame2D, Quad4, Quad9
from fem.sections import Membrane
from fem.functions import build_nodes_from_gmsh, create_elements_from_gmsh, build_load_vector
```

---

## 🔧 Basic Usage

```python
import numpy as np
from fem.core import Node, Material
from fem.sections import Membrane
from fem.elements import Quad4
from fem.functions import build_nodes_from_gmsh, create_elements_from_gmsh, build_load_vector

# Material and section
Steel    = Material(name='Steel', E=200000.0, nu=0.30, rho=0.0)
Plate    = Membrane(name='Plate', thickness=10.0, material=Steel)

# Dictionaries
section_dictionary  = {201: Plate}
load_dictionary     = {50: {'value': 100.0, 'direction': 'x'}}
restrain_dictionary = {101: ['r', 'r']}

# Build model from gmsh mesh
node_map, nodes = build_nodes_from_gmsh('mesh.msh', restrain_dictionary=restrain_dictionary)
elements        = create_elements_from_gmsh('mesh.msh', node_map, section_dictionary, {4: Quad4})

# Assembly and solve
# ..(see examples/ for full workflows)
```

---

## 🛑 Disclaimer

This library is developed for educational purposes in the context of the Finite Element Method course at Universidad de los Andes. Results should always be validated against reference solutions and established FEM software.

The author assumes no responsibility for incorrect use, misinterpretation of results, or consequences of numerical errors.

---

## 👨‍💻 Author

Developed by **Patricio Palacios B. - Nicolas Mora Bowen**
GitHub: [@ppalacios92](https://github.com/ppalacios92)
GitHub: [@nmorabowen](https://github.com/nmorabowen)


---

## 📚 How to Cite

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
Palacios B., P. , Mora Bowen N. (2025). *FEM: A Python library for finite element analysis* [Computer software]. GitHub. https://github.com/ppalacios92/FEM

---

## 📜 License

This project is licensed under the MIT License – see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Feel free to submit pull requests, report bugs, or suggest new features through the GitHub issues page.

---

## 🎉 Get Fun with FEM!

Interactive visualizations included in this library — explore shape functions, Jacobian fields, and stiffness integrands live in Jupyter.

| <img src="docs/images/01.png" width="150"/> | <img src="docs/images/02.png" width="150"/> | <img src="docs/images/03.png" width="150"/> | <img src="docs/images/04.png" width="150"/> |
|:---:|:---:|:---:|:---:|
| <img src="docs/images/05.png" width="150"/> | <img src="docs/images/06.png" width="150"/> | <img src="docs/images/07.png" width="150"/> | <img src="docs/images/08.png" width="150"/> |
