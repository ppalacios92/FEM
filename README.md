# FEM - Finite Element Analysis

Python library for structural analysis using the Finite Element Method.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ppalacios92/FEM.git
cd FEM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Examples are located in the `examples/` folder:
```bash
cd examples
jupyter notebook
```

## Structure

- `fem/` - Main library
- `examples/` - Examples and exercises

## Import modules
```python
from fem.core import Node, Material
from fem.elements import CST, LST, Truss2D
from fem.sections import Membrane
```