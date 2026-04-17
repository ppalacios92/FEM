"""
validation_json.py
==================
Generates JSON reference files for FEM element validation.

Usage
-----
from fem.validation.validation_json import FEMValidator
from fem.core.Node     import Node
from fem.core.Material import Material
from fem.sections.Membrane import Membrane
from fem.elements.CST  import CST

# Build your nodes and section
nodes   = [Node(0, [0,0]), Node(1, [1,0]), Node(2, [0,1])]
section = Membrane('Plate', thickness=10, material=Material('Steel', E=200000, nu=0.3, rho=0))

# Generate reference JSON
FEMValidator.generate(
    nodes        = nodes,
    element_type = CST,
    section      = section,
    label        = 'CST_regular',
    output_dir   = 'examples/validation',
)
"""

import os
import json
import numpy as np
from scipy.special import roots_legendre


class FEMValidator:
    """
    Generates JSON reference files for FEM element validation.

    All logic is encapsulated in class methods — no instantiation needed.
    Supports CST, LST, Quad4, and Quad9 elements.
    """

    # ── public entry point ─────────────────────────────────────────────────

    @classmethod
    def generate(cls,
                 nodes:        list,
                 element_type: type,
                 section:      object,
                 label:        str,
                 output_dir:   str,
                 element_tag:  int = 1):
        """
        Build and write a JSON reference file for a given element.

        Parameters
        ----------
        nodes        : list of Node   Pre-built Node objects with coordinates set
        element_type : type           Element class — CST, LST, Quad4, or Quad9
        section      : Membrane       Section object with material and thickness
        label        : str            Descriptive label written into the JSON
                                      e.g. 'CST_regular', 'Quad4_distorted'
        output_dir   : str            Directory where the JSON will be written.
                                      e.g. 'examples/validation'
        element_tag  : int            Element tag passed to the constructor (default 1)
        """
        name = element_type.__name__

        # Instantiate element
        element = element_type(
            element_tag = element_tag,
            node_list   = nodes,
            section     = section,
        )

        # Dispatch to the correct builder
        builders = {
            'CST'  : cls._build_cst,
            'LST'  : cls._build_isoparametric_tri,
            'Quad4': cls._build_isoparametric_quad,
            'Quad9': cls._build_isoparametric_quad,
        }

        if name not in builders:
            raise ValueError(
                f"Unsupported element type '{name}'. "
                f"Supported: {list(builders.keys())}"
            )

        ref = builders[name](element, label, section)

        # Write JSON
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{label}.json')
        with open(output_path, 'w') as f:
            json.dump(ref, f, indent=2)

        print(f"[FEMValidator] {name} '{label}' → {output_path}")

    # ── element builders ───────────────────────────────────────────────────

    @classmethod
    def _build_cst(cls, element, label, section):
        """
        CST reference — B is constant (evaluated once).
        N evaluated at centroid.
        """
        B             = element.get_B_matrix()
        centroid      = element.get_centroid()
        N_at_centroid = element.get_interpolation_matrix(*centroid)

        return {
            'element'   : 'CST',
            'label'     : label,
            'nodes'     : cls._nodes_to_list(element.nodes),
            'material'  : cls._material_dict(section),
            'thickness' : float(section.thickness),
            'area'      : float(element.area),
            # B is constant for CST — one evaluation covers the whole element
            'B'         : cls._arr(B),
            # N at centroid (barycentric coords = 1/3 each)
            'N_centroid': cls._arr(N_at_centroid),
            'K'         : cls._arr(element.kg),
        }

    @classmethod
    def _build_isoparametric_tri(cls, element, label, section):
        """
        LST reference — N, J, detJ, B at each triangular Gauss point.
        Uses the same integration scheme as the element itself.
        """
        n          = element.sampling_points
        roots, wts = roots_legendre(n)
        xi_pts     = 0.5 * (roots + 1.0)
        w_pts      = 0.5 * wts

        gauss_data = []
        for xi, wx in zip(xi_pts, w_pts):
            for eta_hat, we in zip(xi_pts, w_pts):
                eta    = eta_hat * (1.0 - xi)
                weight = wx * we * (1.0 - xi)
                B, J, J_det, N = element.get_B_matrix(xi, eta)
                gauss_data.append({
                    'xi'    : float(xi),
                    'eta'   : float(eta),
                    'weight': float(weight),
                    'N'     : cls._arr(N),
                    'J'     : cls._arr(J),
                    'detJ'  : float(J_det),
                    'B'     : cls._arr(B),
                })

        return {
            'element'      : element.__class__.__name__,
            'label'        : label,
            'nodes'        : cls._nodes_to_list(element.nodes),
            'material'     : cls._material_dict(section),
            'thickness'    : float(section.thickness),
            'area'         : float(element.area),
            'gauss_points' : gauss_data,
            'K'            : cls._arr(element.kg),
        }

    @classmethod
    def _build_isoparametric_quad(cls, element, label, section):
        """
        Quad4 / Quad9 reference — N, J, detJ, B at each quadrilateral Gauss point.
        Uses the same integration scheme as the element itself.
        """
        n               = element.sampling_points
        roots, weights  = roots_legendre(n)

        gauss_data = []
        for r, wr in zip(roots, weights):
            for s, ws in zip(roots, weights):
                B, J, J_det, N = element.get_B_matrix(r, s)
                gauss_data.append({
                    'zeta'  : float(r),
                    'eta'   : float(s),
                    'weight': float(wr * ws),
                    'N'     : cls._arr(N),
                    'J'     : cls._arr(J),
                    'detJ'  : float(J_det),
                    'B'     : cls._arr(B),
                })

        return {
            'element'      : element.__class__.__name__,
            'label'        : label,
            'nodes'        : cls._nodes_to_list(element.nodes),
            'material'     : cls._material_dict(section),
            'thickness'    : float(section.thickness),
            'area'         : float(element.area),
            'gauss_points' : gauss_data,
            'K'            : cls._arr(element.kg),
        }

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _arr(obj):
        """Convert numpy array to nested list for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    @staticmethod
    def _nodes_to_list(nodes):
        """Extract node coordinates as list of [x, y] pairs."""
        return [node.coordinates[:2].tolist() for node in nodes]

    @staticmethod
    def _material_dict(section):
        """Extract material properties into a plain dict."""
        return {
            'E'  : float(section.material.E),
            'nu' : float(section.material.nu),
            'rho': float(section.material.rho),
        }