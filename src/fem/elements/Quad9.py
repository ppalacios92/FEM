import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.special import roots_legendre
from fem.core.parameters import globalParameters


# ---------------------------------------------------------------------------
# Edge definition for the Quad9 element (local 0-based node indices).
# Nodes 0-3 are corner nodes; nodes 4-7 are mid-side nodes; node 8 is the
# centre node.
#
#   Natural-coordinate layout:
#
#   eta
#   +1   N4(-1,+1) --edge2(N4,N3)-- N3(+1,+1)
#        |          mid-side: N7    |
#      edge3                      edge1
#    (N4,N1)                    (N2,N3)
#     mid: N8                   mid: N6
#        |                        |
#   -1   N1(-1,-1) --edge0(N1,N2)-- N2(+1,-1)
#       -1          mid-side: N5     +1   zeta
#
#   Edge 0 — corner nodes (0, 1), mid-side node 4  : eta  = -1, zeta free
#   Edge 1 — corner nodes (1, 2), mid-side node 5  : zeta = +1, eta  free
#   Edge 2 — corner nodes (2, 3), mid-side node 6  : eta  = +1, zeta free (reversed)
#   Edge 3 — corner nodes (3, 0), mid-side node 7  : zeta = -1, eta  free (reversed)
#
# Each tuple stores (corner_i, corner_j, mid_node_local_index)
# ---------------------------------------------------------------------------

_QUAD9_EDGES = [
    (0, 1, 4),   # bottom  — eta = -1
    (1, 2, 5),   # right   — zeta = +1
    (2, 3, 6),   # top     — eta  = +1  (traversed right→left)
    (3, 0, 7),   # left    — zeta = -1  (traversed top→bottom)
]


class Quad9:
    def __init__(self,
                 element_tag: int,
                 node_list: list,
                 section: object,
                 load_direction: list = None,
                 surface_loads: list = None,
                 type: str = 'planeStress',
                 sampling_points: int = 3,
                 eval_points: list = None,
                 print_summary: bool = False):
        """
        Initialize the Quad9 serendipity isoparametric element.

        Node numbering (local, 0-based):
            0 — corner  (-1, -1)
            1 — corner  (+1, -1)
            2 — corner  (+1, +1)
            3 — corner  (-1, +1)
            4 — mid-side ( 0, -1)   edge 0
            5 — mid-side (+1,  0)   edge 1
            6 — mid-side ( 0, +1)   edge 2
            7 — mid-side (-1,  0)   edge 3
            8 — centre   ( 0,  0)

        Args:
            element_tag     (int)  : Unique element identifier.
            node_list       (list) : List of nine Node objects in the order above.
            section         (obj)  : Section with .thickness and .material.
            load_direction  (list) : [Cx, Cy] body-force direction cosines.
                                     If None, reads globalParameters['gravity'].
            surface_loads   (list) : List of surface load dicts, each with:
                                       'node_indices' : tuple (i, j) — local 0-based
                                                        corner indices of the loaded edge
                                       'value'        : [qx, qy] — force/length
                                     Populated automatically by build_elements.
            type            (str)  : 'planeStress' or 'planeStrain'.
            sampling_points (int)  : Gauss points per direction (default 3 → 3×3 = 9).
            eval_points     (list) : [zeta, eta] for stress/strain recovery.
                                     Default is element centre [0, 0].
            print_summary   (bool) : If True, prints element summary on init.
        """
        if len(node_list) != 9:
            raise ValueError("Quad9 elements must have exactly 9 nodes.")

        self.element_tag     = element_tag
        self.nodes           = node_list
        self.node_list       = node_list          # backwards-compat alias
        self.section         = section
        self.load_direction  = load_direction
        self.surface_loads   = surface_loads if surface_loads is not None else []
        self.type            = type
        self.sampling_points = sampling_points
        self.eval_points     = eval_points if eval_points is not None else [0, 0]
        self.nDof            = 2

        # Load direction — fall back to globalParameters gravity if not provided
        self._initialize_load_direction()

        # Material and section properties
        self.thickness = self.section.thickness
        self.material  = self.section.material
        self.C         = self.material.get_Emat(self.type)

        # Geometry
        self.xy = self.get_xy_matrix()

        # Element calculations — mirrors CST / LST / Quad4 structure exactly
        self.idx           = self.calculate_indices()
        self.kg, self.area = self.get_stiffness_matrix()
        self.F_fe_body     = self.get_body_forces()
        self.F_fe_surface  = self.get_surface_forces()
        self.F_fe_global   = self.F_fe_body + self.F_fe_surface

        if print_summary:
            self.printSummary()

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __str__(self):
        return f"Quad9 Element {self.element_tag}: Nodes {[node.name for node in self.nodes]}"

    def __repr__(self):
        return self.__str__()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_load_direction(self):
        """Reads gravity from globalParameters if load_direction was not given."""
        if self.load_direction is None:
            self.load_direction = globalParameters.get('gravity', [0, 0])

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def calculate_indices(self):
        """
        Returns the global DoF indices for the Quad9 element (length 18).

        Returns:
            idx (np.ndarray): Integer array of length 18 with global DOF indices.
        """
        return np.hstack([node.idx for node in self.nodes]).astype(int)

    def get_xy_matrix(self):
        """
        Returns the 9×2 matrix of nodal coordinates.

        Returns:
            xy (np.ndarray): shape (9, 2) — [[x0,y0], ..., [x8,y8]].
        """
        return np.array([node.coordinates for node in self.nodes], dtype=float)

    def get_centroid(self):
        """
        Computes the centroid by averaging the four corner node coordinates.

        Returns:
            centroid (np.ndarray): (2,) array [x_c, y_c].
        """
        return np.mean(self.xy[:4], axis=0)

    # ------------------------------------------------------------------
    # Shape functions and natural-coordinate derivatives
    # ------------------------------------------------------------------

    @staticmethod
    def get_interpolation_matrix(zeta, eta):
        """
        Evaluates the biquadratic (Lagrangian) shape functions and their
        natural-coordinate derivatives for the 9-node quadrilateral.

        Node ordering (local, 0-based):
            0 (-1,-1)  1 (+1,-1)  2 (+1,+1)  3 (-1,+1)   ← corners
            4 ( 0,-1)  5 (+1, 0)  6 ( 0,+1)  7 (-1, 0)   ← mid-sides
            8 ( 0, 0)                                       ← centre

        Shape functions (Lagrangian product form):
            Corner nodes use the product of two linear-with-zero Lagrange
            polynomials in each direction; mid-side nodes couple a
            quadratic-zero polynomial in one direction with a linear one
            in the other; the centre node is the product of the two
            quadratic-zero polynomials.

        Args:
            zeta (float): Natural coordinate ζ ∈ [-1, +1].
            eta  (float): Natural coordinate η ∈ [-1, +1].

        Returns:
            N          (np.ndarray): Interpolation matrix (2×18).
            dNnatural  (np.ndarray): Derivatives w.r.t. (ζ, η), shape (2×9).
        """
        # --- shape functions ---
        # Corners
        N1 = 0.25 * zeta * (zeta - 1.0) * eta * (eta - 1.0)   # node 0 (-1,-1)
        N2 = 0.25 * zeta * (zeta + 1.0) * eta * (eta - 1.0)   # node 1 (+1,-1)
        N3 = 0.25 * zeta * (zeta + 1.0) * eta * (eta + 1.0)   # node 2 (+1,+1)
        N4 = 0.25 * zeta * (zeta - 1.0) * eta * (eta + 1.0)   # node 3 (-1,+1)
        # Mid-sides
        N5 = 0.50 * (1.0 - zeta**2) * eta * (eta - 1.0)        # node 4 ( 0,-1)
        N6 = 0.50 * zeta * (zeta + 1.0) * (1.0 - eta**2)       # node 5 (+1, 0)
        N7 = 0.50 * (1.0 - zeta**2) * eta * (eta + 1.0)        # node 6 ( 0,+1)
        N8 = 0.50 * zeta * (zeta - 1.0) * (1.0 - eta**2)       # node 7 (-1, 0)
        # Centre
        N9 = (1.0 - zeta**2) * (1.0 - eta**2)                  # node 8 ( 0, 0)

        # Interpolation matrix N (2×18)
        N = np.array([
            [N1, 0,  N2, 0,  N3, 0,  N4, 0,  N5, 0,  N6, 0,  N7, 0,  N8, 0,  N9, 0 ],
            [0,  N1, 0,  N2, 0,  N3, 0,  N4, 0,  N5, 0,  N6, 0,  N7, 0,  N8, 0,  N9]
        ])

        # --- partial derivatives w.r.t. zeta ---
        dN1_dz = 0.25 * (2*zeta - 1.0) * eta * (eta - 1.0)
        dN2_dz = 0.25 * (2*zeta + 1.0) * eta * (eta - 1.0)
        dN3_dz = 0.25 * (2*zeta + 1.0) * eta * (eta + 1.0)
        dN4_dz = 0.25 * (2*zeta - 1.0) * eta * (eta + 1.0)
        dN5_dz = -zeta             * eta * (eta - 1.0)
        dN6_dz = 0.50 * (2*zeta + 1.0) * (1.0 - eta**2)
        dN7_dz = -zeta             * eta * (eta + 1.0)
        dN8_dz = 0.50 * (2*zeta - 1.0) * (1.0 - eta**2)
        dN9_dz = -2.0 * zeta       * (1.0 - eta**2)

        # --- partial derivatives w.r.t. eta ---
        dN1_de = 0.25 * zeta * (zeta - 1.0) * (2*eta - 1.0)
        dN2_de = 0.25 * zeta * (zeta + 1.0) * (2*eta - 1.0)
        dN3_de = 0.25 * zeta * (zeta + 1.0) * (2*eta + 1.0)
        dN4_de = 0.25 * zeta * (zeta - 1.0) * (2*eta + 1.0)
        dN5_de = 0.50 * (1.0 - zeta**2) * (2*eta - 1.0)
        dN6_de = -eta              * zeta * (zeta + 1.0)
        dN7_de = 0.50 * (1.0 - zeta**2) * (2*eta + 1.0)
        dN8_de = -eta              * zeta * (zeta - 1.0)
        dN9_de = -2.0 * eta        * (1.0 - zeta**2)

        dNnatural = np.array([
            [dN1_dz, dN2_dz, dN3_dz, dN4_dz,
             dN5_dz, dN6_dz, dN7_dz, dN8_dz, dN9_dz],
            [dN1_de, dN2_de, dN3_de, dN4_de,
             dN5_de, dN6_de, dN7_de, dN8_de, dN9_de]
        ])

        return N, dNnatural

    # ------------------------------------------------------------------
    # Strain-displacement matrix
    # ------------------------------------------------------------------

    def get_B_matrix(self, zeta, eta):
        """
        Computes the strain-displacement matrix B (3×18), the Jacobian J (2×2),
        its determinant, and the interpolation matrix N at (zeta, eta).

        Args:
            zeta (float): Natural coordinate ζ.
            eta  (float): Natural coordinate η.

        Returns:
            B     (np.ndarray): Strain-displacement matrix (3×18).
            J     (np.ndarray): Jacobian matrix (2×2).
            J_det (float)     : Determinant of J.
            N     (np.ndarray): Interpolation matrix (2×18).
        """
        xy = self.xy   # (9, 2)

        N, dNnatural = self.get_interpolation_matrix(zeta, eta)

        # Jacobian J = dNnatural (2×9) @ xy (9×2)  →  (2×2)
        J     = dNnatural @ xy
        J_det = np.linalg.det(J)

        if J_det <= 0:
            warnings.warn(
                f"Element {self.element_tag}: non-positive Jacobian determinant "
                f"({J_det:.6e}) at (ζ={zeta:.4f}, η={eta:.4f}). "
                f"Check node ordering (counter-clockwise) and mid-node positions."
            )

        # Physical derivatives:  dN/d(x,y) = J^{-1} dN/d(ζ,η)
        dNcart = np.linalg.solve(J, dNnatural)   # (2×9)

        # Assemble B (3×18)
        B = np.zeros((3, 18))
        B[0, 0::2] = dNcart[0, :]   # εxx  ← ∂Ni/∂x
        B[1, 1::2] = dNcart[1, :]   # εyy  ← ∂Ni/∂y
        B[2, 0::2] = dNcart[1, :]   # γxy  ← ∂Ni/∂y (x-DOF columns)
        B[2, 1::2] = dNcart[0, :]   # γxy  ← ∂Ni/∂x (y-DOF columns)

        return B, J, J_det, N

    # ------------------------------------------------------------------
    # Stiffness matrix
    # ------------------------------------------------------------------

    def get_stiffness_matrix(self):
        """
        Computes the element stiffness matrix Ke (18×18) and element area
        by full 3×3 (default) Gauss-Legendre quadrature on [-1,+1]²:

            Ke = ∫∫ B^T C B t dΩ  ≈  Σ_{i,j} w_i w_j B^T C B t |J|

        Returns:
            Ke   (np.ndarray): Stiffness matrix (18×18).
            area (float)     : Element area.
        """
        roots, weights = roots_legendre(self.sampling_points)
        t  = self.thickness
        Ke = np.zeros((18, 18))
        A  = 0.0

        for r, wr in zip(roots, weights):
            for s, ws in zip(roots, weights):
                B, _, J_det, _ = self.get_B_matrix(r, s)
                w = wr * ws
                Ke += w * t * J_det * (B.T @ self.C @ B)
                A  += w * np.abs(J_det)

        return Ke, A

    # ------------------------------------------------------------------
    # Body forces
    # ------------------------------------------------------------------

    def get_body_forces(self):
        """
        Computes the consistent nodal body force vector (18,) by 3×3
        Gauss-Legendre quadrature on [-1,+1]²:

            fe = ∫∫ N^T b t dΩ  ≈  Σ_{i,j} w_i w_j N^T b t |J|

        where b = γ · [Cx, Cy]^T  (unit weight × direction cosines).

        Returns:
            fe (np.ndarray): Body force vector (18,). Zero if load_direction is [0,0].
        """
        b = np.array(self.load_direction, dtype=float)
        if np.allclose(b, 0.0):
            return np.zeros(18)

        gamma = self.material.rho
        t     = self.thickness
        b_vec = (b * gamma).reshape(-1, 1)   # (2,1)

        roots, weights = roots_legendre(self.sampling_points)
        fe = np.zeros((18, 1))

        for r, wr in zip(roots, weights):
            for s, ws in zip(roots, weights):
                _, _, J_det, N = self.get_B_matrix(r, s)
                fe += wr * ws * t * J_det * (N.T @ b_vec)

        return fe.flatten()

    # ------------------------------------------------------------------
    # Surface (traction) forces
    # ------------------------------------------------------------------

    def get_surface_forces(self):
        """
        Computes the consistent nodal surface force vector (18,) by
        integrating the applied traction along each loaded edge using
        1D Gauss-Legendre quadrature with quadratic shape functions.

        Strategy (mirrors LST approach):
        ─────────────────────────────────
        Forces are deposited ONLY onto the 3 nodes of the loaded edge
        (corner i, corner j, mid-side m). The remaining 6 nodes receive
        zero contribution — no full 2D N matrix is used.

        Edge layout (_QUAD9_EDGES, local indices):
            edge 0: corners (0,1), mid-node 4  — eta = -1
            edge 1: corners (1,2), mid-node 5  — zeta = +1
            edge 2: corners (2,3), mid-node 6  — eta = +1  (reversed: 2→3)
            edge 3: corners (3,0), mid-node 7  — zeta = -1 (reversed: 3→0)

        1D quadratic Lagrange shape functions on s ∈ [-1, +1]:
            N_i(s) = 0.5 * s * (s - 1)   → 1 at s = -1  (corner i)
            N_j(s) = 0.5 * s * (s + 1)   → 1 at s = +1  (corner j)
            N_m(s) = 1 - s²              → 1 at s =  0  (mid-side m)

        Geometry map along the edge (isoparametric, quadratic):
            x(s) = N_i * x_i + N_j * x_j + N_m * x_m
        1D Jacobian:
            J_edge = || dx/ds ||

        Nodal contributions (2 DOFs per edge node):
            f_{k,x} += w * N_k(s) * qx * J_edge * t
            f_{k,y} += w * N_k(s) * qy * J_edge * t    for k ∈ {i, j, m}

        Returns:
            fe (np.ndarray): Surface force vector (18,). Zero if no surface loads.
        """
        if not self.surface_loads:
            return np.zeros(18)

        roots, weights = roots_legendre(3)   # 3-point rule: exact for quadratic integrand
        t  = self.thickness
        fe = np.zeros(18)

        for load in self.surface_loads:
            i_corner = load['node_indices'][0]
            j_corner = load['node_indices'][1]
            q        = np.array(load['value'], dtype=float)   # [qx, qy]

            # ── Identify the mid-side node for this edge ──────────────────
            m_local = None
            for (ci, cj, cm) in _QUAD9_EDGES:
                if (ci == i_corner and cj == j_corner) or \
                   (ci == j_corner and cj == i_corner):
                    m_local = cm
                    break

            if m_local is None:
                warnings.warn(
                    f"Element {self.element_tag}: cannot identify mid-side node "
                    f"for edge ({i_corner}, {j_corner}). Edge load skipped."
                )
                continue

            # ── Physical coordinates of the 3 edge nodes ──────────────────
            xi = np.array(self.nodes[i_corner].coordinates, dtype=float)
            xj = np.array(self.nodes[j_corner].coordinates, dtype=float)
            xm = np.array(self.nodes[m_local].coordinates,  dtype=float)

            # ── Local DOF positions in the 18-component element vector ─────
            # Node k → DOFs [2k, 2k+1]
            li = 2 * i_corner
            lj = 2 * j_corner
            lm = 2 * m_local

            # ── 1D Gauss integration along the edge ───────────────────────
            for s, w in zip(roots, weights):
                # 1D quadratic Lagrange shape functions
                Ni = 0.5 * s * (s - 1.0)
                Nj = 0.5 * s * (s + 1.0)
                Nm = 1.0 - s * s

                # 1D Jacobian: || dx/ds ||
                # dNi/ds = s - 0.5,  dNj/ds = s + 0.5,  dNm/ds = -2s
                dxds   = (s - 0.5) * xi + (s + 0.5) * xj + (-2.0 * s) * xm
                J_edge = np.linalg.norm(dxds)

                contrib = w * J_edge * t

                # Deposit forces ONLY on the 3 edge nodes
                fe[li    ] += contrib * Ni * q[0]
                fe[li + 1] += contrib * Ni * q[1]
                fe[lj    ] += contrib * Nj * q[0]
                fe[lj + 1] += contrib * Nj * q[1]
                fe[lm    ] += contrib * Nm * q[0]
                fe[lm + 1] += contrib * Nm * q[1]

        return fe

    # ------------------------------------------------------------------
    # Post-processing: displacements, strains, stresses
    # ------------------------------------------------------------------

    def get_element_displacements(self, u):
        """
        Extracts the element displacement vector from the global vector.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            ue (np.ndarray): Element displacement vector (18,).
        """
        return u[self.idx]

    def get_element_strains(self, u):
        """
        Computes the engineering strain vector at self.eval_points.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            epsilon_e (np.ndarray): Strain vector [εx, εy, γxy] (3,).
            ue        (np.ndarray): Element displacement vector (18,).
        """
        ue          = self.get_element_displacements(u)
        zeta, eta   = self.eval_points
        B, _, _, _  = self.get_B_matrix(zeta, eta)
        epsilon_e   = B @ ue
        return epsilon_e, ue

    def get_element_stress(self, u):
        """
        Computes the stress vector at self.eval_points.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            sigma_e   (np.ndarray): Stress vector [σx, σy, τxy] (3,).
            epsilon_e (np.ndarray): Strain vector [εx, εy, γxy] (3,).
            ue        (np.ndarray): Element displacement vector (18,).
        """
        epsilon_e, ue = self.get_element_strains(u)
        sigma_e       = self.C @ epsilon_e
        return sigma_e, epsilon_e, ue

    # ------------------------------------------------------------------
    # Principal values
    # ------------------------------------------------------------------

    def calculate_principal_stress(self, sigma):
        """
        Computes principal stresses [σ1, σ2] from [σx, σy, τxy] by
        diagonalizing the 2×2 stress tensor.

        Args:
            sigma (np.ndarray): Stress vector [σx, σy, τxy] (3,).

        Returns:
            np.ndarray: [σ1, σ2] sorted in descending order.
        """
        sx, sy, sxy = sigma[0], sigma[1], sigma[2]
        stress_tensor = np.array([[sx,  sxy],
                                  [sxy, sy ]])
        eigenvalues, _ = np.linalg.eigh(stress_tensor)
        return eigenvalues[::-1]   # descending

    def calculate_principal_strain(self, epsilon):
        """
        Computes principal strains [ε1, ε2] from [εx, εy, γxy] by
        diagonalizing the 2×2 strain tensor.

        NOTE: γxy is engineering shear strain; the tensor uses εxy = γxy/2
        as the off-diagonal component.

        Args:
            epsilon (np.ndarray): Strain vector [εx, εy, γxy] (3,).

        Returns:
            np.ndarray: [ε1, ε2] sorted in descending order.
        """
        ex, ey, gxy = epsilon[0], epsilon[1], epsilon[2]
        strain_tensor = np.array([[ex,       gxy / 2.0],
                                  [gxy / 2.0, ey      ]])
        eigenvalues, _ = np.linalg.eigh(strain_tensor)
        return eigenvalues[::-1]   # descending

    # ------------------------------------------------------------------
    # Internal forces
    # ------------------------------------------------------------------

    def get_element_internal_forces(self, u):
        """
        Computes the internal nodal force vector: f_int = Ke · ue.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            fe_int (np.ndarray): Internal force vector (18,).
        """
        ue = self.get_element_displacements(u)
        return self.kg @ ue

    # ------------------------------------------------------------------
    # Results dictionary
    # ------------------------------------------------------------------

    def get_results(self, u):
        """
        Computes and returns all element results as a dictionary.

        Keys:
            'displacement'    : element nodal displacement vector (18,)
            'strain'          : strain vector [εx, εy, γxy] at eval_points (3,)
            'stress'          : stress vector [σx, σy, τxy] at eval_points (3,)
            'principal_stress': [σ1, σ2] sorted descending (2,)
            'principal_strain': [ε1, ε2] sorted descending (2,)
            'internal_forces' : internal nodal force vector (18,)

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            results (dict): All element results.
        """
        sigma_e, epsilon_e, ue = self.get_element_stress(u)
        fe_int                 = self.get_element_internal_forces(u)
        principal_stress       = self.calculate_principal_stress(sigma_e)
        principal_strain       = self.calculate_principal_strain(epsilon_e)

        return {
            'displacement':     ue,
            'strain':           epsilon_e,
            'stress':           sigma_e,
            'principal_stress': principal_stress,
            'principal_strain': principal_strain,
            'internal_forces':  fe_int,
        }

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plotGeometry(self, ax=None, text=False, nodes=True, nodeLabels=False,
                     facecolor='lightgray', edgecolor='k', alpha=0.5):
        """
        Plots the element geometry as a shaded quadrilateral using only
        the four corner nodes.

        Args:
            ax         : matplotlib axis (created if None).
            text       : If True, shows element tag at centroid.
            nodes      : If True, plots all 9 nodes.
            nodeLabels : If True, labels each node with its name.
            facecolor  : Fill colour.
            edgecolor  : Border colour.
            alpha      : Transparency.

        Returns:
            ax: The matplotlib axis.
        """
        if ax is None:
            _, ax = plt.subplots()

        corner_coords = self.xy[[0, 1, 2, 3]]   # only corners for the patch
        polygon = patches.Polygon(corner_coords, closed=True,
                                  facecolor=facecolor, edgecolor=edgecolor,
                                  alpha=alpha)
        ax.add_patch(polygon)

        if nodes or nodeLabels:
            for node in self.nodes:
                node.plotGeometry(ax, text=nodeLabels)

        if text:
            xc, yc = self.get_centroid()
            ax.text(xc, yc, f'{self.element_tag}',
                    fontsize=12, ha='center', va='center')

        return ax

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def printSummary(self):
        """Prints a detailed summary of the Quad9 element."""
        print('-------------------------------------------------------------')
        print(f"Quad9 Element {self.element_tag}")
        print(f"Type:                 {self.type}")
        print(f"Nodes: {[node.name for node in self.nodes]}")
        labels = ['corner'] * 4 + ['mid   '] * 4 + ['centre']
        for i, (node, lbl) in enumerate(zip(self.nodes, labels)):
            print(f"  Node {node.name} ({lbl}): "
                  f"({self.xy[i, 0]:.4f}, {self.xy[i, 1]:.4f})")
        print(f"Thickness:            {self.thickness}")
        print(f"Area:                 {self.area:.6f}")
        print(f"Sampling points:      {self.sampling_points}×{self.sampling_points}")
        print(f"Eval point (ζ, η):   {self.eval_points}")
        print(f"Element DoF indices:  {self.idx}")
        print(f"Gravity direction:    {self.load_direction}")
        print(f"Unit weight (rho):    {self.material.rho}")
        print(f"Surface loads:        {len(self.surface_loads)} edge(s) loaded")
        print(f"F_fe_body:            {np.round(self.F_fe_body, 6)}")
        print(f"F_fe_surface:         {np.round(self.F_fe_surface, 6)}")
        print(f"F_fe_global:          {np.round(self.F_fe_global, 6)}")
        print(f"\nStiffness matrix (18×18):\n{np.round(self.kg, 4)}")
        print('-------------------------------------------------------------\n')