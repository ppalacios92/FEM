import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fem.core.parameters import globalParameters

# ---------------------------------------------------------------------------
# Edge definition for the LST element (local 0-based node indices).
# Nodes 0-2 are corner nodes; nodes 3-5 are mid-side nodes.
#
#   Node 2
#    |  \
#  edge2  edge1
#  (2,0)  (1,2)
#    |        \
#   Node 0 --- Node 1
#       edge0 (0,1)
#
#   Mid-side nodes:
#     Node 3  — mid of edge 0 (between nodes 0 and 1)
#     Node 4  — mid of edge 1 (between nodes 1 and 2)
#     Node 5  — mid of edge 2 (between nodes 2 and 0)
# ---------------------------------------------------------------------------

_LST_EDGES = [
    (0, 1, 3),   # edge 0: corners (0,1), mid-node 3
    (1, 2, 4),   # edge 1: corners (1,2), mid-node 4
    (2, 0, 5),   # edge 2: corners (2,0), mid-node 5
]


class LST:
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
        Initialize the LST (Linear Strain Triangle, 6-node) element.

        Nodes 0-2 are the three corner nodes (counter-clockwise).
        Nodes 3-5 are the mid-side nodes:
            Node 3 — mid of edge (0,1)
            Node 4 — mid of edge (1,2)
            Node 5 — mid of edge (2,0)

        Args:
            element_tag     (int)  : Unique element identifier.
            node_list       (list) : List of six Node objects [corner×3, midside×3].
            section         (obj)  : Section with .thickness and .material.
            load_direction  (list) : [Cx, Cy] body-force direction cosines.
                                     If None, reads globalParameters['gravity'].
            surface_loads   (list) : List of surface load dicts, each with keys:
                                       'node_indices' : tuple (i, j) — local 0-based
                                                        edge node indices
                                       'value'        : [qx, qy] — force/length
                                     Populated automatically by build_elements.
            type            (str)  : 'planeStress' or 'planeStrain'.
            sampling_points (int)  : Gauss points per direction for integration.
                                     Default is 3 (9 total evaluations via Duffy).
            eval_points     (list) : [zeta, eta] natural coords for stress/strain
                                     recovery. Default is element centroid [1/3, 1/3].
            print_summary   (bool) : If True, prints element summary on init.
        """
        if len(node_list) != 6:
            raise ValueError("LST elements must have exactly 6 nodes.")

        self.element_tag    = element_tag
        self.nodes          = node_list
        self.section        = section
        self.load_direction = load_direction
        self.surface_loads  = surface_loads if surface_loads is not None else []
        self.type           = type
        self.sampling_points = sampling_points
        # Default eval point: centroid of reference triangle
        self.eval_points    = eval_points if eval_points is not None else [1/3, 1/3]

        # Load direction — fall back to globalParameters gravity if not provided
        self._initialize_load_direction()

        # Material and section properties
        self.thickness = self.section.thickness
        self.material  = self.section.material
        self.C         = self.material.get_Emat(self.type)

        # Geometry
        self.xy  = self._get_xy_matrix()
        self.compute_area()

        # Element calculations (mirrors CST structure exactly)
        self.idx           = self.calculate_indices()
        self.kg            = self.get_stiffness_matrix()
        self.F_fe_body     = self.get_body_forces()
        self.F_fe_surface  = self.get_surface_forces()
        self.F_fe_global   = self.F_fe_body + self.F_fe_surface

        if print_summary:
            self.printSummary()

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __str__(self):
        return f"LST Element {self.element_tag}: Nodes {[node.name for node in self.nodes]}"

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
        Returns the global DoF indices for the LST element (length 12).

        Returns:
            idx (np.ndarray): Integer array of length 12 with global DOF indices.
        """
        idx = np.hstack([node.idx for node in self.nodes]).astype(int)
        return idx

    def _get_xy_matrix(self):
        """
        Returns the 6×2 matrix of nodal coordinates.

        Returns:
            xy (np.ndarray): shape (6, 2) — [[x1,y1], ..., [x6,y6]].
        """
        return np.array([node.coordinates for node in self.nodes], dtype=float)

    # Public alias used by some callers
    def get_xy_matrix(self):
        return self._get_xy_matrix()

    def compute_area(self):
        """
        Computes and stores the area of the triangle using only the three
        corner nodes (nodes 0-2) via the determinant formula:

            A = 0.5 * det([[1, x1, y1], [1, x2, y2], [1, x3, y3]])

        Node ordering must be counter-clockwise for A > 0.

        Sets:
            self.area (float): Signed area of the triangle.
        """
        x1, y1 = self.nodes[0].coordinates
        x2, y2 = self.nodes[1].coordinates
        x3, y3 = self.nodes[2].coordinates

        self.area = 0.5 * np.linalg.det(np.array([
            [1, x1, y1],
            [1, x2, y2],
            [1, x3, y3]
        ]))

        if self.area <= 0:
            warnings.warn(
                f"Element {self.element_tag} has non-positive area: {self.area:.6f}. "
                f"Check node ordering (should be counter-clockwise)."
            )

    def get_centroid(self):
        """
        Computes the centroid by averaging the three corner node coordinates.

        Returns:
            centroid (np.ndarray): (2,) array [x_c, y_c].
        """
        xy = self._get_xy_matrix()
        return np.mean(xy[:3], axis=0)   # only corner nodes

    # ------------------------------------------------------------------
    # Shape functions and natural-coordinate derivatives
    # ------------------------------------------------------------------

    def get_interpolation_matrix(self, zeta: float, eta: float):
        """
        Evaluates the shape function matrix N (2×12) and the matrix of
        natural-coordinate derivatives dN/d(zeta,eta) (2×6) at the
        parametric point (zeta, eta) of the reference triangle.

        Parametric coordinates:
            xi0 = 1 - zeta - eta   (weight at corner node 0)
            xi1 = zeta             (weight at corner node 1)
            xi2 = eta              (weight at corner node 2)

        Quadratic shape functions (Serendipity on triangle):
            N1 = xi0 (2 xi0 - 1)
            N2 = xi1 (2 xi1 - 1)
            N3 = xi2 (2 xi2 - 1)
            N4 = 4 xi0 xi1          (mid-side 0-1)
            N5 = 4 xi1 xi2          (mid-side 1-2)
            N6 = 4 xi2 xi0          (mid-side 2-0)

        Args:
            zeta (float): Natural coord ζ  ∈ [0, 1].
            eta  (float): Natural coord η  ∈ [0, 1-ζ].

        Returns:
            N        (np.ndarray): Interpolation matrix (2×12).
            dNnatural(np.ndarray): Derivatives w.r.t. (ζ, η), shape (2×6).
        """
        xi0 = 1.0 - zeta - eta
        xi1 = zeta
        xi2 = eta

        # Shape functions
        N1 = xi0 * (2*xi0 - 1)
        N2 = xi1 * (2*xi1 - 1)
        N3 = xi2 * (2*xi2 - 1)
        N4 = 4.0 * xi0 * xi1
        N5 = 4.0 * xi1 * xi2
        N6 = 4.0 * xi2 * xi0

        # Interpolation matrix N (2×12)
        N = np.array([
            [N1, 0, N2, 0, N3, 0, N4, 0, N5, 0, N6, 0],
            [0, N1,  0, N2,  0, N3,  0, N4,  0, N5,  0, N6]
        ])

        # Partial derivatives w.r.t. ζ and η
        # Chain rule:  ∂ξ0/∂ζ = -1,  ∂ξ0/∂η = -1
        #              ∂ξ1/∂ζ = +1,  ∂ξ1/∂η =  0
        #              ∂ξ2/∂ζ =  0,  ∂ξ2/∂η = +1

        # dN1/dζ = (4ξ0 - 1)(∂ξ0/∂ζ) = -(4ξ0 - 1)
        dN1_dz = -(4*xi0 - 1)
        dN1_de = -(4*xi0 - 1)

        # dN2/dζ = (4ξ1 - 1)(∂ξ1/∂ζ) = +(4ξ1 - 1)
        dN2_dz =  (4*xi1 - 1)
        dN2_de =  0.0

        # dN3/dη = (4ξ2 - 1)(∂ξ2/∂η) = +(4ξ2 - 1)
        dN3_dz =  0.0
        dN3_de =  (4*xi2 - 1)

        # N4 = 4ξ0ξ1  →  dN4/dζ = 4(ξ1·(-1) + ξ0·(+1)) = 4(ξ0 - ξ1)
        #               dN4/dη = 4(ξ1·(-1) + ξ0·0)     = -4ξ1
        dN4_dz =  4.0 * (xi0 - xi1)
        dN4_de = -4.0 * xi1

        # N5 = 4ξ1ξ2  →  dN5/dζ = 4(ξ2·(+1) + ξ1·0)  = 4ξ2
        #               dN5/dη = 4(ξ2·0     + ξ1·(+1)) = 4ξ1
        dN5_dz =  4.0 * xi2
        dN5_de =  4.0 * xi1

        # N6 = 4ξ2ξ0  →  dN6/dζ = 4(ξ0·0 + ξ2·(-1))  = -4ξ2
        #               dN6/dη = 4(ξ0·(+1) + ξ2·(-1)) = 4(ξ0 - ξ2)
        dN6_dz = -4.0 * xi2
        dN6_de =  4.0 * (xi0 - xi2)

        dNnatural = np.array([
            [dN1_dz, dN2_dz, dN3_dz, dN4_dz, dN5_dz, dN6_dz],
            [dN1_de, dN2_de, dN3_de, dN4_de, dN5_de, dN6_de]
        ])

        return N, dNnatural

    # ------------------------------------------------------------------
    # Strain-displacement matrix
    # ------------------------------------------------------------------

    def get_B_matrix(self, zeta: float, eta: float):
        """
        Computes the strain-displacement matrix B (3×12), the Jacobian J (2×2),
        its determinant, and the interpolation matrix N at (zeta, eta).

        Unlike the CST, B varies throughout the element because the shape
        functions are quadratic — the strain field is linear (hence "Linear
        Strain Triangle").

        Args:
            zeta (float): Natural coordinate ζ.
            eta  (float): Natural coordinate η.

        Returns:
            B     (np.ndarray): Strain-displacement matrix (3×12).
            J     (np.ndarray): Jacobian matrix (2×2).
            J_det (float)     : Determinant of J.
            N     (np.ndarray): Interpolation matrix (2×12).
        """
        xy = self.xy
        N, dNnatural = self.get_interpolation_matrix(zeta, eta)

        # Jacobian J = dNnatural (2×6) @ xy (6×2)  →  (2×2)
        J     = dNnatural @ xy
        J_det = np.linalg.det(J)

        if J_det <= 0:
            warnings.warn(
                f"Element {self.element_tag}: non-positive Jacobian determinant "
                f"({J_det:.6e}) at (ζ={zeta:.4f}, η={eta:.4f}). "
                f"Check node ordering (counter-clockwise) and mid-node positions."
            )

        # Physical derivatives:  dN/d(x,y) = J^{-1} dN/d(ζ,η)
        dNcart = np.linalg.solve(J, dNnatural)   # (2×6)

        # Assemble B (3×12)
        B = np.zeros((3, 12))
        B[0, 0::2] = dNcart[0, :]   # ∂Ni/∂x  → row εxx, x-DOF columns
        B[1, 1::2] = dNcart[1, :]   # ∂Ni/∂y  → row εyy, y-DOF columns
        B[2, 0::2] = dNcart[1, :]   # ∂Ni/∂y  → row γxy, x-DOF columns
        B[2, 1::2] = dNcart[0, :]   # ∂Ni/∂x  → row γxy, y-DOF columns

        return B, J, J_det, N

    # ------------------------------------------------------------------
    # Stiffness matrix
    # ------------------------------------------------------------------

    def get_stiffness_matrix(self):
        """
        Computes the element stiffness matrix Ke (12×12) by numerical
        integration using a Duffy-transformed Gauss-Legendre rule on the
        reference triangle.

        The Duffy transformation maps the unit square [0,1]² to the
        reference triangle {ζ ≥ 0, η ≥ 0, ζ+η ≤ 1}:

            ζ = ζ̂,   η = η̂ (1 - ζ̂)
            dA = |J_elem| · (1 - ζ̂) · dζ̂ dη̂

        where the factor (1-ζ̂) absorbs the triangular domain collapse.

        Returns:
            Ke (np.ndarray): Stiffness matrix (12×12).
        """
        from scipy.special import roots_legendre
        roots, weights = roots_legendre(self.sampling_points)

        # Map 1D Gauss points from [-1,1] to [0,1]
        xi_pts = 0.5 * (roots + 1.0)
        xi_wts = 0.5 * weights

        t  = self.thickness
        Ke = np.zeros((12, 12))

        for z_hat, w_z in zip(xi_pts, xi_wts):
            for e_hat, w_e in zip(xi_pts, xi_wts):
                # Duffy map to triangular domain
                zeta   = z_hat
                eta    = e_hat * (1.0 - z_hat)
                weight = w_z * w_e * (1.0 - z_hat)

                B, _, J_det, _ = self.get_B_matrix(zeta, eta)
                Ke += weight * t * J_det * (B.T @ self.C @ B)

        return Ke

    # ------------------------------------------------------------------
    # Body forces
    # ------------------------------------------------------------------

    def get_body_forces(self):
        """
        Computes the consistent nodal body force vector (12,) by numerical
        integration using the same Duffy rule as the stiffness matrix:

            fe = ∫ N^T b t dΩ  ≈  Σ_gp  w_gp · N^T(ζ,η) · b · t · |J| · (1-ζ̂)

        where b = γ · [Cx, Cy]^T (unit weight × direction cosines).

        Returns:
            fe (np.ndarray): Body force vector (12,). Zero if load_direction is [0,0].
        """
        from scipy.special import roots_legendre

        b     = np.array(self.load_direction, dtype=float).reshape(-1, 1)
        gamma = self.material.rho
        t     = self.thickness

        roots, weights = roots_legendre(self.sampling_points)
        xi_pts = 0.5 * (roots + 1.0)
        xi_wts = 0.5 * weights

        fe = np.zeros((12, 1))

        for z_hat, w_z in zip(xi_pts, xi_wts):
            for e_hat, w_e in zip(xi_pts, xi_wts):
                zeta   = z_hat
                eta    = e_hat * (1.0 - z_hat)
                weight = w_z * w_e * (1.0 - z_hat)

                _, _, J_det, N = self.get_B_matrix(zeta, eta)
                fe += weight * (N.T @ b) * J_det

        fe = fe * (t * gamma)
        return fe.flatten()

    # ------------------------------------------------------------------
    # Surface (traction) forces
    # ------------------------------------------------------------------

    def get_surface_forces(self):
        """
        Computes the consistent nodal surface force vector (12,) by
        integrating the applied traction along each loaded edge using
        1D Gauss-Legendre quadrature.

        CRITICAL: forces are deposited ONLY onto the 3 nodes of the loaded
        edge (corner i, corner j, mid-side m). The other 3 nodes of the
        element receive zero contribution — no global N_mat is used.

        Edge layout (local indices):
            edge 0: corners (0,1) -> mid-node 3
            edge 1: corners (1,2) -> mid-node 4
            edge 2: corners (2,0) -> mid-node 5

        1D quadratic shape functions on s in [-1,+1]:
            N_i(s) = 0.5*s*(s-1)    ->  1 at s=-1 (corner i)
            N_j(s) = 0.5*s*(s+1)    ->  1 at s=+1 (corner j)
            N_m(s) = 1 - s^2        ->  1 at s= 0 (mid-side m)

        Geometry map:  x(s) = N_i*xi + N_j*xj + N_m*xm
        1D Jacobian:   J = ||dx/ds||

        Nodal contributions (2 DOFs per edge node):
            f_{k,x} += w * N_k * qx * J * t
            f_{k,y} += w * N_k * qy * J * t    for k in {i, j, m} only

        Returns:
            fe (np.ndarray): Surface force vector (12,). Zero if no loads.
        """
        if not self.surface_loads:
            return np.zeros(12)

        from scipy.special import roots_legendre
        roots, weights = roots_legendre(3)   # exact for quadratic integrand
        t  = self.thickness
        fe = np.zeros(12)

        for load in self.surface_loads:
            i_corner = load['node_indices'][0]
            j_corner = load['node_indices'][1]
            q        = np.array(load['value'], dtype=float)   # [qx, qy]

            # Identify mid-side node for this edge (mirrors Quad9 pattern)
            m_local = None
            for (ci, cj, cm) in _LST_EDGES:
                if (ci == i_corner and cj == j_corner) or                    (ci == j_corner and cj == i_corner):
                    m_local = cm
                    break

            if m_local is None:
                warnings.warn(
                    f"Element {self.element_tag}: cannot identify mid-side node "
                    f"for edge ({i_corner}, {j_corner}). Edge load skipped."
                )
                continue

            # Coordinates of the 3 edge nodes
            xi = np.array(self.nodes[i_corner].coordinates, dtype=float)
            xj = np.array(self.nodes[j_corner].coordinates, dtype=float)
            xm = np.array(self.nodes[m_local].coordinates,  dtype=float)

            # Local DOF positions in the 12-component element vector
            # Node k occupies positions [2k, 2k+1]
            li = 2 * i_corner
            lj = 2 * j_corner
            lm = 2 * m_local

            for s, w in zip(roots, weights):
                # 1D quadratic shape functions
                Ni = 0.5 * s * (s - 1.0)
                Nj = 0.5 * s * (s + 1.0)
                Nm = 1.0 - s * s

                # 1D Jacobian: ||dx/ds||
                # dNi/ds = s-0.5,  dNj/ds = s+0.5,  dNm/ds = -2s
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

    def get_element_displacements(self, u):
        """
        Extracts the element displacement vector from the global vector.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            ue (np.ndarray): Element displacement vector (12,).
        """
        return u[self.idx]

    def get_element_strains(self, u):
        """
        Computes the strain vector at self.eval_points.

        For the LST, strains vary linearly throughout the element. The
        default evaluation point is the centroid [1/3, 1/3], which gives
        the average strain — a good representative value for post-processing.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            epsilon_e (np.ndarray): Strain vector [εx, εy, γxy] (3,).
            ue        (np.ndarray): Element displacement vector (12,).
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
            ue        (np.ndarray): Element displacement vector (12,).
        """
        epsilon_e, ue = self.get_element_strains(u)
        sigma_e       = self.C @ epsilon_e   # uses cached self.C
        return sigma_e, epsilon_e, ue

    # ------------------------------------------------------------------
    # Principal values
    # ------------------------------------------------------------------

    def calculate_principal_stress(self, sigma):
        """
        Computes principal stresses [σ1, σ2] from [σx, σy, τxy] by
        diagonalizing the 2×2 stress tensor. σxy is already a tensorial
        component — no correction factor needed.

        Args:
            sigma (np.ndarray): Stress vector [σx, σy, τxy] (3,).

        Returns:
            np.ndarray: [σ1, σ2] sorted in descending order.
        """
        sx, sy, sxy = sigma[0], sigma[1], sigma[2]

        stress_tensor = np.array([[sx,  sxy],
                                  [sxy, sy ]])

        eigenvalues, _ = np.linalg.eigh(stress_tensor)   # eigh: symmetric, real
        return eigenvalues[::-1]   # descending: [σ1, σ2]

    def calculate_principal_strain(self, epsilon):
        """
        Computes principal strains [ε1, ε2] from [εx, εy, γxy] by
        diagonalizing the 2×2 strain tensor.

        IMPORTANT: γxy is the engineering shear strain. The tensor uses
        εxy = γxy/2 as the off-diagonal component. This factor of 1/2 is
        mandatory — omitting it inflates the principal strain magnitudes.

        Args:
            epsilon (np.ndarray): Strain vector [εx, εy, γxy] (3,).

        Returns:
            np.ndarray: [ε1, ε2] sorted in descending order.
        """
        ex, ey, gxy = epsilon[0], epsilon[1], epsilon[2]

        strain_tensor = np.array([[ex,       gxy / 2],
                                  [gxy / 2,  ey     ]])

        eigenvalues, _ = np.linalg.eigh(strain_tensor)   # eigh: symmetric, real
        return eigenvalues[::-1]   # descending: [ε1, ε2]

    # ------------------------------------------------------------------
    # Internal forces
    # ------------------------------------------------------------------

    def get_element_internal_forces(self, u):
        """
        Computes the internal nodal force vector: f_int = Ke · ue.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            fe_int (np.ndarray): Internal force vector (12,).
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
            'displacement'    : element nodal displacement vector (12,)
            'strain'          : strain vector [εx, εy, γxy] at eval_points (3,)
            'stress'          : stress vector [σx, σy, τxy] at eval_points (3,)
            'principal_stress': [σ1, σ2] sorted descending (2,)
            'principal_strain': [ε1, ε2] sorted descending (2,)
            'internal_forces' : internal nodal force vector (12,)

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
        Plots the element geometry as a shaded triangle using only the
        three corner nodes.

        Args:
            ax         : matplotlib axis (created if None).
            text       : If True, shows element tag at centroid.
            nodes      : If True, plots all 6 nodes.
            nodeLabels : If True, labels each node with its name.
            facecolor  : Fill color.
            edgecolor  : Border color.
            alpha      : Transparency.

        Returns:
            ax: The matplotlib axis.
        """
        if ax is None:
            _, ax = plt.subplots()

        coords         = self._get_xy_matrix()
        corner_coords  = coords[:3]   # only corner nodes for the patch

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
        """Prints a detailed summary of the LST element."""
        print('-------------------------------------------------------------')
        print(f"LST Element {self.element_tag}")
        print(f"Type: {self.type}")
        print(f"Nodes: {[node.name for node in self.nodes]}")
        coords = self._get_xy_matrix()
        for i, node in enumerate(self.nodes):
            label = 'corner' if i < 3 else 'mid   '
            print(f"  Node {node.name} ({label}): ({coords[i, 0]:.4f}, {coords[i, 1]:.4f})")
        print(f"Thickness:            {self.thickness}")
        print(f"Area:                 {self.area:.6f}")
        print(f"Eval point (ζ, η):   {self.eval_points}")
        print(f"Element DoF indices:  {self.idx}")
        print(f"Gravity direction:    {self.load_direction}")
        print(f"Unit weight (rho):    {self.material.rho}")
        print(f"Surface loads:        {len(self.surface_loads)} edge(s) loaded")
        print(f"F_fe_body:            {np.round(self.F_fe_body, 6)}")
        print(f"F_fe_surface:         {np.round(self.F_fe_surface, 6)}")
        print(f"F_fe_global:          {np.round(self.F_fe_global, 6)}")
        print(f"\nStiffness matrix (12×12):\n{np.round(self.kg, 4)}")
        print('-------------------------------------------------------------\n')