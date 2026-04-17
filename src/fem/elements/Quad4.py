import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
from scipy.special import roots_legendre
from fem.core.parameters import globalParameters


# Edge definition in natural coordinates (local 0-based node indices).
# Each tuple (i, j) identifies the two corner nodes of the edge and
# implicitly defines which natural coordinate is fixed and which is free:
#
#   Edge 0 — nodes (0, 1) : eta = -1,  zeta free in [-1, +1]  (bottom)
#   Edge 1 — nodes (1, 2) : zeta = +1, eta  free in [-1, +1]  (right)
#   Edge 2 — nodes (2, 3) : eta = +1,  zeta free in [+1, -1]  (top)
#   Edge 3 — nodes (3, 0) : zeta = -1, eta  free in [+1, -1]  (left)
#
#   eta
#   +1   N4(-1,+1) ---edge2--- N3(+1,+1)
#        |                     |
#      edge3                 edge1
#        |                     |
#   -1   N1(-1,-1) ---edge0--- N2(+1,-1)
#       -1          zeta        +1

_QUAD4_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]


class Quad4:
    def __init__(self,
                 element_tag: int,
                 node_list: list,
                 section: object,
                 load_direction: list = None,
                 surface_loads: list = None,
                 type: str = 'planeStress',
                 sampling_points: int = 3,
                 eval_points: list = [0, 0],
                 print_summary: bool = False):
        """
        Initialize the Quad4 isoparametric element with nodes, section properties,
        and optional load direction.

        Args:
            element_tag (int): Unique identifier for the element.
            node_list (list): List of four nodes defining the Quad4 element,
                              ordered counter-clockwise.
            section (object): Section object containing material and thickness.
            load_direction (list, optional): List [Cx, Cy] for gravitational load
                                            direction cosines. If None, reads
                                            globalParameters['gravity'].
            surface_loads (list, optional): List of surface load dictionaries, each
                                            with keys:
                                              'node_indices' : tuple (i, j) — local
                                                               0-based edge node indices
                                              'value'        : [qx, qy] — load vector
                                                               [force/length]
                                            Populated automatically by build_elements
                                            from load_dictionary. Default is None
                                            (no surface loads).
            type (str): 'planeStress' or 'planeStrain'. Default is 'planeStress'.
            sampling_points (int): Number of Gauss integration points per direction.
                                   Default is 2 (2x2 = 4 total points), which is
                                   exact for bilinear elements.
            eval_points (list): Natural coordinates [zeta, eta] at which strains and
                                stresses are evaluated during post-processing.
                                Default is [0, 0] (element centroid in natural space).
            print_summary (bool): If True, prints element summary after initialization.
        """
        if len(node_list) != 4:
            raise ValueError("Quad4 elements must have exactly 4 nodes.")

        self.element_tag   = element_tag
        self.nodes         = node_list
        self.section       = section
        self.load_direction = load_direction
        self.surface_loads  = surface_loads if surface_loads is not None else []
        self.type           = type
        self.sampling_points = sampling_points
        self.eval_points    = eval_points

        # Initialize load direction from globalParameters if not provided
        self._initialize_load_direction()

        # Element geometric and material properties
        self.xy        = self.get_xy_matrix()
        self.thickness = self.section.thickness
        self.material  = self.section.material
        self.C         = self.material.get_Emat(self.type)

        # Element calculations
        self.idx             = self.calculate_indices()
        self.kg, self.area   = self.get_stiffness_matrix()
        self.F_fe_body       = self.get_body_forces()
        self.F_fe_surface    = self.get_surface_forces()
        self.F_fe_global     = self.F_fe_body + self.F_fe_surface

        if print_summary is True:
            self.printSummary()

    def __str__(self):
        return f"Quad4 Element {self.element_tag}: Nodes {[node.name for node in self.nodes]}"

    def __repr__(self):
        return self.__str__()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _initialize_load_direction(self):
        """Initializes load direction from globalParameters gravity if not provided."""
        if self.load_direction is None:
            self.load_direction = globalParameters.get('gravity', [0, 0])

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def calculate_indices(self):
        """
        Returns the global DoF indices for the Quad4 element.

        Returns:
            idx (np.ndarray): Integer array of length 8 with global DOF indices.
        """
        idx = []
        for node in self.nodes:
            for dof in node.idx:
                idx.append(dof)
        return np.array(idx, dtype=int)

    def get_xy_matrix(self):
        """
        Returns the matrix of nodal coordinates of the Quad4 element.

        Returns:
            xy (np.ndarray): 4x2 array with node coordinates
                             [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        """
        xy = np.array([node.coordinates for node in self.nodes])
        return xy

    def get_centroid(self):
        """
        Computes the centroid of the quadrilateral element by averaging
        nodal coordinates.

        Returns:
            centroid (np.ndarray): (2,) array with centroid coordinates [x, y].
        """
        xy = self.get_xy_matrix()
        centroid = np.mean(xy, axis=0)
        return centroid

    # ------------------------------------------------------------------
    # Shape functions and interpolation
    # ------------------------------------------------------------------

    def get_interpolation_matrix(self, zeta, eta):
        """
        Computes the interpolation (shape function) matrix N and the matrix of
        partial derivatives of the shape functions with respect to natural
        coordinates (zeta, eta) for the Quad4 element.

        The shape functions are:
            N1 = (1 - zeta)(1 - eta) / 4    (node 0, bottom-left)
            N2 = (1 + zeta)(1 - eta) / 4    (node 1, bottom-right)
            N3 = (1 + zeta)(1 + eta) / 4    (node 2, top-right)
            N4 = (1 - zeta)(1 + eta) / 4    (node 3, top-left)

        Args:
            zeta (float): Natural coordinate in [-1, 1].
            eta  (float): Natural coordinate in [-1, 1].

        Returns:
            N (np.ndarray): Interpolation matrix (2x8).
            dNnatural (np.ndarray): Partial derivatives of shape functions
                                    with respect to (zeta, eta) (2x4).
        """
        # Shape functions
        N1 = 0.25 * (1 - zeta) * (1 - eta)
        N2 = 0.25 * (1 + zeta) * (1 - eta)
        N3 = 0.25 * (1 + zeta) * (1 + eta)
        N4 = 0.25 * (1 - zeta) * (1 + eta)

        # Partial derivatives with respect to zeta
        dN1dzeta = -0.25 * (1 - eta)
        dN2dzeta =  0.25 * (1 - eta)
        dN3dzeta =  0.25 * (1 + eta)
        dN4dzeta = -0.25 * (1 + eta)

        # Partial derivatives with respect to eta
        dN1deta = -0.25 * (1 - zeta)
        dN2deta = -0.25 * (1 + zeta)
        dN3deta =  0.25 * (1 + zeta)
        dN4deta =  0.25 * (1 - zeta)

        # Interpolation matrix N (2x8)
        N = np.array([
            [N1, 0,  N2, 0,  N3, 0,  N4, 0 ],
            [0,  N1, 0,  N2, 0,  N3, 0,  N4]
        ])

        # Derivatives of N with respect to natural coordinates (2x4)
        dNnatural = np.array([
            [dN1dzeta, dN2dzeta, dN3dzeta, dN4dzeta],
            [dN1deta,  dN2deta,  dN3deta,  dN4deta ]
        ])

        return N, dNnatural

    # ------------------------------------------------------------------
    # Strain-displacement matrix
    # ------------------------------------------------------------------

    def get_B_matrix(self, zeta, eta):
        """
        Computes the strain-displacement matrix B, the Jacobian matrix J,
        and its determinant at a given natural coordinate point (zeta, eta).

        The Jacobian is computed as:
            J = dNnatural @ xy    (2x4) @ (4x2) = (2x2)

        Cartesian derivatives of the shape functions are obtained by solving:
            dNcartesian = J^{-1} @ dNnatural

        The B matrix maps the 8 nodal DOFs to the 3 engineering strain
        components [εx, εy, γxy]:
            B[0, 0::2] = dN/dx    (normal strain x)
            B[1, 1::2] = dN/dy    (normal strain y)
            B[2, 0::2] = dN/dy    (shear — x-DOF contribution)
            B[2, 1::2] = dN/dx    (shear — y-DOF contribution)

        Args:
            zeta (float): Natural coordinate in [-1, 1].
            eta  (float): Natural coordinate in [-1, 1].

        Returns:
            B (np.ndarray): Strain-displacement matrix (3x8).
            J (np.ndarray): Jacobian matrix (2x2).
            J_det (float): Determinant of the Jacobian.
            N (np.ndarray): Interpolation matrix (2x8).
        """
        xy = self.xy
        N, dNnatural = self.get_interpolation_matrix(zeta, eta)

        # Jacobian: J = dN/d(zeta,eta) @ xy  →  (2x4)(4x2) = (2x2)
        J     = np.dot(dNnatural, xy)
        J_det = np.linalg.det(J)

        if J_det <= 0:
            warnings.warn(
                f"Element {self.element_tag} has non-positive Jacobian determinant "
                f"({J_det:.4f}) at (zeta={zeta:.3f}, eta={eta:.3f}). "
                f"Check node ordering (should be counter-clockwise)."
            )

        # Cartesian derivatives: solve J @ dNcartesian = dNnatural
        dNcartesian = np.linalg.solve(J, dNnatural)

        # Assemble B matrix (3x8)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNcartesian[0, :]   # dN/dx  →  εx
        B[1, 1::2] = dNcartesian[1, :]   # dN/dy  →  εy
        B[2, 0::2] = dNcartesian[1, :]   # dN/dy  →  γxy (x-DOF)
        B[2, 1::2] = dNcartesian[0, :]   # dN/dx  →  γxy (y-DOF)

        return B, J, J_det, N

    # ------------------------------------------------------------------
    # Stiffness matrix
    # ------------------------------------------------------------------

    def get_stiffness_matrix(self):
        """
        Computes the element stiffness matrix Ke and the element area
        using Gauss-Legendre numerical integration:

            Ke = t * sum_i sum_j  w_i * w_j * B^T * E * B * |J|

            A  =     sum_i sum_j  w_i * w_j * |J|

        Returns:
            Ke (np.ndarray): Stiffness matrix (8x8).
            A  (float):      Element area.
        """
        roots, weights = roots_legendre(self.sampling_points)
        t  = self.thickness
        Ke = np.zeros((8, 8))
        A  = 0.0

        for r, w_r in zip(roots, weights):
            for s, w_s in zip(roots, weights):
                B, _, J_det, _ = self.get_B_matrix(r, s)
                A  += w_r * w_s * abs(J_det)
                Ke += w_r * w_s * t * (B.T @ self.C @ B) * J_det

        return Ke, A

    # ------------------------------------------------------------------
    # Body forces
    # ------------------------------------------------------------------

    def get_body_forces(self):
        """
        Computes the consistent nodal body force vector using
        Gauss-Legendre numerical integration:

            fe = t * gamma * sum_i sum_j  w_i * w_j * N^T * b * |J|

        where b = [Cx, Cy]^T is the load direction vector and
        gamma = material.rho is the unit weight.

        Returns:
            fe (np.ndarray): Body force vector (8,).
        """
        b     = np.array(self.load_direction, dtype=float).reshape(-1, 1)
        roots, weights = roots_legendre(self.sampling_points)
        t     = self.thickness
        gamma = self.material.rho

        fe = np.zeros((8, 1))

        for r, w_r in zip(roots, weights):
            for s, w_s in zip(roots, weights):
                _, _, J_det, N = self.get_B_matrix(r, s)
                fe += w_r * w_s * (N.T @ b) * J_det

        fe = fe * (t * gamma)
        return fe.flatten()

    # ------------------------------------------------------------------
    # Surface forces
    # ------------------------------------------------------------------

    def get_surface_forces(self):
        """
        Computes the consistent nodal surface force vector by integrating
        the applied traction along each loaded edge using 1D Gauss quadrature:

            fe = t * integral_{-1}^{+1}  N^T(s) * q * |J_edge| ds

        where s is the free natural coordinate along the edge and
        |J_edge| = physical_edge_length / 2 is the 1D Jacobian.

        Each entry in self.surface_loads must have:
            'node_indices' : tuple (i, j)  — local 0-based indices of the two
                                             edge corner nodes, matching one of
                                             the four Quad4 edges in _QUAD4_EDGES.
            'value'        : [qx, qy]      — traction vector [force/length].

        Returns:
            fe (np.ndarray): Surface force vector (8,). Zero if no surface loads.
        """
        if not self.surface_loads:
            return np.zeros(8)

        roots, weights = roots_legendre(self.sampling_points)
        t  = self.thickness
        fe = np.zeros((8, 1))

        for load in self.surface_loads:
            node_i = load['node_indices'][0]   # local index 0–3
            node_j = load['node_indices'][1]   # local index 0–3
            q      = np.array(load['value'], dtype=float).reshape(-1, 1)

            # 1D Jacobian: physical edge length / 2
            xi     = self.xy[node_i]
            xj     = self.xy[node_j]
            J_edge = np.linalg.norm(xj - xi) / 2.0

            # Identify edge index to determine fixed/free natural coordinate
            edge_idx = _QUAD4_EDGES.index((node_i, node_j))

            for s, w in zip(roots, weights):
                # Map free parameter s ∈ [-1,+1] to (zeta, eta) on the edge.
                # Edges 2 and 3 traverse in reverse direction (counter-clockwise
                # ordering), so s is negated to maintain consistent orientation.
                if edge_idx == 0:       # bottom: eta=-1, zeta=s
                    zeta, eta = s,    -1.0
                elif edge_idx == 1:     # right:  zeta=+1, eta=s
                    zeta, eta = 1.0,   s
                elif edge_idx == 2:     # top:    eta=+1, zeta=-s  (reversed)
                    zeta, eta = -s,    1.0
                else:                   # left:   zeta=-1, eta=-s  (reversed)
                    zeta, eta = -1.0,  -s

                N, _ = self.get_interpolation_matrix(zeta, eta)
                fe  += w * (N.T @ q) * J_edge

        fe = fe * t
        return fe.flatten()

    # ------------------------------------------------------------------
    # Post-processing: displacements, strains, stresses
    # ------------------------------------------------------------------

    def get_element_displacements(self, u):
        """
        Extracts the element displacement vector from the global displacement vector.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            ue (np.ndarray): Element displacement vector (8,).
        """
        ue = u[self.idx]
        return ue

    def get_element_strains(self, u):
        """
        Computes the engineering strain vector for the element, evaluated at
        self.eval_points in natural coordinates.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            epsilon_e (np.ndarray): Strain vector [εx, εy, γxy] (3,).
            ue (np.ndarray): Element displacement vector (8,).
        """
        ue        = self.get_element_displacements(u)
        zeta, eta = self.eval_points
        B, _, _, _ = self.get_B_matrix(zeta, eta)
        epsilon_e = B @ ue
        return epsilon_e, ue

    def get_element_stress(self, u):
        """
        Computes the stress vector for the element, evaluated at self.eval_points.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            sigma_e   (np.ndarray): Stress vector [σx, σy, τxy] (3,).
            epsilon_e (np.ndarray): Strain vector [εx, εy, γxy] (3,).
            ue        (np.ndarray): Element displacement vector (8,).
        """
        epsilon_e, ue = self.get_element_strains(u)
        sigma_e       = self.C @ epsilon_e
        return sigma_e, epsilon_e, ue

    # ------------------------------------------------------------------
    # Principal values
    # ------------------------------------------------------------------

    def calculate_principal_stress(self, sigma):
        """
        Computes principal stresses from the stress vector [σx, σy, τxy]
        by diagonalizing the 2x2 stress tensor.

        Args:
            sigma (np.ndarray): Stress vector [σx, σy, τxy] (3,).

        Returns:
            np.ndarray: Principal stresses [σ1, σ2] sorted in descending order.
        """
        sx, sy, sxy = sigma[0], sigma[1], sigma[2]

        stress_tensor = np.array([[sx,  sxy],
                                  [sxy, sy ]])

        eigenvalues, _ = np.linalg.eig(stress_tensor)
        sorted_idx     = np.argsort(eigenvalues)[::-1]
        sigma1, sigma2 = eigenvalues[sorted_idx]

        return np.array([sigma1, sigma2])

    def calculate_principal_strain(self, epsilon):
        """
        Computes principal strains from the strain vector [εx, εy, γxy]
        by diagonalizing the 2x2 strain tensor.

        Note: γxy is the engineering shear strain. The strain tensor uses
        εxy = γxy / 2 as the off-diagonal component.

        Args:
            epsilon (np.ndarray): Strain vector [εx, εy, γxy] (3,).

        Returns:
            np.ndarray: Principal strains [ε1, ε2] sorted in descending order.
        """
        ex, ey, exy = epsilon[0], epsilon[1], epsilon[2]

        strain_tensor = np.array([[ex,      exy / 2],
                                  [exy / 2, ey     ]])

        eigenvalues, _    = np.linalg.eig(strain_tensor)
        sorted_idx        = np.argsort(eigenvalues)[::-1]
        epsilon1, epsilon2 = eigenvalues[sorted_idx]

        return np.array([epsilon1, epsilon2])

    # ------------------------------------------------------------------
    # Internal forces
    # ------------------------------------------------------------------

    def get_element_internal_forces(self, u):
        """
        Computes the internal nodal force vector for the element.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            fe_int (np.ndarray): Internal force vector (8,).
        """
        ue     = self.get_element_displacements(u)
        fe_int = self.kg @ ue
        return fe_int

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self, u):
        """
        Computes and returns all element results as a dictionary:
            - 'displacement'    : element nodal displacement vector (8,)
            - 'strain'          : strain vector at eval_points [εx, εy, γxy] (3,)
            - 'stress'          : stress vector at eval_points [σx, σy, τxy] (3,)
            - 'principal_stress': principal stresses [σ1, σ2] (2,)
            - 'principal_strain': principal strains [ε1, ε2] (2,)
            - 'internal_forces' : internal nodal force vector (8,)

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            results (dict): Dictionary with all element results.
        """
        sigma_e, epsilon_e, ue = self.get_element_stress(u)
        fe_int           = self.get_element_internal_forces(u)
        principal_stress = self.calculate_principal_stress(sigma_e)
        principal_strain = self.calculate_principal_strain(epsilon_e)

        results = {
            'displacement':     ue,
            'strain':           epsilon_e,
            'stress':           sigma_e,
            'principal_stress': principal_stress,
            'principal_strain': principal_strain,
            'internal_forces':  fe_int
        }

        return results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plotGeometry(self, ax=None, text=False, nodes=True, nodeLabels=False,
                     facecolor='lightgray', edgecolor='k', alpha=0.5):
        """
        Plots the geometry of the Quad4 element as a shaded quadrilateral.

        Args:
            ax         (matplotlib axis, optional): Existing axis. If None, a new one is created.
            text       (bool): If True, displays the element tag at its centroid.
            nodes      (bool): If True, plots the nodes.
            nodeLabels (bool): If True, labels each node with its name.
            facecolor  (str):  Fill color of the quadrilateral.
            edgecolor  (str):  Border color of the quadrilateral.
            alpha      (float): Fill transparency.

        Returns:
            ax (matplotlib axis): The axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        coords = self.get_xy_matrix()

        polygon = patches.Polygon(coords, closed=True,
                                  facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        ax.add_patch(polygon)

        if nodes or nodeLabels:
            for node in self.nodes:
                node.plotGeometry(ax, text=nodeLabels)

        if text:
            x_c, y_c = self.get_centroid()
            ax.text(x_c, y_c, f'{self.element_tag}', fontsize=12,
                    ha='center', va='center')

        return ax

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def printSummary(self):
        """
        Prints a detailed summary of the Quad4 element.
        """
        print(f'-------------------------------------------------------------')
        print(f"Quad4 Element {self.element_tag}")
        print(f"Type: {self.type}")
        print(f"Nodes: {[node.name for node in self.nodes]}")

        coords = self.get_xy_matrix()
        for i, node in enumerate(self.nodes):
            print(f"  Node {node.name}: ({coords[i, 0]:.3f}, {coords[i, 1]:.3f})")

        print(f"Thickness:                  {self.section.thickness}")
        print(f"Area:                       {self.area:.4f}")
        print(f"Sampling points (per dir):  {self.sampling_points}")
        print(f"Eval point (natural coords): {self.eval_points}")
        print(f"Element DoF indices:        {self.idx}")
        print(f"Gravity direction:          {self.load_direction}")
        print(f"Unit weight (rho):          {self.material.rho}")
        print(f"Surface loads:              {len(self.surface_loads)} edge(s) loaded")
        print(f"F_fe_body:                  {np.round(self.F_fe_body, 6)}")
        print(f"F_fe_surface:               {np.round(self.F_fe_surface, 6)}")
        print(f"F_fe_global:                {np.round(self.F_fe_global, 6)}")
        print(f"\nStiffness matrix:\n{self.kg}")
        print(f'-------------------------------------------------------------\n')