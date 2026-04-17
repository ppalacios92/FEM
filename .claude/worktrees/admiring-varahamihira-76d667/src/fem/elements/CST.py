import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
from scipy.special import roots_legendre
from fem.core.parameters import globalParameters


# Edge definition for the CST element (local 0-based node indices).
# Each tuple (i, j) identifies the two nodes of the edge:
#
#   Edge 0 — nodes (0, 1) : bottom edge
#   Edge 1 — nodes (1, 2) : right/hypotenuse edge
#   Edge 2 — nodes (2, 0) : left edge
#
#   Node 2
#    |  \
#  edge2  edge1
#    |        \
#   Node 0 --- Node 1
#       edge0

_CST_EDGES = [(0, 1), (1, 2), (2, 0)]


class CST:
    def __init__(self,
                 element_tag: int,
                 node_list: list,
                 section: object,
                 load_direction: list = None,
                 surface_loads: list = None,
                 type: str = 'planeStress',
                 print_summary: bool = False):
        """
        Initialize the CST (Constant Strain Triangle) element with nodes,
        section properties, and optional load direction.

        Args:
            element_tag (int): Unique identifier for the element.
            node_list (list): List of three nodes defining the CST element,
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
            print_summary (bool): If True, prints element summary after initialization.
        """
        if len(node_list) != 3:
            raise ValueError("CST elements must have exactly 3 nodes.")

        self.element_tag    = element_tag
        self.nodes          = node_list
        self.section        = section
        self.load_direction = load_direction
        self.surface_loads  = surface_loads if surface_loads is not None else []
        self.type           = type

        # Initialize load direction from globalParameters if not provided
        self._initialize_load_direction()

        # Element geometric and material properties
        self.thickness = self.section.thickness
        self.material  = self.section.material
        self.C         = self.material.get_Emat(self.type)

        # Element calculations
        self.compute_area()
        self.idx             = self.calculate_indices()
        self.kg              = self.get_stiffness_matrix()
        self.F_fe_body       = self.get_body_forces()
        self.F_fe_surface    = self.get_surface_forces()
        self.F_fe_global     = self.F_fe_body + self.F_fe_surface

        if print_summary is True:
            self.printSummary()

    def __str__(self):
        return f"CST Element {self.element_tag}: Nodes {[node.name for node in self.nodes]}"

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
        Returns the global DoF indices for the CST element.

        Returns:
            idx (np.ndarray): Integer array of length 6 with global DOF indices.
        """
        idx = []
        for node in self.nodes:
            for dof in node.idx:
                idx.append(dof)
        return np.array(idx, dtype=int)

    def get_xy_matrix(self):
        """
        Returns the matrix of nodal coordinates of the CST element.

        Returns:
            xy (np.ndarray): 3x2 array with node coordinates
                             [[x1, y1], [x2, y2], [x3, y3]].
        """
        xy = []
        for node in self.nodes:
            xy.append(node.coordinates)
        return np.array(xy)

    def get_centroid(self):
        """
        Computes the centroid of the triangular element by averaging
        nodal coordinates.

        Returns:
            centroid (np.ndarray): (2,) array with centroid coordinates [x, y].
        """
        xy       = self.get_xy_matrix()
        w        = np.ones((1, 3)) / 3
        centroid = w @ xy
        return centroid.flatten()

    def compute_area(self):
        """
        Computes and stores the area of the CST element using the
        determinant formula:

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
                f"Element {self.element_tag} has non-positive area: {self.area:.4f}. "
                f"Check node ordering (should be counter-clockwise)."
            )

    # ------------------------------------------------------------------
    # Shape functions and interpolation
    # ------------------------------------------------------------------

    def get_interpolation_matrix(self, x: float, y: float):
        """
        Evaluates the shape function matrix N at a physical point (x, y).

        The shape functions for the CST are the triangular coordinates xi_i,
        computed directly from their closed-form expressions:

            xi0 = [(x1*y2 - x2*y1) + (y1 - y2)*x + (x2 - x1)*y] / 2A
            xi1 = [(x2*y0 - x0*y2) + (y2 - y0)*x + (x0 - x2)*y] / 2A
            xi2 = [(x0*y1 - x1*y0) + (y0 - y1)*x + (x1 - x0)*y] / 2A

        The interpolation matrix N is (2x6), structured as:

            N = [[xi0,  0, xi1,  0, xi2,  0],
                 [  0, xi0,  0, xi1,  0, xi2]]

        Args:
            x (float): Physical x-coordinate of the evaluation point.
            y (float): Physical y-coordinate of the evaluation point.

        Returns:
            Nmat (np.ndarray): Shape function matrix of shape (2, 6).
        """
        xy = self.get_xy_matrix()
        x0, y0 = xy[0]
        x1, y1 = xy[1]
        x2, y2 = xy[2]

        xi0 = ((x1*y2 - x2*y1) + (y1 - y2)*x + (x2 - x1)*y) / (2*self.area)
        xi1 = ((x2*y0 - x0*y2) + (y2 - y0)*x + (x0 - x2)*y) / (2*self.area)
        xi2 = ((x0*y1 - x1*y0) + (y0 - y1)*x + (x1 - x0)*y) / (2*self.area)

        Nmat = np.zeros((2, 6))
        Nmat[0, 0::2] = [xi0, xi1, xi2]
        Nmat[1, 1::2] = [xi0, xi1, xi2]

        return Nmat

    # ------------------------------------------------------------------
    # Strain-displacement matrix
    # ------------------------------------------------------------------

    def get_B_matrix(self):
        """
        Computes the strain-displacement matrix B for the CST element.

        Because the shape functions are linear in x and y, B is constant
        throughout the entire element (no dependence on position):

            B = (1 / 2A) * [[b1,  0,  b2,  0,  b3,  0 ],
                             [ 0, c1,   0, c2,   0, c3 ],
                             [c1, b1,  c2, b2,  c3, b3 ]]

        where:
            b1 = y2 - y3,  b2 = y3 - y1,  b3 = y1 - y2
            c1 = x3 - x2,  c2 = x1 - x3,  c3 = x2 - x1

        Returns:
            B (np.ndarray): Strain-displacement matrix (3x6).
        """
        x1, y1 = self.nodes[0].coordinates
        x2, y2 = self.nodes[1].coordinates
        x3, y3 = self.nodes[2].coordinates

        b1 = y2 - y3;  b2 = y3 - y1;  b3 = y1 - y2
        c1 = x3 - x2;  c2 = x1 - x3;  c3 = x2 - x1

        B = (1 / (2 * self.area)) * np.array([
                                                [b1,  0,  b2,  0,  b3,  0],
                                                [ 0, c1,   0, c2,   0, c3],
                                                [c1, b1,  c2, b2,  c3, b3]
        ])

        return B

    # ------------------------------------------------------------------
    # Stiffness matrix
    # ------------------------------------------------------------------

    def get_stiffness_matrix(self):
        """
        Computes the element stiffness matrix Ke using the closed-form
        expression (exact — no numerical integration required):

            Ke = B^T * E * B * t * A

        Returns:
            Ke (np.ndarray): Stiffness matrix (6x6).
        """
        B  = self.get_B_matrix()
        t  = self.thickness
        Ke = B.T @ self.C @ B * self.area * t
        return Ke

    # ------------------------------------------------------------------
    # Body forces
    # ------------------------------------------------------------------

    def get_body_forces(self):
        """
        Computes the consistent nodal body force vector using 1-point
        integration at the centroid of the element:

            fe = (A * t / 3) * [bx, by, bx, by, bx, by]^T

        For a uniform body force, each node receives exactly one-third
        of the total force acting on the element.

        Returns:
            fe (np.ndarray): Body force vector (6,).
        """
        b        = np.array(self.load_direction, dtype=float).reshape(2,)
        centroid = self.get_centroid()
        N        = self.get_interpolation_matrix(*centroid)
        t        = self.thickness
        gamma    = self.material.rho
        
        fe = (N.T @ b) * self.area * t * gamma
        return fe

    # ------------------------------------------------------------------
    # Surface forces
    # ------------------------------------------------------------------

    def get_surface_forces(self):
        """
        Computes the consistent nodal surface force vector by integrating
        the applied traction along each loaded edge using 1D Gauss quadrature:

            fe = t * integral_{-1}^{+1}  N^T(x(s), y(s)) * q * |J_edge| ds

        where s ∈ [-1, +1] is the parametric coordinate along the edge,
        (x(s), y(s)) is the corresponding physical point, and
        |J_edge| = physical_edge_length / 2 is the 1D Jacobian.

        The physical point is interpolated linearly along the edge:
            x(s) = 0.5*(1-s)*xi + 0.5*(1+s)*xj
            y(s) = 0.5*(1-s)*yi + 0.5*(1+s)*yj

        Each entry in self.surface_loads must have:
            'node_indices' : tuple (i, j)  — local 0-based indices of the two
                                             edge nodes, matching one of the
                                             three CST edges in _CST_EDGES.
            'value'        : [qx, qy]      — traction vector [force/length].

        Returns:
            fe (np.ndarray): Surface force vector (6,). Zero if no surface loads.
        """
        if not self.surface_loads:
            return np.zeros(6)

        roots, weights = roots_legendre(2)
        t  = self.thickness
        fe = np.zeros((6, 1))

        for load in self.surface_loads:
            node_i = load['node_indices'][0]
            node_j = load['node_indices'][1]
            q      = np.array(load['value'], dtype=float).reshape(-1, 1)

            # Physical coordinates of edge endpoints
            xi = np.array(self.nodes[node_i].coordinates, dtype=float)
            xj = np.array(self.nodes[node_j].coordinates, dtype=float)

            # 1D Jacobian: physical edge length / 2
            J_edge = np.linalg.norm(xj - xi) / 2.0

            for s, w in zip(roots, weights):
                # Map s ∈ [-1,+1] to physical point on edge
                x_phys = 0.5 * (1 - s) * xi[0] + 0.5 * (1 + s) * xj[0]
                y_phys = 0.5 * (1 - s) * xi[1] + 0.5 * (1 + s) * xj[1]

                N   = self.get_interpolation_matrix(x_phys, y_phys)
                fe += w * (N.T @ q) * J_edge

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
            ue (np.ndarray): Element displacement vector (6,).
        """
        ue = u[self.idx]
        return ue

    def get_element_strains(self, u):
        """
        Computes the engineering strain vector for the element.

        Because B is constant for the CST, the strain is uniform
        throughout the entire element.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            epsilon_e (np.ndarray): Strain vector [εx, εy, γxy] (3,).
            ue        (np.ndarray): Element displacement vector (6,).
        """
        ue        = self.get_element_displacements(u)
        epsilon_e = self.get_B_matrix() @ ue
        return epsilon_e, ue

    def get_element_stress(self, u):
        """
        Computes the stress vector for the element.

        Because B is constant, the stress is also uniform throughout
        the entire element.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            sigma_e   (np.ndarray): Stress vector [σx, σy, τxy] (3,).
            epsilon_e (np.ndarray): Strain vector [εx, εy, γxy] (3,).
            ue        (np.ndarray): Element displacement vector (6,).
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

        eigenvalues, _     = np.linalg.eig(strain_tensor)
        sorted_idx         = np.argsort(eigenvalues)[::-1]
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
            fe_int (np.ndarray): Internal force vector (6,).
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
            - 'displacement'    : element nodal displacement vector (6,)
            - 'strain'          : strain vector [εx, εy, γxy] (3,)
            - 'stress'          : stress vector [σx, σy, τxy] (3,)
            - 'principal_stress': principal stresses [σ1, σ2] (2,)
            - 'principal_strain': principal strains [ε1, ε2] (2,)
            - 'internal_forces' : internal nodal force vector (6,)

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
        Plots the geometry of the CST element as a shaded triangle.

        Args:
            ax         (matplotlib axis, optional): Existing axis. If None, a new one is created.
            text       (bool): If True, displays the element tag at its centroid.
            nodes      (bool): If True, plots the nodes.
            nodeLabels (bool): If True, labels each node with its name.
            facecolor  (str):  Fill color of the triangle.
            edgecolor  (str):  Border color of the triangle.
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
        Prints a detailed summary of the CST element.
        """
        print(f'-------------------------------------------------------------')
        print(f"CST Element {self.element_tag}")
        print(f"Type: {self.type}")
        print(f"Nodes: {[node.name for node in self.nodes]}")

        coords = self.get_xy_matrix()
        for i, node in enumerate(self.nodes):
            print(f"  Node {node.name}: ({coords[i, 0]:.3f}, {coords[i, 1]:.3f})")

        print(f"Thickness:                   {self.section.thickness}")
        print(f"Area:                        {self.area:.4f}")
        print(f"Element DoF indices:         {self.idx}")
        print(f"Gravity direction:           {self.load_direction}")
        print(f"Unit weight (rho):           {self.material.rho}")
        print(f"Surface loads:               {len(self.surface_loads)} edge(s) loaded")
        print(f"F_fe_body:                   {np.round(self.F_fe_body, 6)}")
        print(f"F_fe_surface:                {np.round(self.F_fe_surface, 6)}")
        print(f"F_fe_global:                 {np.round(self.F_fe_global, 6)}")
        print(f"\nStiffness matrix:\n{self.kg}")
        print(f'-------------------------------------------------------------\n')