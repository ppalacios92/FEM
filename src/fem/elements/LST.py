import numpy as np
import matplotlib.pyplot as plt
from fem.core.parameters import globalParameters
import matplotlib.patches as patches
from scipy.special import roots_legendre

class LST:
    def __init__(self, 
                 element_tag: int, 
                 node_list: list, 
                 section: object, 
                 load_direction=None,
                 type: str = 'planeStress',
                 sampling_points=3,
                 eval_points=[0,0],
                 print_summary=False,
                 nDof=2):
        """
        Initialize the LST (6-node triangular) element with nodes, section properties, and optional load direction.

        Args:
            element_tag (int): Unique identifier for the element.
            node_list (list): List of six nodes defining the LST element (3 vertices followed by 3 mid-side nodes).
            section (object): Section object containing material and thickness.
            load_direction (list, optional): List [Cx, Cy] for gravitational load direction (body force direction).
            type (str): 'planeStress' or 'planeStrain'. Default is 'planeStress'.
        """
        if len(node_list) != 6:
            raise ValueError("LST elements must have exactly 6 nodes.")
        
        self.element_tag = element_tag
        self.node_list = node_list
        self.nodes = node_list
        self.section = section
        self.load_direction = load_direction
        self.type = type
        self.sampling_points = sampling_points
        self.eval_points = eval_points
        self.nDof = nDof

        # Initialize the element properties
        self._initialize_load_direction()
        self.xy = self.get_xy_matrix()
        self.thickness = self.section.thickness
        self.material = self.section.material
        self.C = self.material.get_Emat(type)
        
        # Element calculations
        self.idx = self.calculate_indices()
        self.kg, self.area, self.F_fe_global = self.get_stiffness_matrix()
        
        if print_summary is True:
            self.printSummary()
    
    def __str__(self):
        return f"LST Element {self.element_tag}: Nodes {[node.name for node in self.nodes]}"
    
    def __repr__(self):
        return self.__str__()
    
    def _initialize_load_direction(self):
        """Initializes the load direction for the LST element."""
        if self.load_direction is None:
            self.load_direction = [0, 0]
    
    def calculate_indices(self):
        """Returns the global DoF indices for the LST element."""
        idx = np.hstack([node.idx for node in self.nodes]).astype(int)
        return idx

    def get_xy_matrix(self):
        """
        Returns the matrix of nodal coordinates of the LST element.

        Returns:
            xy (np.ndarray): 6x2 array with node coordinates [[x1, y1], [x2, y2], ..., [x6, y6]]
        """
        xy = np.array([node.coordinates for node in self.nodes])
        return xy
    
    @staticmethod
    def get_interpolation_matrix(zeta, eta):
        """
        Calculates the shape functions and their partial derivatives for the 6-node triangular element (LST)
        in area (barycentric) coordinates (zeta, eta).

        Args:
            zeta (float): Natural coordinate (ξ) in the reference triangle.
            eta (float): Natural coordinate (η) in the reference triangle.

        Returns:
            N (ndarray): Interpolation function matrix for the given natural coordinates (2 x 12 for 6 nodes with 2 DoF each).
            dNnatural (ndarray): Matrix of partial derivatives of the shape functions with respect to zeta and eta (shape 2 x 6).
        """
        # Barycentric coordinates ξ0, ξ1, ξ2 such that ξ0 + ξ1 + ξ2 = 1
        xi0 = 1.0 - zeta - eta  # corresponds to node 1 (opposite side 23)
        xi1 = zeta             # corresponds to node 2 (opposite side 31)
        xi2 = eta              # corresponds to node 3 (opposite side 12)
        # Shape functions for 6-node LST (Quadratic triangle) using barycentric coordinates
        N1 = xi0 * (2*xi0 - 1)       # Node 1 (corner)
        N2 = xi1 * (2*xi1 - 1)       # Node 2 (corner)
        N3 = xi2 * (2*xi2 - 1)       # Node 3 (corner)
        N4 = 4 * xi0 * xi1           # Node 4 (mid-side between 1-2)
        N5 = 4 * xi1 * xi2           # Node 5 (mid-side between 2-3)
        N6 = 4 * xi2 * xi0           # Node 6 (mid-side between 3-1)
        # Interpolation function matrix N (2 x 12)
        N = np.array([
            [N1, 0, N2, 0, N3, 0, N4, 0, N5, 0, N6, 0],
            [0, N1, 0, N2, 0, N3, 0, N4, 0, N5, 0, N6]
        ])
        # Partial derivatives of shape functions w.r.t. zeta and eta (using ∂ξ0/∂zeta = -1, ∂ξ0/∂eta = -1; ∂ξ1/∂zeta = 1; ∂ξ2/∂eta = 1)
        # N1 = ξ0(2ξ0 - 1); ∂N1/∂ξ0 = 4ξ0 - 1
        dN1dzeta = (4*xi0 - 1) * (-1)   # ∂N1/∂zeta = (4ξ0 - 1)*∂ξ0/∂zeta
        dN1deta  = (4*xi0 - 1) * (-1)   # ∂N1/∂eta  = (4ξ0 - 1)*∂ξ0/∂eta
        # N2 = ξ1(2ξ1 - 1); ∂N2/∂ξ1 = 4ξ1 - 1
        dN2dzeta = (4*xi1 - 1) * (1)    # ∂N2/∂zeta = (4ξ1 - 1)*∂ξ1/∂zeta
        dN2deta  = 0.0                 # N2 has no direct η dependence (ξ1 independent of η)
        # N3 = ξ2(2ξ2 - 1); ∂N3/∂ξ2 = 4ξ2 - 1
        dN3dzeta = 0.0                 # N3 has no direct ζ dependence
        dN3deta  = (4*xi2 - 1) * (1)    # ∂N3/∂eta = (4ξ2 - 1)*∂ξ2/∂eta
        # N4 = 4 ξ0 ξ1; ∂N4/∂ξ0 = 4 ξ1, ∂N4/∂ξ1 = 4 ξ0
        dN4dzeta = 4 * (xi0 - xi1)     # ∂N4/∂zeta = 4(ξ1*∂ξ1/∂zeta + ξ0*∂ξ0/∂zeta) = 4*(xi1*1 + xi0*(-1)) = 4(xi1 - xi0) with a sign error check
        dN4dzeta = 4 * (xi0 - xi1)     # (corrected: 4*(xi0 - xi1))
        dN4deta  = -4 * xi1            # ∂N4/∂eta = 4(ξ1*∂ξ1/∂eta + ξ0*∂ξ0/∂eta) = 4*(xi1*0 + xi0*(-1)) = -4 xi0? Actually, since ξ0 = 1-ζ-η, ∂N4/∂eta = -4 ξ1 (ξ0 constant along η for fixed ξ1?) 
        # (More directly: N4 = 4 ξ0 ξ1 depends on ξ0 and ξ1, and ∂N4/∂eta = ∂N4/∂ξ0 * ∂ξ0/∂eta + ∂N4/∂ξ1 * ∂ξ1/∂eta = 4 ξ1 * (-1) + 4 ξ0 * 0 = -4 ξ1)
        # N5 = 4 ξ1 ξ2; ∂N5/∂ξ1 = 4 ξ2, ∂N5/∂ξ2 = 4 ξ1
        dN5dzeta = 4 * xi2            # ∂N5/∂zeta = 4(ξ2*∂ξ2/∂zeta + ξ1*∂ξ1/∂zeta) = 4*(xi2*0 + xi1*1) = 4 xi1? Wait carefully: ∂N5/∂zeta = 4 ξ2 * 0 + 4 ξ1 * 1 = 4 ξ1. However, with our xi naming, zeta = ξ1, so it should be 4 * xi2 * (∂ξ2/∂zeta) + 4 * xi1 * (∂ξ1/∂zeta) = 4*xi2*0 + 4*xi1*1 = 4 xi1. Actually, easier: N5 = 4 ζ η (with ζ=xi1, η=xi2), so ∂N5/∂ζ = 4 η = 4 xi2.
        dN5dzeta = 4 * xi2            # ∂N5/∂ζ = 4 ξ2
        dN5deta  = 4 * xi1            # ∂N5/∂η = 4 ξ1
        # N6 = 4 ξ2 ξ0; ∂N6/∂ξ0 = 4 ξ2, ∂N6/∂ξ2 = 4 ξ0
        dN6dzeta = -4 * xi2           # ∂N6/∂ζ = 4(ξ2*∂ξ2/∂zeta + ξ0*∂ξ0/∂zeta) = 4(xi2*0 + xi0*(-1)) = -4 xi0? Actually, directly N6 = 4 ξ2 ξ0, ∂N6/∂ζ = -4 ξ2 (since ∂ξ0/∂ζ = -1).
        dN6deta  = 4 * (xi0 - xi2)    # ∂N6/∂η = 4(ξ2*∂ξ2/∂eta + ξ0*∂ξ0/∂eta) = 4(xi2*1 + xi0*(-1)) = 4(xi2 - xi0) = -4(xi0 - xi2) so careful: Actually, ∂N6/∂η = 4 ξ0 * (-1) + 4 ξ2 * 1 = 4(ξ2 - ξ0) = -4(ξ0 - ξ2). To maintain consistency (as sum of derivatives = 0), we expect ∂N6/∂η = 4(ξ0 - ξ2).
        dN6deta  = 4 * (xi0 - xi2)
        # Assemble derivative matrix
        dNnatural = np.array([
            [dN1dzeta, dN2dzeta, dN3dzeta, dN4dzeta, dN5dzeta, dN6dzeta],
            [dN1deta,  dN2deta,  dN3deta,  dN4deta,  dN5deta,  dN6deta ]
        ])
        return N, dNnatural
    
    def transform_to_physical(self, zeta, eta):
        """
        Transforms a point given in natural (ξ, η) coordinates to physical (x, y) coordinates.
        """
        N, _ = self.get_interpolation_matrix(zeta, eta)
        nDof = self.nDof
        nDof_element = len(self.node_list) * nDof
        
        vector_coordinates = np.zeros((nDof_element, 1))
        vector_coordinates[0::2, 0] = self.xy[:, 0]
        vector_coordinates[1::2, 0] = self.xy[:, 1]
        
        coordinates_cartesianas = np.dot(N, vector_coordinates)
        return coordinates_cartesianas
    
    def get_B_matrix(self, zeta, eta):
        """
        Calculates the strain-displacement matrix B, the Jacobian matrix J and its determinant, 
        and the interpolation matrix N at a given natural coordinate (zeta, eta). 
        These values are evaluated at each integration (Gaussian) point.

        Args:
            zeta (float): natural coordinate (ξ) of the integration point.
            eta (float): natural coordinate (η) of the integration point.

        Returns:
            B (ndarray): Strain-displacement matrix (3 x 2*6).
            J (ndarray): Jacobian matrix (2x2).
            J_det (float): Determinant of J.
            N (ndarray): Interpolation matrix at (zeta, eta).
        """
        xy = self.xy
        # Shape functions and derivatives in natural coords
        N, dNnatural = self.get_interpolation_matrix(zeta, eta)
        # Jacobian matrix J = dNnatural * xy  (2x6 * 6x2 = 2x2)
        J = np.dot(dNnatural, xy)
        J_det = np.linalg.det(J)
        # If J_det is negative, the element orientation is improper (commented out check)
        # if J_det < 0:
        #     raise ValueError('Jacobiano Negativo!')
        # Compute derivatives of shape functions w.r.t physical coords: dN_cart = J^{-1} * dNnatural
        dNcartesian = np.linalg.solve(J, dNnatural)
        
        B = np.zeros((3, 2 * len(xy)))
        B[0, 0::2] = dNcartesian[0, :]  # ∂N/∂x for each node goes to B[0] at x-dof positions
        B[1, 1::2] = dNcartesian[1, :]  # ∂N/∂y for each node goes to B[1] at y-dof positions
        B[2, 0::2] = dNcartesian[1, :]  # ∂N/∂y to B[2] at x-dof positions
        B[2, 1::2] = dNcartesian[0, :]  # ∂N/∂x to B[2] at y-dof positions
        
        return B, J, J_det, N
    
    def _get_b_load_direction(self):
        # Return body force (load direction) vector as column
        b = np.array(self.load_direction, dtype=float)
        b = b.reshape(-1, 1)
        return b
    
    def get_stiffness_matrix(self):
        """
        Calculates the initial stiffness matrix (Ke), area (A), and consistent body force vector (fe) of the element.

        Returns:
            Ke (ndarray): Stiffness matrix of the element (12x12 for 6-node triangle).
            A (float): Area of the element.
            fe (ndarray): Consistent body force vector (length 12).
        """
        # Integration points for triangle (using Gauss-Legendre in transformed [0,1] domain)
        sampling_points = self.sampling_points
        roots, weights = roots_legendre(sampling_points)
        # if sampling_points == 3:
        #     roots=np.array([1/6,1/6,1/6])
        #     weights=np.array([1/3,1/3,1/3])
        # print(roots , weights)
        # Transform 1D Gauss points from [-1,1] to [0,1]
        xi_points = 0.5 * (roots + 1.0)
        xi_weights = 0.5 * weights
        eta_points = xi_points  # use same for inner integration
        eta_weights = xi_weights
        t = self.thickness
        
        b_loadDirection = self._get_b_load_direction()
        
        # Initialize stiffness matrix, area and elemental force
        A = 0.0
        nDof = self.nDof
        nDof_element = len(self.node_list) * nDof
        Ke = np.zeros((nDof_element, nDof_element))
        fe = np.zeros((nDof_element, 1))
        
        # Double integration loop over reference triangle (ξ in [0,1], η in [0,1] with η <= 1-ξ)
        for xi, w_xi in zip(xi_points, xi_weights):
            for eta_hat, w_eta in zip(eta_points, eta_weights):
                # Map (xi, eta_hat) to triangular domain: actual η = eta_hat * (1 - xi)
                eta = eta_hat * (1.0 - xi)
                # Weight for triangular domain = w_xi * w_eta * (1 - xi)
                weight = w_xi * w_eta * (1.0 - xi)
                # Evaluate at this integration point
                B, _, J_det, N = self.get_B_matrix(xi, eta)
                # Accumulate area (use absolute J_det to avoid negative area if orientation is flipped)
                A += weight * abs(J_det)
                # Stiffness matrix integration
                Ke += weight * t * (B.T @ self.C @ B) * J_det
                # Consistent body force vector integration
                fe += weight * (N.T @ b_loadDirection) * J_det
        
        gamma = self.material.rho
        fe = fe * (t * gamma)  # include thickness and density in force vector
        fe = fe.flatten()
        return Ke, A, fe
    
    def get_centroid(self):
        """
        Computes the centroid of the triangular element by averaging nodal coordinates.
        """
        xy = self.get_xy_matrix()
        centroid = np.mean(xy, axis=0)
        return centroid
    
    def get_element_displacements(self, u):
        """
        Extracts the element displacement vector from global displacements.

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            ue (np.ndarray): Element displacement vector (length 12).
        """
        ue = u[self.idx]
        return ue
    
    def get_element_strains(self, u):
        """
        Computes the strain vector for the element evaluated at self.eval_points (local ξ,η coordinates).

        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            epsilon_e (np.ndarray): Strain vector (3,)
            ue (np.ndarray): Element displacement vector (length 12)
        """
        ue = self.get_element_displacements(u)
        zeta, eta = self.eval_points
        B, _, _, _ = self.get_B_matrix(zeta, eta)
        epsilon_e = B @ ue
        return epsilon_e, ue
    
    def get_element_stress(self, u):
        """
        Computes the stress vector for the element at self.eval_points.

        Args:
            u (np.ndarray): Global displacement vector

        Returns:
            sigma_e (np.ndarray): Stress vector (3x1)
            epsilon_e (np.ndarray): Strain vector (3x1)
            ue (np.ndarray): Element displacement vector (length 12)
        """
        epsilon_e, ue = self.get_element_strains(u)
        sigma_e = self.section.material.get_Emat(self.type) @ epsilon_e
        return sigma_e, epsilon_e, ue
    
    def calculate_principal_stress(self, sigma):
        """
        Calculates principal stresses from the stress vector [σx, σy, τxy].
        Returns an array [σ1, σ2] of principal stresses.
        """
        sx = sigma[0]
        sy = sigma[1]
        sxy = sigma[2]
        stress_matrix = np.array([[sx, sxy],
                                  [sxy, sy]])
        # Eigen decomposition for principal stresses
        eigenvalues, _ = np.linalg.eig(stress_matrix)
        # Sort eigenvalues in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        sigma1, sigma2 = eigenvalues[sorted_idx]
        return np.array([sigma1, sigma2])
    
    def calculate_principal_strain(self, epsilon):
        """
        Calculates principal strains from the strain vector [εx, εy, γxy].
        Returns an array [ε1, ε2] of principal strains.
        """
        ex = epsilon[0]
        ey = epsilon[1]
        exy = epsilon[2]
        strain_matrix = np.array([[ex, exy],
                                  [exy, ey]])
        eigenvalues, _ = np.linalg.eig(strain_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        epsilon1, epsilon2 = eigenvalues[sorted_idx]
        return np.array([epsilon1, epsilon2])
    
    def get_element_internal_forces(self, u):
        """
        Computes internal nodal forces for the element (element restoring force).
        
        Args:
            u (np.ndarray): Global displacement vector.

        Returns:
            fe_int (np.ndarray): Internal force vector (length 12).
        """
        ue = self.get_element_displacements(u)
        fe_int = self.kg @ ue
        return fe_int
    
    def get_results(self, u):
        """
        Computes and returns all element results:
            - Displacement (element nodal displacement vector)
            - Strain (at eval point)
            - Stress (at eval point)
            - Principal Stress (at eval point)
            - Principal Strain (at eval point)
            - Internal Forces (element internal force vector)
        """
        sigma_e, epsilon_e, ue = self.get_element_stress(u)
        fe_int = self.get_element_internal_forces(u)
        principal_stress = self.calculate_principal_stress(sigma_e)
        principal_strain = self.calculate_principal_strain(epsilon_e)
        results = {
            'displacement': ue,
            'strain': epsilon_e,
            'stress': sigma_e,
            'principal_stress': principal_stress,
            'principal_strain': principal_strain,
            'internal_forces': fe_int
        }
        return results
    
    def plotGeometry(self, ax=None, text=False, nodes=True, nodeLabels=True, facecolor='lightgray', edgecolor='k', alpha=0.5):
        """
        Plots the triangular element geometry using Matplotlib.
        """
        if ax is None:
            fig, ax = plt.subplots()
        coords = self.get_xy_matrix()  # (6,2) for LST element
        corner_coords = coords[[0, 1, 2]]  # use only the 3 corner node coordinates
        # Create and add the triangular patch
        polygon = patches.Polygon(corner_coords, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        ax.add_patch(polygon)
        # Plot nodes if requested
        if nodes or nodeLabels:
            for node in self.nodes:
                node.plotGeometry(ax, text=nodeLabels)
        # Show element number at centroid if text label is requested
        if text:
            x_c, y_c = self.get_centroid()
            ax.text(x_c, y_c, f'{self.element_tag}', fontsize=12, ha='center', va='center')
        return ax
    
    def printSummary(self):
        """
        Prints a detailed summary of the LST element.
        """
        print(f'-------------------------------------------------------------')
        print(f"LST Element {self.element_tag}")
        print(f"Type: {self.type}")
        print(f"Nodes: {[node.name for node in self.nodes]}")
        coords = self.get_xy_matrix()
        for i, node in enumerate(self.nodes):
            print(f"  Node {node.name}: ({coords[i,0]:.3f}, {coords[i,1]:.3f})")
        print(f"Thickness: {self.section.thickness}")
        print(f"Area: {self.area:.4f}")
        print(f"Element DoF indices: {self.idx}")
        if self.load_direction is not None:
            print(f"Body force direction: {self.load_direction}")
        else:
            print(f"Body force direction: None")
        print(f"\nStiffness matrix (local):\n{self.kg}")
        print(f'-------------------------------------------------------------\n')
