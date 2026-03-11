import numpy as np
import matplotlib.pyplot as plt
from fem.core.parameters import globalParameters
import matplotlib.patches as patches


class CST:
    def __init__(self, 
                 element_tag: int, 
                 node_list: list, 
                 section: object, 
                 load_direction=None,
                 type: str = 'planeStress',
                 print_summary=False):
        """
        Initialize the CST element with nodes, section properties, and optional load direction.

        Args:
            element_tag (int): Unique identifier for the element.
            node_list (list): List of three nodes defining the CST element.
            section (object): Section object containing material and thickness.
            load_direction (list, optional): List [Cx, Cy] for gravitational load direction.
            type (str): 'planeStress' or 'planeStrain'. Default is 'planeStress'.
        """
        if len(node_list) != 3:
            raise ValueError("CST elements must have exactly 3 nodes.")
        
        self.element_tag = element_tag
        self.node_list = node_list
        self.nodes = node_list
        self.section = section
        self.load_direction = load_direction
        self.type = type

        # Initialize the element properties
        self.thickness = self.section.thickness
        self.material  = self.section.material
        self.C         = self.material.get_Emat(self.type)

        self.compute_area()
        self.idx=self.calculate_indices()
        self.kg=self.get_stiffness_matrix()       


        if print_summary is True:
            self.printSummary()

    def __str__(self):
        return f"CST Element {self.element_tag}: Nodes {[node.name for node in self.nodes]}"
    
    def __repr__(self):
        return self.__str__()

    def calculate_indices(self):
        """Returns the global DoF indices for the CST element."""
        idx = np.hstack([node.idx for node in self.nodes]).astype(int)
        return idx

    def get_xy_matrix(self):
        """
        Returns the matrix of nodal coordinates of the CST element.

        Returns:
            X (np.ndarray): 3x2 array with node coordinates [[x1, y1], [x2, y2], [x3, y3]]
        """
        xy = np.array([node.coordinates for node in self.nodes])
        return xy

    def get_centroid(self):
        """
        Computes the centroid of the triangular element using matrix operations.

        Returns:
            centroid (np.ndarray): (2,) array with centroid coordinates [x, y]
        """
        xy = self.get_xy_matrix()
        w = np.ones((1, 3)) / 3
        centroid = w @ xy
        return centroid.flatten()

    def compute_area(self):
        """
        Computes and stores the area of the CST element using determinant-based formula.

        Sets:
            self.area (float): Area of the triangle
        """
        x1, y1 = self.nodes[0].coordinates
        x2, y2 = self.nodes[1].coordinates
        x3, y3 = self.nodes[2].coordinates

        self.area = 0.5 * np.linalg.det(np.array([
            [1, x1, y1],
            [1, x2, y2],
            [1, x3, y3]
        ]))

        # if self.area <= 0:
        #     raise ValueError(f"Element {self.element_tag} has non-positive area: {self.area}")

    def get_interpolation_matrix(self, x: float, y: float):
        """
        Returns the interpolation matrix N at a given point (x, y)
        using matrix operations to compute barycentric coordinates.

        Returns:
            Nmat (2x6 np.array): Interpolation matrix
        """
        coords = self.get_xy_matrix()
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]

        T = np.array([
            [1, x1, y1],
            [1, x2, y2],
            [1, x3, y3]
        ])

        p = np.array([1, x, y])

        try:
            lambdas = np.linalg.solve(T, p)
        except np.linalg.LinAlgError:
            raise ValueError("Singular element (zero area or colinear nodes)")

        Nmat = np.zeros((2, 6))
        Nmat[0, 0::2] = lambdas
        Nmat[1, 1::2] = lambdas

        return Nmat

    def get_B_matrix(self):
        """
        Computes the strain-displacement matrix B for the CST element.

        Returns:
            B (3x6 np.ndarray): Strain-displacement matrix
        """
        x1, y1 = self.nodes[0].coordinates
        x2, y2 = self.nodes[1].coordinates
        x3, y3 = self.nodes[2].coordinates

        b1 = y2 - y3
        b2 = y3 - y1
        b3 = y1 - y2
        c1 = x3 - x2
        c2 = x1 - x3
        c3 = x2 - x1

        B = (1 / (2 * self.area)) * np.array([
            [b1, 0, b2, 0, b3, 0],
            [0, c1, 0, c2, 0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])

        return B

    def get_stiffness_matrix(self):
        """
        Computes the local stiffness matrix for the CST element.

        Returns:
            Ke (6x6 np.array): Local stiffness matrix
        """
        D = self.C
        B = self.get_B_matrix()
        t = self.section.thickness

        Ke = B.T @ D @ B * self.area * t
        return Ke

    def get_body_forces(self):
        """
        Computes the equivalent nodal body force vector using 1-point integration
        at the centroid of the triangular element.

        Returns:
            fb (np.ndarray): 6x1 body force vector (flattened)
        """
        if self.load_direction is None:
            return np.zeros(6)

        b = np.array(self.load_direction).reshape(2,)  # [bx, by]
        centroid = self.get_centroid()
        N = self.get_interpolation_matrix(*centroid)
        t = self.section.thickness

        f_local = (N.T @ b) * self.area * t
        return f_local
    
    def get_element_displacements(self, u):
        """
        Extracts the element displacement vector from global displacements.

        Args:
            u (np.array): Global displacement vector

        Returns:
            ue (np.array): Element displacement vector
        """
        ue = u[self.idx]
        return ue

    def get_element_strains(self, u):
        """
        Computes the strain vector for the element.

        Args:
            u (np.array): Global displacement vector

        Returns:
            epsilon_e (np.array): Strain vector (3x1)
            ue (np.array): Element displacement vector (6x1)
        """
        ue = self.get_element_displacements(u)
        epsilon_e = self.get_B_matrix() @ ue
        return epsilon_e, ue

    def get_element_stress(self, u):
        """
        Computes the stress vector for the element.

        Args:
            u (np.array): Global displacement vector

        Returns:
            sigma_e (np.array): Stress vector (3x1)
            epsilon_e (np.array): Strain vector (3x1)
            ue (np.array): Element displacement vector (6x1)
        """
        epsilon_e, ue = self.get_element_strains(u)
        sigma_e = self.section.material.get_Emat(self.type) @ epsilon_e
        return sigma_e, epsilon_e, ue
    
    def calculate_principal_stress(self, sigma):
        
        sx=sigma[0]
        sy=sigma[1]
        sxy=sigma[2]
        
        stress_matrix = np.array([[sx, sxy],
                                  [sxy, sy]])

        # Diagonalize the stress matrix
        eigenvalues, eigenvectors = np.linalg.eig(stress_matrix)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Principal stresses are the eigenvalues
        sigma1, sigma2 = eigenvalues
        
        return np.array([sigma1,sigma2])
    
    def calculate_principal_strain(self, epsilon):
        
        ex=epsilon[0]
        ey=epsilon[1]
        exy=epsilon[2]
        
        strain_matrix = np.array([[ex, exy],
                                  [exy, ey]])

        # Diagonalize the stress matrix
        eigenvalues, eigenvectors = np.linalg.eig(strain_matrix)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort in descending order
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Principal stresses are the eigenvalues
        epsilon1, epsilon2 = eigenvalues
        
        return np.array([epsilon1,epsilon2])

    def get_element_internal_forces(self, u):
        """
        Computes internal nodal forces for the element.

        Args:
            u (np.array): Global displacement vector

        Returns:
            fe (np.array): Internal force vector (6x1)
        """
        ue = self.get_element_displacements(u)
        fe = self.kg @ ue
        return fe

    def get_results(self, u):
        """
        Computes and stores all element results:
            - Displacement
            - Strain
            - Stress
            - Principal Stress
            - Principal Strain
            - Internal Forces
        """
        sigma_e, epsilon_e, ue = self.get_element_stress(u)
        fe = self.get_element_internal_forces(u)
        principal_stress = self.calculate_principal_stress(sigma_e)
        principal_strain = self.calculate_principal_strain(epsilon_e)

        results = {
            'stress': sigma_e,
            'strain': epsilon_e,
            'displacement': ue,
            'internal_forces': fe,
            'principal_stress': principal_stress,
            'principal_strain': principal_strain
        }

        return results
  
    def plotGeometry(self, ax=None, text=False, nodes=True, nodeLabels=False, facecolor='lightgray', edgecolor='k', alpha=0.5):
        """
        Plots the geometry of the CST element as a shaded triangle.

        Args:
            ax (matplotlib axis, optional): Existing matplotlib axis. If None, a new one is created.
            text (bool): Whether to display the element tag at its centroid.
            nodes (bool): Whether to plot the nodes.
            nodeLabels (bool): Whether to label the nodes with their names.
            facecolor (str): Fill color of the triangle.
            edgecolor (str): Color of the triangle border.
            alpha (float): Transparency of the fill.

        Returns:
            ax (matplotlib axis): The axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        coords = self.get_xy_matrix()  # (3,2)

        # Crear y agregar el parche del triángulo
        polygon = patches.Polygon(coords, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        ax.add_patch(polygon)

        # Dibujar los nodos
        if nodes or nodeLabels:
            for node in self.nodes:
                node.plotGeometry(ax, text=nodeLabels)

        # Mostrar el número del elemento en el centroide
        if text:
            x_c, y_c = self.get_centroid()
            ax.text(x_c, y_c, f'{self.element_tag}', fontsize=12, ha='center', va='center')

        return ax
    
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
            print(f"  Node {node.name}: ({coords[i,0]:.3f}, {coords[i,1]:.3f})")
        
        print(f"Thickness: {self.section.thickness}")
        print(f"Area: {self.area:.4f}")
        print(f"Element DoF indices: {self.calculate_indices()}")
        
        if self.load_direction is not None:
            print(f"Body force direction: {self.load_direction}")
        else:
            print(f"Body force direction: None")
        
        print(f"\nStiffness matrix (local):\n{self.get_stiffness_matrix()}")
        print(f'-------------------------------------------------------------\n')

