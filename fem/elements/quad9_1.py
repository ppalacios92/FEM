import numpy as np
import matplotlib.pyplot as plt
from fem.core.parameters import globalParameters
import matplotlib.patches as patches
from scipy.special import roots_legendre


class Quad9:
    def __init__(self, 
                 element_tag: int, 
                 node_list: list, 
                 section: object, 
                 load_direction=None,
                 type: str = 'planeStress',
                 sampling_points=5,
                 eval_points=[0,0],
                 print_summary=False,
                 nDof=2):
        """
        Initialize the Quad element with nodes, section properties, and optional load direction.

        Args:
            element_tag (int): Unique identifier for the element.
            node_list (list): List of three nodes defining the Quad9 element.
            section (object): Section object containing material and thickness.
            load_direction (list, optional): List [Cx, Cy] for gravitational load direction.
            type (str): 'planeStress' or 'planeStrain'. Default is 'planeStress'.
        """
        if len(node_list) != 9:
            raise ValueError("Quad9 elements must have exactly 9 nodes.")
        
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
        self.xy=self.get_xy_matrix()
        self.thickness = self.section.thickness
        self.material = self.section.material
        self.C=self.material.get_Emat(type)
        
        # Element calculations
        self.idx=self.calculate_indices()
        self.kg, self.area, self.F_fe_global = self.get_stiffness_matrix()
        
        if print_summary is True:
            self.printSummary()

    def __str__(self):
        return f"Quad9 Element {self.element_tag}: Nodes {[node.name for node in self.nodes]}"
    
    def __repr__(self):
        return self.__str__()

    def _initialize_load_direction(self):
        """Initializes the load direction for the Quad9 element."""
        if self.load_direction is None:
            self.load_direction = [0, 0]
    
    def calculate_indices(self):
        """Returns the global DoF indices for the Quad9 element."""
        idx = np.hstack([node.idx for node in self.nodes]).astype(int)
        return idx

    def get_xy_matrix(self):
        """
        Returns the matrix of nodal coordinates of the Quad9 element.

        Returns:
            X (np.ndarray): 4x2 array with node coordinates [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        """
        xy = np.array([node.coordenadas for node in self.nodes])
        return xy
    
    @staticmethod
    def get_interpolation_matrix(zeta, eta):
        """
        Calculates the interpolation functions and their partial derivatives for a quadrilateral element
        in natural coordinates (zeta, eta).

        Args:
            zeta (float): Natural coordinate corresponding to the zeta axis.
            eta (float): Natural coordinate corresponding to the eta axis.

        Returns:
            N (ndarray): Interpolation function matrix for the given natural coordinates.
            dNnatural (ndarray): Matrix of partial derivatives of the interpolation functions with respect to zeta and eta (2x4).

        """
        
        N1 = (1/4) * (zeta**2 - zeta) * (eta**2 - eta)
        N2 = (1/4) * (zeta**2 + zeta) * (eta**2 - eta)
        N3 = (1/4) * (zeta**2 + zeta) * (eta**2 + eta)
        N4 = (1/4) * (zeta**2 - zeta) * (eta**2 + eta)
        N5 = (1/2) * (1 - zeta**2) * (eta**2 - eta)
        N6 = (1/2) * (zeta**2 + zeta) * (1 - eta**2)
        N7 = (1/2) * (1 - zeta**2) * (eta**2 + eta)
        N8 = (1/2) * (zeta**2 - zeta) * (1 - eta**2)
        N9 = (1 - zeta**2) * (1 - eta**2)
        
        # Matriz de funciones de interpolacion
        N=np.array([
            [N1,0,N2,0,N3,0,N4,0,N5,0,N6,0,N7,0,N8,0,N9,0],
            [0,N1,0,N2,0,N3,0,N4,0,N5,0,N6,0,N7,0,N8,0,N9]
        ])
        
        # Partial derivatives with respect to zeta
        dN1dzeta = (1/4) * (2*zeta - 1) * (eta**2 - eta)
        dN2dzeta = (1/4) * (2*zeta + 1) * (eta**2 - eta)
        dN3dzeta = (1/4) * (2*zeta + 1) * (eta**2 + eta)
        dN4dzeta = (1/4) * (2*zeta - 1) * (eta**2 + eta)
        dN5dzeta = (1/2) * (-2*zeta) * (eta**2 - eta)
        dN6dzeta = (1/2) * (2*zeta + 1) * (1 - eta**2)
        dN7dzeta = (1/2) * (-2*zeta) * (eta**2 + eta)
        dN8dzeta = (1/2) * (2*zeta - 1) * (1 - eta**2)
        dN9dzeta = -2*zeta * (1 - eta**2)
        
        # Partial derivatives with respect to eta
        dN1deta = (1/4) * (zeta**2 - zeta) * (2*eta - 1)
        dN2deta = (1/4) * (zeta**2 + zeta) * (2*eta - 1)
        dN3deta = (1/4) * (zeta**2 + zeta) * (2*eta + 1)
        dN4deta = (1/4) * (zeta**2 - zeta) * (2*eta + 1)
        dN5deta = (1/2) * (1 - zeta**2) * (2*eta - 1)
        dN6deta = (1/2) * (zeta**2 + zeta) * (-2*eta)
        dN7deta = (1/2) * (1 - zeta**2) * (2*eta + 1)
        dN8deta = (1/2) * (zeta**2 - zeta) * (-2*eta)
        dN9deta = (1 - zeta**2) * (-2*eta)
        
        # Derivada de N con respecto a eta y zeta
        dNnatural=np.array([
            [dN1dzeta, dN2dzeta, dN3dzeta, dN4dzeta, dN5dzeta, dN6dzeta, dN7dzeta, dN8dzeta, dN9dzeta],
            [dN1deta, dN2deta, dN3deta, dN4deta, dN5deta, dN6deta, dN7deta, dN8deta, dN9deta]
        ])
        
        return N, dNnatural
    
    def transform_to_physical(self,zeta,eta):
        
        N,_=self.get_interpolation_matrix(zeta, eta)
        nDof=self.nDof
        nDof_element=len(self.node_list)*nDof
        
        vector_coordenadas=np.zeros((nDof_element,1))
        vector_coordenadas[0::2,0]=self.xy[:,0]
        vector_coordenadas[1::2,0]=self.xy[:,1]
        
        coordenadas_cartesianas=np.dot(N,vector_coordenadas)
        return coordenadas_cartesianas
        
    
    def get_B_matrix(self,zeta,eta):
        """
        Method to calculate the strain displacement matrix, the Jacobian and it determinant, and the interpolation matrix
        This values are to be evaluated at each Gaussian point

        Args:
            zeta (float): natural coordinate corresponding to a gausssian point
            eta (float): natural coordinate correponding to a gaussian point

        Raises:
            ValueError: Display error when the Jacobian determinate is less than zero

        Returns:
            B (ndarray): strain displacement matrix
            J (ndarray): Jacobian
            J_det (float): Jacobian determinant
            
        """
        
        # Determinamos la matriz de coordenadas xy
        xy=self.xy
        
        # Funciones de interpolacion N(zeta, eta)
        N, dNnatural = self.get_interpolation_matrix(zeta, eta)
        
        # J=PX
        J=np.dot(dNnatural, xy)
        #print(f'La Jacobiana es: {np.round(J,2)}')
        J_det = np.linalg.det(J)
        #print(f'El determinante de la Jacobiana es {np.round(J_det,3)}')
        
        # Si el determinante es menor a zero la forma del elemento es inadecuada
        # if J_det < 0:
            # raise ValueError('Jacobiano Negativo!')
        
        # Derivada de N con respecto a x y y
        # dNnatural = J x dNcartesian        
        dNcartesian=np.linalg.solve(J,dNnatural)
        
        B=np.zeros((3,2*len(xy)))
        B[0, 0::2] = dNcartesian[0, :]
        B[1, 1::2] = dNcartesian[1, :]
        B[2, 0::2] = dNcartesian[1, :]
        B[2, 1::2] = dNcartesian[0, :]
        
        return B, J, J_det, N
    
    def _get_b_load_direction(self):
        b=np.array(self.load_direction)
        b=b.reshape(-1,1)
        return b
    
    def get_stiffness_matrix(self):
        """
        Calculates the initial stiffness matrix and area of the element.
        
        Returns:
            Ke (ndarray): Stiffness matrix of the element.
            A (float): Area of the element.
            B (ndarray): Strain Displacement matrix.
        """

        sampling_points = self.sampling_points
        roots, weights = roots_legendre(sampling_points)
        t = self.thickness
        
        b_loadDirection=self._get_b_load_direction()
        
        # Calculo de la matriz de rigidez y vector de fuerzas
        A = 0
        nDof=self.nDof
        nDof_element=len(self.node_list)*nDof
        Ke = np.zeros((nDof_element, nDof_element))
        fe = np.zeros((nDof_element, 1))

        for r, weight_r in zip(roots, weights):
            for s, weight_s in zip(roots, weights):
            
                b=b_loadDirection
                B, _, J_det, N = self.get_B_matrix(r,s)
                #print(f'Para r={r} y s={s} el valor de Jdet es={J_det}')
                A += weight_r * weight_s * np.abs(J_det)
                Ke += weight_r * weight_s * t * B.T @ self.C @ B * J_det
                fe += weight_r * weight_s  * N.T @ b * J_det
        
        gamma=self.material.rho
        fe=fe*(t*gamma)
        fe=fe.flatten()
        return Ke, A, fe
    
    def get_centroid(self):
        """
        Computes the centroid of the quadrilateral element by averaging nodal coordinates.

        Returns:
            centroid (np.ndarray): (2,) array with centroid coordinates [x, y]
        """
        xy = self.get_xy_matrix()
        centroid = np.mean(xy, axis=0)
        return centroid
    
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
        Computes the strain vector for the element evaluated at self.eval_points.

        Args:
            u (np.ndarray): Global displacement vector

        Returns:
            epsilon_e (np.ndarray): Strain vector (3,)
            ue (np.ndarray): Element displacement vector (8,)
        """
        ue = self.get_element_displacements(u)
        zeta, eta = self.eval_points
        B, _, _, _ = self.get_B_matrix(zeta, eta)
        epsilon_e = B @ ue
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
  
    def plotGeometry(self, ax=None, text=False, nodes=True, nodeLabels=True, facecolor='lightgray', edgecolor='k', alpha=0.5):

        if ax is None:
            fig, ax = plt.subplots()

        coords = self.get_xy_matrix()  # (3,2)
        corner_coords = coords[[0, 1, 2, 3]]  # toma solo los nodos 0,1,2,3
        # Crear y agregar el parche del triángulo
        polygon = patches.Polygon(corner_coords, closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
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
        Prints a detailed summary of the Quad9 element.
        """
        print(f'-------------------------------------------------------------')
        print(f"Quad9 Element {self.element_tag}")
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


