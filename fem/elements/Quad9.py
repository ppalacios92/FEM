"""
Description:
    Este archivo contiene la clase Quad2D para el modulo de elementos finitos.

Date:
    2024-06-12
"""
__author__ = "Nicolás Mora Bowen"
__version__ = "1.1.0"

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
import matplotlib.patches as     patches

class Quad9:
    """
    Quad9 is a class that represents a second order 2D quadrilateral finite element for structural analysis.
    
    Attributes:
        elementTag (int): Identifier for the element.
        node_list (list): Nodes list defining the quadrilateral element. The list is expected to be properly ordered.
        thickness (float): Thickness of the element.
        material (Material): Material properties of the element.
        type (str): Type of analysis ('planeStress' or 'planeStrain').
        samplingPoints (int): Number of sampling points for numerical integration.
        load_direction (list): Direction of the body force applied to the element.
        nodes (ndarray): Array of node coordinates.
        nodes_idx (ndarray): Array of node indices.
        _x, _y (ndarray): Coordinates of the nodes.
        C (ndarray): Constitutive matrix of the material.
        Kg (ndarray): Global stiffness matrix of the element.
        A (float): Area of the element.
        index (ndarray): Indices for the degrees of freedom.
        F_fe_global (ndarray): Global force vector for body forces.
    """
    
    def __init__(self, elementTag, node_list, membrane, type='planeStress', samplingPoints=5, load_direction=None, eval_points=[0,0], nDof=2):
        """
        Initializes the Quad2D element with the given nodes, material, and other properties.
        
        Args:
            elementTag (int): Identifier for the element.
            node_list (list): Nodes list defining the quadrilateral element. The list is expected to be properly ordered.
            membrane (Membrane): Membrane object containing thickness and material.
            type (str): Type of analysis ('planeStress' or 'planeStrain'). Default is 'planeStress'.
            samplingPoints (int): Number of sampling points for numerical integration. Default is 3.
            load_direction (list): Direction of the body force applied to the element. Default is None.
            load_direction (list): List of points where stresses are to be evaluated. Default is [0,0].
        """
        
        # Number of elements validation
        if len(node_list) != 9:
            raise ValueError("node_list must contain exactly 9 nodes.")
        
        self.nDof=nDof
        self.node_list=node_list
        self.nodes = node_list
        self.element_tag = elementTag
        self.thickness = membrane.thickness
        self.material = membrane.material
        self.type = type
        self.samplingPoints = samplingPoints
        self.load_direction = load_direction
        self.eval_points=eval_points    
        self.section=membrane
        
        self.xy=self.get_xy_matrix()
        self.C =self.material.get_Emat(type)
        
        self._initialize_load_direction()
        
        self.kg, self.area, self.F_fe_global= self.calculate_K0()
        self.idx = self.calculate_indices()
        
    def __str__(self):
        # node_name_list=[node.name for node in self.node_list]
        # return f'Quad9 {node_name_list}'
        return f"Quad9 Element {self.element_tag}: Nodes {[node.name for node in self.nodes]}"
    
    def __repr__(self):
        return self.__str__()
    
    def _initialize_load_direction(self):
        """
        Initializes the direction of the body force. If not specified, defaults to [0, 0].
        """
        if self.load_direction is None:
            self.load_direction = [0, 0]
    
    def get_xy_matrix(self):
        """
        Creates the list of node coordinates and their indices.
        
        Returns:
            nodes (ndarray): Array of node coordinates.
            nodes_idx (ndarray): Array of node indices.
        """
        # xy=np.array([node.coord for node in self.node_list])
        xy = np.array([node.coordenadas for node in self.node_list])

        
        return xy
    
    
    
    def calculate_interpolation_functions(self, zeta, eta):
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
    
    def transform_to_physical(self,zeta, eta):
        
        N,_=self.calculate_interpolation_functions(zeta, eta)
        nDof=self.nDof
        nDof_element=len(self.node_list)*nDof
        
        vector_coordenadas=np.zeros((nDof_element,1))
        vector_coordenadas[0::2,0]=self.nodes[:,0]
        vector_coordenadas[1::2,0]=self.nodes[:,1]
        
        coordenadas_cartesianas=np.dot(N,vector_coordenadas)
        return coordenadas_cartesianas
    
    def calculate_B_matrix(self,zeta,eta):
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
        
        N, dNnatural = self.calculate_interpolation_functions(zeta, eta)
        
        # J=PX
        J=np.dot(dNnatural, xy)
        J_det = np.linalg.det(J)
        
        # Si el determinante es menor a zero la forma del elemento es inadecuada
        if J_det < 0:
            print('Jacobiano Negativo!')
        
        # Derivada de N con respecto a x y y
        # dNnatural = J x dNcartesian        
        dNcartesian=np.linalg.solve(J,dNnatural)
        
        B=np.zeros((3,2*len(xy)))
        B[0, 0::2] = dNcartesian[0, :]
        B[1, 1::2] = dNcartesian[1, :]
        B[2, 0::2] = dNcartesian[1, :]
        B[2, 1::2] = dNcartesian[0, :]
        
        return B, J, J_det, N
    
    def calculate_K0(self):
        """
        Calculates the initial stiffness matrix and area of the element.
        
        Returns:
            Ke (ndarray): Stiffness matrix of the element.
            A (float): Area of the element.
            B (ndarray): Strain Displacement matrix.
        """
        nDof=self.nDof
        nDof_element=len(self.node_list)*nDof
        
        C=self.C
        sampling_points = self.samplingPoints
        roots, weights = roots_legendre(sampling_points)
        t = self.thickness
        b=np.array(self.load_direction)
        b=b.reshape(-1, 1)
        
        # Calculo de la matriz de rigidez y vector de fuerzas
        A = 0
        Ke = np.zeros((nDof_element, nDof_element))
        fe = np.zeros((nDof_element, 1))

        for r, weight_r in zip(roots, weights):
            for s, weight_s in zip(roots, weights):
                
                B, _, J_det, N = self.calculate_B_matrix(r,s)
                A += weight_r * weight_s * np.abs(J_det)
                #print(A)
                Ke += weight_r * weight_s * t * B.T @ C @ B * J_det
                fe += weight_r * weight_s  * N.T @ b * J_det
        
        gamma=self.material.rho
        fe=fe*(t*gamma)
        fe=fe.flatten()
        return Ke, A, fe
        
    def calculate_Ke_difference(self, sampling_point_i, sampling_point_j):
        """
        Calculates the percentage difference between the stiffness matrices for two different sampling points.
        
        Args:
            sampling_point_i (int): Number of sampling points for the first matrix.
            sampling_point_j (int): Number of sampling points for the second matrix.
        
        Returns:
            delta_i_j (ndarray): Percentage difference between the stiffness matrices.
            max_diff (float): Maximum percentage difference.
        """
        Ke_i = self.create_stiffness_matrix(sampling_point_i)
        Ke_j = self.create_stiffness_matrix(sampling_point_j)
        delta_i_j = np.round(np.abs(np.divide(Ke_i - Ke_j, Ke_i)) * 100, 2)
        max_diff = np.max(delta_i_j)
        return delta_i_j, max_diff
    
    def calculate_indices(self):
        """
        Calculates the indices of the degrees of freedom for the element.
        
        Returns:
            index (ndarray): Indices of the degrees of freedom.
        """
        index=np.hstack([node.idx for node in self.node_list])

        
        return index
    
    def get_element_displacements(self, u):
        """
        Extracts the displacements of the element from the global displacement vector.
        
        Args:
            u (ndarray): Global displacement vector.
        
        Returns:
            ue (ndarray): Displacement vector of the element.
        """
        index = self.idx
        ue = u[index]
        return ue
    
    def get_element_strains(self, u):
        """
        Calculates the strains in the element.
        
        Args:
            u (ndarray): Global displacement vector.
        
        Returns:
            epsilon_e (ndarray): Strain vector of the element.
            ue (ndarray): Displacement vector of the element.
        """
        ue = self.get_element_displacements(u)
        eval_point=self.eval_points
        B, _, _, _ = self.calculate_B_matrix(eval_point[0],eval_point[1])
        epsilon_e = B @ ue
        return epsilon_e, ue
    
    def get_element_stress(self, u):
        """
        Calculates the stresses in the element.
        
        Args:
            u (ndarray): Global displacement vector.
        
        Returns:
            sigma_e (ndarray): Stress vector of the element.
            epsilon_e (ndarray): Strain vector of the element.
            ue (ndarray): Displacement vector of the element.
        """
    
        epsilon_e, ue = self.get_element_strains(u)
        sigma_e = self.section.material.get_Emat(self.type) @ epsilon_e
        return sigma_e, epsilon_e, ue
    

    
    def set_results(self, stress, strain, displacement, principal_stress, principal_strain):
        """
        Sets the results of the analysis for the element.
        
        Args:
            stress (ndarray): Stress vector of the element.
            strain (ndarray): Strain vector of the element.
            displacement (ndarray): Displacement vector of the element.
            principal_stress (ndarray): Principal stresses of the element.
            principal_strain (ndarray): Principal strains of the element.
        """
        self.sigma = stress
        self.epsilon = strain
        self.displacement = displacement
        self.principal_stress = principal_stress
        self.principal_strain = principal_strain
        
    def calculate_principal_stress(self, sigma):
        """
        Calculates the principal stresses from the stress tensor.
        
        Args:
            sigma (ndarray): Stress tensor of the element.
        
        Returns:
            principal_stress (ndarray): Principal stresses of the element.
        """
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
        """
        Calculates the principal strains from the strain tensor.
        
        Args:
            epsilon (ndarray): Strain tensor of the element.
        
        Returns:
            principal_strain (ndarray): Principal strains of the element.
        """
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
    
    def element_visualization(self, offset=0):
        """
        Visualizes the quadrilateral element.
        
        Args:
            offset (float): Offset for the text labels. Default is 0.
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')

        # Plot nodes
        for n, node in enumerate(self.nodes):
            ax.plot(node[0], node[1], 'ko', ms=6)
            label = str(self.node_list[n].name)
            ax.text(node[0] + offset, node[1] + offset, label, fontsize=10)
        
        # Sampling points for Gaussian quadrature
        sampling_points = self.samplingPoints  # Ensure this is set in the class constructor
        roots, _ = roots_legendre(sampling_points)
        
        # Transform sampling points to physical coordinates
        for r in roots:
            for s in roots:
                coord_cartesianas = self.transform_to_physical(r, s)
                x_physical = coord_cartesianas[0]
                y_physical = coord_cartesianas[1]
                ax.plot(x_physical, y_physical, 'ro', ms=6)  # Plot sampling points in red for distinction

        # Plot the element as a polygon using all nodes
        ax.add_patch(patches.Polygon(xy=self.nodes, edgecolor='black', facecolor='grey', alpha=0.30))
        ax.set_ylabel('Distance [m]')
        ax.set_xlabel('Distance [m]')
        ax.set_title('2D Element')
        ax.grid(True)
        
        plt.show()

    def get_centroid(self):
        """
        Computes the centroid of the Quad9 element using the average of the 4 corner nodes.

        Returns:
            x_c, y_c (float, float): Coordinates of the centroid.
        """
        xy = self.get_xy_matrix()  # ⬅️ asegúrate de usar coordenadas, no nodos
        x_c = np.mean(xy[[0, 2, 6, 8], 0])  # típicamente los nodos de esquina en Quad9
        y_c = np.mean(xy[[0, 2, 6, 8], 1])
        return float(x_c), float(y_c)



    def plotGeometry(self, ax=None, text=False, nodes=True, nodeLabels=False,
                    facecolor='lightgray', edgecolor='k', alpha=0.5):
        """
        Plots the geometry of the Quad9 element in a given matplotlib axis.

        Args:
            ax (matplotlib axis, optional): Existing matplotlib axis. If None, a new one is created.
            text (bool): Whether to display the element tag at its centroid.
            nodes (bool): Whether to plot the nodes.
            nodeLabels (bool): Whether to label the nodes with their names.
            facecolor (str): Fill color of the element.
            edgecolor (str): Color of the element border.
            alpha (float): Transparency of the fill.

        Returns:
            ax (matplotlib axis): The axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()

        # Coordenadas de los nodos
        coords = self.get_xy_matrix()


        # Crear y agregar el parche del contorno con los 4 nodos de borde
        # El orden esperado es: 0-1-2-3 vértices, 4-5-6-7-8 internos (depende del mallado)
        polygon = patches.Polygon(coords[:4], closed=True, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
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