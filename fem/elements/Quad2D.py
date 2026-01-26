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
import matplotlib.patches as patches
import sympy as sp
x, y = sp.symbols('x y')

class Quad2D:
    """
    Quad2D is a class that represents a 2D quadrilateral finite element for structural analysis.
    
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
    
    def __init__(self, elementTag, node_list, membrane, type='planeStress', samplingPoints=3, load_direction=None, eval_points=[0,0], nDof=2):
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
        if len(node_list) != 4:
            raise ValueError("node_list must contain exactly 4 nodes.")
        
        self.nDof=nDof
        self.node_list=node_list
        self.elementTag = elementTag
        self.thickness = membrane.thickness
        self.material = membrane.material
        self.type = type
        self.samplingPoints = samplingPoints
        self.load_direction = load_direction
        self.eval_points=eval_points
        
        self.xy = self.calculate_xy()
        self.C = membrane.material.Emat
        
        self._initialize_load_direction()
        
        self.Kg, self.A, self.F_fe_global = self.calculate_K0()
        self.index = self.calculate_indices()
        
    def __str__(self):
        node_name_list=[node.name for node in self.node_list]
        return f'Quad4 {node_name_list}'
    
    def _initialize_load_direction(self):
        """
        Initializes the direction of the body force. If not specified, defaults to [0, 0].
        """
        if self.load_direction is None:
            self.load_direction = [0, 0]
    
    def calculate_xy(self):
        """
        Creates the list of node coordinates and their indices.
        
        Returns:
            nodes (ndarray): Array of node coordinates.
            nodes_idx (ndarray): Array of node indices.
        """
        xy=np.array([node.coord for node in self.node_list])
        
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
        
        # Funciones de interpolacion N(zeta, eta)
        N1=((1-eta)*(1-zeta))/4
        N2=((1-eta)*(1+zeta))/4
        N3=((1+eta)*(1+zeta))/4
        N4=((1+eta)*(1-zeta))/4
        
        # Derivadas parciales de las funciones de interpolacion en funcion de las coordenadas naturales
        dN1dzeta = -0.25 * (1 - eta)
        dN2dzeta =  0.25 * (1 - eta)
        dN3dzeta =  0.25 * (1 + eta)
        dN4dzeta = -0.25 * (1 + eta)
        
        dN1deta = -0.25 * (1 - zeta)
        dN2deta = -0.25 * (1 + zeta)
        dN3deta =  0.25 * (1 + zeta)
        dN4deta =  0.25 * (1 - zeta)
        
        # Matriz de funciones de interpolacion
        N=np.array([
            [N1,0,N2,0,N3,0,N4,0],
            [0,N1,0,N2,0,N3,0,N4]
        ])

        # Derivada de N con respecto a eta y zeta
        dNnatural=np.array([
            [dN1dzeta, dN2dzeta, dN3dzeta, dN4dzeta],
            [dN1deta, dN2deta, dN3deta, dN4deta]
        ])
        
        return N, dNnatural
    
    def transform_to_physical(self,zeta,eta):
        
        N,_=self.calculate_interpolation_functions(zeta, eta)
        nDof=self.nDof
        nDof_element=len(self.node_list)*nDof
        
        vector_coordenadas=np.zeros((nDof_element,1))
        vector_coordenadas[0::2,0]=self.xy[:,0]
        vector_coordenadas[1::2,0]=self.xy[:,1]
        
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
        
        # Funciones de interpolacion N(zeta, eta)
        N, dNnatural = self.calculate_interpolation_functions(zeta, eta)
        
        # J=PX
        J=np.dot(dNnatural, xy)
        #print(f'La Jacobiana es: {np.round(J,2)}')
        J_det = np.linalg.det(J)
        #print(f'El determinante de la Jacobiana es {np.round(J_det,3)}')
        
        # Si el determinante es menor a zero la forma del elemento es inadecuada
        if J_det < 0:
            raise ValueError('Jacobiano Negativo!')
        
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
    
    def calculate_K0(self):
        """
        Calculates the initial stiffness matrix and area of the element.
        
        Returns:
            Ke (ndarray): Stiffness matrix of the element.
            A (float): Area of the element.
            B (ndarray): Strain Displacement matrix.
        """

        sampling_points = self.samplingPoints
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
                
                B, _, J_det, N = self.calculate_B_matrix(r,s)
                #print(f'Para r={r} y s={s} el valor de Jdet es={J_det}')
                A += weight_r * weight_s * np.abs(J_det)
                Ke += weight_r * weight_s * t * B.T @ self.C @ B * J_det
                fe += weight_r * weight_s  * N.T @ b * J_det
        
        gamma=self.material.gamma
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
        index=np.hstack([node.index for node in self.node_list])
        return index
    
    def get_element_displacements(self, u):
        """
        Extracts the displacements of the element from the global displacement vector.
        
        Args:
            u (ndarray): Global displacement vector.
        
        Returns:
            ue (ndarray): Displacement vector of the element.
        """
        index = self.index
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
        sigma_e = self.material.Emat @ epsilon_e
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
        sx = sigma[0][0]
        sy = sigma[1][0]
        sxy = sigma[2][0]
        
        stress_matrix = np.array([[sx, sxy], [sxy, sy]])
        eigenvalues, _ = np.linalg.eig(stress_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        sigma1, sigma2 = eigenvalues
        
        return np.array([[sigma1], [sigma2]])
    
    def calculate_principal_strain(self, epsilon):
        """
        Calculates the principal strains from the strain tensor.
        
        Args:
            epsilon (ndarray): Strain tensor of the element.
        
        Returns:
            principal_strain (ndarray): Principal strains of the element.
        """
        ex = epsilon[0][0]
        ey = epsilon[1][0]
        exy = epsilon[2][0]
        
        strain_matrix = np.array([[ex, exy], [exy, ey]])
        eigenvalues, _ = np.linalg.eig(strain_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        epsilon1, epsilon2 = eigenvalues
        
        return np.array([[epsilon1], [epsilon2]])
    
    def element_visualization(self, offset=0):
        # SE TIENE QUE REVISAR!!!!
        """
        Visualizes the quadrilateral element.
        
        Args:
            offset (float): Offset for the text labels. Default is 0.
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')

        for n, node in enumerate(self.xy):
            ax.plot(node[0], node[1], 'ko', ms=6)
            label = f'{self.node_list[n].name}'
            ax.text(node[0] + offset, node[1] + offset, label, fontsize=10)

        x = self.xy[:, 0]
        y = self.xy[:, 1]

        polygon = patches.Polygon(xy=list(zip(x, y)), edgecolor='black', facecolor='grey', alpha=0.30)
        ax.add_patch(polygon)
        ax.set_ylabel('Distance [m]')
        ax.set_xlabel('Distance [m]')
        ax.set_title('2D Element')
        ax.grid(True)
        
        plt.show()
