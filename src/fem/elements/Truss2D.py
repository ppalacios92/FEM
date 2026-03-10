import numpy as np
import matplotlib.pyplot as plt
from fem.core.parameters import globalParameters

class Truss2D:
    def __init__(self, coord_inicial, coord_final, E, A, printSummary=True):
        global globalParameters
        
        self.coord_inicial = coord_inicial
        self.coord_final = coord_final
        self.E = E
        self.A = A
        
        # Geometry properties
        self.longitud, self.angle, self.angle_degrees=self._geometria()
        
        # Basic system
        self.kb=self.stiffness_matrix_basic()
        self.Tbl=self.basicLocalTransformation()
        
        # Local system
        self.kl=self.localStiffnessMatrix()
        self.Tlg=self.localGlobalTransformation()
        
        # Global system
        self.kg=self.globalStiffnessMatrix()
        
        # Element indices
        self.idx, self.restrain=self._elementIndices()
        
        if printSummary:
            self.printSummary()
        
    def __str__(self):
        return f"Truss2D: {self.coord_inicial.name} -> {self.coord_final.name}"
    
    def _geometria(self):
        vector=self.coord_final.coordenadas-self.coord_inicial.coordenadas
        longitud=np.linalg.norm(vector)
        angle=np.arctan2(vector[1],vector[0])
        angle_degrees=np.degrees(angle)
        
        return longitud, angle, angle_degrees
    
    def stiffness_matrix_basic(self):
        kb=np.array([[self.A*self.E/self.longitud]])
        return kb
        
    def basicLocalTransformation(self):
        Tbl=np.array([[-1,1]])
        return Tbl
        
    def localStiffnessMatrix(self):
        kl=self.Tbl.T@self.kb@self.Tbl
        return kl
        
    def localGlobalTransformation(self):
        c=np.cos(self.angle)
        s=np.sin(self.angle)
        Tlg=np.array([[c,s,0,0],
                      [0,0,c,s]])
        return Tlg
    
    def globalStiffnessMatrix(self):
        kg=self.Tlg.T@self.kl@self.Tlg
        return kg
    
    def _elementIndices(self):
        idx=np.concatenate([self.coord_inicial.idx,self.coord_final.idx])
        restraint=np.concatenate([self.coord_inicial.restrain,self.coord_final.restrain])
        return idx, restraint
    
    def _extractDisplacements(self, u):
        # Given the complete displacement vector, extract the relevant displacements
        ue=u[self.idx]
        return ue
    
    def _calculateLocalForces(self, u):
        ue=self._extractDisplacements(u)
        ue_local=self.Tlg@ue
        fe_local=self.kl@ue_local
        
        return fe_local, ue_local
    
    def _calculateBasicForces(self, u):
        _, ue_local=self._calculateLocalForces(u)
        ue_basic=self.Tbl@ue_local
        fe_basic=self.kb@ue_basic
        
        return fe_basic, ue_basic
    
    def forceRecovery(self, u, printSummary=True):
        
        fe_basic, ue_basic=self._calculateBasicForces(u)
        fe_local, ue_local=self._calculateLocalForces(u)
        ue_global=u[self.idx]
        fe_global=self.kg@ue_global
        
        self.fe_basic=fe_basic
        self.fe_local=fe_local
        self.fe_global=fe_global
        
        self.ue_global=ue_global

        self.ue_local=ue_local
        self.ue_basic=ue_basic
        
        print(f'-------------------------------------------------------------')
        print(f'Forces for element Truss2D: {self.coord_inicial.name} -> {self.coord_final.name}')
        print(f'Basic forces: {fe_basic}')
        print(f'Local forces: {fe_local}')
        print(f'Global forces: {fe_global}')
        print(f'\n')
        print(f'Basic displacements: {ue_basic}')
        print(f'Local displacements: {ue_local}')
        print(f'Global displacements: {ue_global}')
        print(f'-------------------------------------------------------------\n')
        
    def plotGeometry(self, ax=None, text=False, nodes=True, nodeLabels=False):
        if ax is None:
            fig, ax = plt.subplots()
        
        x=np.array([self.coord_inicial.coordenadas[0],self.coord_final.coordenadas[0]])
        y=np.array([self.coord_inicial.coordenadas[1],self.coord_final.coordenadas[1]])
        
        ax.plot(x, y, 'k-')
        
        if nodes:
            self.coord_inicial.plotGeometry(ax, text=nodeLabels)
            self.coord_final.plotGeometry(ax, text=nodeLabels)
        
        if text:
            ax.text(np.mean(x), np.mean(y), f'{self.coord_inicial.name}->{self.coord_final.name}', fontsize=12)
        
        return ax
    
    def printSummary(self):
        print(f'-------------------------------------------------------------')
        print(f"Truss2D: {self.coord_inicial.name} -> {self.coord_final.name}")
        print(f"Longitud: {self.longitud}")
        print(f"Ángulo: {self.angle_degrees}")
        print(f"Element indices: {self.idx}")
        print(f"Restrain: {self.restrain}")
        
        print(f'\n')
        print(f"Stiffness matrix basic:\n {self.kb}")
        print(f"Basic local transformation:\n {self.Tbl}")
        print(f"Local stiffness matrix:\n {self.kl}")
        print(f"Local global transformation:\n {self.Tlg}")
        print(f"Global stiffness matrix:\n {self.kg}")
        print(f'-------------------------------------------------------------\n')
        