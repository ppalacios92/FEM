import numpy as np
import matplotlib.pyplot as plt
from fem.core.parameters import globalParameters

class Node:
    def __init__(self, name, coordenadas, nodalLoad=None, restrain=None, printSummary=False):
        # This is a dict defined in the main file to store the global parameters
        global Global
        
        self.name = name
        self.coordenadas = np.array(coordenadas)
        
        # Get the indices for the node, python count from 0
        self.idx=self._indices()
        
        # Define the nodal load during the instantiation
        if nodalLoad is not None:
            self.nodalLoad = np.array(nodalLoad)
        else:
            self.nodalLoad = np.zeros(globalParameters['nDoF'])
            
        # Define the boundary conditions
        
        if restrain is not None:
            if isinstance(restrain, list) and len(restrain)==globalParameters['nDoF']:
                self.restrain = np.array(restrain)
            else:
                raise ValueError('Las restricciones deben ser una lista y tener el tamaño correcto')
        else:
            self.restrain=np.full(globalParameters['nDoF'],'f')
            
        if printSummary:
            self.printSummary()
    
    def __str__(self):
        return "Node %d at %s" % (self.name, self.coordenadas)
    
    def __repr__(self):
        return self.__str__()
    
    def set_restrain(self, restrain:list):
        restrain = np.array(restrain)
        
        if restrain.shape[0] != globalParameters['nDoF']:
            raise ValueError('Las restricciones deben tener el tamaño correcto')
        
        if not np.all(np.isin(restrain, ['f', 'r'])):
            raise ValueError("Las restricciones solo pueden ser 'f' (free) o 'r' (restrained)")
        
        self.restrain = restrain

        
    def set_nodalLoad(self, nodalLoad):
        if isinstance(nodalLoad, list) and len(nodalLoad)==globalParameters['nDoF']:
            self.nodalLoad = np.array(nodalLoad)
        else:
            raise ValueError('La carga nodal debe ser una lista y tener el tamaño correcto')
        
    def _indices(self):
        idx = np.arange(globalParameters['nDoF']) + (globalParameters['nDoF'] * (self.name-1)).astype(int)
        return idx

    def plotGeometry(self, ax=None, text=False):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.coordenadas[0], self.coordenadas[1], 'ro')
        if text:
            ax.text(self.coordenadas[0], self.coordenadas[1], f'{self.name}', fontsize=8)
        return ax
    
    def printSummary(self):
        print(f'--------------------------------------------')
        print("Node %d at %s" % (self.name, self.coordenadas))
        print("Indices: ", self.idx)
        print("Nodal Load: ", self.nodalLoad)
        print("Restrain: ", self.restrain)
        print(f'--------------------------------------------\n')
         

    
