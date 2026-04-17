import numpy as np
import matplotlib.pyplot as plt

class Material:
    def __init__(self, name, E, nu, rho, printSummary=False):
        self.name = name
        self.E = E
        self.nu = nu
        self.rho = rho
        
        if printSummary:
            self.printSummary()

    def __str__(self):
        return f"{self.name}"
    
    def __repr__(self):
        return self.__str__()
    
    def get_Emat(self, type: str):
        if type == 'planeStress':
            factor = self.E / (1 - self.nu ** 2)
            Emat = np.array([
                [factor, self.nu * factor, 0],
                [self.nu * factor, factor, 0],
                [0, 0, (1 - self.nu) * factor / 2]
            ])
        
        elif type == 'planeStrain':
            factor = self.E / ((1 + self.nu) * (1 - 2 * self.nu))
            Emat = factor * np.array([
                [1 - self.nu, self.nu, 0],
                [self.nu, 1 - self.nu, 0],
                [0, 0, (1 - 2 * self.nu) / 2]
            ])

        elif type == 'frame':
            Emat = self.E  # 

        elif type == 'solid':
            # 3D linear isotropic elasticity matrix (6x6)
            lmda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            mu = self.E / (2 * (1 + self.nu))
            Emat = np.array([
                [lmda + 2*mu, lmda, lmda, 0, 0, 0],
                [lmda, lmda + 2*mu, lmda, 0, 0, 0],
                [lmda, lmda, lmda + 2*mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu]
            ])
        else:
            raise ValueError(f"Unknown Material: {type}")
        
        return Emat

    def printSummary(self):
        print(f'-------------------------------------------------------------')
        print(f"Material properties:")
        print(f"Name: {self.name}")
        print(f"E: {self.E:.2e}")
        print(f"nu: {self.nu:.3f}")
        print(f"rho: {self.rho:.1f}")
        print(f'-------------------------------------------------------------\n')
