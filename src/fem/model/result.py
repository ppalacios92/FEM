"""FEMResult — stores FEM results for a single load step."""

import numpy as np


class FEMResult:
    """
    Stores all results for one load step.

    Parameters
    ----------
    step    : int     Load step index.
    time    : float   Load factor or time associated with this step.
    u       : ndarray (system_nDof,) displacements.
    F       : ndarray (system_nDof,) applied forces.
    R       : ndarray (system_nDof,) reactions.
    u_3d    : ndarray (n_nodes, 3) displacements in gmsh format.
    F_3d    : ndarray (n_nodes, 3) forces in gmsh format.
    R_3d    : ndarray (n_nodes, 3) reactions in gmsh format.
    sigma            : ndarray (n_elements, n_comp) stresses.
    epsilon          : ndarray (n_elements, n_comp) strains.
    vm               : ndarray (n_elements,) von Mises stress.
    sigma_principal  : ndarray (n_elements, 2) principal stresses — 2D only.
    epsilon_principal: ndarray (n_elements, 2) principal strains — 2D only.
    sigma_nodal      : ndarray (n_nodes, n_comp) nodal averaged stresses.
    epsilon_nodal    : ndarray (n_nodes, n_comp) nodal averaged strains.
    vm_nodal         : ndarray (n_nodes,) nodal averaged von Mises.
    """

    def __init__(self, step, time,
                 u, F, R,
                 u_3d, F_3d, R_3d,
                 sigma, epsilon, vm,
                 sigma_principal=None, epsilon_principal=None,
                 sigma_nodal=None, epsilon_nodal=None, vm_nodal=None):

        self.step = step
        self.time = time

        # global vectors
        self.u = u
        self.F = F
        self.R = R

        # gmsh format
        self.u_3d = u_3d
        self.F_3d = F_3d
        self.R_3d = R_3d

        # per element
        self.sigma   = sigma
        self.epsilon = epsilon
        self.vm      = vm
        self.sigma_principal   = sigma_principal
        self.epsilon_principal = epsilon_principal

        # nodal averaged
        self.sigma_nodal   = sigma_nodal
        self.epsilon_nodal = epsilon_nodal
        self.vm_nodal      = vm_nodal

    def save(self, path: str):
        """Save result to HDF5 file. path should end in .h5"""
        import h5py
        with h5py.File(path, 'w') as f:
            f.attrs['step'] = self.step
            f.attrs['time'] = self.time
            for name in ('u', 'F', 'R', 'u_3d', 'F_3d', 'R_3d',
                         'sigma', 'epsilon', 'vm'):
                arr = getattr(self, name)
                if arr is not None:
                    f.create_dataset(name, data=arr)
            for name in ('sigma_principal', 'epsilon_principal',
                         'sigma_nodal', 'epsilon_nodal', 'vm_nodal'):
                arr = getattr(self, name)
                if arr is not None:
                    f.create_dataset(name, data=arr)

    @classmethod
    def load(cls, path: str):
        """Load result from HDF5 file."""
        import h5py
        with h5py.File(path, 'r') as f:
            step = int(f.attrs['step'])
            time = float(f.attrs['time'])
            data = {k: f[k][()] for k in f.keys()}
        return cls(
            step=step, time=time,
            u=data['u'], F=data['F'], R=data['R'],
            u_3d=data['u_3d'], F_3d=data['F_3d'], R_3d=data['R_3d'],
            sigma=data['sigma'], epsilon=data['epsilon'], vm=data['vm'],
            sigma_principal=data.get('sigma_principal'),
            epsilon_principal=data.get('epsilon_principal'),
            sigma_nodal=data.get('sigma_nodal'),
            epsilon_nodal=data.get('epsilon_nodal'),
            vm_nodal=data.get('vm_nodal'),
        )

    def __repr__(self):
        return f"FEMResult(step={self.step}, time={self.time:.4f})"
