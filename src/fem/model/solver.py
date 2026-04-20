"""Solver — stiffness assembly, linear solve, and result extraction."""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from fem.model.result import FEMResult


class Solver:
    """
    Assembles K, solves the system, and extracts FEMResult objects.

    Parameters
    ----------
    node_map     : dict   {tag: Node}
    elements     : ndarray of FEM element objects
    system_nDof  : int
    analysis_type: str   'planeStress', 'planeStrain', or '3D'
    """

    def __init__(self, node_map, elements, system_nDof, analysis_type):
        self.node_map     = node_map
        self.elements     = elements
        self.system_nDof  = system_nDof
        self.analysis_type = analysis_type
        self.K            = None

    # -------------------------------------------------------------------------
    # K assembly
    # -------------------------------------------------------------------------

    def assemble_K(self):
        """Assemble global stiffness matrix. Stores result in self.K."""
        K = lil_matrix((self.system_nDof, self.system_nDof))
        for elem in self.elements:
            for i_local, i_global in enumerate(elem.idx):
                for j_local, j_global in enumerate(elem.idx):
                    K[i_global, j_global] += elem.kg[i_local, j_local]
        self.K = K.tocsr()

    # -------------------------------------------------------------------------
    # Linear solve
    # -------------------------------------------------------------------------

    def solve(self, F: np.ndarray, step=0, time=0.0,
              node_tags=None, element_tags=None) -> FEMResult:
        """
        Solve K u = F and extract results.

        Parameters
        ----------
        F            : ndarray (system_nDof,) force vector.
        step         : int    Load step index.
        time         : float  Load factor or time.
        node_tags    : ndarray node tags in gmsh order.
        element_tags : list   element tags.

        Returns
        -------
        FEMResult
        """
        if self.K is None:
            self.assemble_K()

        dof_flags       = np.concatenate([n.restrain for n in self.node_map.values()])
        free_dofs       = np.where(dof_flags == 'f')[0]
        restrained_dofs = np.where(dof_flags == 'r')[0]

        Kff = self.K[free_dofs[:, None], free_dofs]
        Kfr = self.K[free_dofs[:, None], restrained_dofs]
        Krf = self.K[restrained_dofs[:, None], free_dofs]
        Krr = self.K[restrained_dofs[:, None], restrained_dofs]

        ur = np.zeros(len(restrained_dofs))
        uf = spsolve(Kff, F[free_dofs] - Kfr @ ur)

        u = np.zeros(self.system_nDof)
        u[free_dofs]       = uf
        u[restrained_dofs] = ur

        Rr = Krf @ uf + Krr @ ur
        R  = np.zeros(self.system_nDof)
        R[restrained_dofs] = Rr

        return self._build_result(u, F, R, step, time, node_tags, element_tags)

    # -------------------------------------------------------------------------
    # Result extraction
    # -------------------------------------------------------------------------

    def _build_result(self, u, F, R, step, time,
                      node_tags, element_tags) -> FEMResult:
        """Build FEMResult from solution vectors."""
        n_nodes    = len(self.node_map)
        n_elements = len(self.elements)
        nDOF       = self.system_nDof // n_nodes

        # gmsh format arrays
        u_3d = np.zeros((n_nodes, 3))
        F_3d = np.zeros((n_nodes, 3))
        R_3d = np.zeros((n_nodes, 3))
        for i, node in enumerate(self.node_map.values()):
            n = min(nDOF, 3)
            u_3d[i, :n] = u[node.idx[:n]]
            F_3d[i, :n] = F[node.idx[:n]]
            R_3d[i, :n] = R[node.idx[:n]]

        # element results
        n_comp = 6 if self.analysis_type == '3D' else 3
        sigma            = np.zeros((n_elements, n_comp))
        epsilon          = np.zeros((n_elements, n_comp))
        vm               = np.zeros(n_elements)
        sigma_principal  = np.zeros((n_elements, 2)) if n_comp == 3 else None
        epsilon_principal= np.zeros((n_elements, 2)) if n_comp == 3 else None

        for i, elem in enumerate(self.elements):
            res = elem.get_results(u)
            s   = res['stress'].flatten()
            e   = res['strain'].flatten()
            nc  = min(len(s), n_comp)
            sigma[i, :nc]   = s[:nc]
            epsilon[i, :nc] = e[:nc]
            sxx, syy = sigma[i, 0], sigma[i, 1]
            sxy = sigma[i, 3] if n_comp == 6 else sigma[i, 2]
            vm[i] = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)
            if sigma_principal is not None:
                sp = res.get('principal_stress')
                ep = res.get('principal_strain')
                if sp is not None:
                    sigma_principal[i]   = np.array(sp).flatten()[:2]
                if ep is not None:
                    epsilon_principal[i] = np.array(ep).flatten()[:2]

        # nodal averages
        sigma_nodal, epsilon_nodal, vm_nodal = self._nodal_averages(
            node_tags, element_tags, sigma, epsilon, vm)

        return FEMResult(
            step=step, time=time,
            u=u, F=F, R=R,
            u_3d=u_3d, F_3d=F_3d, R_3d=R_3d,
            sigma=sigma, epsilon=epsilon, vm=vm,
            sigma_principal=sigma_principal,
            epsilon_principal=epsilon_principal,
            sigma_nodal=sigma_nodal,
            epsilon_nodal=epsilon_nodal,
            vm_nodal=vm_nodal,
        )

    def _nodal_averages(self, node_tags, element_tags, sigma, epsilon, vm):
        """Average element results to nodes using connectivity."""
        if node_tags is None or element_tags is None:
            return None, None, None

        mesh       = None  # connectivity from elements directly
        n_nodes    = len(node_tags)
        n_comp     = sigma.shape[1]
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        s_sum  = np.zeros((n_nodes, n_comp))
        e_sum  = np.zeros((n_nodes, n_comp))
        vm_sum = np.zeros(n_nodes)
        count  = np.zeros(n_nodes)

        for i, (elem, etag) in enumerate(zip(self.elements, element_tags)):
            for node in elem.nodes:
                j = tag_to_idx.get(node.name)
                if j is not None:
                    s_sum[j]  += sigma[i]
                    e_sum[j]  += epsilon[i]
                    vm_sum[j] += vm[i]
                    count[j]  += 1

        count[count == 0] = 1
        return s_sum / count[:, None], e_sum / count[:, None], vm_sum / count

    # -------------------------------------------------------------------------
    # OpenSees result extraction
    # -------------------------------------------------------------------------

    def extract_opensees_results(self, ops, mesh, solid_group_name,
                                  node_tags, element_tags,
                                  F_3d, step=0, time=0.0,
                                  material=None) -> FEMResult:

        # normalize to list
        if not isinstance(solid_group_name, list):
            solid_group_name = [solid_group_name]

        ops.reactions()
        nDOF    = self.system_nDof // len(mesh.nodes)
        n_nodes = len(mesh.nodes)

        u_3d = np.zeros((n_nodes, 3))
        R_3d = np.zeros((n_nodes, 3))
        for i, tag in enumerate(mesh.nodes):
            for j in range(nDOF):
                u_3d[i, j] = ops.nodeDisp(tag, j + 1)
                R_3d[i, j] = ops.nodeReaction(tag, j + 1)

        # collect elements from all solid groups
        all_elem_tags = []
        all_conn      = []
        for name in solid_group_name:
            pg = mesh.physical_groups.get(name)
            if pg is None:
                continue
            all_elem_tags.extend(pg.elements['element_tags'])
            all_conn.extend(pg.elements.get('connectivity', []))

        n_elements = len(all_elem_tags)
        n_comp     = 6 if self.analysis_type == '3D' else 3
        sigma      = np.zeros((n_elements, n_comp))
        vm         = np.zeros(n_elements)

        for i, etag in enumerate(all_elem_tags):
            stress = ops.eleResponse(etag, 'stresses')
            if not stress:
                stress = ops.eleResponse(etag, 'stressAtNodes')
            if stress:
                s  = np.array(stress)
                nn = len(ops.eleNodes(etag))
                
                # if total components match 3 or 6 — direct stress vector
                if len(s) in (3, 6):
                    nc = len(s)
                else:
                    nc = len(s) // nn
                
                if nc < 3:
                    continue
                
                if len(s) != nc:
                    s = s.reshape(-1, nc).mean(axis=0)
                
                nc = min(nc, n_comp)
                sigma[i, :nc] = s[:nc]
                sxx, syy = s[0], s[1]
                sxy = s[3] if nc == 6 else s[2]
                vm[i] = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

    

        epsilon = self._strains_from_hooke(sigma, material)

        # nodal averages
        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
        s_sum      = np.zeros((n_nodes, n_comp))
        e_sum      = np.zeros((n_nodes, n_comp))
        vm_sum     = np.zeros(n_nodes)
        count      = np.zeros(n_nodes)

        for i, conn in enumerate(all_conn):
            for tag in conn:
                j = tag_to_idx.get(int(tag))
                if j is not None:
                    s_sum[j]  += sigma[i]
                    e_sum[j]  += epsilon[i] if epsilon is not None else 0
                    vm_sum[j] += vm[i]
                    count[j]  += 1

        count[count == 0] = 1
        sigma_nodal   = s_sum / count[:, None]
        epsilon_nodal = e_sum / count[:, None] if epsilon is not None else None
        vm_nodal      = vm_sum / count

        u_global = np.zeros(self.system_nDof)
        R_global = np.zeros(self.system_nDof)
        F_global = np.zeros(self.system_nDof)
        for i, node in enumerate(self.node_map.values()):
            n = min(nDOF, 3)
            u_global[node.idx[:n]] = u_3d[i, :n]
            R_global[node.idx[:n]] = R_3d[i, :n]
            F_global[node.idx[:n]] = F_3d[i, :n]

        return FEMResult(
            step=step, time=time,
            u=u_global, F=F_global, R=R_global,
            u_3d=u_3d, F_3d=F_3d, R_3d=R_3d,
            sigma=sigma, epsilon=epsilon, vm=vm,
            sigma_nodal=sigma_nodal,
            epsilon_nodal=epsilon_nodal,
            vm_nodal=vm_nodal,
        )



    def _strains_from_hooke(self, sigma, material):
        """Compute strains from stresses using Hooke's law."""
        if material is None:
            return None
        E, nu     = material.E, material.nu
        n_comp    = sigma.shape[1]
        epsilon   = np.zeros_like(sigma)
        if n_comp == 6:
            sxx, syy, szz = sigma[:,0], sigma[:,1], sigma[:,2]
            sxy, syz, sxz = sigma[:,3], sigma[:,4], sigma[:,5]
            epsilon[:,0] = (sxx - nu*(syy+szz)) / E
            epsilon[:,1] = (syy - nu*(sxx+szz)) / E
            epsilon[:,2] = (szz - nu*(sxx+syy)) / E
            epsilon[:,3] = 2*sxy*(1+nu) / E
            epsilon[:,4] = 2*syz*(1+nu) / E
            epsilon[:,5] = 2*sxz*(1+nu) / E
        elif self.analysis_type == 'planeStress':
            sxx, syy, sxy = sigma[:,0], sigma[:,1], sigma[:,2]
            epsilon[:,0] = (sxx - nu*syy) / E
            epsilon[:,1] = (syy - nu*sxx) / E
            epsilon[:,2] = 2*sxy*(1+nu) / E
        elif self.analysis_type == 'planeStrain':
            sxx, syy, sxy = sigma[:,0], sigma[:,1], sigma[:,2]
            epsilon[:,0] = ((1-nu**2)*sxx - nu*(1+nu)*syy) / E
            epsilon[:,1] = ((1-nu**2)*syy - nu*(1+nu)*sxx) / E
            epsilon[:,2] = 2*sxy*(1+nu) / E
        return epsilon
