"""FEMModel — orchestrates mesh, BCs, elements, solver, and results."""

import numpy as np
import matplotlib.pyplot as plt
from fem.model.bcs import BoundaryConditions
from fem.model.solver import Solver
from fem.model.result import FEMResult
from fem.utils.functions import build_elements
from fem.core.parameters import globalParameters


class FEMModel:
    """
    Full FEM model container. Orchestrates BCs, elements, solver, and results.

    Parameters
    ----------
    mesh                : GMSHtools
    section_dictionary  : dict  {phys_id or name: section}
    restrain_dictionary : dict  {phys_id or name: ['r'/'f', ...]}
    load_dictionary     : dict  {phys_id or name: {'value': float, 'direction': str}}
    element_class_map   : dict  {n_nodes: ElementClass}  None = OpenSees only
    analysis_type       : str   'planeStress', 'planeStrain', '3D'
    sampling_points     : int   Gauss points per direction
    eval_points         : list  [zeta, eta] for stress recovery
    consistent_loads    : bool  Use shape-function loads (requires element_class_map)
    """

    def __init__(self, 
                mesh, 
                section_dictionary, 
                restrain_dictionary,
                load_dictionary, 
                element_class_map=None,
                analysis_type='planeStress', 
                sampling_points=3,
                eval_points=None,
                consistent_loads=True,
                verbose=True):

        self.mesh              = mesh
        self.mesh_file         = getattr(mesh, '_file', None)
        self.section_dictionary = section_dictionary
        self.analysis_type      = analysis_type
        self.results_fem        = []   # FEMResult from own solver
        self.results_opensees   = []   # FEMResult from OpenSees
        self._last_source       = None # 'fem' or 'opensees'

        # boundary conditions and load vectors
        self._bcs = BoundaryConditions(
            mesh, restrain_dictionary, load_dictionary, section_dictionary)

        self.node_map        = self._bcs.node_map
        self.system_nDof     = self._bcs.system_nDof
        self.node_tags       = self._bcs.node_tags
        self.restrained_nodes = self._bcs.restrained_nodes
        self.F_nodal         = self._bcs.F_nodal

        # build FEM elements if element_class_map provided
        self.elements    = None
        self.element_tags = None
        if element_class_map is not None:
            kwargs = dict(
                mesh               = mesh,
                node_map           = self.node_map,
                section_dictionary = section_dictionary,
                element_class_map  = element_class_map,
                load_dictionary    = load_dictionary,
                type               = analysis_type,
            )
            if sampling_points is not None:
                kwargs['sampling_points'] = sampling_points
            if eval_points is not None:
                kwargs['eval_points'] = eval_points
            self.elements     = build_elements(**kwargs)
            self.element_tags = [e.element_tag for e in self.elements]

        # load vector
        if consistent_loads and self.elements is not None:
            self.F_load = self._bcs.build_F_consistent(self.elements)
        else:
            self.F_load = self._bcs.F_load

        # solver
        self._solver = Solver(
            self.node_map, self.elements, self.system_nDof, analysis_type)

        # sdummary
        if verbose:
            self._print_summary()


    # -------------------------------------------------------------------------
    # Shortcuts to last result
    # -------------------------------------------------------------------------

    @property
    def results(self):
        """Active results list — last source used (fem or opensees)."""
        if self._last_source == 'opensees':
            return self.results_opensees
        return self.results_fem

    @property
    def u(self):
        return self.results[-1].u if self.results else None

    @property
    def sigma(self):
        return self.results[-1].sigma if self.results else None

    @property
    def vm(self):
        return self.results[-1].vm if self.results else None

    # -------------------------------------------------------------------------
    # Solver — own solver
    # -------------------------------------------------------------------------

    def solve_static(self, n_steps=1, load_factor=1.0, verbose=True):
        """
        Solve the FEM system — own solver.

        Parameters
        ----------
        n_steps      : int    Number of load steps. 1 = single static solve.
        load_factor  : float  Total load factor applied over all steps.
        verbose      : bool   Print summary after solving.
        """
        self._solver.assemble_K()
        for step in range(n_steps):
            time  = (step + 1) / n_steps * load_factor
            F_inc = self.F_load * time
            result = self._solver.solve(
                F=F_inc, step=step, time=time,
                node_tags=self.node_tags,
                element_tags=self.element_tags,
            )
            self.results_fem.append(result)

        self._last_source = 'fem'
        if verbose:
            sep = '--' * 40
            r   = self.results_fem[-1]
            print(f"\n  SOLVE SUMMARY")
            print(sep)
            print(f"  Steps solved     : {len(self.results_fem)}")
            print(f"  Last load factor : {r.time:.4f}")
            print(f"  Max |ux|         : {np.max(np.abs(r.u[0::2])):.6f}")
            print(f"  Max |uy|         : {np.max(np.abs(r.u[1::2])):.6f}")
            print(f"  Max Von Mises    : {np.max(r.vm):.4f}")
            Fx = r.F[0::2].sum();  Rx = r.R[0::2].sum()
            Fy = r.F[1::2].sum();  Ry = r.R[1::2].sum()
            print(f"\n  --- Equilibrium ---")
            print(f"  Applied  Fx: {Fx:+.4f}   Reaction Rx: {Rx:+.4f}   Balance: {Fx+Rx:.4e}")
            print(f"  Applied  Fy: {Fy:+.4f}   Reaction Ry: {Ry:+.4f}   Balance: {Fy+Ry:.4e}")
            print(sep)
            print()


    # -------------------------------------------------------------------------
    # OpenSees
    # -------------------------------------------------------------------------

    def set_results_opensees(self, ops, solid_group_name=None,
                              step=0, time=0.0, material=None):
        """
        Extract results from solved OpenSees model and append to self.results.

        Parameters
        ----------
        ops              : opensees module passed by user.
        solid_group_name : str or int  physical group with solid elements.
        step, time       : int, float
        material         : Material with E, nu — needed for strain computation.
        """
        # infer solid groups from section_dictionary if not provided
        if solid_group_name is None:
            # infer all solid groups from mesh — dim 2 or 3
            solid_group_name = list({
                pg.name for pg in self.mesh.physical_groups.values()
                if isinstance(pg.name, str) and pg.dim >= 2
            })
        elif not isinstance(solid_group_name, list):
            solid_group_name = [solid_group_name]

        # store element tags from solid groups for gmsh
        if self.element_tags is None:
            all_tags = []
            for name in solid_group_name:
                pg = self.mesh.physical_groups.get(name)
                if pg is not None:
                    all_tags.extend(pg.elements['element_tags'])
            self.element_tags = all_tags

        nDOF    = self.system_nDof // len(self.mesh.nodes)
        n_nodes = len(self.mesh.nodes)
        F_3d    = np.zeros((n_nodes, 3))
        for i, tag in enumerate(self.mesh.nodes):
            f = self.F_nodal.get(tag, np.zeros(nDOF))
            F_3d[i, :len(f)] = f[:nDOF]

        result = self._solver.extract_opensees_results(
            ops=ops,
            mesh=self.mesh,
            solid_group_name=solid_group_name,
            node_tags=self.node_tags,
            element_tags=self.element_tags,
            F_3d=F_3d,
            step=step,
            time=time,
            material=material,
        )
        self.results_opensees.append(result)
        self._last_source = 'opensees'
        return result

    # -------------------------------------------------------------------------
    # Result queries
    # -------------------------------------------------------------------------

    def get_node(self, tag=None, tags=None, x=None, y=None, z=None,
                 locations=None, step=-1, source=None):
        """
        Return node info and displacements for one or more nodes.

        Parameters
        ----------
        tag       : int          Single node gmsh tag.
        tags      : list[int]    Multiple node tags.
        x, y, z  : float        Coordinates to find nearest single node.
        locations : list[(x,y)] Multiple coordinate pairs.
        step      : int          Index into results.
        source    : str          'fem', 'opensees', or None=last used.
        """
        all_tags = []
        if tag is not None:
            all_tags.append(tag)
        if tags is not None:
            all_tags += list(tags)
        if x is not None or y is not None:
            all_tags.append(self._find_nearest_node(x, y, z))
        if locations is not None:
            for loc in locations:
                xi = loc[0] if len(loc) > 0 else 0.
                yi = loc[1] if len(loc) > 1 else 0.
                zi = loc[2] if len(loc) > 2 else 0.
                all_tags.append(self._find_nearest_node(xi, yi, zi))

        results_list = []
        sep = '--' * 30
        print(f"\n  NODE RESULTS  (step={step})")
        print(sep)
        print(f"  {'Tag':>6}  {'Coordinates':30s}  {'ux':>12}  {'uy':>12}")
        print(f"  {'-'*6}  {'-'*30}  {'-'*12}  {'-'*12}")

        _results = self._get_results(source)
        for t in all_tags:
            node   = self.node_map[t]
            result = _results[step] if _results else None
            u      = result.u[node.idx] if result is not None else None
            ux     = f"{u[0]:+.6f}" if u is not None else '-'
            uy     = f"{u[1]:+.6f}" if u is not None and len(u) > 1 else '-'
            print(f"  {t:>6}  {str(node.coordinates.round(4)):30s}  {ux:>12}  {uy:>12}")
            results_list.append({'tag': t, 'coordinates': node.coordinates,
                                 'u': u, 'restrain': node.restrain})

        print(sep)
        return results_list if len(results_list) > 1 else results_list[0]


    def _find_nearest_node(self, x, y, z=None):
        """Find nearest node tag to given coordinates."""
        target    = np.array([x or 0., y or 0., z or 0.])
        best_tag  = None
        best_dist = np.inf
        for t, n in self.node_map.items():
            c    = np.array(list(n.coordinates) + [0.]*(3-len(n.coordinates)))
            dist = np.linalg.norm(c - target)
            if dist < best_dist:
                best_dist = dist
                best_tag  = t
        return best_tag



    def get_element(self, tag=None, tags=None, x=None, y=None,
                    locations=None, step=-1, source=None):
        """
        Return element info and results for one or more elements.

        Parameters
        ----------
        tag       : int          Single element tag.
        tags      : list[int]    Multiple element tags.
        x, y      : float        Centroid coordinates to find nearest element.
        locations : list[(x,y)]  Multiple centroid coordinate pairs.
        step      : int          Index into results.
        source    : str          'fem', 'opensees', or None=last used.
        """
        if self.elements is None:
            raise RuntimeError("No FEM elements — use element_class_map.")

        all_idx = []
        if tag is not None:
            all_idx.append(self._find_nearest_element(tag=tag))
        if tags is not None:
            for t in tags:
                all_idx.append(self._find_nearest_element(tag=t))
        if x is not None or y is not None:
            all_idx.append(self._find_nearest_element(x=x, y=y))
        if locations is not None:
            for loc in locations:
                xi = loc[0] if len(loc) > 0 else 0.
                yi = loc[1] if len(loc) > 1 else 0.
                all_idx.append(self._find_nearest_element(x=xi, y=yi))

        _results = self._get_results(source)
        result   = _results[step] if _results else None

        n_comp   = result.sigma.shape[1] if result is not None else 3
        s_labels = ['Sxx','Syy','Szz','Sxy','Syz','Sxz'] if n_comp==6 else ['Sxx','Syy','Sxy']
        e_labels = ['Exx','Eyy','Ezz','Exy','Eyz','Exz'] if n_comp==6 else ['Exx','Eyy','Exy']

        sep = '--' * 40
        print(f"\n  ELEMENT RESULTS  (step={step})")
        print(sep)
        header = f"  {'Tag':>6}  {'Centroid':25s}  {'Type':8s}"
        for name in s_labels:
            header += f"  {name:>10}"
        header += f"  {'VM':>10}"
        print(header)
        print(f"  {'-'*6}  {'-'*25}  {'-'*8}" + f"  {'-'*10}"*(len(s_labels)+1))

        results_list = []
        for idx in all_idx:
            elem     = self.elements[idx]
            centroid = np.mean([n.coordinates for n in elem.nodes], axis=0)
            sigma    = result.sigma[idx]   if result is not None else None
            epsilon  = result.epsilon[idx] if result is not None else None
            vm       = result.vm[idx]      if result is not None else None

            row = f"  {elem.element_tag:>6}  {str(centroid.round(2)):25s}  {type(elem).__name__:8s}"
            if sigma is not None:
                for val in sigma:
                    row += f"  {val:>10.4f}"
                row += f"  {vm:>10.4f}"
            print(row)

            results_list.append({
                'tag'     : elem.element_tag,
                'centroid': centroid,
                'sigma'   : sigma,
                'epsilon' : epsilon,
                'vm'      : vm,
            })

        print(sep)
        return results_list if len(results_list) > 1 else results_list[0]


    def _find_nearest_element(self, tag=None, x=None, y=None):
        """Find element index by tag or nearest centroid."""
        if tag is not None:
            return next(i for i, e in enumerate(self.elements)
                        if e.element_tag == tag)
        best_idx  = 0
        best_dist = np.inf
        target    = np.array([x or 0., y or 0.])
        for i, elem in enumerate(self.elements):
            c    = np.mean([n.coordinates for n in elem.nodes], axis=0)
            dist = np.linalg.norm(c[:2] - target)
            if dist < best_dist:
                best_dist = dist
                best_idx  = i
        return best_idx




    def node_history(self, tag, component='uy', source=None):
        """
        Return time history of a displacement component at a node.

        Parameters
        ----------
        tag       : int   Node gmsh tag.
        component : str   'ux', 'uy', 'uz'
        source    : str   'fem' or 'opensees'. None = last used.
        """
        comp_map = {'ux': 0, 'uy': 1, 'uz': 2}
        i        = comp_map.get(component, component)
        node     = self.node_map[tag]
        results  = self._get_results(source)
        return np.array([r.u[node.idx[i]] for r in results])

    def element_history(self, tag, component='sxx', source=None):
        """
        Return time history of a stress/strain component at an element.

        Parameters
        ----------
        tag       : int  Element tag.
        component : str  'sxx','syy','sxy','vm','exx','eyy','exy'
        source    : str  'fem' or 'opensees'. None = last used.
        """
        comp_map = {'sxx': (0,'sigma'), 'syy': (1,'sigma'), 'sxy': (2,'sigma'),
                    'exx': (0,'epsilon'), 'eyy': (1,'epsilon'), 'exy': (2,'epsilon'),
                    'vm': (0,'vm')}
        ci, field = comp_map[component]
        idx       = next(i for i, e in enumerate(self.elements) if e.element_tag == tag)
        results   = self._get_results(source)
        if field == 'vm':
            return np.array([r.vm[idx] for r in results])
        return np.array([getattr(r, field)[idx, ci] for r in results])

    def _get_results(self, source=None) -> list:
        """Return results list for given source ('fem', 'opensees', or None=last)."""
        if source == 'fem':
            return self.results_fem
        if source == 'opensees':
            return self.results_opensees
        return self.results  # last used

    # -------------------------------------------------------------------------
    # Modal — OpenSees
    # -------------------------------------------------------------------------

    def set_modal_results(self, ops, n_modes=6):
        """
        Run eigenanalysis and store modal results.

        Parameters
        ----------
        ops     : opensees module passed by user.
        n_modes : int  Number of modes to compute.
        """
        from fem.model.modal_result import ModalResult

        eigenvalues = ops.eigen(n_modes)
        omega       = np.sqrt(np.array(eigenvalues))
        freq        = omega / (2 * np.pi)
        period      = 1 / freq

        sep = '--' * 40
        print(f"\n  MODAL ANALYSIS")
        print(sep)
        print(f"  {'Mode':>6}  {'Freq [Hz]':>12}  {'Period [s]':>12}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}")

        n_nodes = len(self.mesh.nodes)
        modal_results = []
        for mode in range(1, n_modes + 1):
            u_3d = np.zeros((n_nodes, 3))
            for i, tag in enumerate(self.mesh.nodes.keys()):
                nDOF = self.system_nDof // n_nodes
                for j in range(min(nDOF, 3)):
                    u_3d[i, j] = ops.nodeEigenvector(tag, mode, j + 1)

            mr = ModalResult(
                mode   = mode,
                freq   = freq[mode - 1],
                period = period[mode - 1],
                omega  = omega[mode - 1],
                u_3d   = u_3d,
            )
            modal_results.append(mr)
            print(f"  {mode:>6}  {freq[mode-1]:>12.4f}  {period[mode-1]:>12.4f}")

        self.results_opensees_modal = modal_results
        print(sep)
        print()

    def plot_modal(self, modes=None, n_steps=30, disp_factor=50):
        """
        Animate modal shapes in gmsh. One animated view per mode.

        Parameters
        ----------
        modes      : list[int]  Mode numbers to plot. None = all.
        n_steps    : int        Animation frames per mode.
        disp_factor: float      Displacement scale factor.
        """
        import gmsh

        if not hasattr(self, 'results_opensees_modal'):
            raise RuntimeError("No modal results — run set_modal_results() first.")

        modal = self.results_opensees_modal
        if modes is None:
            modes = [mr.mode for mr in modal]

        gmsh.initialize()
        gmsh.open(self.mesh._file)

        for mr in modal:
            if mr.mode not in modes:
                continue
            view = gmsh.view.add(f"Mode {mr.mode}  T={mr.period:.4f}s  f={mr.freq:.4f}Hz")
            for step in range(n_steps):
                scale = np.cos(step / n_steps * 2 * np.pi)
                gmsh.view.addHomogeneousModelData(
                    tag=view, step=step, time=float(step),
                    modelName=gmsh.model.getCurrent(),
                    dataType="NodeData", numComponents=-1,
                    tags=self.node_tags, data=(mr.u_3d * scale).reshape(-1))
            gmsh.view.option.setNumber(view, "VectorType",         5)
            gmsh.view.option.setNumber(view, "DisplacementFactor", disp_factor)
            gmsh.view.option.setNumber(view, "Visible",            0)

        # make first mode visible
        all_views = gmsh.view.getTags()
        if len(all_views) > 0:
            gmsh.view.option.setNumber(all_views[0], "Visible", 1)

        gmsh.fltk.run()
        gmsh.finalize()



    def check_mesh(self):
        """Check mesh integrity — orphan nodes, missing sections, element counts."""
        mesh = self.mesh
        sep  = '--' * 40

        print(f"\n  MESH DIAGNOSTICS")
        print(sep)
        print(f"  Nodes            : {len(mesh.nodes)}")
        print(f"  system_nDof      : {self.system_nDof}")
        print(f"  Elements         : {len(self.elements) if self.elements is not None else 'None (OpenSees)'}")
        print(f"  Physical groups  : {len(mesh._physical_raw)}")

        # orphan nodes
        print(f"\n  --- Orphan nodes ---")
        connected = set()
        for group in mesh.elements.values():
            for conn in group['connectivity']:
                connected.update(conn)
        orphans = set(mesh.nodes.keys()) - connected
        if orphans:
            print(f"  WARNING: {len(orphans)} orphan nodes found: {sorted(orphans)[:10]}")
            # remove orphan nodes from mesh
            for tag in orphans:
                del mesh.nodes[tag]
            # rebuild node_map and system_nDof
            from fem.utils.functions import plan
            mesh.node_map, mesh.system_nDof = plan(mesh)
            self.node_map    = self._bcs.node_map = mesh.node_map
            self.system_nDof = mesh.system_nDof
            self.node_tags   = self._bcs.node_tags = np.array(sorted(mesh.nodes.keys()))
        
            self._solver.node_map    = self.node_map
            self._solver.system_nDof = self.system_nDof

            print(f"  Removed {len(orphans)} orphan nodes — node_map rebuilt.")
        else:
            print(f"  OK — no orphan nodes")

        # physical groups
        print(f"\n  --- Physical groups ---")
        print(f"  {'ID':>6}  {'Dim':>4}  {'Name':20s}  {'Elements':>10}  {'Nodes':>8}  {'Section'}")
        print(f"  {'-'*6}  {'-'*4}  {'-'*20}  {'-'*10}  {'-'*8}  {'-'*10}")
        for phys_id, group in mesh.elements.items():
            pg      = mesh.physical_groups.get(phys_id)
            name    = pg.name if pg else str(phys_id)
            n_elem  = len(group['element_tags'])
            n_nodes = len(pg.nodes) if pg else '-'
            has_sec = phys_id in (self.section_dictionary or {})
            sec_str = 'OK' if has_sec else ('N/A' if group['dim'] < 2 else 'MISSING')
            print(f"  {phys_id:>6}  {group['dim']:>4}  {name:20s}  {n_elem:>10}  {n_nodes:>8}  {sec_str}")

        # restrained nodes
        print(f"\n  --- Restrained nodes ---")
        print(f"  {'Tag':>6}  {'x':>12}  {'y':>12}  {'Condition'}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}")
        for tag, cond in self._bcs.restrained_nodes.items():
            node = self.node_map[tag]
            x, y = node.coordinates[0], node.coordinates[1]
            print(f"  {tag:>6}  {x:>12.4f}  {y:>12.4f}  {cond}")

        # load summary
        print(f"\n  --- Load summary ---")
        nz = np.where(np.abs(self.F_load) > 0)[0]
        print(f"  Non-zero DOFs    : {len(nz)}")
        print(f"  Total Fx         : {self.F_load[0::2].sum():+.4f}")
        print(f"  Total Fy         : {self.F_load[1::2].sum():+.4f}")

        print(sep)
        print()


    # -------------------------------------------------------------------------
    # Visualization — matplotlib (delegates to plotting.py)
    # -------------------------------------------------------------------------

    def plot(self, step=-1, **kwargs):
        """Plot mesh geometry."""
        from fem.utils.plotting import plot_mesh
        nodes = list(self.node_map.values())
        elems = self.elements if self.elements is not None else []
        plot_mesh(nodes=nodes, elements=elems, **kwargs)

    def plot_loads(self, **kwargs):
        """Plot applied load arrows."""
        from fem.utils.plotting import plot_loads_2d
        nodes = list(self.node_map.values())
        elems = self.elements if self.elements is not None else []
        plot_loads_2d(nodes=nodes, elements=elems,
                      F_load=self.F_load, **kwargs)

    def plot_deformed(self, sfac=1.0, step=-1, source=None, **kwargs):
        """Plot deformed shape. source: 'fem', 'opensees', or None=last."""
        from fem.utils.plotting import plot_deformed
        results = self._get_results(source)
        if not results:
            raise RuntimeError("No results — run solve_static() first.")
        nodes = list(self.node_map.values())
        plot_deformed(nodes=nodes, elements=self.elements,
                      u=results[step].u, sfac=sfac, **kwargs)

    def plot_field(self, component='sxx', step=-1, source=None, **kwargs):
        """Plot stress or strain field. source: 'fem', 'opensees', or None=last."""
        results = self._get_results(source)
        if not results:
            raise RuntimeError("No results — run solve_static() or set_results_opensees() first.")
        nodes  = list(self.node_map.values())
        result = results[step]
        if self.elements is not None:
            from fem.utils.plotting import plot_field_2d
            plot_field_2d(nodes=nodes, elements=self.elements,
                          u=result.u, component=component, **kwargs)
        else:
            from fem.utils.plotting import plot_field_3d
            plot_field_3d(nodes=nodes, result=result,
                          component=component, **kwargs)


    def plot_node_history(self, tags=None, locations=None, component='uy',
                          source=None, figsize=(10, 5), ax=None, **kwargs):
        results = self._get_results(source)
        if not results:
            raise RuntimeError("No results — run solve_static() first.")

        times = [r.time for r in results]

        all_tags = []
        if tags is not None:
            all_tags += [tags] if isinstance(tags, int) else list(tags)
        if locations is not None:
            for loc in locations:
                all_tags.append(self._find_nearest_node(loc[0], loc[1]))

        # print summary
        sep = '--' * 30
        print(f"\n  NODE HISTORY  —  {component}")
        print(sep)
        print(f"  {'Tag':>6}  {'Coordinates':30s}  {'Min':>12}  {'Max':>12}  {'Final':>12}")
        print(f"  {'-'*6}  {'-'*30}  {'-'*12}  {'-'*12}  {'-'*12}")
        for tag in all_tags:
            history = self.node_history(tag=tag, component=component)
            node    = self.node_map[tag]
            print(f"  {tag:>6}  {str(node.coordinates.round(2)):30s}"
                  f"  {np.min(history):>+12.6f}  {np.max(history):>+12.6f}  {history[-1]:>+12.6f}")
        print(sep)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for tag in all_tags:
            history = self.node_history(tag=tag, component=component)
            node    = self.node_map[tag]
            label   = f"Node {tag} ({node.coordinates[0]:.1f}, {node.coordinates[1]:.1f})"
            ax.plot(times, history, 'o-', label=label, **kwargs)

        ax.set_xlabel('Load factor')
        ax.set_ylabel(component)
        ax.set_title(f'Node history — {component}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    def plot_element_history(self, tags=None, locations=None, component='sxx',
                             source=None, figsize=(10, 5), ax=None, **kwargs):
        results = self._get_results(source)
        if not results:
            raise RuntimeError("No results — run solve_static() first.")
        if self.elements is None:
            raise RuntimeError("No FEM elements available.")

        times = [r.time for r in results]

        all_tags = []
        if tags is not None:
            all_tags += [tags] if isinstance(tags, int) else list(tags)
        if locations is not None:
            for loc in locations:
                idx = self._find_nearest_element(x=loc[0], y=loc[1])
                all_tags.append(self.elements[idx].element_tag)

        # print summary
        sep = '--' * 40
        print(f"\n  ELEMENT HISTORY  —  {component}")
        print(sep)
        print(f"  {'Tag':>6}  {'Centroid':25s}  {'Type':8s}  {'Min':>12}  {'Max':>12}  {'Final':>12}")
        print(f"  {'-'*6}  {'-'*25}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")
        for tag in all_tags:
            history  = self.element_history(tag=tag, component=component)
            idx      = self._find_nearest_element(tag=tag)
            elem     = self.elements[idx]
            centroid = np.mean([n.coordinates for n in elem.nodes], axis=0)
            print(f"  {tag:>6}  {str(centroid.round(2)):25s}  {type(elem).__name__:8s}"
                  f"  {np.min(history):>+12.6f}  {np.max(history):>+12.6f}  {history[-1]:>+12.6f}")
        print(sep)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        for tag in all_tags:
            history  = self.element_history(tag=tag, component=component)
            idx      = self._find_nearest_element(tag=tag)
            elem     = self.elements[idx]
            centroid = np.mean([n.coordinates for n in elem.nodes], axis=0)
            label    = f"Elem {tag} ({centroid[0]:.1f}, {centroid[1]:.1f})"
            ax.plot(times, history, 'o-', label=label, **kwargs)

        ax.set_xlabel('Load factor')
        ax.set_ylabel(component)
        ax.set_title(f'Element history — {component}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()



            
    # -------------------------------------------------------------------------
    # Visualization — gmsh (delegates to visualization.py)
    # -------------------------------------------------------------------------

    def plot2gmsh(self, 
                step=-1,
                source=None,
                disp_factor=10,
                show_disp=True, show_loads=True, show_reactions=True,
                show_stress=True, show_strain=True, show_vm=True,
                show_averaged=True):
        """Send results to gmsh for visualization.
        
        Parameters
        ----------
        source : str  'fem' or 'opensees'. None = last used.
        """
        from fem.utils.visualization import results2gmsh
        results = self._get_results(source)
        if not results:
            raise RuntimeError("No results — run solve_static() or set_results_opensees() first.")
        r = results[step]
        results2gmsh(
            output_file       =  self.mesh._file,
            mesh              = self.mesh,
            node_tags         = self.node_tags,
            element_tags_list = self.element_tags,
            u_3d              = r.u_3d,
            F_3d              = r.F_3d,
            R_3d              = r.R_3d,
            sigma_gmsh        = r.sigma,
            epsilon_gmsh      = r.epsilon,
            von_mises_gmsh    = r.vm,
            disp_factor       = disp_factor,
            show_disp         = show_disp,
            show_loads        = show_loads,
            show_reactions    = show_reactions,
            show_stress       = show_stress,
            show_strain       = show_strain,
            show_vm           = show_vm,
            show_averaged     = show_averaged,
        )

    def plot2gmsh_animate(self, disp_factor=10, source=None):
        """Send all result steps to gmsh as animation.
        
        Parameters
        ----------
        source : str  'fem' or 'opensees'. None = last used.
        """
        from fem.utils.visualization import animate_nodal_view
        results = self._get_results(source)
        if not results:
            raise RuntimeError("No results — run solve_static() or set_results_opensees() first.")
        data_steps = [r.u_3d for r in results]
        times      = [r.time for r in results]
        animate_nodal_view(
            output_file = self.mesh._file,
            node_tags   = self.node_tags,
            data_steps  = data_steps,
            view_name   = "Displacements",
            vector_type = 5,
            disp_factor = disp_factor,
            times       = times,
        )

    # -------------------------------------------------------------------------
    # IO
    # -------------------------------------------------------------------------

    def save(self, path: str):
        """
        Save model results to HDF5.

        Parameters
        ----------
        path : str  Output file path ending in .h5
        """
        import h5py
        with h5py.File(path, 'w') as f:
            f.attrs['analysis_type'] = self.analysis_type
            f.attrs['system_nDof']   = self.system_nDof
            f.attrs['mesh_file']     = self.mesh_file or ''

            # mesh geometry
            mg = f.create_group('mesh')
            tags   = np.array(list(self.mesh.nodes.keys()))
            coords = np.array(list(self.mesh.nodes.values()))
            mg.create_dataset('node_tags', data=tags)
            mg.create_dataset('coordinates', data=coords)
            if self.node_tags is not None:
                mg.create_dataset('node_tags_ordered', data=self.node_tags)
            if self.element_tags is not None:
                mg.create_dataset('element_tags', data=np.array(self.element_tags))

            # results
            def _write_results(group, results_list):
                for r in results_list:
                    sg = group.create_group(f'step_{r.step}')
                    sg.attrs['step'] = r.step
                    sg.attrs['time'] = r.time
                    for name in ('u', 'F', 'R', 'u_3d', 'F_3d', 'R_3d',
                                 'sigma', 'epsilon', 'vm',
                                 'sigma_principal', 'epsilon_principal',
                                 'sigma_nodal', 'epsilon_nodal', 'vm_nodal'):
                        arr = getattr(r, name)
                        if arr is not None:
                            sg.create_dataset(name, data=arr)

            _write_results(f.create_group('results_fem'),       self.results_fem)
            _write_results(f.create_group('results_opensees'),  self.results_opensees)

    @classmethod
    def load_results(cls, path: str) -> dict:
        """
        Load results from HDF5.

        Returns
        -------
        dict with keys 'fem' and 'opensees', each a list of FEMResult.
        """
        import h5py

        def _read_group(grp):
            results = []
            for key in sorted(grp.keys()):
                sg   = grp[key]
                step = int(sg.attrs['step'])
                time = float(sg.attrs['time'])
                data = {k: sg[k][()] for k in sg.keys()}
                results.append(FEMResult(
                    step=step, time=time,
                    u=data['u'], F=data['F'], R=data['R'],
                    u_3d=data['u_3d'], F_3d=data['F_3d'], R_3d=data['R_3d'],
                    sigma=data['sigma'], epsilon=data['epsilon'], vm=data['vm'],
                    sigma_principal=data.get('sigma_principal'),
                    epsilon_principal=data.get('epsilon_principal'),
                    sigma_nodal=data.get('sigma_nodal'),
                    epsilon_nodal=data.get('epsilon_nodal'),
                    vm_nodal=data.get('vm_nodal'),
                ))
            return results

        with h5py.File(path, 'r') as f:
            return {
                'fem'      : _read_group(f['results_fem'])      if 'results_fem'      in f else [],
                'opensees' : _read_group(f['results_opensees']) if 'results_opensees' in f else [],
            }

    def __repr__(self):
        n_el = len(self.elements) if self.elements is not None else 0
        return (f"FEMModel | nodes={len(self.node_map)}"
                f" | elements={n_el}"
                f" | steps_fem={len(self.results_fem)}"
                f" | steps_opensees={len(self.results_opensees)}"
                f" | analysis={self.analysis_type}")


    def _print_summary(self):
        sep = '--' * 40
        print(f"\n  FEM MODEL SUMMARY")
        print(sep)
        print(f"  Analysis type    : {self.analysis_type}")
        print(f"  Nodes            : {len(self.node_map)}")
        print(f"  system_nDof      : {self.system_nDof}")
        print(f"  Elements         : {len(self.elements) if self.elements is not None else 'None (OpenSees)'}")
        print(f"  Steps FEM        : {len(self.results_fem)}")
        print(f"  Steps OpenSees   : {len(self.results_opensees)}")

        print(f"\n  --- Sections ---")
        for phys_id, section in (self.section_dictionary or {}).items():
            pg   = self.mesh.physical_groups.get(phys_id)
            name = pg.name if pg else str(phys_id)
            t    = getattr(section, 'thickness', '-')
            E    = getattr(section.material, 'E', '-') if hasattr(section, 'material') else '-'
            nu   = getattr(section.material, 'nu', '-') if hasattr(section, 'material') else '-'
            print(f"  [{phys_id}] {name:20s}  t={t}  E={E:.2f}  nu={nu:.3f}")

        print(f"\n  --- Restrained nodes ---")
        for tag, cond in self._bcs.restrained_nodes.items():
            node = self.node_map[tag]
            print(f"  Node {tag:>6}  {str(node.coordinates):30s}  {cond}")

        print(f"\n  --- Loaded nodes (dim=0) ---")
        for node in self.node_map.values():
            if np.any(np.abs(node.nodalLoad) > 0):
                print(f"  Node {node.name:>6}  {str(node.coordinates):30s}  F={node.nodalLoad}")

        print(f"\n  --- Load vector ---")
        total = np.zeros(globalParameters['nDoF'])
        for node in self.node_map.values():
            total += node.nodalLoad
        nz = np.where(np.abs(self.F_load) > 0)[0]
        print(f"  Non-zero DOFs in F_load : {len(nz)}")
        print(f"  Total applied force     : {self.F_load[0::2].sum():.4f} (x)  {self.F_load[1::2].sum():.4f} (y)")
        print(sep)
        print()