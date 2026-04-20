"""
Gmsh Visualization Utilities for FEM Results
"""

import numpy as np
import gmsh


def add_element_data_view(viewname, 
                          element_tags, 
                          data, 
                          visible=False):
    viewnum = gmsh.view.add(viewname)
    gmsh.view.addHomogeneousModelData(
        tag=viewnum,
        step=0,
        time=0,
        modelName=gmsh.model.getCurrent(),
        dataType="ElementData",
        numComponents=-1,
        tags=element_tags,
        data=data.reshape((-1))
    )
    gmsh.view.option.setNumber(viewnum, "Visible", 1 if visible else 0)

    return viewnum


def add_node_data_view(viewname,
                       node_tags,
                       data,
                       visible=False,
                       vector_type=None,
                       factor=None,
                       arrow_size_max=None,
                       arrow_size_min=None,
                       deformed_view=None):  
    """
    Add a nodal data view to the current Gmsh model.

    Args:
        viewname       (str)        : Label shown in the Gmsh view panel.
        node_tags      (np.ndarray) : Array of node tags matching the model.
        data           (np.ndarray) : Data array, shape (n_nodes, n_components).
                                      1 component → scalar, 3 components → vector.
        visible        (bool)       : Whether the view is visible on creation.
                                      Default is False.
        vector_type    (int|None)   : Gmsh VectorType option.
                                        3 → displacement (deformed shape)
                                        4 → arrows at nodes
                                        5 → arrows + displacement
                                      Default is None (not set).
        factor         (float|None) : DisplacementFactor — scales the vector
                                      for deformed-shape views (vector_type=5).
                                      Default is None (not set).
        arrow_size_max (int|None)   : Maximum arrow size in pixels (ArrowSizeMax).
                                      Useful when force magnitudes span many orders
                                      of magnitude and small arrows are invisible.
                                      Default is None (not set → Gmsh default).
        arrow_size_min (int|None)   : Minimum arrow size in pixels (ArrowSizeMin).
                                      Default is None (not set → Gmsh default).

    Returns:
        viewnum (int): Tag of the created Gmsh view.
    """
    viewnum = gmsh.view.add(viewname)
    gmsh.view.addHomogeneousModelData(
        tag=viewnum,
        step=0,
        time=0,
        modelName=gmsh.model.getCurrent(),
        dataType="NodeData",
        numComponents=-1,
        tags=node_tags,
        data=data.reshape((-1))
    )

    gmsh.view.option.setNumber(viewnum, "Visible",       1 if visible else 0)
    gmsh.view.option.setNumber(viewnum, "GlyphLocation", 2)

    if vector_type is not None:
        gmsh.view.option.setNumber(viewnum, "VectorType",         vector_type)

    if factor is not None:
        gmsh.view.option.setNumber(viewnum, "DisplacementFactor", factor)

    if arrow_size_max is not None:
        gmsh.view.option.setNumber(viewnum, "ArrowSizeMax",       arrow_size_max)

    if arrow_size_min is not None:
        gmsh.view.option.setNumber(viewnum, "ArrowSizeMin",       arrow_size_min)

    if deformed_view is not None and factor is not None:
        gmsh.plugin.setNumber("Warp", "View",      viewnum)
        gmsh.plugin.setNumber("Warp", "OtherView", deformed_view)
        gmsh.plugin.setNumber("Warp", "Factor",    factor)
        gmsh.plugin.run("Warp")

    return viewnum


def compute_nodal_average(mesh,
                          element_tags_list,
                          element_data):
    """
    Compute nodal average of element data using raw mesh connectivity.

    Works for both FEM own solver and OpenSees flows — only needs the mesh.

    Parameters
    ----------
    mesh              : GMSHtools   mesh with node_map and elements
    element_tags_list : list[int]   element tags matching element_data order
    element_data      : np.ndarray  (n_elements,)  scalar value per element

    Returns
    -------
    np.ndarray  (n_nodes,)  averaged nodal values in mesh.nodes order
    """
    # build {element_tag: connectivity} across all groups
    tag_to_conn = {}
    for group in mesh.elements.values():
        for tag, conn in zip(group['element_tags'], group['connectivity']):
            tag_to_conn[tag] = conn

    # node tag → position index in mesh.nodes order
    all_node_tags = list(mesh.nodes.keys())
    node_idx      = {tag: i for i, tag in enumerate(all_node_tags)}
    nNodes        = len(all_node_tags)

    nodal_values = np.zeros(nNodes)
    count        = np.zeros(nNodes)

    for elem_tag, value in zip(element_tags_list, element_data):
        if elem_tag not in tag_to_conn:
            continue
        for tag in tag_to_conn[elem_tag]:
            if tag in node_idx:
                idx               = node_idx[tag]
                nodal_values[idx] += value
                count[idx]        += 1

    count[count == 0] = 1
    nodal_values /= count

    return nodal_values

def results2gmsh(output_file, mesh,
                 node_tags, element_tags_list,
                 u_3d, F_3d, R_3d,
                 sigma_gmsh,
                 epsilon_gmsh    = None,
                 von_mises_gmsh  = None,
                 disp_factor     = 10,
                 show_disp       = True,
                 show_loads      = True,
                 show_reactions  = True,
                 show_stress     = True,
                 show_strain     = True,
                 show_vm         = True,
                 show_averaged   = True):
    """
    Visualize FEM results in gmsh from pre-computed arrays.

    Parameters
    ----------
    output_file       : str           Path to .msh file
    mesh              : GMSHtools     Mesh object
    node_tags         : np.ndarray    Node tags in gmsh order
    element_tags_list : list          Element tags
    u_3d              : np.ndarray    Displacements (n_nodes, 3)
    F_3d              : np.ndarray    Applied loads (n_nodes, 3)
    R_3d              : np.ndarray    Reactions (n_nodes, 3)
    sigma_gmsh        : np.ndarray    Stresses (n_elements, 3 or 6)
    epsilon_gmsh      : np.ndarray    Strains (n_elements, 3 or 6) — optional
    von_mises_gmsh    : np.ndarray    Von Mises (n_elements,) — optional
    disp_factor       : float         Displacement scale factor
    show_*            : bool          Toggle each view
    """
    n_comp = sigma_gmsh.shape[1]
    if n_comp == 6:
        stress_labels = ['Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Sxz']
        strain_labels = ['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz']
    else:
        stress_labels = ['Sxx', 'Syy', 'Sxy']
        strain_labels = ['Exx', 'Eyy', 'Exy']

    # compute von Mises if not provided
    if von_mises_gmsh is None:
        sxx = sigma_gmsh[:, 0]
        syy = sigma_gmsh[:, 1]
        sxy = sigma_gmsh[:, 3] if n_comp == 6 else sigma_gmsh[:, 2]
        von_mises_gmsh = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

    gmsh.initialize()
    gmsh.open(output_file)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0)

    disp_view = None

    if show_disp:
        disp_view = add_node_data_view("Displacements", node_tags, u_3d,
                                       vector_type=5, factor=disp_factor)

    if show_loads and np.any(F_3d != 0):
        add_node_data_view("Applied Loads", node_tags, F_3d,
                           arrow_size_max=60, arrow_size_min=20)

    if show_reactions:
        add_node_data_view("Reactions", node_tags, R_3d)

    if show_stress:
        for col, name in enumerate(stress_labels):
            add_element_data_view(f"Stress {name}", element_tags_list, sigma_gmsh[:, col])

    if show_strain and epsilon_gmsh is not None:
        for col, name in enumerate(strain_labels):
            add_element_data_view(f"Strain {name}", element_tags_list, epsilon_gmsh[:, col])

    if show_vm:
        add_element_data_view("Von Mises", element_tags_list, von_mises_gmsh)

    if show_averaged and disp_view is not None:
        for col, name in enumerate(stress_labels):
            nodal = compute_nodal_average(mesh, element_tags_list, sigma_gmsh[:, col])
            add_node_data_view(f"{name} Averaged", node_tags, nodal,
                               deformed_view=disp_view)
        vm_nodal = compute_nodal_average(mesh, element_tags_list, von_mises_gmsh)
        add_node_data_view("Von Mises Averaged", node_tags, vm_nodal,
                           deformed_view=disp_view)
        if epsilon_gmsh is not None:
            for col, name in enumerate(strain_labels):
                nodal = compute_nodal_average(mesh, element_tags_list, epsilon_gmsh[:, col])
                add_node_data_view(f"{name} Averaged", node_tags, nodal,
                                   deformed_view=disp_view)

    gmsh.fltk.run()
    gmsh.finalize()


def opensees2gmsh(output_file, mesh, ops, solid_group_name,
                  F_nodal       = None,
                  disp_factor   = 10,
                  material      = None,
                  analysis_type = 'planeStress',
                  show_disp     = True,
                  show_loads    = True,
                  show_reactions= True,
                  show_stress   = True,
                  show_strain   = True,
                  show_vm       = True,
                  show_averaged = True):
    """
    Extract results from OpenSees and visualize in gmsh.

    Parameters
    ----------
    output_file      : str        Path to .msh file
    mesh             : GMSHtools  Mesh object
    ops              : module     OpenSees module
    solid_group_name : str        Physical group name for solid elements
    F_nodal          : dict       {tag: np.array} from build_load_vector (optional)
    disp_factor      : float      Displacement scale factor
    material         : Material   Material object with E and nu (for strain computation)
    analysis_type    : str        'planeStress' or 'planeStrain' (2D only)
    show_*           : bool       Toggle each view
    """
    # --- Extract nodal results ---
    nDOF = mesh.system_nDof // len(mesh.nodes)
    ops.reactions()

    n_nodes   = len(mesh.nodes)
    node_tags = np.array(list(mesh.nodes.keys()))
    u_3d      = np.zeros((n_nodes, 3))
    R_3d      = np.zeros((n_nodes, 3))
    F_3d      = np.zeros((n_nodes, 3))

    for i, tag in enumerate(mesh.nodes):
        for j in range(nDOF):
            u_3d[i, j] = ops.nodeDisp(tag, j+1)
            R_3d[i, j] = ops.nodeReaction(tag, j+1)

    if F_nodal is not None:
        for i, tag in enumerate(mesh.nodes):
            f = F_nodal.get(tag, np.zeros(nDOF))
            F_3d[i, :len(f)] = f[:nDOF]

    # --- Extract element results ---
    element_tags_list   = mesh.physical_groups[solid_group_name].elements['element_tags']
    n_elements          = len(element_tags_list)
    sigma_gmsh          = np.zeros((n_elements, 6))
    strain_gmsh         = np.zeros((n_elements, 6))
    von_mises_gmsh      = np.zeros(n_elements)
    n_stress_components = 3
    compute_strain      = material is not None

    for i, elem_tag in enumerate(element_tags_list):
        # stress = ops.eleResponse(elem_tag, 'stress')
        stress = ops.eleResponse(elem_tag, 'stresses')
        if stress:
            s      = np.array(stress)
            n_comp = 6 if len(s) % 6 == 0 and len(s) >= 6 else 3
            n_stress_components = n_comp
            s      = s.reshape(-1, n_comp).mean(axis=0)
            sigma_gmsh[i, :n_comp] = s

            sxx, syy = s[0], s[1]
            sxy      = s[3] if n_comp == 6 else s[2]
            von_mises_gmsh[i] = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)

            if compute_strain:
                E, nu = material.E, material.nu
                if n_comp == 6:
                    szz, sxy, syz, sxz = s[2], s[3], s[4], s[5]
                    strain_gmsh[i, 0] = (sxx - nu*(syy+szz)) / E
                    strain_gmsh[i, 1] = (syy - nu*(sxx+szz)) / E
                    strain_gmsh[i, 2] = (szz - nu*(sxx+syy)) / E
                    strain_gmsh[i, 3] = 2*sxy*(1+nu) / E
                    strain_gmsh[i, 4] = 2*syz*(1+nu) / E
                    strain_gmsh[i, 5] = 2*sxz*(1+nu) / E
                elif analysis_type == 'planeStress':
                    sxy = s[2]
                    strain_gmsh[i, 0] = (sxx - nu*syy) / E
                    strain_gmsh[i, 1] = (syy - nu*sxx) / E
                    strain_gmsh[i, 2] = 2*sxy*(1+nu) / E
                elif analysis_type == 'planeStrain':
                    sxy = s[2]
                    strain_gmsh[i, 0] = ((1-nu**2)*sxx - nu*(1+nu)*syy) / E
                    strain_gmsh[i, 1] = ((1-nu**2)*syy - nu*(1+nu)*sxx) / E
                    strain_gmsh[i, 2] = 2*sxy*(1+nu) / E

    # trim arrays to detected component count
    sigma_gmsh = sigma_gmsh[:, :n_stress_components]
    epsilon_gmsh = strain_gmsh[:, :n_stress_components] if compute_strain else None

    results2gmsh(
        output_file       = output_file,
        mesh              = mesh,
        node_tags         = node_tags,
        element_tags_list = element_tags_list,
        u_3d              = u_3d,
        F_3d              = F_3d,
        R_3d              = R_3d,
        sigma_gmsh        = sigma_gmsh,
        epsilon_gmsh      = epsilon_gmsh,
        von_mises_gmsh    = von_mises_gmsh,
        disp_factor       = disp_factor,
        show_disp         = show_disp,
        show_loads        = show_loads,
        show_reactions    = show_reactions,
        show_stress       = show_stress,
        show_strain       = show_strain,
        show_vm           = show_vm,
        show_averaged     = show_averaged,
    )



def animate_nodal_view(output_file, node_tags, data_steps,
                       view_name   = "Animation",
                       vector_type = 5,
                       disp_factor = 1.0,
                       times       = None):
    """
    Create animated gmsh view from a list of nodal arrays.

    Parameters
    ----------
    output_file : str          Path to .msh file.
    node_tags   : ndarray      Node tags matching the model.
    data_steps  : list         List of ndarray (n_nodes, 3), one per frame.
    view_name   : str          Label shown in gmsh.
    vector_type : int          Gmsh VectorType (5 = deformed shape).
    disp_factor : float        Displacement scale factor.
    times       : list[float]  Time labels per frame. Defaults to 0,1,2,...
    """
    if times is None:
        times = list(range(len(data_steps)))

    gmsh.initialize()
    gmsh.open(output_file)

    view = gmsh.view.add(view_name)
    for step, (data, t) in enumerate(zip(data_steps, times)):
        gmsh.view.addHomogeneousModelData(
            tag           = view,
            step          = step,
            time          = float(t),
            modelName     = gmsh.model.getCurrent(),
            dataType      = "NodeData",
            numComponents = -1,
            tags          = node_tags,
            data          = np.array(data).reshape(-1),
        )

    gmsh.view.option.setNumber(view, "VectorType",         vector_type)
    gmsh.view.option.setNumber(view, "DisplacementFactor", disp_factor)

    gmsh.fltk.run()
    gmsh.finalize()


def animate_results(output_file, mesh, node_tags, element_tags, results,
                    disp_factor    = 10,
                    show_disp      = True,
                    show_stress    = True,
                    show_strain    = True,
                    show_vm        = True,
                    show_averaged  = True,
                    deformed_shape = True):
    """
    Send all result steps to gmsh as animated views.

    Parameters
    ----------
    output_file    : str
    mesh           : GMSHtools
    node_tags      : ndarray
    element_tags   : list
    results        : list[FEMResult]
    disp_factor    : float
    show_*         : bool
    deformed_shape : bool  Apply warp to nodal averaged views.
    """
    n_comp = results[0].sigma.shape[1]
    if n_comp == 6:
        stress_labels = ['Sxx', 'Syy', 'Szz', 'Sxy', 'Syz', 'Sxz']
        strain_labels = ['Exx', 'Eyy', 'Ezz', 'Exy', 'Eyz', 'Exz']
    else:
        stress_labels = ['Sxx', 'Syy', 'Sxy']
        strain_labels = ['Exx', 'Eyy', 'Exy']

    times = [r.time for r in results]

    gmsh.initialize()
    gmsh.open(output_file)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0)

    def _node_view(name, data_steps, vector_type=1, factor=1.0):
        view = gmsh.view.add(name)
        for step, (data, t) in enumerate(zip(data_steps, times)):
            gmsh.view.addHomogeneousModelData(
                tag=view, step=step, time=float(t),
                modelName=gmsh.model.getCurrent(),
                dataType="NodeData", numComponents=-1,
                tags=node_tags, data=np.array(data).reshape(-1))
        gmsh.view.option.setNumber(view, "VectorType",         vector_type)
        gmsh.view.option.setNumber(view, "DisplacementFactor", factor)
        gmsh.view.option.setNumber(view, "Visible", 0)
        return view

    def _elem_view(name, data_steps):
        view = gmsh.view.add(name)
        for step, (data, t) in enumerate(zip(data_steps, times)):
            gmsh.view.addHomogeneousModelData(
                tag=view, step=step, time=float(t),
                modelName=gmsh.model.getCurrent(),
                dataType="ElementData", numComponents=-1,
                tags=element_tags, data=np.array(data).reshape(-1))
        gmsh.view.option.setNumber(view, "Visible", 0)
        return view

    def _warp(view, disp_view):
        gmsh.plugin.setNumber("Warp", "View",      view)
        gmsh.plugin.setNumber("Warp", "OtherView", disp_view)
        gmsh.plugin.setNumber("Warp", "Factor",    disp_factor)
        gmsh.plugin.run("Warp")

    disp_view = None
    if show_disp:
        disp_view = _node_view("Displacements",
                               [r.u_3d for r in results],
                               vector_type=5, factor=disp_factor)
        gmsh.view.option.setNumber(disp_view, "Visible", 1)

    if show_stress:
        for col, name in enumerate(stress_labels):
            _elem_view(f"Stress {name}", [r.sigma[:, col] for r in results])

    if show_strain and results[0].epsilon is not None:
        for col, name in enumerate(strain_labels):
            _elem_view(f"Strain {name}", [r.epsilon[:, col] for r in results])

    if show_vm:
        _elem_view("Von Mises", [r.vm for r in results])

    if show_averaged and results[0].sigma_nodal is not None:
        for col, name in enumerate(stress_labels):
            view = _node_view(f"{name} Averaged",
                              [r.sigma_nodal[:, col] for r in results])
            if deformed_shape and disp_view is not None:
                _warp(view, disp_view)

        view = _node_view("Von Mises Averaged", [r.vm_nodal for r in results])
        if deformed_shape and disp_view is not None:
            _warp(view, disp_view)

        if results[0].epsilon_nodal is not None:
            for col, name in enumerate(strain_labels):
                view = _node_view(f"{name} Averaged",
                                  [r.epsilon_nodal[:, col] for r in results])
                if deformed_shape and disp_view is not None:
                    _warp(view, disp_view)

    gmsh.fltk.run()
    gmsh.finalize()
