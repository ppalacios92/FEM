# %%
from fem import (
    # Gmsh tools
    GMSHtools,
    # Visualization — Gmsh
    add_element_data_view, add_node_data_view, compute_nodal_average,
    results2gmsh,
    # Plotting — matplotlib
    plot_gmsh_mesh,opensees2gmsh,
    # Parameters
    globalParameters,
)

import os
import numpy as np
import matplotlib.pyplot as plt
import gmsh

np.set_printoptions(suppress=True, precision=6, linewidth=400)

# %%
globalParameters['nDoF'] = 3
globalParameters['nDIM'] = 3

# %%
# General model parameters
output_file = 'example_PP.msh'

# %%
# read mesh — node_map and system_nDof auto-generated
mesh = GMSHtools(output_file)

# %%
load_dictionary = {
                22:   {'value': 100, 'direction': '-z'},     
}

# %%
# build lumped nodal force vector
F_nodal = mesh.build_load_vector(load_dictionary)

# assemble to global vector
F_load = np.zeros(mesh.system_nDof)
for tag, f_vec in F_nodal.items():
    if tag in mesh.node_map:
        F_load[mesh.node_map[tag].idx[:len(f_vec)]] += f_vec
F_load[np.abs(F_load) < 1e-4] = 0.0

# %%
# %matplotlib widget

plot_gmsh_mesh(mesh,
               show_node_labels   = False,
               show_element_labels= False,
               show_node_points   = False,
               view_3d            = True,   elev= 45, azim= -45,
               figsize            = (12, 8))

# %% [markdown]
# ## Opensees

# %%
import opensees as ops
import opsvis as opsv

ops.wipe()
ops.model('basicBuilder', '-ndm', 3, '-ndf', 3)

# %%
# Nodes
for tag, (x, y, z) in mesh.nodes.items():
    ops.node(tag, x, y , z)

# %%
# Boundary conditions
fixed_nodes = set()
for tag in mesh.physical_groups['support'].nodes:
    if tag not in fixed_nodes:
        fixed_nodes.add(tag)
        # ops.fix(tag, 1, 1, 1, 1, 1, 1)
        ops.fix(tag, 1, 1, 1)

# %%
# Material
E = 3500      
nu = 0.36     
rho = 2400e-9  
ops.nDMaterial('ElasticIsotropic', 1, E, nu, rho)


# %%
# group = mesh.physical_groups['solid'].elements
# for elem_tag, conn in zip(group['element_tags'], group['connectivity']):
#     n1, n2, n3, n4 = conn
#     ops.element('FourNodeTetrahedron', elem_tag, n1, n2, n3, n4, 1)

group = mesh.physical_groups['solid'].elements
for elem_tag, conn in zip(group['element_tags'], group['connectivity']):
    ops.element('TenNodeTetrahedron', elem_tag, *conn, 1)
    

# %%
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
for tag, force in F_nodal.items():
    if np.any(np.abs(force) > 0):
        ops.load(tag, *force.tolist())


# %%
NstepGravity=10
DGravity=1/NstepGravity

ops.system("FullGeneral")
ops.numberer("Plain")
ops.constraints("Plain")
ops.integrator("LoadControl", DGravity )
ops.test("NormUnbalance", 1.0e-6, 100 , 0)
ops.algorithm("Newton")
ops.analysis("Static")

ops.analyze(NstepGravity)

# %%
# elem_tag = mesh.physical_groups['solid'].elements['element_tags'][0]
# print(ops.eleResponse(elem_tag, 'stresses'))
# print(ops.eleResponse(elem_tag, 'stress'))
# print(ops.eleResponse(elem_tag, 'stressAtNodes'))

# %% [markdown]
# ## Vertical Analysis

# %%


opensees2gmsh(
    output_file      = output_file,
    mesh             = mesh,
    ops              = ops,
    solid_group_name = 'solid',
    F_nodal          = F_nodal,
    # nDOF             = 3,
    disp_factor      = 10,
    show_disp        = True,
    show_loads       = True,
    show_reactions   = True,
    show_stress      = True,
    show_vm          = True,
    show_averaged    = True,
)


