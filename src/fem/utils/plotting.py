"""
FEM Plotting Utilities — matplotlib-based visualization for 2D FEM results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.collections as mc
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -- Internal helpers ----------------------------------------------------------

def _get_node_coords(nodes: list, u: np.ndarray = None, sfac: float = 1.0) -> np.ndarray:
    """
    Return node coordinates, optionally shifted by scaled displacements.
    sfac is applied for plotting only — original data is never modified.
    """
    coords = np.array([n.coordinates[:2] for n in nodes], dtype=float)
    if u is not None and sfac != 0:
        for i, node in enumerate(nodes):
            coords[i, 0] += sfac * u[node.idx[0]]
            coords[i, 1] += sfac * u[node.idx[1]]
    return coords


def _get_triangulation(nodes: list, elements: list,
                        u: np.ndarray = None, sfac: float = 1.0) -> mtri.Triangulation:
    """Build a matplotlib Triangulation from FEM nodes and elements."""
    coords    = _get_node_coords(nodes, u, sfac)
    node_map  = {node.name: i for i, node in enumerate(nodes)}
    triangles = []

    for element in elements:
        idx = [node_map[n.name] for n in element.nodes]
        n   = len(idx)
        if n == 3:
            triangles.append(idx[:3])
        elif n == 6:
            triangles.append([idx[0], idx[1], idx[2]])
        elif n == 4:
            triangles.append([idx[0], idx[1], idx[2]])
            triangles.append([idx[0], idx[2], idx[3]])
        elif n == 9:
            triangles.append([idx[0], idx[1], idx[8]])
            triangles.append([idx[1], idx[2], idx[8]])
            triangles.append([idx[2], idx[3], idx[8]])
            triangles.append([idx[3], idx[0], idx[8]])

    return mtri.Triangulation(coords[:, 0], coords[:, 1], triangles)


def _extract_field(elements: list, u: np.ndarray, component: str) -> np.ndarray:
    """Extract a scalar field per element from FEM results."""
    stress_map = {'sxx': 0, 'syy': 1, 'sxy': 2}
    strain_map = {'exx': 0, 'eyy': 1, 'exy': 2}
    values     = np.zeros(len(elements))

    for i, element in enumerate(elements):
        results = element.get_results(u)
        if component in stress_map:
            values[i] = results['stress'].flatten()[stress_map[component]]
        elif component in strain_map:
            values[i] = results['strain'].flatten()[strain_map[component]]
        elif component == 'vmis':
            sxx, syy, sxy = results['stress'].flatten()
            values[i] = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
        elif component == 's1':
            values[i] = results['principal_stress'].flatten()[0]
        elif component == 's2':
            values[i] = results['principal_stress'].flatten()[1]
        elif component == 'e1':
            values[i] = results['principal_strain'].flatten()[0]
        elif component == 'e2':
            values[i] = results['principal_strain'].flatten()[1]
        else:
            raise ValueError(
                f"Unknown component '{component}'. "
                f"Valid: 'sxx','syy','sxy','vmis','s1','s2','exx','eyy','exy','e1','e2'"
            )
    return values


def _nodal_average(nodes: list, elements: list, element_values: np.ndarray) -> np.ndarray:
    """Compute nodal average of an element scalar field."""
    node_map    = {node.name: i for i, node in enumerate(nodes)}
    nNodes      = len(nodes)
    nodal_sum   = np.zeros(nNodes)
    nodal_count = np.zeros(nNodes)

    for element, value in zip(elements, element_values):
        for node in element.nodes:
            idx = node_map[node.name]
            nodal_sum[idx]   += value
            nodal_count[idx] += 1

    nodal_count[nodal_count == 0] = 1
    return nodal_sum / nodal_count


def _add_colorbar(fig, ax, mappable, label: str):
    """Add a colorbar that matches the axes height exactly."""
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes('right', size='4%', pad=0.08)
    fig.colorbar(mappable, cax=cax, label=label)


def _draw_element_edges(ax, nodes, elements, u=None, sfac=1.0,
                        color='k', linewidth=0.3, alpha=0.4, zorder=3):
    """Draw element boundary edges over a plot."""
    coords   = _get_node_coords(nodes, u, sfac)
    node_map = {node.name: i for i, node in enumerate(nodes)}
    polygons = [coords[[node_map[n.name] for n in el.nodes], :] for el in elements]
    ax.add_collection(mc.PolyCollection(polygons,
                                        facecolors='none',
                                        edgecolors=color,
                                        linewidths=linewidth,
                                        alpha=alpha,
                                        zorder=zorder))


def _draw_node_points(ax, nodes, u=None, sfac=1.0,
                      color='k', markersize=1.5, alpha=0.5, zorder=4):
    """Draw node points over a plot."""
    coords = _get_node_coords(nodes, u, sfac)
    ax.plot(coords[:, 0], coords[:, 1], '.',
            color=color, markersize=markersize, alpha=alpha, zorder=zorder)


def _draw_supports(ax, nodes, u=None, sfac=1.0):
    """Draw support symbols on all restrained nodes."""
    for node in nodes:
        if hasattr(node, 'restrain') and node.restrain is not None:
            if any(r == 'r' for r in node.restrain):
                _draw_support(ax, node, u=u, sfac=sfac)


def _draw_support(ax, node, color='mediumpurple', size=6, u=None, sfac=1.0):
    """
    Draw support symbol based on 2D restraint conditions.

    rx & ry → square (pinned)
    rx only → triangle up
    ry only → triangle right

    If u and sfac are provided, symbol is placed at deformed position.
    """
    x, y  = node.coordinates[0], node.coordinates[1]
    if u is not None:
        x += sfac * u[node.idx[0]]
        y += sfac * u[node.idx[1]]
    restr = list(node.restrain)
    rx    = restr[0] == 'r'
    ry    = restr[1] == 'r'
    if rx and ry:
        ax.plot(x, y, 's', color=color, markersize=size, zorder=6)
    elif rx:
        ax.plot(x, y, '^', color=color, markersize=size, zorder=6)
    elif ry:
        ax.plot(x, y, '>', color=color, markersize=size, zorder=6)


def _save_figure(fig, save: str):
    """Save figure at 300 dpi."""
    import os
    os.makedirs(os.path.dirname(save) if os.path.dirname(save) else '.', exist_ok=True)
    fig.savefig(save, dpi=300, bbox_inches='tight')


# -- Public functions ----------------------------------------------------------

def plot_mesh(nodes=None,
              elements=None,
              show_node_labels: bool = False,
              show_element_labels: bool = False,
              show_supports: bool = True,
              show_element_edges: bool = True,
              show_node_points: bool = True,
              figsize: tuple = (12, 8),
              ax=None,
              save: str = None):
    """
    Plot the FEM mesh.

    Parameters
    ----------
    nodes               : list of Node
    elements            : list of Element
    show_node_labels    : bool
    show_element_labels : bool
    show_supports       : bool
    show_element_edges  : bool
    show_node_points    : bool
    figsize             : tuple
    ax                  : matplotlib Axes or None
    save                : str or None
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=figsize)

    if elements is not None and nodes is not None:
        coords   = _get_node_coords(nodes)
        node_map = {node.name: i for i, node in enumerate(nodes)}
        polygons = [coords[[node_map[n.name] for n in el.nodes], :] for el in elements]

        ax.add_collection(mc.PolyCollection(polygons,
                                            facecolors='#f0f0f0',
                                            edgecolors='tab:blue' if show_element_edges else 'none',
                                            linewidths=0.4,
                                            zorder=1))

        if show_element_labels:
            for element in elements:
                x_c, y_c = element.get_centroid()
                ax.text(x_c, y_c, f'{element.element_tag}',
                        color='tab:blue', ha='center', va='center', fontsize=6, zorder=3)

    if nodes is not None:
        if show_node_points:
            coords = _get_node_coords(nodes)
            ax.plot(coords[:, 0], coords[:, 1], '.',
                    color='tab:blue', markersize=2, zorder=2)

        if show_node_labels:
            for node in nodes:
                x, y = node.coordinates[:2]
                ax.text(x, y, f' {node.name}',
                        color='tab:blue', fontsize=6,
                        ha='left', va='bottom', zorder=3)

        if show_supports:
            _draw_supports(ax, nodes)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Mesh')

    if save:
        _save_figure(fig, save)

    plt.show()


def plot_loads_2d(nodes: list,
                  elements: list,
                  F_load: np.ndarray,
                  show_element_edges: bool = True,
                  show_node_points: bool = True,
                  show_supports: bool = True,
                  figsize: tuple = (12, 8),
                  ax=None,
                  save: str = None):
    """
    Plot applied nodal loads as arrows over the mesh background.

    Arrows are normalized — all the same length regardless of magnitude.
    Direction only. For quantitative values use the load vector directly.

    Parameters
    ----------
    nodes               : list of Node
    elements            : list of Element
    F_load              : np.ndarray   Global load vector
    show_element_edges  : bool
    show_node_points    : bool
    show_supports       : bool
    figsize             : tuple
    ax                  : matplotlib Axes or None
    save                : str or None
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=figsize)

    if elements is not None and nodes is not None:
        coords   = _get_node_coords(nodes)
        node_map = {node.name: i for i, node in enumerate(nodes)}
        polygons = [coords[[node_map[n.name] for n in el.nodes], :] for el in elements]
        ax.add_collection(mc.PolyCollection(polygons,
                                            facecolors='#f0f0f0',
                                            edgecolors='#888888' if show_element_edges else 'none',
                                            linewidths=0.4,
                                            zorder=1))

    if show_node_points:
        _draw_node_points(ax, nodes, color='#444444', markersize=2)

    if show_supports:
        _draw_supports(ax, nodes)

    # Collect loaded nodes — normalize to unit vectors
    loaded_nodes = [(node, F_load[node.idx[0]], F_load[node.idx[1]])
                    for node in nodes
                    if abs(F_load[node.idx[0]]) > 0 or abs(F_load[node.idx[1]]) > 0]

    if loaded_nodes:
        xs  = np.array([n.coordinates[0] for n, _, _ in loaded_nodes])
        ys  = np.array([n.coordinates[1] for n, _, _ in loaded_nodes])
        fxs = np.array([fx for _, fx, _ in loaded_nodes], dtype=float)
        fys = np.array([fy for _, _, fy in loaded_nodes], dtype=float)

        # Normalize each vector to unit length
        mags = np.sqrt(fxs**2 + fys**2)
        mags[mags == 0] = 1
        uxs = fxs / mags
        uys = fys / mags

        # Arrow length = 5% of the model bounding box diagonal
        all_coords = _get_node_coords(nodes)
        bbox_diag  = np.sqrt((np.ptp(all_coords[:, 0]))**2 + (np.ptp(all_coords[:, 1]))**2)
        arrow_len  = 0.05 * bbox_diag

        ax.quiver(xs, ys, uxs, uys,
                  scale=1.0 / arrow_len,
                  scale_units='xy',
                  angles='xy',
                  color='tab:blue',
                  width=0.002,
                  headwidth=4,
                  headlength=5,
                  zorder=5)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Applied Loads')

    if save:
        _save_figure(fig, save)

    plt.show()


def plot_deformed(nodes: list,
                  elements: list,
                  u: np.ndarray,
                  component: str = 'umag',
                  sfac: float = 1.0,
                  cmap: str = 'jet',
                  limit: tuple = None,
                  show_element_edges: bool = True,
                  show_node_points: bool = False,
                  show_supports: bool = True,
                  figsize: tuple = (12, 8),
                  ax=None,
                  save: str = None):
    """
    Plot undeformed mesh (transparent background) with deformed mesh on top,
    colored by displacement component using a smooth filled contour surface.

    Parameters
    ----------
    nodes               : list of Node
    elements            : list of Element
    u                   : np.ndarray      Global displacement vector
    component           : str             'ux', 'uy', 'umag'
    sfac                : float           Displacement scale factor (plot only)
    cmap                : str             Matplotlib colormap or 'basic'
    limit               : tuple or None   (min, max) — fixes cmap range; outside → tab:red
    show_element_edges  : bool
    show_node_points    : bool
    show_supports       : bool
    figsize             : tuple
    ax                  : matplotlib Axes or None
    save                : str or None
    """
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=figsize)

    node_map = {node.name: i for i, node in enumerate(nodes)}

    # ── Undeformed — transparent background ───────────────────────────────
    coords_orig = _get_node_coords(nodes)
    polygons_orig = [coords_orig[[node_map[n.name] for n in el.nodes], :] for el in elements]
    ax.add_collection(mc.PolyCollection(polygons_orig,
                                        facecolors='#f0f0f0',
                                        edgecolors='tab:blue',
                                        linewidths=0.4,
                                        alpha=0.3,
                                        zorder=1))

    # ── Displacement values ────────────────────────────────────────────────
    disp_vals = np.zeros(len(nodes))
    for i, node in enumerate(nodes):
        ux = u[node.idx[0]]
        uy = u[node.idx[1]]
        if component == 'ux':
            disp_vals[i] = ux
        elif component == 'uy':
            disp_vals[i] = uy
        elif component == 'umag':
            disp_vals[i] = np.sqrt(ux**2 + uy**2)
        else:
            raise ValueError(f"Unknown component '{component}'. Valid: 'ux','uy','umag'")

    vmin      = limit[0] if limit is not None else disp_vals.min()
    vmax      = limit[1] if limit is not None else disp_vals.max()
    use_basic = (cmap == 'basic')
    cmap_obj  = plt.get_cmap('Greys') if use_basic else plt.get_cmap(cmap)
    norm      = plt.Normalize(vmin=vmin, vmax=vmax)

    coords_def = _get_node_coords(nodes, u, sfac)
    triang_def = _get_triangulation(nodes, elements, u, sfac)

    if not use_basic:
        # Smooth contourf surface
        tcf = ax.tricontourf(triang_def, disp_vals, levels=50,
                             cmap=cmap, norm=norm, zorder=2)

        # Out-of-range overlay
        if limit is not None:
            out_polys = []
            for el in elements:
                idx = [node_map[n.name] for n in el.nodes]
                if np.mean(disp_vals[idx]) < vmin or np.mean(disp_vals[idx]) > vmax:
                    out_polys.append(coords_def[idx, :])
            if out_polys:
                ax.add_collection(mc.PolyCollection(out_polys,
                                                    facecolors='tab:red',
                                                    edgecolors='none',
                                                    alpha=0.7, zorder=3))

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        _add_colorbar(fig, ax, sm, component)

    else:
        polygons_def = []
        colors_def   = []
        for el in elements:
            idx      = [node_map[n.name] for n in el.nodes]
            avg_disp = np.mean(disp_vals[idx])
            polygons_def.append(coords_def[idx, :])
            outside = (avg_disp < vmin or avg_disp > vmax) if limit is not None else False
            colors_def.append('tab:red' if outside else '#d0d0d0')
        ax.add_collection(mc.PolyCollection(polygons_def,
                                            facecolors=colors_def,
                                            edgecolors='none',
                                            alpha=0.9, zorder=2))

    if show_element_edges:
        _draw_element_edges(ax, nodes, elements, u, sfac,
                            color='k', linewidth=0.3, alpha=0.4, zorder=4)

    if show_node_points:
        _draw_node_points(ax, nodes, u, sfac, zorder=5)

    if show_supports:
        _draw_supports(ax, nodes, u=u, sfac=sfac)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    limit_str = f'  |  limit={limit}' if limit is not None else ''
    ax.set_title(f'Deformed  |  {component}  |  sfac={sfac}{limit_str}')

    if save:
        _save_figure(fig, save)

    plt.show()


def plot_field_2d(nodes: list,
                  elements: list,
                  u: np.ndarray,
                  component: str = 'vmis',
                  result_type: str = 'nodal_avg',
                  deformed: bool = False,
                  sfac: float = 1.0,
                  limit: tuple = None,
                  levels: int = None,
                  cmap: str = 'jet',
                  show_element_edges: bool = True,
                  show_node_points: bool = False,
                  show_supports: bool = True,
                  figsize: tuple = (12, 8),
                  ax=None,
                  save: str = None):
    """
    Plot a scalar stress or strain field over the FEM mesh.

    Renders a smooth tricontourf surface first, then draws element edges
    and optional isolines on top so individual elements remain visible.

    Parameters
    ----------
    nodes               : list of Node
    elements            : list of Element
    u                   : np.ndarray         Global displacement vector
    component           : str                'sxx','syy','sxy','vmis','s1','s2',
                                             'exx','eyy','exy','e1','e2'
    result_type         : str                'nodal_avg' or 'element'
    deformed            : bool               Plot over deformed shape
    sfac                : float              Displacement scale factor (plot only)
    limit               : tuple or None      (min, max) — fixes cmap range; outside → tab:red
                                             Use cmap='basic' for gray/red mode.
    levels              : int or None        Number of isolines drawn on top
    cmap                : str                Matplotlib colormap or 'basic'
    show_element_edges  : bool
    show_node_points    : bool
    show_supports       : bool
    figsize             : tuple
    ax                  : matplotlib Axes or None
    save                : str or None
    """
    u_plot       = u if deformed else None
    element_vals = _extract_field(elements, u, component)
    node_map     = {node.name: i for i, node in enumerate(nodes)}

    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=figsize)

    vmin      = limit[0] if limit is not None else element_vals.min()
    vmax      = limit[1] if limit is not None else element_vals.max()
    use_basic = (cmap == 'basic')
    cmap_obj  = plt.get_cmap('Greys') if use_basic else plt.get_cmap(cmap)
    norm      = plt.Normalize(vmin=vmin, vmax=vmax)

    # ── Nodal values for smooth surface ───────────────────────────────────
    nodal_vals = _nodal_average(nodes, elements, element_vals)
    triang     = _get_triangulation(nodes, elements, u_plot, sfac)
    coords     = _get_node_coords(nodes, u_plot, sfac)

    if not use_basic:
        # Smooth filled contour surface
        ax.tricontourf(triang, nodal_vals, levels=50,
                       cmap=cmap, norm=norm, zorder=1)

        # Isolines on top
        if levels is not None:
            ax.tricontour(triang, nodal_vals, levels=levels,
                          colors='k', linewidths=0.5, alpha=0.5,
                          norm=norm, zorder=3)

        # Out-of-range overlay
        if limit is not None:
            out_polys = []
            for el, val in zip(elements, element_vals):
                if val < vmin or val > vmax:
                    idx = [node_map[n.name] for n in el.nodes]
                    out_polys.append(coords[idx, :])
            if out_polys:
                ax.add_collection(mc.PolyCollection(out_polys,
                                                    facecolors='tab:red',
                                                    edgecolors='none',
                                                    alpha=0.7, zorder=4))

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        _add_colorbar(fig, ax, sm, component)

    else:
        # Basic mode — gray + red for out-of-range
        polygons_f = []
        colors_f   = []
        for el, val in zip(elements, element_vals):
            idx = [node_map[n.name] for n in el.nodes]
            polygons_f.append(coords[idx, :])
            outside = (val < vmin or val > vmax) if limit is not None else False
            colors_f.append('tab:red' if outside else '#d0d0d0')
        ax.add_collection(mc.PolyCollection(polygons_f,
                                            facecolors=colors_f,
                                            edgecolors='none',
                                            alpha=0.9, zorder=1))

    # ── Element edges on top of field ─────────────────────────────────────
    if show_element_edges:
        _draw_element_edges(ax, nodes, elements, u_plot, sfac,
                            color='k', linewidth=0.3, alpha=0.35, zorder=5)

    if show_node_points:
        _draw_node_points(ax, nodes, u_plot, sfac, zorder=6)

    if show_supports:
        _draw_supports(ax, nodes, u=u_plot, sfac=sfac if deformed else 1.0)

    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    limit_str  = f'  |  limit={limit}' if limit is not None else ''
    levels_str = f'  |  levels={levels}' if levels is not None else ''
    ax.set_title(f'{component}  |  {result_type}'
                 + (f'  |  deformed x{sfac}' if deformed else '')
                 + limit_str + levels_str)

    if save:
        _save_figure(fig, save)

    plt.show()