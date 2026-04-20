"""
FEM Plotting Utilities — matplotlib-based visualization for 2D FEM results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.collections as mc
from mpl_toolkits.axes_grid1 import make_axes_locatable


# -- Internal helpers 
def _get_node_coords(nodes: list, u: np.ndarray = None, sfac: float = 1.0) -> np.ndarray:
    """
    Return node coordinates as (N, 3) array, padding z=0 if not present.
    sfac is applied for plotting only — original data is never modified.
    """
    coords = np.zeros((len(nodes), 3), dtype=float)
    for i, n in enumerate(nodes):
        c = n.coordinates
        coords[i, 0] = c[0]
        coords[i, 1] = c[1]
        coords[i, 2] = c[2] if len(c) > 2 else 0.0

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
    polygons = [coords[[node_map[n.name] for n in _corner_nodes(el)], :2] for el in elements]
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


# Number of corner nodes per gmsh element type — used ONLY to build clean
# polygons for plot_gmsh_mesh. Mid-side / centre nodes are excluded from
# the polygon outline so the element shape renders correctly.
# Node and label plotting always shows ALL nodes (no filtering).
#
#   type  2  — 3-node triangle  (CST)       → 3 corners
#   type  9  — 6-node triangle  (LST)       → 3 corners  [0..2]
#   type  3  — 4-node quad      (Quad4)     → 4 corners
#   type 16  — 8-node quad      (Serendip.) → 4 corners  [0..3]
#   type 10  — 9-node quad      (Quad9)     → 4 corners  [0..3]
#
_GMSH_CORNER_COUNT = {
    2:  3,   # tri3  / CST
    9:  3,   # tri6  / LST
    3:  4,   # quad4
    16: 4,   # quad8 (serendipity)
    10: 4,   # quad9
}


def _get_corner_count(gmsh_type: int, n_nodes: int) -> int:
    """
    Return the number of corner nodes for a given gmsh element type.
    Used only for polygon construction — NOT for filtering node plots.
    Falls back to n_nodes for unknown / linear element types.
    """
    return _GMSH_CORNER_COUNT.get(gmsh_type, n_nodes)


# Corner node count by Python element class name
_CLASS_CORNER_COUNT = {'CST': 3, 'LST': 3, 'Quad4': 4, 'Quad9': 4}

def _corner_nodes(el):
    """Return only the corner nodes of an element (strips mid-side nodes)."""
    n = _CLASS_CORNER_COUNT.get(type(el).__name__, len(el.nodes))
    return el.nodes[:n]


# --------------------------------------
def plot_mesh(nodes=None,
              elements=None,
              show_node_labels: bool = False,
              show_element_labels: bool = False,
              show_supports: bool = True,
              show_element_edges: bool = True,
              show_node_points: bool = True,
              view_3d: bool = False,
              elev: float = 30,
              azim: float = -60,
              figsize: tuple = (12, 8),
              ax=None,
              xlim=None, ylim=None,
              save: str = None):

    if view_3d:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
    else:
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=figsize)

    if elements is not None and nodes is not None:
        coords   = _get_node_coords(nodes)
        node_map = {node.name: i for i, node in enumerate(nodes)}

        for el in elements:
            corners = _corner_nodes(el)
            idx  = [node_map[n.name] for n in corners]
            poly = coords[idx, :2]
            if view_3d:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                poly3d = np.array([
                    (n.coordinates[0], n.coordinates[1], n.coordinates[2] if len(n.coordinates) > 2 else 0.0)
                    for n in corners
                ])
                verts  = [[(x, y, z) for x, y, z in poly3d]]
                ax.add_collection3d(Poly3DCollection(verts,
                                                     facecolors='#f0f0f0',
                                                     edgecolors='tab:blue' if show_element_edges else 'none',
                                                     linewidths=0.4))
            else:
                ax.add_collection(mc.PolyCollection([poly],
                                                    facecolors='#f0f0f0',
                                                    edgecolors='tab:blue' if show_element_edges else 'none',
                                                    linewidths=0.4,
                                                    zorder=1))

        if show_element_labels:
            for element in elements:
                x_c, y_c = element.get_centroid()
                if view_3d:
                    coords3d = np.array([(n.coordinates[0], n.coordinates[1], n.coordinates[2] if len(n.coordinates) > 2 else 0.0) for n in element.nodes])
                    z_c      = coords3d[:, 2].mean()
                    ax.text(x_c, y_c, z_c, f'{element.element_tag}',
                            color='tab:blue', ha='center', va='center', fontsize=6)
                else:
                    ax.text(x_c, y_c, f'{element.element_tag}',
                            color='tab:blue', ha='center', va='center', fontsize=6, zorder=3)

    if nodes is not None:
        if show_node_points:
            if view_3d:
                coords3d = np.array([(n.coordinates[0], n.coordinates[1], n.coordinates[2] if len(n.coordinates) > 2 else 0.0) for n in nodes])
                ax.scatter(coords3d[:, 0], coords3d[:, 1], coords3d[:, 2],
                           color='tab:blue', s=4, zorder=2)
            else:
                coords2d = _get_node_coords(nodes)
                ax.plot(coords2d[:, 0], coords2d[:, 1], '.',
                        color='tab:blue', markersize=2, zorder=2)

        if show_node_labels:
            for node in nodes:
                x = node.coordinates[0]
                y = node.coordinates[1]
                z = node.coordinates[2] if len(node.coordinates) > 2 else 0.0
                if view_3d:
                    ax.text(x, y, z, f' {node.name}',
                            color='tab:blue', fontsize=6)
                else:
                    ax.text(x, y, f' {node.name}',
                            color='tab:blue', fontsize=6,
                            ha='left', va='bottom', zorder=3)

        if show_supports:
            _draw_supports(ax, nodes)

    if not view_3d:
        ax.set_aspect('equal')
        ax.autoscale()
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            margin = max((x_max - x_min), (y_max - y_min)) * 0.05
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

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
               view_3d: bool = False,
               elev: float = 30,
               azim: float = -60,
               figsize: tuple = (12, 8),
               ax=None,
               xlim=None, ylim=None,
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
    xlim,ylim           : axis lims
    save                : str or None
    """

    if view_3d:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
    else:
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=figsize)

    if elements is not None and nodes is not None:
        coords   = _get_node_coords(nodes)
        node_map = {node.name: i for i, node in enumerate(nodes)}
        polygons = [coords[[node_map[n.name] for n in _corner_nodes(el)], :2] for el in elements]

        if view_3d:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            for el in elements:
                idx    = [node_map[n.name] for n in _corner_nodes(el)]
                poly3d = np.array([
                    (n.coordinates[0], n.coordinates[1], n.coordinates[2] if len(n.coordinates) > 2 else 0.0)
                    for n in _corner_nodes(el)
                ])
                verts = [[(x, y, z) for x, y, z in poly3d]]
                ax.add_collection3d(Poly3DCollection(verts,
                                                     facecolors='#f0f0f0',
                                                     edgecolors='#888888' if show_element_edges else 'none',
                                                     linewidths=0.4))
        else:
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

            mags    = np.sqrt(fxs**2 + fys**2)
            max_mag = mags.max()
            max_mag = max_mag if max_mag > 0 else 1.0
            ratios  = mags / max_mag         

            uxs = fxs / mags
            uys = fys / mags

            all_coords = _get_node_coords(nodes)
            bbox_diag  = np.sqrt(np.ptp(all_coords[:, 0])**2 + np.ptp(all_coords[:, 1])**2)
            arrow_len  = 0.05 * bbox_diag

            cmap_obj = plt.get_cmap('Reds')
            colors   = [cmap_obj(0.3 + 0.7 * r) for r in ratios]

            for i in range(len(xs)):
                length = arrow_len * max(ratios[i], 0.25)
                ax.quiver(xs[i], ys[i], uxs[i], uys[i],
                          scale=1.0 / length,
                          scale_units='xy',
                          angles='xy',
                          color=colors[i],
                          width=0.002,
                          headwidth=4,
                          headlength=5,
                          zorder=5)


    if not view_3d:
        ax.set_aspect('equal')
        ax.autoscale()
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            margin = max((x_max - x_min), (y_max - y_min)) * 0.05
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

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
                  view_3d: bool = False,
                  elev: float = 30,
                  azim: float = -60,
                  figsize: tuple = (12, 8),
                  ax=None,
                  xlim=None, ylim=None,
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
    limit               : tuple or None   (min, max) — fixes cmap range; outside -> tab:red
    show_element_edges  : bool
    show_node_points    : bool
    show_supports       : bool
    view_3d             : bool
    elev                : float
    azim                : float
    figsize             : tuple
    ax                  : matplotlib Axes or None
    xlim,ylim           : axis lims
    save                : str or None
    """
    if view_3d:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
    else:
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=figsize)

    node_map = {node.name: i for i, node in enumerate(nodes)}

    # Displacement values
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

    coords_orig = _get_node_coords(nodes)
    coords_def  = _get_node_coords(nodes, u, sfac)

    if view_3d:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Undeformed — transparent
        for el in elements:
            idx    = [node_map[n.name] for n in _corner_nodes(el)]
            poly3d = coords_orig[idx, :3]
            verts  = [[(x, y, z) for x, y, z in poly3d]]
            ax.add_collection3d(Poly3DCollection(verts,
                                                 facecolors='#f0f0f0',
                                                 edgecolors='tab:blue',
                                                 linewidths=0.4,
                                                 alpha=0.2))

        # Deformed — colored by avg disp
        for el in elements:
            idx      = [node_map[n.name] for n in _corner_nodes(el)]
            poly3d   = coords_def[idx, :3]
            avg_disp = np.mean(disp_vals[idx])
            color    = cmap_obj(norm(avg_disp))
            verts    = [[(x, y, z) for x, y, z in poly3d]]
            ax.add_collection3d(Poly3DCollection(verts,
                                                 facecolors=color,
                                                 edgecolors='k' if show_element_edges else 'none',
                                                 linewidths=0.3,
                                                 alpha=0.8))

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=component, shrink=0.5)

    else:
        # Undeformed — transparent background
        polygons_orig = [coords_orig[[node_map[n.name] for n in _corner_nodes(el)], :2] for el in elements]
        ax.add_collection(mc.PolyCollection(polygons_orig,
                                            facecolors='#f0f0f0',
                                            edgecolors='tab:blue',
                                            linewidths=0.4,
                                            alpha=0.3,
                                            zorder=1))

        triang_def = _get_triangulation(nodes, elements, u, sfac)

        if not use_basic:
            ax.tricontourf(triang_def, disp_vals, levels=50,
                           cmap=cmap, norm=norm, zorder=2)

            if limit is not None:
                out_polys = []
                for el in elements:
                    idx = [node_map[n.name] for n in _corner_nodes(el)]
                    if np.mean(disp_vals[idx]) < vmin or np.mean(disp_vals[idx]) > vmax:
                        out_polys.append(coords_def[idx, :2])
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
                idx      = [node_map[n.name] for n in _corner_nodes(el)]
                avg_disp = np.mean(disp_vals[idx])
                polygons_def.append(coords_def[idx, :2])
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
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            margin = max((x_max - x_min), (y_max - y_min)) * 0.05
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

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
               view_3d: bool = False,
               elev: float = 30,
               azim: float = -60,
               figsize: tuple = (12, 8),
               ax=None,
               xlim=None, ylim=None,
               save: str = None):
    """
    Plot a scalar stress or strain field over the FEM mesh.

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
    limit               : tuple or None      (min, max) — fixes cmap range; outside -> tab:red
    levels              : int or None        Number of isolines drawn on top
    cmap                : str                Matplotlib colormap or 'basic'
    show_element_edges  : bool
    show_node_points    : bool
    show_supports       : bool
    view_3d             : bool
    elev                : float
    azim                : float
    figsize             : tuple
    ax                  : matplotlib Axes or None
    xlim,ylim           : axis lims
    save                : str or None
    """
    if view_3d:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
    else:
        fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots(figsize=figsize)

    u_plot       = u if deformed else None
    element_vals = _extract_field(elements, u, component)
    node_map     = {node.name: i for i, node in enumerate(nodes)}

    vmin      = limit[0] if limit is not None else element_vals.min()
    vmax      = limit[1] if limit is not None else element_vals.max()
    use_basic = (cmap == 'basic')
    cmap_obj  = plt.get_cmap('Greys') if use_basic else plt.get_cmap(cmap)
    norm      = plt.Normalize(vmin=vmin, vmax=vmax)

    coords = _get_node_coords(nodes, u_plot, sfac)

    if view_3d:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        for el, val in zip(elements, element_vals):
            idx    = [node_map[n.name] for n in _corner_nodes(el)]
            poly3d = coords[idx, :3]
            color  = cmap_obj(norm(val))
            verts  = [[(x, y, z) for x, y, z in poly3d]]
            ax.add_collection3d(Poly3DCollection(verts,
                                                 facecolors=color,
                                                 edgecolors='k' if show_element_edges else 'none',
                                                 linewidths=0.3,
                                                 alpha=0.9))

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=component, shrink=0.5)

    else:
        nodal_vals = _nodal_average(nodes, elements, element_vals)
        triang     = _get_triangulation(nodes, elements, u_plot, sfac)

        if not use_basic:
            ax.tricontourf(triang, nodal_vals, levels=50,
                           cmap=cmap, norm=norm, zorder=1)

            if levels is not None:
                ax.tricontour(triang, nodal_vals, levels=levels,
                              colors='k', linewidths=0.5, alpha=0.5,
                              norm=norm, zorder=3)

            if limit is not None:
                out_polys = []
                for el, val in zip(elements, element_vals):
                    if val < vmin or val > vmax:
                        idx = [node_map[n.name] for n in _corner_nodes(el)]
                        out_polys.append(coords[idx, :2])
                if out_polys:
                    ax.add_collection(mc.PolyCollection(out_polys,
                                                        facecolors='tab:red',
                                                        edgecolors='none',
                                                        alpha=0.7, zorder=4))

            sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            _add_colorbar(fig, ax, sm, component)

        else:
            polygons_f = []
            colors_f   = []
            for el, val in zip(elements, element_vals):
                idx = [node_map[n.name] for n in _corner_nodes(el)]
                polygons_f.append(coords[idx, :2])
                outside = (val < vmin or val > vmax) if limit is not None else False
                colors_f.append('tab:red' if outside else '#d0d0d0')
            ax.add_collection(mc.PolyCollection(polygons_f,
                                                facecolors=colors_f,
                                                edgecolors='none',
                                                alpha=0.9, zorder=1))

        if show_element_edges:
            _draw_element_edges(ax, nodes, elements, u_plot, sfac,
                                color='k', linewidth=0.3, alpha=0.35, zorder=5)

        if show_node_points:
            _draw_node_points(ax, nodes, u_plot, sfac, zorder=6)

        if show_supports:
            _draw_supports(ax, nodes, u=u_plot, sfac=sfac if deformed else 1.0)

        ax.set_aspect('equal')
        ax.autoscale()
        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        else:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            margin = max((x_max - x_min), (y_max - y_min)) * 0.05
            ax.set_xlim(x_min - margin, x_max + margin)
            ax.set_ylim(y_min - margin, y_max + margin)

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



# -----------------------------------------------------------------
def plot_gmsh_mesh(mesh: dict,
                   show_node_labels: bool = False,
                   show_element_labels: bool = False,
                   show_node_points: bool = True,
                   view_3d: bool = False,
                   elev: float = 30,
                   azim: float = -60,
                   figsize: tuple = (12, 8),
                   save: str = None):

    # nodes    = mesh['nodes']
    # elements = mesh['elements']
    # phys_grp = mesh['physical_groups']
    nodes    = mesh.nodes
    elements = mesh.elements
    phys_grp = mesh._physical_raw


    if view_3d:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
    else:
        fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10.colors

    for k, (phys_id, elem_data) in enumerate(elements.items()):
        dim          = elem_data['dim']
        n_nodes      = elem_data['n_nodes']
        connectivity = elem_data['connectivity']
        color        = colors[k % len(colors)]
        name         = phys_grp[phys_id]['name']

        # --- dim=0: points ---
        if dim == 0:
            for conn in connectivity:
                for tag in conn:
                    x, y, z = nodes[tag]
                    if view_3d:
                        ax.scatter(x, y, z, color=color, s=20, zorder=5, label=name)
                    else:
                        ax.plot(x, y, 'o', color=color, markersize=5, zorder=5, label=name)
            name = None  # evitar duplicados en leyenda

        # --- dim=1: lines ---
        elif dim == 1:
            for i, conn in enumerate(connectivity):
                xs = [nodes[tag][0] for tag in conn]
                ys = [nodes[tag][1] for tag in conn]
                zs = [nodes[tag][2] for tag in conn]
                lbl = name if i == 0 else None
                if view_3d:
                    ax.plot(xs, ys, zs, color=color, linewidth=1.5, zorder=4, label=lbl)
                else:
                    ax.plot(xs, ys, color=color, linewidth=1.5, zorder=4, label=lbl)

        # --- dim=2: surfaces ---
        elif dim == 2:
            polygons    = []
            polygons_3d = []
            centroids   = []

            # Use only corner nodes for the polygon outline.
            # Higher-order elements (LST, Quad9…) store mid-side / centre
            # nodes after the corners — using all nodes deforms the polygon.
            gmsh_type = elem_data.get('gmsh_type', None)
            n_corners = _get_corner_count(gmsh_type, n_nodes) if gmsh_type else n_nodes

            for i, conn in enumerate(connectivity):
                corner_conn = conn[:n_corners]
                if view_3d:
                    c = np.array([[nodes[tag][0], nodes[tag][1], nodes[tag][2]] for tag in corner_conn])
                    polygons_3d.append(c)
                else:
                    c = np.array([[nodes[tag][0], nodes[tag][1]] for tag in corner_conn])
                    polygons.append(c)
                if show_element_labels:
                    centroids.append((c.mean(axis=0), elem_data['element_tags'][i]))

            if view_3d:
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                verts = [[(x, y, z) for x, y, z in poly] for poly in polygons_3d]
                ax.add_collection3d(Poly3DCollection(verts,
                                                     facecolors='#f0f0f0',
                                                     edgecolors=color,
                                                     linewidths=0.4,
                                                     label=name))
            else:
                ax.add_collection(mc.PolyCollection(polygons,
                                                    facecolors='#f0f0f0',
                                                    edgecolors=color,
                                                    linewidths=0.4,
                                                    zorder=1,
                                                    label=name))

            if show_element_labels:
                for centroid, tag in centroids:
                    if view_3d:
                        ax.text(centroid[0], centroid[1], centroid[2], str(tag),
                                fontsize=5, ha='center', va='center', color=color)
                    else:
                        ax.text(centroid[0], centroid[1], str(tag),
                                fontsize=5, ha='center', va='center', color=color, zorder=3)

        # --- dim=3: volumes (solo 3D) ---
        elif dim == 3 and view_3d:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            for i, conn in enumerate(connectivity):
                c     = np.array([[nodes[tag][0], nodes[tag][1], nodes[tag][2]] for tag in conn])
                verts = [[(x, y, z) for x, y, z in c]]
                lbl   = name if i == 0 else None
                ax.add_collection3d(Poly3DCollection(verts,
                                                     facecolors='#f0f0f0',
                                                     edgecolors=color,
                                                     linewidths=0.2,
                                                     label=lbl))

    if show_node_points:
        for tag, (x, y, z) in nodes.items():
            if view_3d:
                ax.scatter(x, y, z, color='k', s=2, zorder=2)
            else:
                ax.plot(x, y, '.', color='k', markersize=2, zorder=2)

    if show_node_labels:
        for tag, (x, y, z) in nodes.items():
            if view_3d:
                ax.text(x, y, z, f' {tag}', fontsize=5, color='k')
            else:
                ax.text(x, y, f' {tag}', fontsize=5,
                        ha='left', va='bottom', color='k', zorder=3)

    if not view_3d:
        ax.set_aspect('equal')
        ax.autoscale()
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        margin = max((x_max - x_min), (y_max - y_min)) * 0.05
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('gmsh Mesh')

    # leyenda sin duplicados
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7)
    ax.axis('equal')

    if view_3d:
        # ax.set_box_aspect([1, 1, 1])          # axis equal in 3D
        ax.xaxis.pane.fill = False            # transparent background panes
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # ax.xaxis.pane.set_edgecolor('white')  # white pane edges
        # ax.yaxis.pane.set_edgecolor('white')
        # ax.zaxis.pane.set_edgecolor('white')
        # ax.grid(False)    
        ax.grid(True, alpha=0.8)                    # no grid
        fig.patch.set_facecolor('white')      # white figure background

    if save:
        _save_figure(fig, save)
    plt.show()

# -- Field extraction from FEMResult (no elements needed) ---------------------

def _extract_field_from_result(result, component: str, n_nodes: int) -> np.ndarray:
    """
    Extract nodal scalar field from a FEMResult object.
    Used when FEM elements are not available (OpenSees flow).

    Parameters
    ----------
    result    : FEMResult
    component : str   'sxx','syy','sxy','sxz','syz','szz',
                      'exx','eyy','exy','vm','vmis'
    n_nodes   : int

    Returns
    -------
    np.ndarray (n_nodes,)
    """
    stress_map = {'sxx': 0, 'syy': 1, 'sxy': 2,
                  'szz': 2, 'sxz': 4, 'syz': 5}
    strain_map = {'exx': 0, 'eyy': 1, 'exy': 2,
                  'ezz': 2, 'exz': 4, 'eyz': 5}

    if component in ('vm', 'vmis'):
        if result.vm_nodal is not None:
            return result.vm_nodal
        raise ValueError("vm_nodal not available in result.")

    if component in stress_map and result.sigma_nodal is not None:
        col = stress_map[component]
        if col < result.sigma_nodal.shape[1]:
            return result.sigma_nodal[:, col]

    if component in strain_map and result.epsilon_nodal is not None:
        col = strain_map[component]
        if col < result.epsilon_nodal.shape[1]:
            return result.epsilon_nodal[:, col]

    raise ValueError(
        f"Component '{component}' not found in result. "
        f"Valid: 'sxx','syy','sxy','szz','exx','eyy','exy','vm'"
    )


# -- 3D field plot -------------------------------------------------------------

def plot_field_3d(nodes: list,
                  result,
                  component: str = 'vm',
                  deformed: bool = False,
                  sfac: float = 1.0,
                  cmap: str = 'turbo',
                  limit: tuple = None,
                  elev: float = 30,
                  azim: float = -60,
                  figsize: tuple = (12, 8),
                  point_size: float = 10.0,
                  ax=None,
                  save: str = None):
    """
    Plot a scalar field over 3D nodes using a scatter plot.
    Works with OpenSees results (no FEM elements needed).

    Parameters
    ----------
    nodes     : list of Node
    result    : FEMResult
    component : str    'sxx','syy','sxy','szz','exx','eyy','exy','vm'
    deformed  : bool   Plot over deformed shape
    sfac      : float  Displacement scale factor
    cmap      : str    Matplotlib colormap
    limit     : tuple  (vmin, vmax) — fixes colormap range
    elev,azim : float  3D view angles
    figsize   : tuple
    point_size: float  Scatter point size
    ax        : matplotlib Axes3D or None
    save      : str or None
    """
    n_nodes = len(nodes)
    values  = _extract_field_from_result(result, component, n_nodes)

    # node coordinates
    coords = np.array([n.coordinates for n in nodes], dtype=float)
    if len(coords[0]) == 2:
        z_col = np.zeros((n_nodes, 1))
        coords = np.hstack([coords, z_col])

    # deformed shape
    if deformed and result.u is not None:
        node_map_idx = {n.name: i for i, n in enumerate(nodes)}
        for i, node in enumerate(nodes):
            dof = node.idx
            nd  = min(len(dof), 3)
            coords[i, :nd] += sfac * result.u[dof[:nd]]

    vmin = limit[0] if limit is not None else values.min()
    vmax = limit[1] if limit is not None else values.max()

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    ax.view_init(elev=elev, azim=azim)

    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                    c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=point_size, depthshade=True)

    plt.colorbar(sc, ax=ax, label=component, shrink=0.5, pad=0.1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    deformed_str = f'  |  deformed x{sfac}' if deformed else ''
    ax.set_title(f'{component}{deformed_str}')

    if save:
        _save_figure(fig, save)

    plt.show()