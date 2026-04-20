"""BoundaryConditions — node planning, restraints, and load assembly."""

import numpy as np
from fem.core.Node import Node
from fem.core.parameters import globalParameters


class BoundaryConditions:
    """
    Builds node_map, applies restraints and nodal loads, assembles load vectors.

    Parameters
    ----------
    mesh                : GMSHtools
    restrain_dictionary : dict  {phys_id or name: ['r'/'f', ...]}
    load_dictionary     : dict  {phys_id or name: {'value': float, 'direction': str}}
    section_dictionary  : dict  {phys_id: section}  needed for thickness in dim=1 lumped
    """

    # corner node count per gmsh element type
    _CORNER = {1: 2, 2: 3, 3: 4, 4: 4, 8: 3, 9: 3, 10: 4, 16: 4}
    _CORNER_3D = {4: 4, 5: 8, 6: 6, 11: 4}

    def __init__(self, mesh, restrain_dictionary, load_dictionary,
                 section_dictionary=None):

        self._mesh               = mesh
        self._restrain_dict      = restrain_dictionary
        self._load_dict          = load_dictionary
        self._section_dictionary = section_dictionary

        nDoF = globalParameters['nDoF']
        nDIM = globalParameters['nDIM']

        # build node_map
        all_tags     = sorted(mesh.nodes.keys())
        tag_to_index = {tag: i for i, tag in enumerate(all_tags)}

        self.node_map = {}
        for tag in all_tags:
            i      = tag_to_index[tag]
            coords = list(mesh.nodes[tag][:nDIM])
            node   = Node(name=int(tag), coordinates=coords)
            node.idx = np.array([nDoF * i + j for j in range(nDoF)])
            self.node_map[tag] = node

        self.system_nDof = len(all_tags) * nDoF
        self.node_tags   = np.array(all_tags)

        self._apply_restraints()
        self._apply_nodal_loads()
        self._build_F_lumped()

    # -------------------------------------------------------------------------
    # Public attributes
    # -------------------------------------------------------------------------

    @property
    def restrained_nodes(self) -> dict:
        """Return {tag: condition} for all restrained nodes — ready for OpenSees."""
        result = {}
        for tag, node in self.node_map.items():
            if any(r == 'r' for r in node.restrain):
                result[tag] = node.restrain.tolist()
        return result

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _apply_restraints(self):
        mesh = self._mesh
        for key, condition in self._restrain_dict.items():
            pg = mesh.physical_groups.get(key)
            if pg is None:
                continue
            for conn in pg.elements.get('connectivity', []):
                for tag in conn:
                    if tag in self.node_map:
                        self.node_map[tag].set_restrain(condition)

    def _apply_nodal_loads(self):
        """Apply dim=0 point loads to node.nodalLoad."""
        nDoF = globalParameters['nDoF']
        mesh = self._mesh
        for key, load_spec in self._load_dict.items():
            pg = mesh.physical_groups.get(key)
            if pg is None or pg.dim != 0:
                continue
            d = self._direction_to_vector(load_spec['direction'], nDoF)
            for tag in pg.nodes:
                if tag in self.node_map:
                    self.node_map[tag].set_nodal_load(load_spec['value'] * d)

    def _build_F_lumped(self):
        """
        Build lumped load vector F_load and F_nodal dict.
        dim=0 → from node.nodalLoad
        dim=1 → q * L * thickness / n_seg
        dim=2 → q * area / n_corners
        dim=3 → q * vol / n_corners
        """
        nDoF = globalParameters['nDoF']
        mesh = self._mesh

        F = np.zeros(self.system_nDof)

        # dim=0 — already in nodalLoad
        for node in self.node_map.values():
            F[node.idx] += node.nodalLoad

        # dim=1/2/3 — lumped from geometry
        nodal_forces = {}
        for key, load_spec in self._load_dict.items():
            pg = mesh.physical_groups.get(key)
            if pg is None or pg.dim == 0:
                continue

            dim        = pg.dim
            load_value = load_spec['value']
            d          = self._direction_to_vector(load_spec['direction'], nDoF)

            thickness = 1.0
            if self._section_dictionary:
                for sec in self._section_dictionary.values():
                    if hasattr(sec, 'thickness'):
                        thickness = sec.thickness
                        break

            if dim == 1:
                group     = pg.elements
                gmsh_type = group.get('gmsh_type', 1)
                n_seg     = self._CORNER.get(gmsh_type, 2)
                accum     = {}
                for conn in group.get('connectivity', []):
                    corners = conn[:n_seg]
                    pts     = [np.array(mesh.nodes[t]) for t in corners]
                    L       = np.linalg.norm(pts[1] - pts[0])
                    f       = load_value * L * thickness / n_seg
                    for tag in corners:
                        accum[tag] = accum.get(tag, 0.0) + f
                for tag, f in accum.items():
                    nodal_forces[tag] = nodal_forces.get(tag, np.zeros(nDoF))
                    nodal_forces[tag][:len(d)] += f * d

            elif dim == 2:
                group     = pg.elements
                gmsh_type = group.get('gmsh_type', 2)
                n_corners = self._CORNER.get(gmsh_type, 3)
                for conn in group.get('connectivity', []):
                    corners = conn[:n_corners]
                    pts     = [np.array(mesh.nodes[t]) for t in corners[:3]]
                    area    = 0.5 * np.linalg.norm(
                        np.cross(pts[1] - pts[0], pts[2] - pts[0]))
                    f = load_value * area / n_corners
                    for tag in corners:
                        nodal_forces[tag] = nodal_forces.get(tag, np.zeros(nDoF))
                        nodal_forces[tag][:len(d)] += f * d

            elif dim == 3:
                group     = pg.elements
                gmsh_type = group.get('gmsh_type', 4)
                n_corners = self._CORNER_3D.get(gmsh_type, 4)
                for conn in group.get('connectivity', []):
                    corners = conn[:n_corners]
                    pts     = [np.array(mesh.nodes[t]) for t in corners[:4]]
                    vol     = abs(np.dot(pts[1] - pts[0],
                                        np.cross(pts[2] - pts[0],
                                                 pts[3] - pts[0]))) / 6.0
                    f = load_value * vol / n_corners
                    for tag in corners:
                        nodal_forces[tag] = nodal_forces.get(tag, np.zeros(nDoF))
                        nodal_forces[tag][:len(d)] += f * d

        # assemble dim=1/2/3 into F_load
        for tag, f_vec in nodal_forces.items():
            if tag in self.node_map:
                F[self.node_map[tag].idx[:len(f_vec)]] += f_vec


        for tag, node in self.node_map.items():
            if np.any(np.abs(node.nodalLoad) > 0):
                nodal_forces[tag] = nodal_forces.get(tag, np.zeros(nDoF))
                nodal_forces[tag] += node.nodalLoad

        F[np.abs(F) < 1e-10] = 0.0
        self.F_load  = F
        self.F_nodal = nodal_forces

        # self.F_load    = F
        # self.F_nodal   = nodal_forces  # {tag: force} for OpenSees

    def build_F_consistent(self, elements) -> np.ndarray:
        """
        Build consistent load vector using element shape functions.
        Replaces F_load for dim=1 loads. dim=0 always stays nodal.

        Parameters
        ----------
        elements : array of FEM element objects with F_fe_global attribute.

        Returns
        -------
        np.ndarray (system_nDof,)
        """
        F = np.zeros(self.system_nDof)

        # dim=0 from nodalLoad
        for node in self.node_map.values():
            F[node.idx] += node.nodalLoad

        # consistent surface/body forces from elements
        for elem in elements:
            F[elem.idx] += elem.F_fe_global

        F[np.abs(F) < 1e-10] = 0.0
        return F
        
    @staticmethod
    def _direction_to_vector(direction, nDoF: int) -> np.ndarray:
        """Convert direction string or angle to unit vector."""
        if nDoF >= 3:
            mapping = {
                'x': [1, 0, 0], '-x': [-1, 0, 0],
                'y': [0, 1, 0], '-y': [0, -1, 0],
                'z': [0, 0, 1], '-z': [0, 0, -1],
            }
            if direction in mapping:
                return np.array(mapping[direction], dtype=float)
            rad = np.radians(float(direction))
            return np.array([np.cos(rad), np.sin(rad), 0.0])
        else:
            mapping = {'x': 0., '-x': 180., 'y': 90., '-y': 270.}
            if direction in mapping:
                angle = mapping[direction]
            else:
                angle = float(direction)
            rad = np.radians(angle)
            return np.array([np.cos(rad), np.sin(rad)])
