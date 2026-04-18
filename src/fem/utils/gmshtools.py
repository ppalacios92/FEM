# =============================================================================
# Based on the course by José Antonio Abell, Universidad de los Andes, Chile
# =============================================================================

import gmsh
import numpy as np
from fem.core.parameters import globalParameters
from fem.core.Node import Node

# -- Gmsh element type reference -----------------------------------------------

GMSH_ELEMENT_INFO = {
    #  gmsh_type    name                           n_nodes
    1  : ('2-node line'                  , 2  ),
    2  : ('3-node triangle'              , 3  ),
    3  : ('4-node quadrangle'            , 4  ),
    4  : ('4-node tetrahedron'           , 4  ),
    5  : ('8-node hexahedron'            , 8  ),
    6  : ('6-node prism'                 , 6  ),
    7  : ('5-node pyramid'               , 5  ),
    8  : ('3-node line'                  , 3  ),
    9  : ('6-node triangle'              , 6  ),
    10 : ('9-node quadrilateral'         , 9  ),
    11 : ('10-node tetrahedron'          , 10 ),
    12 : ('27-node hexahedron'           , 27 ),
    13 : ('18-node prism'                , 18 ),
    14 : ('14-node pyramid'              , 14 ),
    15 : ('1-node point'                 , 1  ),
    16 : ('8-node quadrangle'            , 8  ),
    17 : ('20-node hexahedron'           , 20 ),
    18 : ('15-node prism'                , 15 ),
    19 : ('13-node pyramid'              , 13 ),
}


# -- Utility -------------------------------------------------------------------

def get_element_info(gmsh_type: int) -> tuple:
    """
    Return the name and number of nodes for a given gmsh element type.

    Parameters
    ----------
    gmsh_type : int
        Gmsh element type integer code.

    Returns
    -------
    tuple  (name: str, n_nodes: int)

    References
    ----------
    https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
    """
    if gmsh_type not in GMSH_ELEMENT_INFO:
        raise NotImplementedError(
            f"gmsh_type={gmsh_type} is not in GMSH_ELEMENT_INFO. "
            f"See https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format"
        )
    return GMSH_ELEMENT_INFO[gmsh_type]


# -- Main mesh class -----------------------------------------------------------

class GMSHtools:
    """
    Read a gmsh mesh file and expose its contents as a structured object.

    Automatically runs plan() after reading the mesh, so node_map and
    system_nDof are always available as attributes. Use apply_restraints()
    to set boundary conditions on node_map nodes when needed.

    Parameters
    ----------
    file : str
        Path to the .msh file.

    Attributes
    ----------
    nodes : dict  {tag: (x, y, z)}
        Raw node coordinates for every node in the mesh.

    elements : dict  {phys_id: {
                    'dim'         : int,
                    'gmsh_type'   : int,
                    'n_nodes'     : int,
                    'element_tags': list[int],
                    'connectivity': list[list[int]]
                }}
        Raw element connectivity grouped by physical group id.

    physical_groups : dict  {phys_id: PhysicalGroup, name: PhysicalGroup}
        Physical groups accessible by integer id or by name string.

    node_map : dict  {gmsh_tag: Node}
        Node objects with consecutive DOF indices assigned. Auto-built by plan().

    system_nDof : int
        Total number of DOFs in the system. Auto-built by plan().

    Examples
    --------
    mesh = GMSHtools('model.msh')

    mesh.nodes                          # {tag: (x, y, z)}
    mesh.elements                       # {phys_id: raw connectivity}
    mesh.physical_groups[201]           # PhysicalGroup by id
    mesh.physical_groups['Head_5mm']    # PhysicalGroup by name
    mesh.physical_groups[201].nodes     # {tag: (x,y,z)} in that group
    mesh.physical_groups[201].elements  # raw element data for that group
    mesh.physical_groups[201].dim       # 1 or 2
    mesh.node_map                       # {gmsh_tag: Node} with DOF indices
    mesh.system_nDof                    # total DOFs in the system
    """

    def __init__(self, file: str):
        gmsh.initialize()
        gmsh.open(file)

        self.nodes           = _read_nodes()
        self._physical_raw   = _read_physical_groups()
        self.elements        = _read_elements(self._physical_raw)
        self.physical_groups = self._build_physical_groups()
        self.section_dictionary = None

        gmsh.finalize()

        # auto DOF numbering — node_map and system_nDof always available
        from fem.utils.functions import plan
        self.node_map, self.system_nDof = plan(self)

        self._print_summary()

    def _build_physical_groups(self) -> dict:
        """
        Build the physical_groups dict accessible by id and by name.

        Returns
        -------
        dict  {phys_id: PhysicalGroup, name: PhysicalGroup}
            Same object is registered under both keys.
        """
        groups = {}
        for pg_id, pg_data in self._physical_raw.items():
            name      = pg_data['name']
            dim       = pg_data['dim']
            elem_data = self.elements.get(pg_id, {})

            node_tags = set()
            for conn in elem_data.get('connectivity', []):
                node_tags.update(conn)
            nodes_in_group = {t: self.nodes[t] for t in node_tags
                              if t in self.nodes}

            obj            = _PhysicalGroup(pg_id, name, dim,
                                            elem_data, nodes_in_group)
            groups[pg_id]  = obj
            groups[name]   = obj

        return groups

    def _print_summary(self):
        pg_raw = self._physical_raw

        print('  MESH SUMMARY')

        print(f"\n  === NODES ===  ({len(self.nodes)} total — showing first 3)")
        print(f"  {'Tag':>6}   {'x':>12}   {'y':>12}   {'z':>12}")
        print('--' * 40)
        for tag, (x, y, z) in list(self.nodes.items())[:3]:
            print(f"  {tag:>6}   {x:>12.4f}   {y:>12.4f}   {z:>12.4f}")
        print('--' * 40)

        print(f"\n  === PHYSICAL GROUPS ===  ({len(pg_raw)} total)")
        print(f"  {'ID':>6}   {'Dim':>4}   {'Name'}")
        print('--' * 40)
        for phys_id, info in pg_raw.items():
            print(f"  {phys_id:>6}   {info['dim']:>4}   '{info['name']}'")
        print('--' * 40)

        print(f"\n  === ELEMENTS ===  ({len(self.elements)} groups)")
        print(f"  {'ID':>6}   {'Dim':>4}   {'Type':>6}   {'Nodes/el':>8}"
              f"   {'N elements':>10}   {'Name'}")
        print('--' * 40)
        for phys_id, group in self.elements.items():
            name = pg_raw[phys_id]['name']
            print(f"  {phys_id:>6}   {group['dim']:>4}   {group['gmsh_type']:>6}"
                  f"   {group['n_nodes']:>8}   {len(group['element_tags']):>10}"
                  f"   '{name}'")
        print('--' * 40)
        print()

    def __repr__(self):
        n_pg = len(self._physical_raw)
        return (f"GMSHtools | nodes={len(self.nodes)}"
                f" | physical_groups={n_pg}")

    def apply_boundary_conditions(self, restrain_dictionary: dict, load_dictionary: dict, section_dictionary: dict = None, verbose=True):
        """
        Initialize nodes and apply Dirichlet (restraints) and Neumann (nodal loads)
        boundary conditions.

        Parameters
        ----------
        restrain_dictionary : dict  {phys_id: ['r'/'f', ...]}
        load_dictionary     : dict  {phys_id: {'value': float, 'direction': str}}
        section_dictionary  : dict  {phys_id: section object}  (optional)
        verbose             : bool  Print summary. Default True.
        """
        from fem.core.parameters import globalParameters
        from fem.core.Node import Node

        nDoF     = globalParameters['nDoF']
        nDIM     = globalParameters['nDIM']
        all_tags = sorted(self.nodes.keys())
        tag_to_index = {tag: i for i, tag in enumerate(all_tags)}

        self.node_map          = {}
        self.section_dictionary = section_dictionary

        for tag in all_tags:
            i      = tag_to_index[tag]
            coords = list(self.nodes[tag][:nDIM])
            node   = Node(name=int(tag), coordinates=coords)
            node.idx = np.array([nDoF * i + j for j in range(nDoF)])
            self.node_map[tag] = node

        self.system_nDof = len(all_tags) * nDoF

        # Dirichlet — restraints
        for phys_id, condition in restrain_dictionary.items():
            if phys_id not in self.elements:
                continue
            for conn in self.elements[phys_id]['connectivity']:
                for tag in conn:
                    if tag in self.node_map:
                        self.node_map[tag].set_restrain(condition)

        # Neumann — nodal loads dim=0 only
        for phys_id, load_spec in load_dictionary.items():
            pg = self.physical_groups.get(phys_id)
            if pg is None or pg.dim != 0:
                continue
            d = self._direction_to_vector(load_spec['direction'], nDoF)
            for tag in pg.nodes:
                if tag in self.node_map:
                    self.node_map[tag].set_nodal_load(load_spec['value'] * d)

        if verbose:
            print(f"\n  === BOUNDARY CONDITIONS ===")
            print(f"  Nodes created  : {len(self.node_map)}")
            print(f"  system_nDof    : {self.system_nDof}")

            print(f"\n  --- Restrained nodes ---")
            print(f"  {'Tag':>6}   {'x':>12}   {'y':>12}   {'Restraints'}")
            print('--' * 40)
            for tag, node in self.node_map.items():
                if any(r == 'r' for r in node.restrain):
                    print(f"  {tag:>6}   {node.coordinates[0]:>12.4f}   {node.coordinates[1]:>12.4f}   {node.restrain.tolist()}")
            print('--' * 40)

            print(f"\n  --- Loaded nodes (dim=0) ---")
            print(f"  {'Tag':>6}   {'x':>12}   {'y':>12}   {'Fx':>12}   {'Fy':>12}")
            print('--' * 40)
            for tag, node in self.node_map.items():
                if np.any(np.abs(node.nodalLoad) > 0):
                    fx = node.nodalLoad[0]
                    fy = node.nodalLoad[1] if len(node.nodalLoad) > 1 else 0.0
                    print(f"  {tag:>6}   {node.coordinates[0]:>12.4f}   {node.coordinates[1]:>12.4f}   {fx:>12.4f}   {fy:>12.4f}")
            print('--' * 40)

            if section_dictionary:
                print(f"\n  --- Sections ---")
                print(f"  {'Phys ID':>8}   {'Name':>20}   {'Thickness':>12}   {'E':>12}   {'nu':>6}")
                print('--' * 40)
                for phys_id, section in section_dictionary.items():
                    name = getattr(section, 'name', '-')
                    t    = getattr(section, 'thickness', '-')
                    E    = getattr(section.material, 'E', '-') if hasattr(section, 'material') else '-'
                    nu   = getattr(section.material, 'nu', '-') if hasattr(section, 'material') else '-'
                    print(f"  {phys_id:>8}   {name:>20}   {t:>12}   {E:>12.4f}   {nu:>6.3f}")
                print('--' * 40)

            print()



    def build_load_vector(self, load_dictionary: dict) -> dict:
        """
        Build lumped nodal force dictionary from load_dictionary.

        Detects the physical group dimension automatically:
          - dim=0  ->  ignored — already applied via apply_boundary_conditions
          - dim=1  ->  line load distributed equally along each segment.
                       value [N/mm] — load per unit length.
                       Force per node = value * L / n_nodes_per_segment
          - dim=2  ->  pressure distributed equally among corner nodes.
                       value [N/mm²] — pressure, multiplied by element area.
                       Force per corner node = value * area / n_corner_nodes

        Parameters
        ----------
        load_dictionary : dict  {phys_id or name: {'value': float, 'direction': str}}

        Returns
        -------
        dict  {gmsh_tag: np.array of shape (nDoF,)}
        """
        from fem.core.parameters import globalParameters
        nDoF         = globalParameters['nDoF']
        nodal_forces = {}

        # corner nodes per gmsh_type — higher order elements use only corners for lumped loads
        corner_count = {1: 2, 2: 3, 3: 4, 4: 4, 8: 3, 9: 3, 10: 4, 16: 4}
        corner_count_3d = {4: 4, 5: 8, 6: 6, 11: 4}

        for key, load_spec in load_dictionary.items():
            pg = self.physical_groups.get(key)
            if pg is None:
                continue

            dim        = pg.dim
            load_value = load_spec['value']
            direction  = load_spec['direction']
            d          = self._direction_to_vector(direction, nDoF)

            thickness = 1.0
            if self.section_dictionary:
                for section in self.section_dictionary.values():
                    if hasattr(section, 'thickness'):
                        thickness = section.thickness
                        break

            if dim == 0:
                # for tag in pg.nodes:
                #     nodal_forces[tag] = nodal_forces.get(tag, np.zeros(nDoF))
                #     nodal_forces[tag][:len(d)] += load_value * d
                continue

            elif dim == 1:
                group     = pg.elements
                gmsh_type = group['gmsh_type']
                n_seg     = corner_count.get(gmsh_type, 2)
                accum     = {}
                for connectivity in group['connectivity']:
                    corners = connectivity[:n_seg]
                    pts     = [np.array(self.nodes[t]) for t in corners]
                    L       = np.linalg.norm(pts[1] - pts[0])
                    f = load_value * L * thickness / n_seg
                    for tag in corners:
                        accum[tag] = accum.get(tag, 0.0) + f
                for tag, f in accum.items():
                    nodal_forces[tag] = nodal_forces.get(tag, np.zeros(nDoF))
                    nodal_forces[tag][:len(d)] += f * d

            elif dim == 2:
                group     = pg.elements
                gmsh_type = group['gmsh_type']
                n_corners = corner_count.get(gmsh_type, 3)
                for connectivity in group['connectivity']:
                    corners = connectivity[:n_corners]
                    pts     = [np.array(self.nodes[t]) for t in corners[:3]]
                    area    = 0.5 * np.linalg.norm(np.cross(pts[1]-pts[0], pts[2]-pts[0]))
                    f       = load_value * area / n_corners
                    for tag in corners:
                        nodal_forces[tag] = nodal_forces.get(tag, np.zeros(nDoF))
                        nodal_forces[tag][:len(d)] += f * d
            
            elif dim == 3:
                group     = pg.elements
                gmsh_type = group['gmsh_type']
                n_corners = corner_count_3d.get(gmsh_type, 4)
                for connectivity in group['connectivity']:
                    corners = connectivity[:n_corners]
                    pts     = [np.array(self.nodes[t]) for t in corners[:4]]
                    vol     = abs(np.dot(pts[1]-pts[0],
                                         np.cross(pts[2]-pts[0], pts[3]-pts[0]))) / 6.0
                    f       = load_value * vol / n_corners
                    for tag in corners:
                        nodal_forces[tag] = nodal_forces.get(tag, np.zeros(nDoF))
                        nodal_forces[tag][:len(d)] += f * d


        return nodal_forces

    @staticmethod
    def _direction_to_vector(direction, nDoF: int) -> np.ndarray:
        if nDoF >= 3:
            mapping = {'x': [1,0,0], '-x': [-1,0,0], 'y': [0,1,0],
                       '-y': [0,-1,0], 'z': [0,0,1], '-z': [0,0,-1]}
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

# -- Physical group container --------------------------------------------------


class _PhysicalGroup:
    """
    Container for a single gmsh physical group.

    Attributes
    ----------
    id       : int    Gmsh physical group id.
    name     : str    Physical group name as defined in gmsh.
    dim      : int    Dimension (0=point, 1=line, 2=surface, 3=volume).
    elements : dict   Raw element data (gmsh_type, connectivity, etc.).
    nodes    : dict   {tag: (x,y,z)} of all nodes in this group.
    """

    def __init__(self, id, name, dim, elements, nodes):
        self.id       = id
        self.name     = name
        self.dim      = dim
        self.elements = elements
        self.nodes    = nodes

    def __repr__(self):
        n_el = len(self.elements.get('connectivity', []))
        return (f"PhysicalGroup(id={self.id}, name='{self.name}', "
                f"dim={self.dim}, n_elements={n_el}, "
                f"n_nodes={len(self.nodes)})")


# -- Internal readers ----------------------------------------------------------

def _read_nodes() -> dict:
    """
    Read all nodes from the open gmsh model.

    Returns
    -------
    dict  {tag: (x, y, z)}
        Full 3D coordinates for every node in the mesh.
    """
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)

    return {
        int(tag): (float(x), float(y), float(z))
        for tag, (x, y, z) in zip(node_tags, coords)
    }


def _read_physical_groups() -> dict:
    """
    Read all physical groups from the open gmsh model.

    Returns
    -------
    dict  {phys_id: {'name': str, 'dim': int}}
    """
    groups = {}
    for dim, phys_id in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, phys_id)
        groups[int(phys_id)] = {'name': name, 'dim': int(dim)}
    return groups


def _read_elements(physical_groups: dict) -> dict:
    """
    Read element connectivity for every physical group from the open gmsh model.

    Each physical group may contain multiple entities. All elements across
    entities belonging to the same physical group are concatenated.

    Parameters
    ----------
    physical_groups : dict  {phys_id: {'name': str, 'dim': int}}
        Output of _read_physical_groups.

    Returns
    -------
    dict  {phys_id: {
        'dim'         : int,
        'gmsh_type'   : int,
        'n_nodes'     : int,
        'element_tags': list[int],
        'connectivity': list[list[int]]
    }}

    Notes
    -----
    Physical groups with mixed element types are not supported.
    Only the first element type found per group is used.
    """
    elements = {}

    for phys_id, group_info in physical_groups.items():
        dim         = group_info['dim']
        entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_id)

        all_element_tags = []
        all_connectivity = []
        gmsh_type        = None
        n_nodes          = None

        for ent_tag in entity_tags:
            elem_types, elem_tags_list, node_tags_list = \
                gmsh.model.mesh.getElements(dim, ent_tag)

            if len(elem_types) == 0:
                continue

            et        = int(elem_types[0])
            elem_tags = elem_tags_list[0]
            node_tags = node_tags_list[0]
            _, nn     = get_element_info(et)

            if gmsh_type is None:
                gmsh_type = et
                n_nodes   = nn

            connectivity = node_tags.reshape(-1, nn).astype(int).tolist()

            all_element_tags.extend(elem_tags.astype(int).tolist())
            all_connectivity.extend(connectivity)

        if gmsh_type is None:
            continue

        elements[phys_id] = {
            'dim'         : dim,
            'gmsh_type'   : gmsh_type,
            'n_nodes'     : n_nodes,
            'element_tags': all_element_tags,
            'connectivity': all_connectivity,
        }

    return elements