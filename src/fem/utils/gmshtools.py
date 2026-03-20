import gmsh
import numpy as np


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


def read_mesh(file: str) -> dict:
    """
    Read a gmsh mesh file and return all raw mesh data.

    Opens and closes gmsh once. All coordinates are stored as (x, y, z)
    regardless of problem dimension. No FEM objects are created here.

    Parameters
    ----------
    file : str
        Path to the .msh file.

    Returns
    -------
    mesh : dict with keys:
        'nodes'           : dict  {tag: (x, y, z)}
        'physical_groups' : dict  {phys_id: {'name': str, 'dim': int}}
        'elements'        : dict  {phys_id: {
                                'dim'         : int,
                                'gmsh_type'   : int,
                                'n_nodes'     : int,
                                'element_tags': list[int],
                                'connectivity': list[list[int]]
                            }}
    """
    gmsh.initialize()
    gmsh.open(file)

    nodes           = _read_nodes()
    physical_groups = _read_physical_groups()
    elements        = _read_elements(physical_groups)

    gmsh.finalize()

    #  Summary

    print('  MESH SUMMARY')

    print(f"\n  === NODES ===  ({len(nodes)} total — showing first 3)")
    print(f"  {'Tag':>6}   {'x':>12}   {'y':>12}   {'z':>12}")
    print('--'*40)
    for tag, (x, y, z) in list(nodes.items())[:3]:
        print(f"  {tag:>6}   {x:>12.4f}   {y:>12.4f}   {z:>12.4f}")
    print('--'*40)

    print(f"\n  === PHYSICAL GROUPS ===  ({len(physical_groups)} total)")
    print(f"  {'ID':>6}   {'Dim':>4}   {'Name'}")
    print('--'*40)
    for phys_id, info in physical_groups.items():
        print(f"  {phys_id:>6}   {info['dim']:>4}   '{info['name']}'")
    print('--'*40)

    print(f"\n  === ELEMENTS ===  ({len(elements)} groups)")
    print(f"  {'ID':>6}   {'Dim':>4}   {'Type':>6}   {'Nodes/el':>8}   {'N elements':>10}   {'Name'}")
    print('--'*40)
    for phys_id, group in elements.items():
        name = physical_groups[phys_id]['name']
        print(f"  {phys_id:>6}   {group['dim']:>4}   {group['gmsh_type']:>6}   "
              f"{group['n_nodes']:>8}   {len(group['element_tags']):>10}   '{name}'")
    print('--'*40)
    print()

    return {
        'nodes'           : nodes,
        'physical_groups' : physical_groups,
        'elements'        : elements,
    }


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
            elem_types, elem_tags_list, node_tags_list = gmsh.model.mesh.getElements(dim, ent_tag)

            if len(elem_types) == 0:
                continue

            # Use first element type found — mixed groups not supported
            et            = int(elem_types[0])
            elem_tags     = elem_tags_list[0]
            node_tags     = node_tags_list[0]
            _, nn         = get_element_info(et)

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