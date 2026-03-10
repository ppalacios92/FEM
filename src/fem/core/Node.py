import numpy as np
import matplotlib.pyplot as plt
from fem.core.parameters import globalParameters


class Node:
    """
    2D structural node for the Direct Stiffness Method (DSM).

    Each node is assigned a set of global DOF indices based on its name
    and the number of DOFs per node defined in globalParameters['nDoF'].

    DOF assignment (plain numbering)
    ---------------------------------
    Node name n  →  global DOFs  [nDoF*n,  nDoF*n+1,  ...,  nDoF*(n+1)-1]

    Nodes are 0-based: the first node must be named 0.

    For nDoF = 2  (truss / continuum):   [u, v]
    For nDoF = 3  (frame):               [u, v, theta]

    Restraint convention
    --------------------
    'f'  →  free DOF        (unknown displacement, known applied force)
    'r'  →  restrained DOF  (prescribed displacement = 0, unknown reaction)
    """

    
    # Constructor
    

    def __init__(self, name: int, coordinates: list,
                 nodal_load: list = None,
                 restrain: list = None,
                 print_summary: bool = False):
        """
        Parameters
        ----------
        name          : int   Node number (0-based).
        coordinates   : list  [x, y] position in global frame.
        nodal_load    : list  Applied nodal load, length = nDoF. Default: zeros.
        restrain      : list  BC flags, length = nDoF. Each entry 'f' or 'r'.
                              Default: all free.
        print_summary : bool  Print node summary on creation.
        """
        nDoF = globalParameters['nDoF']

        self.name        = name
        self.coordinates = np.array(coordinates, dtype=float)

        # ── Global DOF indices ─────────────────────────────────────────────
        self.idx = self._compute_indices()

        # ── Nodal load ─────────────────────────────────────────────────────
        if nodal_load is not None:
            if len(nodal_load) != nDoF:
                raise ValueError(
                    f"nodal_load must have length {nDoF}, got {len(nodal_load)}."
                )
            self.nodalLoad = np.array(nodal_load, dtype=float)
        else:
            self.nodalLoad = np.zeros(nDoF)

        # ── Boundary conditions ────────────────────────────────────────────
        if restrain is not None:
            if len(restrain) != nDoF:
                raise ValueError(
                    f"restrain must have length {nDoF}, got {len(restrain)}."
                )
            if not all(r in ('f', 'r') for r in restrain):
                raise ValueError(
                    "restrain entries must be 'f' (free) or 'r' (restrained)."
                )
            self.restrain = np.array(restrain)
        else:
            self.restrain = np.full(nDoF, 'f')

        if print_summary:
            self.print_summary()

    
    # String representation
    

    def __str__(self):
        return f"Node {self.name} at {self.coordinates}"

    def __repr__(self):
        return self.__str__()

    
    # DOF index computation
    

    def _compute_indices(self) -> np.ndarray:
        """
        Compute the global DOF indices for this node.

        Node naming is 0-based:
            node 0  →  DOFs [0, 1, 2]
            node 1  →  DOFs [3, 4, 5]
            node n  →  DOFs [nDoF*n, ..., nDoF*(n+1)-1]

        Returns
        -------
        idx : np.ndarray (nDoF,)
        """
        nDoF = globalParameters['nDoF']
        return np.arange(nDoF) + int(nDoF * self.name)

    
    # Setters
    

    def set_restrain(self, restrain: list):
        """Set or update the boundary condition flags."""
        nDoF = globalParameters['nDoF']
        if len(restrain) != nDoF:
            raise ValueError(
                f"restrain must have length {nDoF}, got {len(restrain)}."
            )
        if not all(r in ('f', 'r') for r in restrain):
            raise ValueError(
                "restrain entries must be 'f' (free) or 'r' (restrained)."
            )
        self.restrain = np.array(restrain)

    def set_nodal_load(self, nodal_load: list):
        """Set or update the nodal load vector."""
        nDoF = globalParameters['nDoF']
        if len(nodal_load) != nDoF:
            raise ValueError(
                f"nodal_load must have length {nDoF}, got {len(nodal_load)}."
            )
        self.nodalLoad = np.array(nodal_load, dtype=float)

    
    # Plotting
    

    def plotGeometry(self, ax=None, text: bool = False,
                     color: str = 'tomato', markersize: int = 6):
        """Plot the node position on a matplotlib axis."""
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.coordinates[0], self.coordinates[1],
                'o', color=color, markersize=markersize, zorder=5)
        if text:
            ax.text(self.coordinates[0], self.coordinates[1],
                    f'  {self.name}', fontsize=9, va='center')
        return ax

    
    # Summary
    

    def print_summary(self):
        """Print a formatted summary of the node properties."""
        sep = '─' * 48
        print(sep)
        print(f"Node {self.name}")
        print(f"  Coordinates : {self.coordinates}")
        print(f"  DOF indices : {self.idx}")
        print(f"  Nodal load  : {self.nodalLoad}")
        print(f"  Restraints  : {self.restrain}")
        print(sep + '\n')

    
    # Backwards compatibility aliases
    

    def printSummary(self):
        """Alias for print_summary() — kept for backwards compatibility."""
        self.print_summary()

    def set_nodalLoad(self, nodal_load: list):
        """Alias for set_nodal_load() — kept for backwards compatibility."""
        self.set_nodal_load(nodal_load)