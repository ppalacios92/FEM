import numpy as np
import matplotlib.pyplot as plt


class Truss2D:
    """
    2D axial truss element for the Direct Stiffness Method (DSM).

    Degrees of freedom per node: [u, v, theta]  ->  nDoF = 3
    Element DOF order (local):   [u_i, v_i, theta_i, u_j, v_j, theta_j]

    The truss carries axial load only. Bending stiffness terms are zero
    in kb, which propagates zeros into kl and kg automatically through
    the same transformation pipeline as Frame2D.

    Coordinate systems
    ------------------
    Basic   : deformational DOFs  -- {q0: axial, q1: 0, q2: 0}
    Local   : along element axis  -- {u_i, v_i, theta_i, u_j, v_j, theta_j}
    Global  : structure axes      -- {u_i, v_i, theta_i, u_j, v_j, theta_j} rotated

    Transformation chain
    --------------------
    kb  (3x3)  basic stiffness  (axial only, bending rows/cols = 0)
    Tbl (3x6)  basic  <- local     :  q = Tbl · u_local
    kl  (6x6)  local stiffness     :  kl = Tbl^T · kb · Tbl
    Tlg (6x6)  local  <- global    :  u_local = Tlg · u_global
    kg  (6x6)  global stiffness    :  kg = Tlg^T · kl · Tlg
    """

    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------

    def __init__(self, node_i, node_j, material, A: float,
                 print_summary: bool = False):
        """
        Parameters
        ----------
        node_i        : Node      Start node (nDoF = 3)
        node_j        : Node      End node   (nDoF = 3)
        material      : Material  Material object with attributes E, rho
        A             : float     Cross-sectional area [m^2]
        print_summary : bool      Print element summary on creation.
        """
        self.node_i   = node_i
        self.node_j   = node_j
        self.material = material
        self.A        = A
        self.E        = material.get_Emat('frame')

        # Self-weight per unit length: w_self = rho * A
        self.w_self = material.rho * A

        # Geometry
        self.L, self.angle, self.angle_deg = self._compute_geometry()

        # Stiffness matrices (basic -> local -> global)
        self.kb  = self._basic_stiffness()
        self.Tbl = self._basic_local_transformation()
        self.kl  = self._local_stiffness()
        self.Tlg = self._local_global_transformation()
        self.kg  = self._global_stiffness()

        # Fixed-end force vector in global frame
        self.fe = self._compute_fixed_end_forces()

        # DOF indices and restraints
        self.idx      = np.concatenate([node_i.idx, node_j.idx]).astype(int)
        self.restrain = np.concatenate([node_i.restrain, node_j.restrain])

        if print_summary:
            self.print_summary()

    # --------------------------------------------------------------------------
    # String representation
    # --------------------------------------------------------------------------

    def __str__(self):
        return f"Truss2D: node {self.node_i.name} -> node {self.node_j.name}"

    def __repr__(self):
        return self.__str__()

    # --------------------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------------------

    def _compute_geometry(self):
        """Compute element length, angle (rad) and angle (degrees)."""
        delta = self.node_j.coordinates - self.node_i.coordinates
        L     = np.linalg.norm(delta)
        angle = np.arctan2(delta[1], delta[0])
        return L, angle, np.degrees(angle)

    # --------------------------------------------------------------------------
    # Stiffness matrices
    # --------------------------------------------------------------------------

    def _basic_stiffness(self) -> np.ndarray:
        """
        Basic stiffness matrix kb (3x3).

        The truss carries axial load only. Bending terms are zero,
        keeping the same 3x3 structure as Frame2D for pipeline compatibility.

            kb = [ EA/L   0     0  ]
                 [  0     0     0  ]
                 [  0     0     0  ]
        """
        E, A, L = self.E, self.A, self.L
        self.I=1e-10
        I=self.I
        return np.array([
            [E*A/L,       0,          0      ],
            [  0,    4*E*I/L,    2*E*I/L    ],
            [  0,    2*E*I/L,    4*E*I/L    ]
        ])
    def _basic_local_transformation(self) -> np.ndarray:
        """
        Basic-to-local transformation Tbl (3x6).
        Identical to Frame2D.
        Columns: [u_i, v_i, theta_i, u_j, v_j, theta_j]
        """
        L = self.L
        return np.array([
            [-1,    0,    0,   1,    0,    0],
            [ 0,  1/L,    1,   0,  -1/L,   0],
            [ 0,  1/L,    0,   0,  -1/L,   1]
        ])

    def _local_stiffness(self) -> np.ndarray:
        """Local stiffness matrix kl (6x6): kl = Tbl^T · kb · Tbl"""
        return self.Tbl.T @ self.kb @ self.Tbl

    def _local_global_transformation(self) -> np.ndarray:
        """Local-to-global transformation Tlg (6x6). Identical to Frame2D."""
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        return np.array([
            [ c,  s,  0,  0,  0,  0],
            [-s,  c,  0,  0,  0,  0],
            [ 0,  0,  1,  0,  0,  0],
            [ 0,  0,  0,  c,  s,  0],
            [ 0,  0,  0, -s,  c,  0],
            [ 0,  0,  0,  0,  0,  1]
        ])

    def _global_stiffness(self) -> np.ndarray:
        """Global stiffness matrix kg (6x6): kg = Tlg^T · kl · Tlg"""
        return self.Tlg.T @ self.kl @ self.Tlg

    # --------------------------------------------------------------------------
    # Fixed-end force vector (self-weight)
    # --------------------------------------------------------------------------

    def _compute_fixed_end_forces(self) -> np.ndarray:
        """
        Compute the element fixed-end force vector in the global frame.
        Truss carries axial load only — self-weight projected onto axial axis.
        No transverse fixed-end forces (no bending).

        Returns
        -------
        fe : np.ndarray (6,)  Fixed-end forces in global DOF order.
        """
        s = np.sin(self.angle)
        w_axial = self.w_self * s
        L = self.L
        f_local = -1*np.array([
            -w_axial * L / 2,
            0.0,
            0.0,
            -w_axial * L / 2,
            0.0,
            0.0,
        ])
        return self.Tlg.T @ f_local

    # --------------------------------------------------------------------------
    # Force recovery
    # --------------------------------------------------------------------------

    def _get_local_displacements(self, u: np.ndarray):
        """Extract element global displacements and rotate to local frame."""
        ue_global = u[self.idx]
        ue_local  = self.Tlg @ ue_global
        return ue_global, ue_local

    def _get_basic_displacements(self, u: np.ndarray):
        """Project local displacements to basic (deformational) frame."""
        _, ue_local = self._get_local_displacements(u)
        ue_basic    = self.Tbl @ ue_local
        return ue_basic

    def get_results(self, u: np.ndarray) -> dict:
        """
        Recover all element forces and displacements from the global
        displacement vector u.

        Parameters
        ----------
        u : np.ndarray  Global displacement vector (all DOFs)

        Returns
        -------
        dict with keys:
            'ue_global'  (6,)  element displacements in global frame
            'ue_local'   (6,)  element displacements in local frame
            'ue_basic'   (3,)  basic deformations  [axial, 0, 0]
            'fe_basic'   (3,)  basic forces        [N, 0, 0]
            'fe_local'   (6,)  local forces        [N_i, V_i, M_i, N_j, V_j, M_j]
            'fe_global'  (6,)  global forces
        """
        ue_global, ue_local = self._get_local_displacements(u)
        ue_basic  = self.Tbl @ ue_local
        fe_basic  = self.kb  @ ue_basic
        fe_local  = self.kl  @ ue_local
        fe_global = self.kg  @ ue_global

        return {
            'ue_global' : ue_global,
            'ue_local'  : ue_local,
            'ue_basic'  : ue_basic,
            'fe_basic'  : fe_basic,
            'fe_local'  : fe_local,
            'fe_global' : fe_global,
        }

    def get_axial_force(self, u: np.ndarray) -> float:
        """
        Return the axial force N in the element.
        Positive = tension, negative = compression.
        """
        results = self.get_results(u)
        return float(results['fe_basic'][0])

    # --------------------------------------------------------------------------
    # Internal forces
    # --------------------------------------------------------------------------

    def get_internal_forces(self, u: np.ndarray, n_points: int = 50):
        """
        Compute axial force N(x), shear V(x) and bending moment M(x).
        Truss carries axial only — V and M are zero everywhere.

        Returns
        -------
        x  : np.ndarray (n,)   Positions along element [0, L]
        N  : np.ndarray (n,)   Axial force  (positive = tension)
        V  : np.ndarray (n,)   Shear force  (always zero)
        M  : np.ndarray (n,)   Bending moment (always zero)
        """
        N_val = self.get_axial_force(u)
        x     = np.linspace(0, self.L, n_points)
        N     = np.full(n_points, N_val)
        V     = np.zeros(n_points)
        M     = np.zeros(n_points)
        return x, N, V, M

    # --------------------------------------------------------------------------
    # Deformed shape
    # --------------------------------------------------------------------------

    def get_deformed_shape(self, u: np.ndarray, n_points: int = 50):
        """
        Compute the deformed shape using linear Lagrange interpolation.
        Truss has no bending so both u and v vary linearly along the element.

        Returns
        -------
        x_def : np.ndarray (n,)  Deformed x coordinates in global frame
        y_def : np.ndarray (n,)  Deformed y coordinates in global frame
        """
        results  = self.get_results(u)
        ue_local = results['ue_local']

        u_i, v_i = ue_local[0], ue_local[1]
        u_j, v_j = ue_local[3], ue_local[4]

        xi = np.linspace(0, 1, n_points)
        L  = self.L

        N1 = 1 - xi
        N2 = xi

        u_local = N1 * u_i + N2 * u_j
        v_local = N1 * v_i + N2 * v_j

        c = np.cos(self.angle)
        s = np.sin(self.angle)

        x0 = self.node_i.coordinates[0] + xi * L * c
        y0 = self.node_i.coordinates[1] + xi * L * s

        x_def = x0 + c * u_local - s * v_local
        y_def = y0 + s * u_local + c * v_local

        return x_def, y_def

    # --------------------------------------------------------------------------
    # Plotting
    # --------------------------------------------------------------------------

    def _draw_support(self, ax, node, color='mediumpurple', size=10):
        x, y  = node.coordinates[0], node.coordinates[1]
        restr = list(node.restrain)
        if restr == ['r', 'r', 'r']:
            ax.plot(x, y, 's', color=color,     markersize=size, zorder=4)
        elif restr == ['r', 'r', 'f']:
            ax.plot(x, y, '^', color=color,     markersize=size, zorder=4)
        elif 'r' in restr:
            ax.plot(x, y, 'o', color=color,     markersize=size, zorder=4)
        else:
            ax.plot(x, y, 'o', color='tab:red', markersize=size/2, zorder=4)


    def plot_geometry(self, ax=None, show_nodes: bool = True,
                      node_labels: bool = False, element_label: bool = False,
                      color: str = 'tab:orange', lw: float = 2.0,
                      color_by_section: bool = False,
                      section_colors: dict = None):
        """
        Plot undeformed element geometry.
        
        Parameters
        ----------
        color_by_section : bool   If True, color is determined by section area
        section_colors   : dict   {area_value: color} mapping. 
                                  e.g. {A_column: 'steelblue', A_beam: 'tomato', A_diag: 'seagreen'}
        """
        if ax is None:
            _, ax = plt.subplots()

        if color_by_section and section_colors is not None:
            plot_color = section_colors.get(self.A, color)
        else:
            plot_color = color

        xi = self.node_i.coordinates
        xj = self.node_j.coordinates
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], color=plot_color, lw=lw)

        if show_nodes:
            self._draw_support(ax, self.node_i)
            self._draw_support(ax, self.node_j)

        if element_label:
            xm = 0.5 * (xi + xj)
            ax.text(xm[0], xm[1],
                    f'{self.node_i.name}->{self.node_j.name}',
                    fontsize=9, ha='center', va='bottom')

        ax.grid(False)
        return ax
        

    def plot_deformed(self, u: np.ndarray, scale: float = 1.0,
                      ax=None, n_points: int = 50,
                      color: str = 'steelblue', lw: float = 1.5):
        """Plot deformed shape of the element."""
        if ax is None:
            _, ax = plt.subplots()

        u_plot = u.copy()
        u_plot[self.idx] = u[self.idx] * scale

        x_def, y_def = self.get_deformed_shape(u_plot, n_points=n_points)
        ax.plot(x_def, y_def, color=color, lw=lw, linestyle='--')
        ax.plot(x_def, y_def, 'bo', markersize=4, zorder=5)
        ax.grid(False)
        return ax

    def plot_axial(self, u: np.ndarray, ax=None,
                   scale: float = 1.0, n_points: int = 50,
                   color: str = 'tomato', fill: bool = True):
        """Plot axial force diagram along the element."""
        ax = self._plot_diagram(u, 'N', ax, scale, n_points, color, fill)
        return ax

    def plot_shear(self, u: np.ndarray, ax=None,
                   scale: float = 1.0, n_points: int = 50,
                   color: str = 'steelblue', fill: bool = True):
        """Shear force diagram — always zero for truss."""
        ax = self._plot_diagram(u, 'V', ax, scale, n_points, color, fill)
        return ax

    def plot_moment(self, u: np.ndarray, ax=None,
                    scale: float = 1.0, n_points: int = 50,
                    color: str = 'seagreen', fill: bool = True):
        """Bending moment diagram — always zero for truss."""
        ax = self._plot_diagram(u, 'M', ax, scale, n_points, color, fill)
        return ax

    def _plot_diagram(self, u, diagram_type, ax, scale, n_points, color, fill):
        """Internal helper to draw N / V / M diagrams. Identical to Frame2D."""
        if ax is None:
            _, ax = plt.subplots()

        x, N, V, M = self.get_internal_forces(u, n_points=n_points)
        diagrams = {'N': N, 'V': V, 'M': M}
        values   = diagrams[diagram_type] * scale

        c = np.cos(self.angle)
        s = np.sin(self.angle)

        xi_start = self.node_i.coordinates
        x_base   = xi_start[0] + x * c
        y_base   = xi_start[1] + x * s
        x_diag   = x_base - values * s
        y_diag   = y_base + values * c

        ax.plot(x_base, y_base, 'k-', lw=1)
        ax.plot(x_diag, y_diag, color=color, lw=1.5)

        if fill:
            ax.fill(
                np.concatenate([x_base, x_diag[::-1]]),
                np.concatenate([y_base, y_diag[::-1]]),
                color=color, alpha=0.25)

        ax.text(x_diag[0],  y_diag[0],  f'{values[0]/scale:.2f}',  fontsize=7)
        ax.text(x_diag[-1], y_diag[-1], f'{values[-1]/scale:.2f}', fontsize=7)
        ax.grid(False)
        return ax

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------

    def print_summary(self):
        """Print a full summary of element properties and matrices."""
        sep = '-' * 64
        print(sep)
        print(f"Truss2D  :  node {self.node_i.name} -> node {self.node_j.name}")
        print(f"  Length      : {self.L:.4f} m")
        print(f"  Angle       : {self.angle_deg:.4f} deg")
        print(f"  Material    : {self.material.name}  (E={self.E:.3e}, rho={self.material.rho:.3e})")
        print(f"  A           : {self.A:.3e} m^2")
        print(f"  Self-weight : {self.w_self:.6f} kN/m")
        print(f"  fe (global) : {np.round(self.fe, 4)}")
        print(f"  DOF indices : {self.idx}")
        print(f"  Restraints  : {self.restrain}")
        print(f"\n  kb  (basic stiffness 3x3):\n{np.round(self.kb,  4)}")
        print(f"\n  Tbl (basic<-local  3x6):\n{np.round(self.Tbl, 4)}")
        print(f"\n  kl  (local stiffness 6x6):\n{np.round(self.kl,  4)}")
        print(f"\n  Tlg (local<-global 6x6):\n{np.round(self.Tlg, 4)}")
        print(f"\n  kg  (global stiffness 6x6):\n{np.round(self.kg,  4)}")
        print(sep + '\n')