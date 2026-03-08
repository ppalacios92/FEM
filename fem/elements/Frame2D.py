import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Frame2D:
    """
    2D Euler-Bernoulli beam-column element for the Direct Stiffness Method (DSM).

    Degrees of freedom per node: [u, v, theta]  →  nDoF = 3
    Element DOF order (local):   [u_i, v_i, θ_i, u_j, v_j, θ_j]

    Coordinate systems
    ------------------
    Basic   : deformational DOFs — {q0: axial, q1: rotation_i, q2: rotation_j}
    Local   : along element axis — {u_i, v_i, θ_i, u_j, v_j, θ_j}
    Global  : structure axes     — {u_i, v_i, θ_i, u_j, v_j, θ_j} rotated

    Transformation chain
    --------------------
    kb  (3×3)  basic stiffness
    Tbl (3×6)  basic  ← local      :  q = Tbl · u_local
    kl  (6×6)  local stiffness     :  kl = Tbl^T · kb · Tbl
    Tlg (6×6)  local  ← global     :  u_local = Tlg · u_global
    kg  (6×6)  global stiffness    :  kg = Tlg^T · kl · Tlg
    """

    # --------------------------------------------------------------------------
    # Constructor
    # --------------------------------------------------------------------------

    def __init__(self, node_i, node_j, material, A: float, I: float,
                 w: float = 0.0, g: float = 9.81e-3,
                 print_summary: bool = False):
        """
        Parameters
        ----------
        node_i   : Node      Start node (nDoF = 3)
        node_j   : Node      End node   (nDoF = 3)
        material : Material  Material object with attributes E, rho
        A        : float     Cross-sectional area  [m²]
        I        : float     Second moment of area [m⁴]
        w        : float     Additional uniform distributed load perpendicular
                             to element axis, local y [force/length]. Default: 0.
        g        : float     Gravity acceleration [kN/kg]. Default: 9.81e-3.
        print_summary : bool Print element summary on creation.
        """
        self.node_i    = node_i
        self.node_j    = node_j
        self.material  = material
        self.A         = A
        self.I         = I
        self.E         = material.get_Emat('frame')

        # Self-weight per unit length: w_self = rho * g * A  [kN/m]
        self.w_self    = material.rho * g * A

        # Additional distributed load (local y, user-supplied)
        self.w_applied = w

        # Geometry
        self.L, self.angle, self.angle_deg = self._compute_geometry()

        # Stiffness matrices (basic → local → global)
        self.kb  = self._basic_stiffness()
        self.Tbl = self._basic_local_transformation()
        self.kl  = self._local_stiffness()
        self.Tlg = self._local_global_transformation()
        self.kg  = self._global_stiffness()

        # Fixed-end force vector in global frame
        # Combines self-weight (projected to local axes) + user distributed load
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
        return f"Frame2D: node {self.node_i.name} → node {self.node_j.name}"

    def __repr__(self):
        return self.__str__()

    # --------------------------------------------------------------------------
    # Geometry
    # --------------------------------------------------------------------------

    def _compute_geometry(self):
        """Compute element length, angle (rad) and angle (degrees)."""
        delta  = self.node_j.coordinates - self.node_i.coordinates
        L      = np.linalg.norm(delta)
        angle  = np.arctan2(delta[1], delta[0])
        return L, angle, np.degrees(angle)

    # --------------------------------------------------------------------------
    # Stiffness matrices
    # --------------------------------------------------------------------------

    def _basic_stiffness(self) -> np.ndarray:
        """
        Basic stiffness matrix kb (3×3).

        Basic DOFs: q = [axial_deformation, rotation_i, rotation_j]

            kb = [ EA/L      0           0      ]
                 [  0     4EI/L        2EI/L    ]
                 [  0     2EI/L        4EI/L    ]
        """
        E, A, I, L = self.E, self.A, self.I, self.L
        return np.array([
            [E*A/L,       0,          0      ],
            [  0,    4*E*I/L,    2*E*I/L    ],
            [  0,    2*E*I/L,    4*E*I/L    ]
        ])

    def _basic_local_transformation(self) -> np.ndarray:
        """
        Basic-to-local transformation Tbl (3×6).

        Relates basic deformations q to local displacements u_local:
            q = Tbl · u_local

            q0 (axial)      =  -u_i  +  u_j
            q1 (rotation_i) =  (1/L)·v_i  + θ_i  - (1/L)·v_j
            q2 (rotation_j) =  (1/L)·v_i          - (1/L)·v_j  + θ_j

        Columns: [u_i, v_i, θ_i, u_j, v_j, θ_j]
        """
        L = self.L
        return np.array([
            [-1,    0,    0,   1,    0,    0],
            [ 0,  1/L,    1,   0,  -1/L,   0],
            [ 0,  1/L,    0,   0,  -1/L,   1]
        ])

    def _local_stiffness(self) -> np.ndarray:
        """Local stiffness matrix kl (6×6): kl = Tbl^T · kb · Tbl"""
        return self.Tbl.T @ self.kb @ self.Tbl

    def _local_global_transformation(self) -> np.ndarray:
        """
        Local-to-global transformation Tlg (6×6).

        Rotates local axes to global axes:
            u_local = Tlg · u_global

            [ c   s  0  0  0  0 ]
            [-s   c  0  0  0  0 ]
            [ 0   0  1  0  0  0 ]
            [ 0   0  0  c  s  0 ]
            [ 0   0  0 -s  c  0 ]
            [ 0   0  0  0  0  1 ]
        """
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
        """Global stiffness matrix kg (6×6): kg = Tlg^T · kl · Tlg"""
        return self.Tlg.T @ self.kl @ self.Tlg

    # --------------------------------------------------------------------------
    # Consistent load vector (distributed load)
    # --------------------------------------------------------------------------

    def _compute_fixed_end_forces(self) -> np.ndarray:
        """
        Compute the element fixed-end force vector in the global frame,
        combining self-weight and any user-applied distributed load.

        Self-weight acts in global -Y direction and is projected onto the
        local element axes:
            w_perp  = -w_self * cos(angle)   [local y — transverse]
            w_axial = -w_self * sin(angle)   [local x — axial]

        The user distributed load w_applied acts directly in local y.

        Returns
        -------
        fe : np.ndarray (6,)  Fixed-end forces in global DOF order.
        """
        c = np.cos(self.angle)
        s = np.sin(self.angle)

        # Self-weight projected to local axes
        w_sw_perp  = -self.w_self * c
        w_sw_axial = -self.w_self * s

        # Total transverse load in local y
        w_total = w_sw_perp + self.w_applied

        L = self.L
        f_local = np.array([
            -w_sw_axial * L / 2,
            -w_total    * L / 2,
            -w_total    * L**2 / 12,
            -w_sw_axial * L / 2,
            -w_total    * L / 2,
             w_total    * L**2 / 12
        ])
        return self.Tlg.T @ f_local

    def get_fixed_end_forces(self, w: float = 0.0, p: float = 0.0) -> np.ndarray:
        """
        Consistent (fixed-end) nodal force vector in the GLOBAL system
        for an arbitrary uniform distributed load. Utility method — does not
        include self-weight. Use self.fe for the assembled load vector.

        Parameters
        ----------
        w : float   Load perpendicular to element axis (local y) [force/length]
        p : float   Load along element axis (local x) [force/length]

        Returns
        -------
        f_fixed : np.ndarray (6,)  Fixed-end forces in global DOF order.
        """
        L = self.L
        f_local = np.array([
            -p * L / 2,
            -w * L / 2,
            -w * L**2 / 12,
            -p * L / 2,
            -w * L / 2,
             w * L**2 / 12
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
            'ue_basic'   (3,)  basic deformations  [axial, θ_i, θ_j]
            'fe_basic'   (3,)  basic forces        [N, M_i, M_j]
            'fe_local'   (6,)  local forces        [N_i,V_i,M_i, N_j,V_j,M_j]
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

    # --------------------------------------------------------------------------
    # Internal force diagrams (axial, shear, bending moment)
    # --------------------------------------------------------------------------

    def get_internal_forces(self, u: np.ndarray, n_points: int = 50):
        """
        Compute axial force N(x), shear force V(x) and bending moment M(x)
        along the element length.

        The distributed loads are taken directly from the element:
            w_total  : total transverse load in local y
                       = self-weight projected perpendicular to axis + w_applied
            w_axial  : self-weight projected along the element axis

        Parameters
        ----------
        u        : np.ndarray  Global displacement vector
        n_points : int         Number of evaluation points along the element

        Returns
        -------
        x  : np.ndarray (n,)   Positions along element [0, L]
        N  : np.ndarray (n,)   Axial force  (positive = tension)
        V  : np.ndarray (n,)   Shear force
        M  : np.ndarray (n,)   Bending moment
        """
        results  = self.get_results(u)
        fe_local = results['fe_local']

        # End forces in local frame (sign convention: positive = acting on element)
        N_i = fe_local[0]   # axial at i
        V_i = fe_local[1]   # shear at i
        M_i = fe_local[2]   # moment at i

        # Self-weight projected onto local axes
        # Global -Y weight projected: transverse component = -w_self * cos(angle)
        #                             axial component      = -w_self * sin(angle)
        w_perp  = -self.w_self * np.cos(self.angle)   # local y (transverse)
        w_axial = -self.w_self * np.sin(self.angle)   # local x (axial)

        # Total transverse distributed load in local y
        w_total = w_perp + self.w_applied

        x = np.linspace(0, self.L, n_points)

        N = -N_i - w_axial * x                      # axial varies if element is inclined
        V =  V_i - w_total * x                      # shear varies linearly
        M = -M_i + V_i * x - 0.5 * w_total * x**2  # moment parabolic

        return x, N, V, M

    # --------------------------------------------------------------------------
    # Deformed shape
    # --------------------------------------------------------------------------

    def get_deformed_shape(self, u: np.ndarray, n_points: int = 50):
        """
        Compute the deformed shape of the element using cubic Hermite interpolation.

        Parameters
        ----------
        u        : np.ndarray  Global displacement vector
        n_points : int         Number of interpolation points

        Returns
        -------
        x_def : np.ndarray (n,)  Deformed x coordinates in global frame
        y_def : np.ndarray (n,)  Deformed y coordinates in global frame
        """
        results  = self.get_results(u)
        ue_local = results['ue_local']

        u_i, v_i, th_i = ue_local[0], ue_local[1], ue_local[2]
        u_j, v_j, th_j = ue_local[3], ue_local[4], ue_local[5]

        xi = np.linspace(0, 1, n_points)   # natural coordinate ξ = x/L
        L  = self.L

        # Hermite shape functions for transverse displacement v(ξ)
        H1 =  1 - 3*xi**2 + 2*xi**3
        H2 =  L * (xi - 2*xi**2 + xi**3)
        H3 =  3*xi**2 - 2*xi**3
        H4 =  L * (-xi**2 + xi**3)

        # Linear shape function for axial displacement u(ξ)
        N1_ax = 1 - xi
        N2_ax = xi

        u_local = N1_ax * u_i + N2_ax * u_j
        v_local = H1 * v_i + H2 * th_i + H3 * v_j + H4 * th_j

        # Rotate back to global frame
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

    def plot_geometry(self, ax=None, show_nodes: bool = True,
                      node_labels: bool = False, element_label: bool = False,
                      color: str = 'k', lw: float = 2.0):
        """Plot undeformed element geometry."""
        if ax is None:
            _, ax = plt.subplots()

        xi = self.node_i.coordinates
        xj = self.node_j.coordinates
        ax.plot([xi[0], xj[0]], [xi[1], xj[1]], color=color, lw=lw)

        if show_nodes:
            self.node_i.plotGeometry(ax, text=node_labels)
            self.node_j.plotGeometry(ax, text=node_labels)

        if element_label:
            xm = 0.5 * (xi + xj)
            ax.text(xm[0], xm[1],
                    f'{self.node_i.name}→{self.node_j.name}',
                    fontsize=9, ha='center', va='bottom')
            
        ax.grid(False)
        return ax

    def plot_deformed(self, u: np.ndarray, scale: float = 1.0,
                      ax=None, n_points: int = 50,
                      color: str = 'steelblue', lw: float = 1.5):
        """
        Plot deformed shape of the element.

        Parameters
        ----------
        u     : global displacement vector
        scale : amplification factor for displacements
        """
        if ax is None:
            _, ax = plt.subplots()

        u_plot = u.copy()
        u_plot[self.idx] = u[self.idx] * scale

        x_def, y_def = self.get_deformed_shape(u_plot, n_points=n_points)
        ax.plot(x_def, y_def, color=color, lw=lw, linestyle='--')

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
        """Plot shear force diagram along the element."""
        ax = self._plot_diagram(u, 'V', ax, scale, n_points, color, fill)
        return ax

    def plot_moment(self, u: np.ndarray, ax=None,
                    scale: float = 1.0, n_points: int = 50,
                    color: str = 'seagreen', fill: bool = True):
        """Plot bending moment diagram along the element."""
        ax = self._plot_diagram(u, 'M', ax, scale, n_points, color, fill)
        return ax

    def _plot_diagram(self, u, diagram_type, ax, scale, n_points, color, fill):
        """Internal helper to draw N / V / M diagrams in local coordinates."""
        if ax is None:
            _, ax = plt.subplots()

        x, N, V, M = self.get_internal_forces(u, n_points=n_points)
        diagrams = {'N': N, 'V': V, 'M': M}
        values   = diagrams[diagram_type] * scale

        c = np.cos(self.angle)
        s = np.sin(self.angle)

        # Baseline in global coords
        xi = self.node_i.coordinates
        x_base = xi[0] + x * c
        y_base = xi[1] + x * s

        # Offset perpendicular to element axis
        x_diag = x_base - values * s
        y_diag = y_base + values * c

        ax.plot(x_base, y_base, 'k-', lw=1)
        ax.plot(x_diag, y_diag, color=color, lw=1.5)

        if fill:
            ax.fill(
                np.concatenate([x_base, x_diag[::-1]]),
                np.concatenate([y_base, y_diag[::-1]]),
                color=color, alpha=0.25
            )

        ax.text(x_diag[0],  y_diag[0],  f'{values[0]:.2f}',  fontsize=7)
        ax.text(x_diag[-1], y_diag[-1], f'{values[-1]:.2f}', fontsize=7)
        ax.grid(False)

        return ax

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------

    def print_summary(self):
        """Print a full summary of element properties and matrices."""
        sep = '-' * 64
        print(sep)
        print(f"Frame2D  :  node {self.node_i.name} → node {self.node_j.name}")
        print(f"  Length      : {self.L:.4f} m")
        print(f"  Angle       : {self.angle_deg:.4f} deg")
        print(f"  Material    : {self.material.name}  (E={self.E:.3e}, rho={self.material.rho:.1f})")
        print(f"  A={self.A:.3e} m²   I={self.I:.3e} m⁴")
        print(f"  Self-weight : {self.w_self:.4f} kN/m  |  Applied w: {self.w_applied:.4f} kN/m")
        print(f"  fe (global) : {np.round(self.fe, 4)}")
        print(f"  DOF indices : {self.idx}")
        print(f"  Restraints  : {self.restrain}")
        print(f"\n  kb  (basic stiffness 3×3):\n{np.round(self.kb,  4)}")
        print(f"\n  Tbl (basic←local  3×6):\n{np.round(self.Tbl, 4)}")
        print(f"\n  kl  (local stiffness 6×6):\n{np.round(self.kl,  4)}")
        print(f"\n  Tlg (local←global 6×6):\n{np.round(self.Tlg, 4)}")
        print(f"\n  kg  (global stiffness 6×6):\n{np.round(self.kg,  4)}")
        print(sep + '\n')