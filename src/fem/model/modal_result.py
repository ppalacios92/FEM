"""ModalResult — stores eigenvector for a single vibration mode."""

import numpy as np


class ModalResult:
    """
    Stores modal analysis result for one mode.

    Parameters
    ----------
    mode   : int     Mode number (1-based).
    freq   : float   Frequency [Hz].
    period : float   Period [s].
    omega  : float   Angular frequency [rad/s].
    u_3d   : ndarray (n_nodes, 3) eigenvector in gmsh format.
    """

    def __init__(self, mode, freq, period, omega, u_3d):
        self.mode   = mode
        self.freq   = freq
        self.period = period
        self.omega  = omega
        self.u_3d   = u_3d

    def __repr__(self):
        return (f"ModalResult(mode={self.mode}, "
                f"freq={self.freq:.4f}Hz, period={self.period:.4f}s)")
