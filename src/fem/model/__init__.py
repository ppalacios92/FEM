"""FEM Model module — FEMModel, FEMResult, ModalResult."""
from .result import FEMResult
from .modal_result import ModalResult
from .bcs import BoundaryConditions
from .solver import Solver
from .fem_model import FEMModel

__all__ = ['FEMResult', 'ModalResult', 'BoundaryConditions', 'Solver', 'FEMModel']
