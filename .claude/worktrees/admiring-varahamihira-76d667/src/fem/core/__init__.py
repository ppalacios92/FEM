"""FEM Core Module"""

from .Node import Node
from .Material import Material
from .parameters import globalParameters

__all__ = ['Node', 'Material', 'globalParameters']
