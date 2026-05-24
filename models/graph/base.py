from abc import ABC, abstractmethod

import numpy as np
from torch_geometric.data import Data

class GraphBuilder(ABC):
    @abstractmethod
    def build(self, r3: np.ndarray, k3: np.ndarray) -> Data:
        raise NotImplementedError()
