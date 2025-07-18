from abc import ABC, abstractmethod
import numpy as np

class AeroModel(ABC):
    """Interface for sail / rudder force models."""

    @abstractmethod
    def compute(self, state: np.ndarray,
                      inputs: dict,
                      env: dict,
                      params: dict) -> np.ndarray:
        """
        Return np.array([X_S, Y_S, N_S])
        """