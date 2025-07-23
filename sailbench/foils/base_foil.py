from abc import ABC, abstractmethod
import numpy as np

class FoilModel(ABC):
    """Interface for sail, rudder and keel foil models."""

    @abstractmethod
    def compute(self, state: np.ndarray,
                      inputs: dict,
                      env: dict,
                      params: dict) -> np.ndarray:
        """
        Return np.array([X_S, Y_S, N_S])
        """