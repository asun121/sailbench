from abc import ABC, abstractmethod
import numpy as np

class HydroModel(ABC):
    """Interface for any hydrodynamic force model."""

    @abstractmethod
    def compute(self, state: np.ndarray,
                      inputs: dict,
                      env: dict,
                      params: dict) -> np.ndarray:
        """
        Return np.array([X_H, Y_H, N_H])  (body-frame forces & yaw moment)
        """
