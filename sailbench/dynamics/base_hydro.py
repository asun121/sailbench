from abc import ABC, abstractmethod

class HydroModel(ABC):
    @abstractmethod
    def compute(self, state, inputs, env, params):
        """Return np.array([X_H, Y_H, N_H])"""
        pass
