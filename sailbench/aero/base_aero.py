from abc import ABC, abstractmethod

class AeroModel(ABC):
    @abstractmethod
    def compute(self, state, inputs, env, params):
        """Return np.array([X_S, Y_S, N_S])"""
        pass
