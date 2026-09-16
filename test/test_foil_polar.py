"""Tests for the tabulated foil polar backing Foil.cl_cd."""

from typing import Any

import numpy as np
import pytest

from sailbench.models import foil as foil_mod
from sailbench.models.foil import Foil
from sailbench.models.model import State
from sailbench.tf.tf_tree import TFTree2D


class _Foil(Foil):
    """Concrete Foil; the polar is what is under test, not a force model."""

    def compute(self, state: State, tf_tree: TFTree2D) -> np.ndarray:
        """Unused by these tests."""
        del state, tf_tree
        return np.zeros(2, dtype=float)


@pytest.fixture
def params() -> dict[str, Any]:
    """Full-range polar parameters, as the boat configs use."""
    return {
        "airfoil_name": "NACA0012",
        "alpha_min": -179,
        "alpha_max": 179,
        "res": [5e5],
    }


class TestFoilPolar:
    """Tabulated polar behaviour."""

    def test_matches_neuralfoil(self, params: dict[str, Any]) -> None:
        """Interpolated coefficients should track NeuralFoil across the whole range."""
        foil = _Foil(params)
        re = foil.get_reynolds()

        alphas = np.linspace(params["alpha_min"], params["alpha_max"], 257)
        exact = foil.foil.get_aero_from_neuralfoil(alpha=alphas, Re=re, mach=0.0)
        cl_exact = np.asarray(exact["CL"], dtype=float)
        cd_exact = np.asarray(exact["CD"], dtype=float)

        for alpha_deg, cl_ref, cd_ref in zip(alphas, cl_exact, cd_exact):
            cl, cd = foil.cl_cd(float(np.radians(alpha_deg)), re)
            assert abs(cl - cl_ref) < 1e-2
            assert abs(cd - cd_ref) < 1e-2

    def test_alpha_is_clamped_to_range(self, params: dict[str, Any]) -> None:
        """Angles beyond the tabulated range clamp to the endpoint, not extrapolate."""
        foil = _Foil(params)
        re = foil.get_reynolds()

        at_max = foil.cl_cd(float(np.radians(params["alpha_max"])), re)
        beyond = foil.cl_cd(float(np.radians(params["alpha_max"] + 50.0)), re)

        assert beyond == pytest.approx(at_max)

    def test_table_is_built_once_and_shared(self, params: dict[str, Any]) -> None:
        """Two foils with identical parameters should share one tabulated polar."""
        foil_mod.clear_polar_cache()
        re = float(params["res"][0])

        _Foil(params).cl_cd(0.1, re)
        assert len(foil_mod._POLAR_CACHE) == 1

        _Foil(params).cl_cd(0.2, re)
        assert len(foil_mod._POLAR_CACHE) == 1

        # A different Reynolds number is a genuinely different polar.
        _Foil(params).cl_cd(0.1, re * 2.0)
        assert len(foil_mod._POLAR_CACHE) == 2

    def test_symmetric_section_has_no_lift_at_zero(self, params: dict[str, Any]) -> None:
        """A symmetric NACA section makes no lift at zero incidence."""
        foil = _Foil(params)
        cl, cd = foil.cl_cd(0.0, foil.get_reynolds())

        assert abs(cl) < 1e-2
        assert cd > 0.0
