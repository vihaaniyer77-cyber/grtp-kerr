"""
tests/conftest.py
=================
Shared pytest fixtures for the GRTP-Kerr test suite.
"""

import numpy as np
import pytest

from grtp.spacetime.ergosphere import event_horizon, ergosphere_radius
from grtp.fields.xpoint import XPointField
from grtp.integrator.solver import ParticleIntegrator


@pytest.fixture(scope="session")
def schwarzschild():
    """Schwarzschild black hole parameters (a=0, M=1)."""
    return {"M": 1.0, "a": 0.0}


@pytest.fixture(scope="session")
def kerr_high_spin():
    """High-spin Kerr black hole (a=0.9, M=1)."""
    return {"M": 1.0, "a": 0.9}


@pytest.fixture(scope="session")
def kerr_extremal():
    """Near-extremal Kerr (a=0.999, M=1)."""
    return {"M": 1.0, "a": 0.999}


@pytest.fixture
def xpoint_field(kerr_high_spin):
    """Standard X-point field at the ergosphere midpoint for a=0.9."""
    M, a = kerr_high_spin["M"], kerr_high_spin["a"]
    r_X  = 0.5 * (event_horizon(M, a) + ergosphere_radius(np.pi / 2, M, a))
    return XPointField(M=M, a=a, r_X=r_X, B0=1.0, E0=0.2, lam=0.4)


@pytest.fixture
def integrator(kerr_high_spin, xpoint_field):
    """Standard particle integrator (a=0.9, q/m=1)."""
    M, a = kerr_high_spin["M"], kerr_high_spin["a"]
    return ParticleIntegrator(
        M=M, a=a, q_over_m=1.0, field=xpoint_field,
        r_escape=20.0, rtol=1e-10, atol=1e-12,
    )


@pytest.fixture
def geodesic_integrator(schwarzschild, xpoint_field):
    """Geodesic integrator (q=0, Schwarzschild) for conservation tests."""
    M, a = schwarzschild["M"], schwarzschild["a"]
    field_sch = XPointField(M=M, a=a, r_X=5.5, B0=0.0, E0=0.0, lam=1.0)
    return ParticleIntegrator(
        M=M, a=a, q_over_m=0.0, field=field_sch,
        r_escape=50.0, rtol=1e-10, atol=1e-12,
    )
