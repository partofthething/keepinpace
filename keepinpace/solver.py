"""Solve kinetics equation."""

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from armi.nuclearDataIO import dlayxs


class KineticsSolver:
    r"""
    Solves the point kinetics equations.

    .. math::
        \frac{dn(t)}{dt} = \frac{\rho(t) - 1}{\Lambda} n(t) + \sum_{i=i}^6 \lambda_i C_i(t) + S(t)  \\
        \frac{dC_i(t)}{dt} = -\lambda_i C_i(t) + \frac{\beta_i}{\beta \Lambda} n(t)

    Where, 
    :math:`l` 
        The neutron generation time
    :math:`\Lambda=\frac{l}{k\beta}` 
        the normalized neutron generation time,
     """

    def __init__(
        self,
        delayedNeutronData: dlayxs.Dlayxs,
        externalReactivity: Callable,
        normalizedGenerationTime: float,
    ):
        self._delayedNeutronData = delayedNeutronData
        self.externalReactivity = externalReactivity
        self._normalizedGenerationTime = normalizedGenerationTime
        numPrecursorGroups = len(self._delayedNeutronData.precursorDecayConstants)
        self.state = np.zeros(1 + numPrecursorGroups)
        self.initial = np.zeros(1 + numPrecursorGroups)
        self._setInitialConditions()
        self.state = self.initial[:]

    def _setInitialConditions(self):
        """Set initial conditions and save for relative plotting."""
        self.initial[0] = 1.0
        for i, (betaFrac, decayConst) in enumerate(
            zip(
                self._delayedNeutronData.delayedNeutronFractions,
                self._delayedNeutronData.precursorDecayConstants,
            )
        ):
            self.initial[i + 1] = betaFrac / self._normalizedGenerationTime / decayConst

    def _system_rhs(self, t, y):
        """Evaluate the RHS of the PKE equation for the current state."""
        result = np.zeros(y.shape)
        precursorDecay = self._delayedNeutronData.precursorDecayConstants
        # dn/dt term
        result[0] = (
            self.externalReactivity(t) - 1
        ) / self._normalizedGenerationTime * y[0] + (y[1:] * precursorDecay).sum()
        # dCi/dt terms
        result[1:] = (
            -y[1:] * precursorDecay
            + self._delayedNeutronData.delayedNeutronFractions
            / self._normalizedGenerationTime
            * y[0]
        )
        return result

    def solve(self, start, end):
        # LSODA solver is "infinitely" faster than default RK45 for this problem
        result = solve_ivp(
            self._system_rhs, (start, end), self.state, method="LSODA", vectorize=False
        )
        times, values = result.t, result.y
        self.state = values  # save for continuation runs
        return times, values
