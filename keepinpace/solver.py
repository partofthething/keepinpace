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

    def __init__(self,
                 delayedNeutronData:dlayxs.Dlayxs,
                 externalReactivity: Callable,
                 normalizedGenerationTime: float,
                 delayedNeutronFractions: np.array):

        self._delayedNeutronData = delayedNeutronData
        self.externalReactivity = externalReactivity
        self._normalizedGenerationTime = normalizedGenerationTime
        self._delayedNeutronFractions = delayedNeutronFractions
        self.state = np.zeros(1 + self.numPrecursorGroups)

        # steady-state initial conditions require
        # derivatives to equal zero.
        self.initial = np.zeros(1 + self.numPrecursorGroups)
        self.initial[0] = 1.0  # initial guess

        for i in range(self.numPrecursorGroups):
            self.initial[i + 1] = self._delayedNeutronFractions[i] / self._normalizedGenerationTime / self._delayedNeutronData.precursorDecayConstants[i]

        self.state = self.initial[:]
    @property
    def numPrecursorGroups(self):
        return len(self._delayedNeutronData.precursorDecayConstants)

    def _system_rhs(self, t, y):
        """Evaluate the RHS of the PKE equation for the current state."""
        result = np.zeros(len(y))
        precursorDecay = self._delayedNeutronData.precursorDecayConstants
        # dn/dt term
        # first term is (prompt neutrons this generation - prompt neutrons last generation)
        # plus production of delayed neutrons from precursor decay
        dn = ((self.externalReactivity(t) - 1) / self._normalizedGenerationTime * y[0] +
             (y[1:] * precursorDecay).sum())
        # dCi/dt terms
        result[1:] = -y[1:] * precursorDecay + self._delayedNeutronFractions / self._normalizedGenerationTime * y[0]
        result[0] = dn
        return result
    
    def solve(self, start, end):
        result = solve_ivp(self._system_rhs, (start, end), self.state)
        times, values = result.t, result.y
        self.state = values
        return times, values

