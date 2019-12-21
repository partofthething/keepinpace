"""Check kinetics solve."""
import unittest
import math
import cProfile

import numpy as np
import matplotlib.pyplot as plt

import armi

armi.configure(armi.apps.App())
from armi.nuclearDataIO import dlayxs
from armi.nuclearDataIO.tests import test_xsLibraries
from armi.nucDirectory import nuclideBases

from keepinpace import solver


class TestKineticsSolve(unittest.TestCase):
    def testSolver(self):
        """Run a sample kinetics problem."""

        def rho(t):
            """Reactivity driver"""
            return 0.2 * math.copysign(1, math.sin(math.pi * t / 1.0))

        adelay = _getDelayedNeutronData()
        s = solver.KineticsSolver(adelay, rho, 1e-7 / 0.0035)
        # cProfile.runctx('t, values = s.solve(0, 7)', locals(), globals())
        t, values = s.solve(0, 15)
        _plot(t, values, s)
        # TODO: do comparison against actual analytic solution for certain delayed neutron numbers


def _getDelayedNeutronData():
    """Grab some generic delayed neutron data from ARMI."""
    delay = dlayxs.readBinary(test_xsLibraries.DLAYXS_MCC3)
    # blend it to be all U235 as a sample
    fracs = dict(zip(delay.keys(), np.zeros(len(delay))))
    u235 = nuclideBases.byName["U235"]
    fracs[u235] = 1.0
    delay.nuclideContributionFractions = fracs
    adelay = delay.generateAverageDelayedNeutronConstants()
    # slap made-up beta fractions onto the data structure too
    adelay.delayedNeutronFractions = np.array([0.2, 0.2, 0.1, 0.1, 0.2, 0.2])
    return adelay


def _plot(t, values, solver):
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.plot(t, values[0], "-", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:blue"
    for i, v in enumerate(values[1:]):
        ax2.plot(t, v / solver.initial[i + 1], "--", label=f"Precursor {i+1}")
    ax2.tick_params(axis="y", labelcolor=color)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
