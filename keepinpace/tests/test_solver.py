"""Check kinetics solve."""
import unittest

import numpy as np
import matplotlib.pyplot as plt

import armi
armi.configure(armi.apps.App())
from armi.nuclearDataIO import dlayxs
from armi.nuclearDataIO.tests import test_xsLibraries
from armi.nucDirectory import nuclideBases

from keepinpace import solver


class TestKineticsSolve(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testStep(self):

        def rho(t):
            if t <= 0:
                return 0.0
            elif t < 1.0:
                return 0.10
            elif t < 2.0:
                return -0.10
            else:
                return 0.05

        delay = dlayxs.readBinary(test_xsLibraries.DLAYXS_MCC3)
        fracs = dict(zip(delay.keys(), np.zeros(len(delay))))
        u235 = nuclideBases.byName["U235"]
        fracs[u235] = 1.0
        delay.nuclideContributionFractions = fracs
        adelay = delay.generateAverageDelayedNeutronConstants()
        s = solver.KineticsSolver(adelay, rho, 1e-7 / 0.0035, np.array([0.2, 0.2, 0.1, 0.1, 0.2, 0.2]))
        t, values = s.solve(0, 5)

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot(t, values[0], '-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        for i, v in enumerate(values[1:]):
            ax2.plot(t, v / s.initial[i + 1], label=f'Precursor {i}')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
