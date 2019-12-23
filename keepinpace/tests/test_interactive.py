"""See if we can make this interactive locally."""
import unittest

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from keepinpace.tests.test_solver import getTestDelayedNeutronData

from keepinpace import solver
from keepinpace import interactive

rhoNow = 0.0  # global :E
time = 0.0
STEP = 5


class TestRawInteractivity(unittest.TestCase):
    """Test the basic principle."""

    def testInteractiveSolve(self):
        """Run a kinetics problem with live user-specified reactivity"""
        global rhoNow
        global time
        def rho(t):
            return rhoNow

        adelay = getTestDelayedNeutronData()
        s = solver.KineticsSolver(adelay, rho, 1e-7 / 0.0035)

        fig, ax = plt.subplots()
        t, values = s.solve(time, time + STEP)
        time += STEP
        l, = ax.plot(t, values[0], lw=2)

        sliderAx = plt.axes([0.25, .03, 0.50, 0.02])
        slider = Slider(sliderAx, r'$\rho$', -1, 1, valinit=0.0)

        def update(val):
            global rhoNow
            global time
            rhoNow = slider.val
            t, values = s.solve(time, time + STEP)
            time += STEP
            # l.set_xdata(t)
            # l.set_ydata(values[0])
            l, = ax.plot(t, values[0], lw=2)
            print(t[-1], values[0][-1])
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        # call update function on slider value change
        slider.on_changed(update)
        # plt.show()


class TestInteractiveKinetics(unittest.TestCase):
    """Test the interactive solver."""

    def testInteractiveSolve(self):
        adelay = getTestDelayedNeutronData()
        i = interactive.InteractiveKineticsMPL(adelay, 1e-7 / 0.0035)
        i.run()


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
