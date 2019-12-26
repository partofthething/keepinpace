"""
Solves kinetics equation *interactively*.

The idea here is to track the past of what actually happened
and plot it with a solid line and also have a predicted
future with a dotted line based on the most up-to-date
solution of the PKE with the current programmed reactivity.
Whenever the reactivity input changes, the future is update.

Meanwhile, the future flows into the present in real-time,
just like in reality.

First implementation will display with matplotlib but
the idea here obviously is to pass the data to some
sweet-ass HTML5/Javascript widget
"""
from collections import namedtuple
import array
import time
import random

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation as animation
import numpy as np

from armi.nuclearDataIO import dlayxs

from .solver import KineticsSolver

TimeData = namedtuple("TimeData", ["times", "vals"])

HORIZON = 20.0
REFRESH_THRESHOLD = 0.5


class InteractiveKinetics:
    """
    Wrapper around a pure kinetics solver that tracks past and future.

    This is able to provide data for plotting during interactive kinetics sessions.
    """

    def __init__(self,
                 delayedNeutronData: dlayxs.Dlayxs,
                 normalizedGenerationTime: float,
        ):
        self.solver = KineticsSolver(
            delayedNeutronData,
            self.reactivity,
            normalizedGenerationTime
        )
        self._currentRho = 0.0
        self._solutionIndex = 0  # for getting proper initial conditions in recompute
        self.past = TimeData(array.array('f'), array.array('f'))
        fTimes, self._fullSolution = self.solver.solve(0.0, HORIZON)
        self.future = TimeData(array.array('f', fTimes),
                               array.array('f', self._fullSolution[0]))
        
        self._startTime = time.time()
        
    def reactivity(self, t):
        """Get reactivity right now."""
        return self._currentRho

    def step(self, frame=0):
        """
        Advance one point in time.
        
        Redraw plot at refresh intervals.

        Might be fun to make a noisy perturbation to reactivity at these steps
        and recompute for a more realistic experience. But then we'll have
        to recompute the future at every single frame, which is probably
        too much."""

        if not self.future.times:
            return

        if self.currentTime < self.future.times[0]:
            return
        self.past.times.append(self.future.times.pop(0))
        self.past.vals.append(self.future.vals.pop(0))
        self._solutionIndex += 1
        self._updatePowerPlot()
        self._updateContributionPlot()

    def recomputeFuture(self, duration):
        """Update the future based on current reactivity"""
        initialTime = self.currentTime
        # update initial condition
        self.solver.state = self._fullSolution[:, self._solutionIndex]
        fTimes, self._fullSolution = self.solver.solve(initialTime, initialTime + duration)
        self.future = TimeData(array.array('f', fTimes),
                               array.array('f', self._fullSolution[0]))
        self._solutionIndex = 0
        
    @property
    def currentTime(self):
        # return self.future.times[0]
        return time.time() - self._startTime

    def _updatePowerPlot(self):
        pass

    def _updateContributionPlot(self):
        pass


class InteractiveKineticsMPL(InteractiveKinetics):
    """
    Implementation of interactive kinetics that plots with Matplotlib.

    This can be run locally on computers with Python and matplotlib.

    It is a prototype of the ultimate system, which will have
    UI elements and control in a web-browser, allowing anyone
    and everyone to run their own reactor simulations.
    """

    def __init__(self, *args, **kwargs):
        InteractiveKinetics.__init__(self, *args, **kwargs)
        self.slider = None
        self.fig = None
        self.ax = None
        self.lastRefresh = time.time()
        self.pastLine = None
        self.futureLine = None
        self.msg = None

    def run(self):
        self.fig, self.ax = plt.subplots(2)
        self._initPowerPlot()
        self._initContributionPlot()
        sliderAx = plt.axes([0.25, .03, 0.50, 0.02])
        self.slider = Slider(sliderAx, r'$\rho$', -5, 1, valinit=0.0)
        
        def updateRho(val):
            """Called when slider changes."""
            newRho = self.slider.val
            self._computePromptJump(newRho)
            self._currentRho = newRho
            self.recomputeFuture(HORIZON)
        self.slider.on_changed(updateRho)

        ani = animation.FuncAnimation(self.fig, self.step, interval=30)
        plt.show()
        
    def _computePromptJump(self, newRho):
        """
        Show where the PJ approximation would put us.

        This code doesn't actually use the prompt jump approximation,
        it just does this to help with intuition.
        """
        pj = (1 - self._currentRho) / (1 - newRho)
        self.ax[0].plot(self.currentTime, self.past.vals[-1] * pj, '.', color='red')

    def _initPowerPlot(self):
        """Make time history of neutron population/power."""
        # do copies because arrays give bufferErrors if they're resized while exporting buffers
        ax = self.ax[0]
        self.pastLine, = ax.plot(
            self.past.times[:], self.past.vals[:], '-', lw=2
        )
        self.futureLine, = ax.plot(
            self.future.times[:], self.future.vals[:], '--', lw=1
        )
        self.msg = ax.text(self.currentTime, 1.01, f"{self.currentTime:.2f}\n{self.future.vals[0]:.2f}")

    def _initContributionPlot(self):
        """Make bar graph showing where changes in neutron population are coming from."""
        ax = self.ax[1]
        labels = ["Prompt fission"] + [f"Precursor Grp {i}" for i in range(self.solver.numPrecursorGroups)]
        x = np.arange(len(labels))  
        width = 0.7
        vals = self._getContributions()
        self._contributionBars = ax.bar(x - width/2, vals, width)
        ax.set_xlabel("What's driving neutron population changes")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        
    def _getContributions(self):
        state = self._fullSolution[:, self._solutionIndex]
        fission = (self._currentRho - 1
        ) / self.solver._normalizedGenerationTime * state[0]
        precursorDecays = state[1:] * self.solver._delayedNeutronData.precursorDecayConstants
        return [fission] + list(precursorDecays)
            
    def _updatePowerPlot(self):
        self.futureLine.set_xdata(self.future.times[:])
        self.futureLine.set_ydata(self.future.vals[:])
        self.pastLine.set_xdata(self.past.times[:])
        self.pastLine.set_ydata(self.past.vals[:])
        self.ax[0].relim()
        self.ax[0].autoscale_view()
        if not self.future.times:
            # rewind for initial condition
            self._solutionIndex -= 1
            self.recomputeFuture(HORIZON)
        self.ax[0].set_xlim([self.future.times[0] - 20, self.future.times[0] + 20])
        self.fig.canvas.draw_idle()
        self.msg.set_text(f"t={self.currentTime:.2f}\nn={self.future.vals[0]:.2f}\nrho={self._currentRho}")
        self.msg.set_x(self.currentTime)
            
    def _updateContributionPlot(self):
        vals = self._getContributions()
        for bar, val in zip(self._contributionBars, vals):
            bar.set_height(val)
        self.ax[1].relim()
        self.ax[1].autoscale_view()

