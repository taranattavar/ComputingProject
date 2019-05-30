"""
Module to run the simulation of 2D collisions.
"""

import simulation as sim

S = sim.Simulation(num=10, T=300)

S.run(num_frames=100000, animate=True)
