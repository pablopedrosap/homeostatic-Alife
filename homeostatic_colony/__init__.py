"""
Homeostatic Colony — ALife / Embodied-AI Experiment
====================================================
A Gymnasium-compatible environment for studying homeostatic control
and viability-driven reinforcement learning.

Agents must regulate internal state (Energy, Temperature, Damage)
to survive in a nutrient/toxin field. Two reward modes are supported:
  - "extrinsic": standard nutrient-collection reward
  - "homeostatic": drive-reduction reward based on deviation from viable range

This project makes NO claims about consciousness or subjective experience.
"Fear-like" behaviour refers strictly to predictive risk anticipation.
"""

__version__ = "0.1.0"
__author__ = "Homeostatic Colony Project"
