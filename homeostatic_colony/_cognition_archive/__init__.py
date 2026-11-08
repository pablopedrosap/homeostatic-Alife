"""Cognition-oriented modules for functional self-maintenance experiments.

The components here are *proxy functional mechanisms*:
- affective state estimation from homeostasis + prediction
- persistent self-state and autobiographical memory
- intrinsic action modulation for self-regulation

They are designed for scientific probing of consciousness-relevant mechanisms
without making claims of literal consciousness.
"""

from .affective_core import AffectiveConfig, AffectiveCore, AffectiveState
from .self_model import RecurrentSelfModel, SelfModelConfig, SelfModelOutput
from .self_state import EpisodeSummary, SelfState, SelfStateStore
from .intrinsic_controller import ControllerDiagnostics, IntrinsicController
from .multi_timescale import FastAdaptState, MultiTimescaleAdapter

__all__ = [
    "AffectiveConfig",
    "AffectiveCore",
    "AffectiveState",
    "RecurrentSelfModel",
    "SelfModelConfig",
    "SelfModelOutput",
    "EpisodeSummary",
    "SelfState",
    "SelfStateStore",
    "ControllerDiagnostics",
    "IntrinsicController",
    "FastAdaptState",
    "MultiTimescaleAdapter",
]
