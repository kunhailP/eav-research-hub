"""Estimand-Aware Validation (EAV).

misclass  -- population algebra of group-specific misclassification (theory, docs/02_theory.md)
audit     -- audit-sample criteria: error regression, EAV-plugin, EAV-efficiency, EAV-structural
simulate  -- data-generating processes for the simulation studies in sims/
"""
from . import audit, misclass, simulate

__all__ = ["audit", "misclass", "simulate"]
