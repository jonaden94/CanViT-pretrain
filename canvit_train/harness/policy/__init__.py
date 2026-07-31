"""Viewpoint-selection policy: the learned alternative to random glimpses.

- ``build.py`` — ``build_policy`` (assembles scorer + objective from the spec) and
                 ``check_credit_regime``
- ``joint.py`` — ``JointPolicy``: drives glimpse choice during task training
- ``rl.py``    — the objectives (``QReg``, ``PG``, ``VPG``) and their losses

``build``'s public API is re-exported here, so ``from canvit_train.harness.policy import
build_policy`` works exactly as it did when this was a single module.
"""

from canvit_train.harness.policy.build import build_policy, check_credit_regime

__all__ = ["build_policy", "check_credit_regime"]
