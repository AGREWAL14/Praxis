"""NB08c again, against records whose IAT unit cannot be read off the data.

The same notebook and the same checks as `nb08c`, run against a fixture built the other
way. The product of raw IAT and raw Rate is spread over four decades instead of sitting on
one, so the unit rule resolves nothing, every time figure stays in IAT units and the file
records the unit as unresolved. The budgets are set so that both readings put the same group
ahead, which is the other half of the agreement branch.

Two profiles rather than one because a branch that is never taken is a branch that has never
run. The resolved path does arithmetic the unresolved path does not, and the unresolved path
writes a null where the resolved one writes a number.
"""

from __future__ import annotations

from .nb08c import FIXTURE, NB08c

PROFILE = NB08c(
    name="nb08c_unresolved",
    notebook="AG_PRAXIS_NB08c_observation_time.ipynb",
    fast_env="0",
    training_path="none",
    max_traces=0,
    scale_asserts=[],
    # 50 records for the volumetric classes against 25 for the low-rate ones puts the
    # volumetric group ahead on reading A, and the hundredfold extraction window leaves it
    # ahead on reading B. The two readings agree.
    fixture={**FIXTURE, "iat_evidence": "scattered",
             "saturation_budgets": {"volumetric": 50, "low_rate": 25, "neither": 10}},
    expect_concentrated=False,
    expect_agreement=True,
)
