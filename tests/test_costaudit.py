"""A regression guard on invariant 1's *completeness*, not just its values.

Invariant 1 says tier costs are estimated from measured telemetry rather than authored. That
was always checked for the tiers the model knows about, and never for whether it knows about
every cost. It did not: reconciliation costs 16-39 us per agent and was charged nowhere, which
silently flattered every configuration that churns and produced two contradictory conclusions
about the surrogate before anyone noticed.

The general form is a cost model whose residual correlates with an observable it does not
track. bench/costaudit.py measures exactly that, and these tests pin it: one asserts the
diagnostic still has the power to see a term it is known to be missing, the other is an xfail
that will announce itself the moment the model is completed.
"""
import pytest

from bench.costaudit import residual_correlation

THRESHOLD = 0.2


def test_audit_detects_the_known_missing_term():
    """The diagnostic has to be able to see a cost that is known to be absent, or its clean
    bill of health means nothing. sim/tiered.py does not charge reconciliation, so the
    residual must correlate with the reconciliation count."""
    c = residual_correlation()
    assert abs(c) > THRESHOLD, (
        f"residual correlation with reconciliations is {c:.3f}; either the cost model now "
        f"charges reconciliation -- in which case invert this test -- or the audit has lost "
        f"its power to detect a missing term")


@pytest.mark.xfail(reason="sim/tiered.py prices steady-state tier cost only; the amortised "
                          "R * e / cap reconciliation term is a recorded OPEN QUESTION because "
                          "adding it invalidates the recorded sweep. This xfail turns into an "
                          "XPASS the moment that batch lands.", strict=True)
def test_cost_model_is_complete():
    c = residual_correlation()
    assert abs(c) <= THRESHOLD, (
        f"the cost model's residual correlates {c:.3f} with an observable it does not price")
