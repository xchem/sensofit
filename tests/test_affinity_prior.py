import numpy as np
import pytest
from sensofit.affinity_prior import estimate_affinity_area_prior


def _prior(*, concentration=1e-6, retained_response=5.0, association_response=None,
           dissociation_rate=0.03, active_drift_per_second=0.0, active_noise_scale=0.0,
           prefit_thresholds=None):
    t = np.linspace(0.0, 80.0, 801)
    injection, rinse, rinse_end = 20.0, 45.0, 80.0
    association = (t >= injection) & (t < rinse)
    common = np.where(association, 20.0, np.where(t >= rinse, -3.0, 0.0))
    if association_response is None:
        association_response = 2.0 * retained_response
    binding = np.where(association, association_response, np.where(
        t >= rinse, retained_response * np.exp(-dissociation_rate * (t - rinse)), 0.0))
    return estimate_affinity_area_prior(dict(
        time=t, concentration_M=concentration, raw_reference=100.0 + common,
        raw_active=(100.0 + common + binding
                    + active_drift_per_second * np.clip(t - injection, 0.0, None)
                    + active_noise_scale * np.sin(2.7 * t)),
        markers={"Injection": injection, "Rinse": rinse, "RinseEnd": rinse_end},
    ), prefit_thresholds=prefit_thresholds,
        end_guard_s=1.0, tail_endpoint_window_s=2.0)


@pytest.mark.parametrize("low, high, evidence", [
    ({"retained_response": 0.2}, {"retained_response": 8.0},
     "early_dissociation_between_mean"),
    ({"concentration": 1e-4}, {"concentration": 1e-7}, "pKD_lower"),
    ({"association_response": 1.0, "retained_response": 1.0},
     {"association_response": 10.0, "retained_response": 10.0}, "association_selectivity"),
    ({"association_response": 10.0, "retained_response": 0.5},
     {"association_response": 10.0, "retained_response": 5.0}, "retention_ratio"),
    ({"dissociation_rate": 0.20}, {"dissociation_rate": 0.01}, "tail_survival_ratio"),
])
def test_area_prior_increases_with_binding_evidence(low, high, evidence):
    low, high = _prior(**low), _prior(**high)
    assert low.usable and high.usable
    assert high.score > low.score
    if evidence == "pKD_lower":
        assert high.pKD_lower >= low.pKD_lower
    else:
        assert getattr(high, evidence) > getattr(low, evidence)


def test_concentration_term_is_tempered():
    assert np.isclose(_prior(concentration=1e-6).score - _prior(concentration=1e-5).score, 0.2)


def test_tail_drift_correction_recovers_binder_like_negative_endpoint():
    drifted = _prior(association_response=15.0, retained_response=12.0,
                     dissociation_rate=0.01, active_drift_per_second=-0.18)
    assert drifted.tail_drift_corrected
    assert 0.0 < drifted.tail_drift_weight <= 1.0
    assert drifted.raw_tail_dissociation_between_mean < 0.0
    assert drifted.tail_dissociation_between_mean > 0.0


def test_tail_drift_correction_requires_early_retention():
    weak = _prior(association_response=10.0, retained_response=0.2,
                  dissociation_rate=0.20, active_drift_per_second=-0.15)
    assert weak.retention_ratio < 0.25
    assert not weak.tail_drift_corrected
    assert weak.tail_drift_weight == 0.0
    assert weak.tail_dissociation_between_mean == weak.raw_tail_dissociation_between_mean


@pytest.fixture
def decay_priors():
    return [_prior(association_response=10.0, retained_response=8.0,
                   dissociation_rate=rate, active_noise_scale=0.5) for rate in (0.5, 0.01)]


def test_early_decay_evidence_rewards_reference_adjusted_decay(decay_priors):
    negligible = _prior(association_response=0.1, retained_response=0.1,
                       dissociation_rate=0.5, active_noise_scale=0.5)
    decaying = decay_priors[0]
    assert decaying.early_decay_delta_bic > negligible.early_decay_delta_bic
    assert decaying.early_decay_evidence > negligible.early_decay_evidence
    for prior in decay_priors:
        assert 0.0 <= prior.early_decay_evidence < 1.0


def test_persistence_evidence_rewards_absolute_retained_tail(decay_priors):
    fast, slow = decay_priors
    assert slow.retained_tail_fraction > fast.retained_tail_fraction
    assert slow.persistence_evidence > fast.persistence_evidence


def test_area_prior_is_unusable_without_positive_concentration():
    prior = _prior(concentration=0.0)
    assert not prior.usable
    assert prior.reason == "non_positive_concentration"
    assert np.isnan(prior.pKD_lower)


@pytest.mark.parametrize("offsets, regime", [
    ((1, 2, 3), "no_binding"), ((0, 1, 2), "weak"),
    ((-1, 0, 1), "medium"), ((-2, -1, 0), "tight"),
])
def test_custom_thresholds_change_regime_without_changing_score(offsets, regime):
    score = _prior().score
    prior = _prior(prefit_thresholds=score + np.asarray(offsets))
    assert prior.score == score
    assert prior.regime == regime
