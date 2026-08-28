"""Scoring the inference-time experiment.

Torch-free, like `test_routing.py`: these are the functions that turn raw counts
into the sentences the report will make, and a wrong confidence interval is not
something a run would ever complain about.

The test that earns its keep is
`test_the_interval_uses_tracks_not_images`. GTSRB's test set is 421 physical
signs photographed 30 times each, so treating 12630 images as 12630 independent
observations produces an interval roughly five times too narrow - narrow enough
to make a two-point difference between two adaptation methods look real. Every
other test here protects an arithmetic detail; that one protects a claim.
"""

import numpy as np
import pytest

from iot.gtsrb_c import REVISIT_SUFFIX, condition_name, parse_condition
from iot.metrics import (
    ConditionRow,
    accuracy_interval,
    bank_footprint,
    classification_metrics,
    difference_is_significant,
    effective_sample_size,
    latency_summary,
    retention,
    routing_report,
    rows_to_csv,
    summarize_by_family,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# Condition names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('condition, corruption, severity', [
    ('fog_s5', 'fog', 5),
    ('gaussian_noise_s1', 'gaussian_noise', 1),
    ('jpeg_compression_s3', 'jpeg_compression', 3),
    ('clean', 'clean', None),
])
def test_a_condition_name_round_trips(condition, corruption, severity):
    assert parse_condition(condition) == (corruption, severity)
    if severity is not None:
        assert condition_name(corruption, severity) == condition


def test_the_second_visit_to_clean_is_still_clean():
    """The stream revisits `clean` at the end to measure what was forgotten.

    That row needs a distinct label, but it must still resolve to the same bank
    entry - otherwise the oracle would treat the revisit as an unseen corruption
    and quietly fall back to blind adaptation on it.
    """
    assert parse_condition('clean' + REVISIT_SUFFIX) == ('clean', None)
    assert parse_condition('fog_s3' + REVISIT_SUFFIX) == ('fog', 3)


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

def test_perfect_and_useless_predictions():
    truth = [0, 1, 2, 0, 1, 2]
    assert classification_metrics(truth, truth)['accuracy'] == 1.0
    assert classification_metrics(truth, truth)['macro_f1'] == 1.0

    wrong = [1, 2, 0, 1, 2, 0]
    assert classification_metrics(truth, wrong)['accuracy'] == 0.0


def test_macro_f1_falls_where_weighted_f1_does_not():
    """The reason both averages are reported rather than one.

    A model that gets the common class right and abandons the rare one keeps a
    high weighted F1 - most images are the common class - while the macro
    average, which counts classes rather than images, records the collapse.
    Corruption hits the rare classes first, so quoting only the weighted number
    would hide the effect the experiment is looking for.
    """
    truth = [0] * 90 + [1] * 10
    predicted = [0] * 100                      # class 1 given up on entirely
    scores = classification_metrics(truth, predicted)

    assert scores['accuracy'] == pytest.approx(0.90)
    assert scores['weighted_f1'] > 0.85
    assert scores['macro_f1'] < 0.55


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        classification_metrics([0, 1, 2], [0, 1])


# ---------------------------------------------------------------------------
# The effective sample size - the one that changes conclusions
# ---------------------------------------------------------------------------

def test_the_interval_uses_tracks_not_images():
    """A 2000-image condition is worth about 420 independent observations.

    12630 GTSRB test images are 421 signs photographed 30 times, and a
    subsample of 2000 still touches almost every sign - so the effective count
    is set by the number of signs, not by how many files were drawn. The
    practical consequence is the width of the interval: about +-4.8 points at
    an accuracy of 0.5, against +-2.2 if the images were counted as
    independent.
    """
    n_eff = effective_sample_size(2000)
    assert 400 < n_eff < 425

    honest = accuracy_interval(0.5, 2000)
    naive = wilson_interval(1000, 2000)
    honest_width = honest[1] - honest[0]
    naive_width = naive[1] - naive[0]

    assert honest_width > 2 * naive_width
    assert honest_width == pytest.approx(0.095, abs=0.01)


def test_subsampling_costs_almost_nothing():
    """Why 2000 images per condition and not 12630.

    Going from a sixth of the set to the whole of it buys a few percent of
    effective sample size, because the ceiling is 421 signs either way. This is
    the argument that made the dataset 2.1 GB instead of 13 GB.
    """
    small = effective_sample_size(2000)
    whole = effective_sample_size(12630)
    assert whole / small < 1.1


def test_rho_zero_gives_back_the_image_count():
    """The assumption is a dial, and the conservative end is the default.

    `rho = 1` treats thirty frames of one sign as one observation; `rho = 0`
    treats them as thirty. The truth is in between and is not measurable, since
    GTSRB stripped the track ids from the test filenames - so the report states
    the assumption instead of pretending it away.
    """
    assert effective_sample_size(2000, rho=0.0) == pytest.approx(2000)
    assert effective_sample_size(2000, rho=1.0) < 425


def test_wilson_stays_inside_zero_one():
    """Where the textbook interval runs off the end of the scale.

    Under severe corruption an arm can sit near 0.02, and at n_eff ~ 420 the
    normal approximation would report a lower bound below zero.
    """
    low, high = accuracy_interval(0.02, 2000)
    assert 0.0 <= low < 0.02 < high <= 1.0

    low, high = accuracy_interval(0.99, 2000)
    assert high <= 1.0


def test_two_points_of_difference_are_not_a_result():
    """The claim §5 of the development notes makes, as an executable check."""
    verdict = difference_is_significant(0.52, 0.50, 2000)
    assert not verdict['significant']
    assert verdict['half_width'] > 0.02

    # Fifteen points, on the other hand, clears the bar comfortably.
    assert difference_is_significant(0.65, 0.50, 2000)['significant']


def test_retention_separates_a_ten_point_gain_from_a_ten_point_gain():
    assert retention(0.30, 0.90) == pytest.approx(1 / 3)
    assert retention(0.80, 0.90) == pytest.approx(8 / 9)
    with pytest.raises(ValueError):
        retention(0.5, 0.0)


# ---------------------------------------------------------------------------
# The efficiency budget
# ---------------------------------------------------------------------------

def resnet18_like_state(channels=(64, 64, 64, 128, 4480)):
    return {f'bn{i}': (np.zeros(c), np.ones(c)) for i, c in enumerate(channels)}


def test_the_bank_is_the_size_the_write_up_claims():
    """~38 KB a state, ~460 KB for twelve - the number quoted as an argument.

    Reported at float32, the precision a device stores them at, even though the
    states are carried in float64 here because that is what they are accumulated
    in. Quoting the float64 size would double the figure for no reason.
    """
    bank = [resnet18_like_state() for _ in range(12)]
    footprint = bank_footprint(bank)

    assert footprint['channels_per_state'] == 4800
    assert footprint['kilobytes_per_state'] == pytest.approx(37.5)
    assert footprint['total_kilobytes'] == pytest.approx(450.0)


def test_states_from_different_models_cannot_be_a_bank():
    with pytest.raises(ValueError, match="different channel counts"):
        bank_footprint([resnet18_like_state(), resnet18_like_state((32, 32))])


def test_latency_reports_the_typical_batch_and_the_bad_one():
    """The first batch pays for CUDA setup and would swallow a short mean."""
    durations = [1.0] + [0.01] * 99
    summary = latency_summary(durations, images=100 * 128)

    assert summary['first_batch_ms'] == pytest.approx(1000.0)
    assert summary['median_ms_per_batch'] == pytest.approx(10.0)
    assert summary['mean_ms_per_batch'] > summary['median_ms_per_batch']


# ---------------------------------------------------------------------------
# The routing, scored apart from the accuracy it produces
# ---------------------------------------------------------------------------

def test_hits_and_refusals_are_not_averaged_together():
    """A router has two jobs and one number would hide the second.

    On a corruption the bank contains, the right answer is to pick it. On one
    it does not, the right answer is to refuse. A single "accuracy" over both
    would reward a router that is confidently wrong on everything new.
    """
    report = routing_report(
        chosen=['fog', 'fog', 'snow', None, None, 'fog'],
        truth=['fog', 'fog', 'fog', 'fog', None, None],
    )
    # Four batches had a right answer: two hits, one wrong pick, one refusal.
    assert report['num_routable'] == 4
    assert report['hit_rate'] == pytest.approx(0.5)
    assert report['fallback_rate_when_routable'] == pytest.approx(0.25)

    # Two had none: one refusal (right), one confident pick (wrong).
    assert report['num_unroutable'] == 2
    assert report['refusal_rate'] == pytest.approx(0.5)


def test_a_router_that_always_refuses_scores_zero_and_one():
    report = routing_report(chosen=[None] * 4, truth=['fog', 'fog', None, None])
    assert report['hit_rate'] == 0.0
    assert report['refusal_rate'] == 1.0
    assert report['fallback_rate'] == 1.0


def test_mismatched_decision_lists_raise():
    with pytest.raises(ValueError):
        routing_report(chosen=['fog'], truth=['fog', 'snow'])


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def row(arm, condition, correct, n=2000, family='weather', macro=0.5):
    corruption, severity = parse_condition(condition)
    return ConditionRow(
        arm=arm, condition=condition, corruption=corruption, severity=severity,
        family=family, seen=True, n_images=n, correct=correct,
        accuracy=correct / n, macro_f1=macro, weighted_f1=macro,
        extra={'state_loads': 3})


def test_families_are_weighted_by_images_not_by_conditions():
    """Three severities of fog must not outvote one condition of snow per image.

    The mean is over images, so a family that happens to have more rows in the
    table does not get more weight for it.
    """
    rows = [row('blind', 'fog_s1', 1800), row('blind', 'fog_s5', 200, n=1000)]
    summary = summarize_by_family(rows, 'blind')['weather']

    assert summary['num_conditions'] == 2
    assert summary['n_images'] == 3000
    assert summary['accuracy'] == pytest.approx(2000 / 3000)


def test_clean_is_never_folded_into_the_corruption_average():
    rows = [row('blind', 'clean', 1900, family='clean'),
            row('blind', 'fog_s3', 900)]
    summary = summarize_by_family(rows, 'blind')

    assert set(summary) == {'clean', 'weather'}
    assert summary['weather']['accuracy'] == pytest.approx(0.45)


def test_every_arm_lands_in_one_csv(tmp_path):
    """One file, so comparing arms is a groupby and not a join.

    Four files could silently disagree about which subsample they were built
    from; one cannot.
    """
    rows = [row('source', 'fog_s3', 400), row('routed', 'fog_s3', 900)]
    path = rows_to_csv(rows, str(tmp_path / 'conditions.csv'))

    text = open(path).read()
    header = text.splitlines()[0].split(',')
    assert 'arm' in header and 'accuracy_low' in header
    assert 'state_loads' in header          # `extra` flattened into columns
    assert len(text.splitlines()) == 3


def test_writing_nothing_raises_instead_of_leaving_an_empty_file(tmp_path):
    with pytest.raises(ValueError):
        rows_to_csv([], str(tmp_path / 'empty.csv'))
