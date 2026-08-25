"""Tests for the FIPA maths. These are the specification of `fipa.py`.

They run without torch, which is the reason the maths lives in a torch-free
module at all: the part of FIPA that is easy to get subtly wrong is the part we
can actually execute on the development machine.

Two of them are the ones the whole method rests on:

  the factors    `U^T U ~= I_r`, and the low-rank reconstruction explains a
                 reasonable fraction of the gradients' variance.
  the aggregation on a toy problem with small p, the low-rank result must match
                 the same quantity computed with explicit p x p matrices and
                 `np.linalg.pinv`.

The full-rank reference below is the honest version of the formulas - slow,
memory-hungry, obviously correct. `preconditioned_sum` is the fast version that
must agree with it. If the two ever disagree, the reference is right.
"""

import numpy as np
import pytest

from fipa import (
    ClientFactors,
    explained_variance_ratio,
    fipa_aggregate,
    flatten_weights,
    preconditioned_sum,
    top_r_factors,
    unflatten_weights,
)


# ---------------------------------------------------------------------------
# The slow, obviously-correct reference implementation
# ---------------------------------------------------------------------------

def full_rank_preconditioned_sum(clients, rcond=1e-12):
    """`sum_m B_m Delta_m` computed the naive way, with explicit p x p matrices.

    Only usable for tiny p - which is the point: it is the ground truth the
    low-rank path is checked against on a toy problem.

        H_m = U_m diag(L_m) U_m^T
        H   = sum_m a_m H_m
        B_m = a_m H^+ H_m
    """
    total = sum(c.n_samples for c in clients)
    p = clients[0].delta.shape[0]

    hessians = []
    for c in clients:
        weight = c.n_samples / total
        h_m = c.directions @ np.diag(c.curvature) @ c.directions.T
        hessians.append((weight, h_m))

    consensus = np.zeros((p, p))
    for weight, h_m in hessians:
        consensus += weight * h_m

    consensus_pinv = np.linalg.pinv(consensus, rcond=rcond, hermitian=True)

    result = np.zeros(p)
    for (weight, h_m), c in zip(hessians, clients):
        result += weight * (consensus_pinv @ (h_m @ c.delta))
    return result


def make_client(rng, p, r, n_samples, directions=None, curvature=None):
    """A `ClientFactors` with random but valid factors: U orthonormal, L >= 0."""
    if directions is None:
        directions, _ = np.linalg.qr(rng.standard_normal((p, r)))
    if curvature is None:
        curvature = np.sort(rng.uniform(0.1, 2.0, size=r))[::-1].copy()
    return ClientFactors(
        delta=rng.standard_normal(p),
        directions=directions,
        curvature=curvature,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# flatten / unflatten - the translation to and from `get_weights` format
# ---------------------------------------------------------------------------

def test_flatten_unflatten_is_a_round_trip():
    """`set_weights` copies tensor by tensor in order: nothing may be reordered."""
    weights = [
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.arange(4, dtype=np.float32),
        np.arange(6, dtype=np.float32).reshape(2, 3),
    ]
    flat, shapes = flatten_weights(weights)

    assert flat.shape == (22,)
    assert shapes == [(3, 4), (4,), (2, 3)]

    restored = unflatten_weights(flat, shapes)
    assert len(restored) == len(weights)
    for original, back in zip(weights, restored):
        np.testing.assert_array_equal(original, back)


def test_flatten_preserves_parameter_order():
    """The first tensor's values must come first: `set_weights` zips by position."""
    weights = [np.full((2, 2), 1.0), np.full(3, 2.0)]
    flat, _ = flatten_weights(weights)
    np.testing.assert_array_equal(flat, [1, 1, 1, 1, 2, 2, 2])


def test_unflatten_returns_float32():
    """`get_weights` produces float32; the checkpoint and `set_weights` expect it."""
    flat, shapes = flatten_weights([np.zeros((2, 2), dtype=np.float32)])
    restored = unflatten_weights(flat, shapes)
    assert restored[0].dtype == np.float32


def test_unflatten_rejects_a_size_mismatch():
    """A silent mismatch would build a model that loads, trains, and is wrong."""
    with pytest.raises(ValueError):
        unflatten_weights(np.zeros(5), [(2, 2)])


# ---------------------------------------------------------------------------
# top_r_factors - compressing the curvature
# ---------------------------------------------------------------------------

def test_directions_are_orthonormal():
    """`U^T U ~= I_r`: the directions must be a basis, not just any r vectors."""
    rng = np.random.default_rng(0)
    directions, _ = top_r_factors(rng.standard_normal((40, 25)), rank=5)

    assert directions.shape == (25, 5)
    np.testing.assert_allclose(directions.T @ directions, np.eye(5), atol=1e-8)


def test_curvature_is_non_negative_and_decreasing():
    """Eigenvalues of a FIM are >= 0; the truncation keeps the largest ones."""
    rng = np.random.default_rng(1)
    _, curvature = top_r_factors(rng.standard_normal((40, 25)), rank=5)

    assert curvature.shape == (5,)
    assert np.all(curvature >= 0)
    assert np.all(np.diff(curvature) <= 1e-12)


def test_factors_match_the_explicit_fim_on_a_toy_problem():
    """The identity the whole method rests on: H = (1/n) G^T G = V diag(s^2/n) V^T.

    Computed both ways on a p small enough to build H explicitly.
    """
    rng = np.random.default_rng(2)
    n, p, r = 30, 12, 4
    G = rng.standard_normal((n, p))

    directions, curvature = top_r_factors(G, rank=r)

    fim = (G.T @ G) / n
    reference_eigenvalues = np.sort(np.linalg.eigvalsh(fim))[::-1][:r]
    np.testing.assert_allclose(curvature, reference_eigenvalues, rtol=1e-6)

    # Each returned direction must be an eigenvector of the FIM with the
    # matching eigenvalue: H u = lambda u.
    for i in range(r):
        u = directions[:, i]
        np.testing.assert_allclose(fim @ u, curvature[i] * u, atol=1e-6)


def test_a_planted_low_rank_subspace_is_recovered():
    """If the gradients really live in a 3-D subspace, the top 3 directions span it.

    This is the assumption FIPA is betting on - a sharply decaying FIM spectrum -
    so it is worth testing that the estimator finds the structure when it exists.
    """
    rng = np.random.default_rng(3)
    p, k = 20, 3
    basis, _ = np.linalg.qr(rng.standard_normal((p, k)))
    coefficients = rng.standard_normal((50, k)) * np.array([5.0, 3.0, 1.0])
    G = coefficients @ basis.T

    directions, curvature = top_r_factors(G, rank=k)

    # Same span: projecting the planted basis onto the recovered one changes
    # nothing. Comparing the projectors avoids depending on sign or ordering.
    planted_projector = basis @ basis.T
    found_projector = directions @ directions.T
    np.testing.assert_allclose(found_projector, planted_projector, atol=1e-6)
    assert np.all(curvature > 0)


def test_low_rank_reconstruction_explains_most_of_the_variance():
    """The other half of what makes truncation legitimate: r directions, most
    of the signal."""
    rng = np.random.default_rng(4)
    p, k = 30, 3
    basis, _ = np.linalg.qr(rng.standard_normal((p, k)))
    G = rng.standard_normal((60, k)) @ basis.T * 10.0
    G = G + rng.standard_normal((60, p)) * 0.01     # a little isotropic noise

    _, curvature = top_r_factors(G, rank=k)

    total_variance = np.sum((G ** 2)) / G.shape[0]  # = trace of the full FIM
    assert curvature.sum() / total_variance > 0.99
    # Same quantity, through the helper the client will log on every round.
    assert explained_variance_ratio(G, curvature) > 0.99


def test_explained_variance_is_one_when_the_gradients_are_exactly_low_rank():
    """Nothing was thrown away: the ratio must be 1, not merely close to it."""
    rng = np.random.default_rng(40)
    p, k = 20, 4
    basis, _ = np.linalg.qr(rng.standard_normal((p, k)))
    G = rng.standard_normal((50, k)) @ basis.T

    _, curvature = top_r_factors(G, rank=k)
    assert explained_variance_ratio(G, curvature) == pytest.approx(1.0, abs=1e-9)


def test_explained_variance_falls_when_the_rank_is_too_small():
    """The signal the truncation is discarding has to show up as a lower ratio.

    Isotropic gradients are the worst case for FIPA: no direction dominates, so
    keeping 2 of 10 dimensions explains about 2/10 of the variance. If a run
    reports something like this, `fipa_rank` is too small for that model.
    """
    rng = np.random.default_rng(41)
    G = rng.standard_normal((200, 10))

    _, curvature = top_r_factors(G, rank=2)
    ratio = explained_variance_ratio(G, curvature)
    assert 0.15 < ratio < 0.5


def test_explained_variance_of_zero_gradients_is_zero():
    """A client whose gradients all vanished has no variance to explain: 0, not 0/0."""
    assert explained_variance_ratio(np.zeros((10, 5)), np.zeros(3)) == 0.0


def test_explained_variance_rejects_a_non_matrix():
    with pytest.raises(ValueError):
        explained_variance_ratio(np.zeros(5), np.zeros(2))


def test_rank_larger_than_available_is_clamped():
    """Asking for more directions than gradients is a config mistake, not a crash."""
    rng = np.random.default_rng(5)
    directions, curvature = top_r_factors(rng.standard_normal((4, 20)), rank=10)
    assert directions.shape[1] == curvature.shape[0] <= 4


def test_top_r_factors_is_reproducible():
    """A randomized SVD is still randomized. Same seed, same factors."""
    rng = np.random.default_rng(6)
    G = rng.standard_normal((40, 25))
    first = top_r_factors(G, rank=5, random_state=7)
    second = top_r_factors(G, rank=5, random_state=7)
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])


@pytest.mark.parametrize("bad_G,rank", [
    (np.zeros((0, 5)), 2),
    (np.zeros(5), 2),
    (np.zeros((5, 5)), 0),
])
def test_top_r_factors_rejects_nonsense(bad_G, rank):
    with pytest.raises(ValueError):
        top_r_factors(bad_G, rank=rank)


# ---------------------------------------------------------------------------
# preconditioned_sum - the aggregation itself
# ---------------------------------------------------------------------------

def test_low_rank_matches_full_rank_on_a_toy_problem():
    """The one test that has to pass: fast path == honest path.

    Small p so the p x p reference is buildable; several clients with different
    subspaces, different ranks and different sample counts, so nothing degenerate
    can hide a mistake.
    """
    rng = np.random.default_rng(10)
    p = 15
    clients = [
        make_client(rng, p, r=3, n_samples=100),
        make_client(rng, p, r=4, n_samples=250),
        make_client(rng, p, r=2, n_samples=50),
    ]

    fast = preconditioned_sum(clients)
    slow = full_rank_preconditioned_sum(clients)

    assert fast.shape == (p,)
    np.testing.assert_allclose(fast, slow, atol=1e-8)


def test_identical_curvature_degenerates_to_projected_fedavg():
    """The defence at the oral: FIPA is FedAvg when clients are interchangeable.

    If every client reports the same (U, L), the preconditioner collapses to
    a_m times the projector onto the common subspace, so the result is the
    size-weighted average of the deltas, projected onto that subspace.
    """
    rng = np.random.default_rng(11)
    p, r = 12, 4
    shared_directions, _ = np.linalg.qr(rng.standard_normal((p, r)))
    shared_curvature = np.array([4.0, 3.0, 2.0, 1.0])

    sizes = [100.0, 300.0, 600.0]
    clients = [
        make_client(rng, p, r, n, directions=shared_directions,
                    curvature=shared_curvature)
        for n in sizes
    ]

    total = sum(sizes)
    fedavg_delta = sum((c.n_samples / total) * c.delta for c in clients)
    projector = shared_directions @ shared_directions.T

    np.testing.assert_allclose(preconditioned_sum(clients),
                               projector @ fedavg_delta, atol=1e-8)


def test_directions_nobody_has_curvature_on_are_dropped():
    """A delta component outside every client's subspace must not reach the model.

    This is the whole selling point: what a client's data say nothing about
    enters as noise under FedAvg and is zeroed here. It is also where a missing
    `rtol` cut would show up, as a huge number instead of a zero.
    """
    rng = np.random.default_rng(12)
    p, r = 10, 3
    # One orthonormal basis of the whole space, split in two: the first r columns
    # are the client's curvature directions, the rest span exactly what is left.
    basis, _ = np.linalg.qr(rng.standard_normal((p, p)))
    directions = basis[:, :r]
    stray = basis[:, r:] @ rng.standard_normal(p - r)   # lives outside the subspace

    client = ClientFactors(delta=stray, directions=directions,
                           curvature=np.array([3.0, 2.0, 1.0]), n_samples=100)

    np.testing.assert_allclose(preconditioned_sum([client]), np.zeros(p), atol=1e-10)


def test_a_zero_curvature_direction_does_not_explode():
    """H is singular by construction; inverting its null space must not amplify.

    Without the relative threshold, `1 / 1e-18` turns a direction nobody has
    information about into an astronomically large update.
    """
    rng = np.random.default_rng(13)
    p = 8
    directions, _ = np.linalg.qr(rng.standard_normal((p, 3)))
    client = ClientFactors(delta=rng.standard_normal(p), directions=directions,
                           curvature=np.array([1.0, 0.5, 0.0]), n_samples=100)

    result = preconditioned_sum([client])
    assert np.all(np.isfinite(result))
    assert np.linalg.norm(result) < 10 * np.linalg.norm(client.delta)


def test_single_client_recovers_its_own_delta_projected():
    """With one client, H = a_1 H_1 and B_1 is exactly the projector onto U_1."""
    rng = np.random.default_rng(14)
    p, r = 9, 4
    client = make_client(rng, p, r, n_samples=42)
    projector = client.directions @ client.directions.T

    np.testing.assert_allclose(preconditioned_sum([client]),
                               projector @ client.delta, atol=1e-10)


def test_zero_total_size_returns_a_zero_update():
    """A round where every client reports train_size 0 must not divide by zero."""
    rng = np.random.default_rng(15)
    clients = [make_client(rng, 6, 2, n_samples=0)]
    np.testing.assert_allclose(preconditioned_sum(clients), np.zeros(6), atol=0)


def test_float32_factors_are_computed_in_double_precision():
    """The real inputs are float32; the aggregation must not be.

    `get_weights` (model_manager.py:214) returns float32, so the deltas and the
    curvature factors reach the server in single precision. If the QR and the
    eigendecomposition inside `preconditioned_sum` are allowed to run in float32,
    the pseudo-inverse - which amplifies precisely the smallest eigenvalues -
    turns that into a visible error in the update.

    The spectrum here spans 3.0 down to 1e-4 on purpose: badly conditioned, like
    a real head's. Measured, on this fixture: about 7e-5 relative error when the
    computation runs in float32, about 5e-8 when the factors are cast at the
    door. The threshold sits between the two, so a regression fails the test.
    """
    rng = np.random.default_rng(17)
    p, r, n_clients = 4000, 5, 6
    curvature = np.array([3.0, 1.0, 0.3, 0.05, 1e-4])

    in_double, in_single = [], []
    for m in range(n_clients):
        directions, _ = np.linalg.qr(rng.standard_normal((p, r)))
        delta = rng.standard_normal(p) * 0.01
        size = 100.0 * (m + 1)
        in_double.append(ClientFactors(delta, directions, curvature, size))
        in_single.append(ClientFactors(delta.astype(np.float32),
                                       directions.astype(np.float32),
                                       curvature.astype(np.float32), size))

    reference = preconditioned_sum(in_double)
    from_float32 = preconditioned_sum(in_single)

    assert from_float32.dtype == np.float64
    relative_error = (np.linalg.norm(reference - from_float32)
                      / np.linalg.norm(reference))
    assert relative_error < 1e-6


def test_empty_round_raises():
    with pytest.raises(ValueError):
        preconditioned_sum([])


def test_mismatched_parameter_count_raises():
    """Two clients on different models is a bug worth failing loudly on."""
    rng = np.random.default_rng(16)
    with pytest.raises(ValueError):
        preconditioned_sum([make_client(rng, 8, 2, 10), make_client(rng, 9, 2, 10)])


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------

def test_fipa_aggregate_adds_the_increment_and_keeps_the_shapes():
    """`theta <- theta + increment`, in the framework's list-of-arrays format."""
    rng = np.random.default_rng(20)
    global_weights = [rng.standard_normal((3, 4)).astype(np.float32),
                      rng.standard_normal(5).astype(np.float32)]
    p = 17
    clients = [make_client(rng, p, 3, 100), make_client(rng, p, 3, 200)]

    updated = fipa_aggregate(global_weights, clients)

    assert [w.shape for w in updated] == [(3, 4), (5,)]
    expected_flat = (flatten_weights(global_weights)[0]
                     + preconditioned_sum(clients))
    np.testing.assert_allclose(flatten_weights(updated)[0], expected_flat, rtol=1e-6)


def test_fipa_aggregate_leaves_theta_untouched_when_there_is_no_movement():
    """No local movement, no update - and in particular no rescaling of theta.

    FIPA's payload is the finished model (`aggregation_policy.py:53`,
    denominator 1.0). If this test ever fails by a constant factor, someone has
    reintroduced an average where there is none.
    """
    rng = np.random.default_rng(21)
    global_weights = [rng.standard_normal((2, 3)).astype(np.float32)]
    p = 6
    directions, _ = np.linalg.qr(rng.standard_normal((p, 2)))
    clients = [ClientFactors(delta=np.zeros(p), directions=directions,
                             curvature=np.array([2.0, 1.0]), n_samples=n)
               for n in (10, 20)]

    updated = fipa_aggregate(global_weights, clients)
    np.testing.assert_allclose(updated[0], global_weights[0], atol=1e-7)