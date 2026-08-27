"""Encrypted FIPA: the fused operator, the fixed-point grid, and the equivalence.

The criterion these tests exist to enforce is one sentence: **the encrypted
route must produce the same numbers as the plaintext one.** Every shortcut that
would make the server cheaper - truncating the consensus curvature, routing
through R^s in two steps - was rejected for that reason, because a shortcut that
changes the number also removes the only check that keeps the rest trustworthy.

They run on the development machine with no torch and no GPU: `fipa.py` and
`fipa_encrypted.py` import numpy, scikit-learn and `phe`, and nothing else. A
real 128-bit Paillier keypair is generated once for the module - the same length
the Trusted Authority uses (`trusted_authority.py`), so the encoding ceiling
under test is the real one and not a comfortable one.

What is covered:
  - `preconditioners` is the plaintext rule, fused: `sum_m P_m z_m` equals
    `preconditioned_sum` exactly, and works on records with no delta at all;
  - the fixed-point grid agrees with `phe`'s own encoding, and pins the exponent
    where `phe`'s default would let the value choose it;
  - a round on ciphertexts reproduces the plaintext round, across magnitudes
    that the default encoding cannot survive;
  - the two moments where the model is not already on the grid: the warmup
    boundary, and a plaintext round 0;
  - the guards fire instead of aggregating an increment of zeros.
"""

import numpy as np
import pytest
from phe import EncodedNumber, paillier

import fipa
import fipa_encrypted

# 128 bits, as `TrustedAuthority.__init__` defaults to. Generating a keypair is
# fast at this length; it is also not a cryptographically meaningful one, which
# is a caveat for the report and not for these tests - what matters here is that
# the encoding ceiling `n/3` is the same one a real run works against.
PUBLIC_KEY, PRIVATE_KEY = paillier.generate_paillier_keypair(n_length=128)

# Small enough that a round is milliseconds, big enough that the joint subspace
# (s = M * r = 6) is a genuine subspace of R^p rather than all of it.
P = 20
RANK = 2


def toy_client(seed: int, delta_scale: float = 1e-2, curvature_scale: float = 1.0):
    """One client's contribution: a movement and `RANK` curvature directions.

    Returns `(delta, U, L)` with `U` orthonormal columns, which is what
    `top_r_factors` produces from a real gradient matrix.
    """
    rng = np.random.default_rng(seed)
    directions, _ = np.linalg.qr(rng.normal(size=(P, RANK)))
    curvature = np.array([3.0, 1.0])[:RANK] * curvature_scale
    delta = rng.normal(0.0, delta_scale, P)
    return delta, directions, curvature


def round_factors(seeds, sizes, delta_scale=1e-2, curvature_scale=1.0):
    """A whole round, in both record shapes.

    Returns `(plaintext_factors, encrypted_factors, projections)`:
      - `plaintext_factors` carry the delta, as the plaintext branch builds them;
      - `encrypted_factors` carry `delta=None`, as the encrypted branch does;
      - `projections` are the `z_m = U_m^T Delta_m` the clients would compute.
    """
    plaintext, encrypted, projections = [], [], []
    for seed, size in zip(seeds, sizes):
        delta, directions, curvature = toy_client(seed, delta_scale, curvature_scale)
        plaintext.append(fipa.ClientFactors(delta, directions, curvature, float(size)))
        encrypted.append(fipa.ClientFactors(None, directions, curvature, float(size)))
        projections.append(fipa.project_delta(directions, delta))
    return plaintext, encrypted, projections


def decrypt(model):
    """The flat parameter vector a client would reconstruct from the wire format."""
    values, _ = fipa_encrypted.flatten_model(model)
    return np.array([PRIVATE_KEY.decrypt(value) for value in values])


def encrypt_model(theta):
    """`Enc(theta)` in the `{'shape', 'values'}` form the encrypted path uses."""
    return [{'shape': (theta.size,),
             'values': [PUBLIC_KEY.encrypt(float(v)) for v in theta]}]


# ---------------------------------------------------------------------------
# The fused operator is the plaintext rule, rearranged
# ---------------------------------------------------------------------------


def test_the_fused_operator_reproduces_the_plaintext_increment():
    """`sum_m P_m z_m == preconditioned_sum`, which is the whole design.

    Not an approximation and not a variant of the algorithm: `P_m` is the same
    three steps - project onto the joint basis, apply `H^+`, come back to R^p -
    multiplied out before any ciphertext is touched, so that a ciphertext is
    multiplied once instead of three times in cascade.
    """
    plaintext, encrypted, projections = round_factors([1, 2, 3], [40, 60, 100])

    fused = np.zeros(P)
    for operator, projection in zip(fipa.preconditioners(encrypted), projections):
        fused += operator @ projection

    np.testing.assert_allclose(fused, fipa.preconditioned_sum(plaintext), atol=1e-12)


def test_the_operator_needs_no_delta():
    """The encrypted server holds `Enc(z_m)` and no delta in any form.

    Records with `delta=None` have to be first-class, not a special case patched
    around: the curvature is all the consensus and the preconditioner ever read.
    """
    plaintext, encrypted, _ = round_factors([4, 5], [30, 70])

    with_delta = list(fipa.preconditioners(plaintext))
    without = list(fipa.preconditioners(encrypted))

    for a, b in zip(with_delta, without):
        np.testing.assert_array_equal(a, b)


def test_the_plaintext_route_still_refuses_records_without_a_delta():
    # It cannot build `v` in R^p from a projection, and silently treating a
    # missing delta as zero would aggregate a client that contributes nothing.
    _, encrypted, _ = round_factors([6], [10])
    curvature = fipa.consensus_curvature(encrypted)

    with pytest.raises(ValueError, match="delta"):
        fipa.project(encrypted, curvature.basis)


def test_project_delta_keeps_exactly_what_the_rule_uses():
    """The r numbers are lossless for the aggregation, not a summary of the delta.

    Reconstructing the delta from `z_m` loses everything orthogonal to `U_m` -
    but `H_m Delta_m` only ever sees the part inside `U_m`, so the two agree.
    """
    delta, directions, curvature = toy_client(7)
    z = fipa.project_delta(directions, delta)

    full = directions @ (curvature * (directions.T @ delta))
    from_projection = directions @ (curvature * z)
    np.testing.assert_allclose(full, from_projection, atol=1e-15)


# ---------------------------------------------------------------------------
# The fixed-point grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0.05, -3e-4, 1.0, -7.25, 3e-8])
def test_the_grid_agrees_with_phe_own_encoding(value):
    """Two ways of putting a float on the same grid must give the same integer.

    `_fixed_point_matrix` computes the integers in numpy for speed - `phe`'s own
    `encode` goes through `fractions.Fraction` and costs fourteen times more per
    entry, which on the 712k entries of one preconditioner is a fifth of the
    round. The saving is only legitimate if the two agree exactly, which they do
    because 16^-13 is 2^-52: dividing a float64 by it changes the binary
    exponent and cannot round.
    """
    exponent = fipa_encrypted.E_OPERATOR
    reference = EncodedNumber.encode(PUBLIC_KEY, float(value),
                                     precision=fipa_encrypted.BASE ** exponent)
    produced = fipa_encrypted._fixed_point_matrix(
        PUBLIC_KEY, np.array([value]), exponent)

    assert int(produced[0]) % PUBLIC_KEY.n == reference.encoding
    assert reference.exponent == exponent


MAGNITUDES = [5.0, 3e-2, 3e-4, 3e-6, 3e-8]

# The grid's absolute resolution: 16^-13 = 2^-52. Everything is represented to
# within half of this, and nothing is represented better - which is the trade
# the fixed point makes, and the reason the assertions below are absolute and
# not relative.
GRID = float(np.ldexp(1.0, fipa_encrypted.LOG2_BASE * fipa_encrypted.E_PROJECTION))


def test_the_client_pins_the_exponent_whatever_the_magnitude():
    """`phe`'s default reads the exponent off the value; the grid does not.

    This is the failure the module exists to prevent, in miniature. Adding
    ciphertexts aligns them to the *smallest* exponent in the sum, multiplying
    every other integer by 16^difference: with the default encoding one client
    whose movement is tiny drags the whole round's integers up by orders of
    magnitude, and nothing signals it. On the grid there is nothing to align,
    across eight orders of magnitude of `z`.
    """
    default_exponents, pinned_exponents = set(), set()
    for magnitude in MAGNITUDES:
        z = np.full(RANK, magnitude)

        default_exponents.update(c.exponent
                                 for c in (PUBLIC_KEY.encrypt(float(v)) for v in z))
        pinned = fipa_encrypted.encrypt_projection(PUBLIC_KEY, z)
        pinned_exponents.update(c.exponent for c in pinned)

        # Pinning changes the representation, not the number - to within the
        # grid, which is what "fixed point" means and what the declared limit is.
        np.testing.assert_allclose([PRIVATE_KEY.decrypt(c) for c in pinned], z,
                                   rtol=0, atol=GRID)

    assert pinned_exponents == {fipa_encrypted.E_PROJECTION}
    assert len(default_exponents) > 1, "phe's default would not follow the value"


def test_the_product_always_lands_on_the_model_exponent():
    """Pinned on both sides, so `E_PROJECTION + E_OPERATOR` is the only outcome.

    Across four orders of magnitude in the operator and eight in the projection.
    That constant is what makes the sums exact integer additions, round after
    round, instead of a drift that only shows up on some data.
    """
    for z_scale in (1e-1, 1e-5, 1e-9):
        for operator_scale in (1e2, 1e-2, 1e-5):
            projection = fipa_encrypted.encrypt_projection(
                PUBLIC_KEY, np.full(RANK, z_scale))
            operator = np.full((3, RANK), operator_scale)
            result = fipa_encrypted.accumulate_increment(operator, projection)

            assert {c.exponent for c in result} == {fipa_encrypted.E_MODEL}


def test_an_operator_entirely_below_the_grid_is_refused():
    # Encoding it would give an increment of exactly zero, which looks like a
    # plateau and not like a bug.
    projection = fipa_encrypted.encrypt_projection(PUBLIC_KEY, np.full(RANK, 1e-2))
    operator = np.full((3, RANK), 1e-20)

    with pytest.raises(ValueError, match="rounds to zero"):
        fipa_encrypted.accumulate_increment(operator, projection)


def test_a_projection_below_the_grid_is_refused():
    with pytest.raises(ValueError, match="rounds to zero"):
        fipa_encrypted.encrypt_projection(PUBLIC_KEY, np.array([1e-2, 1e-20]))


# ---------------------------------------------------------------------------
# The equivalence, end to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("delta_scale", [1e-1, 1e-2, 1e-4, 1e-6])
def test_an_encrypted_round_reproduces_the_plaintext_round(delta_scale):
    """The criterion. Same clients, same curvature, two routes, same model.

    The magnitudes are the point of the parametrisation: with `phe`'s default
    encoding the three smallest of these overflow the 128-bit modulus - silently
    on some keys - because the naive route multiplies the same ciphertext three
    times over. On the grid all four land on the same exponent and the same
    numbers.
    """
    theta = np.linspace(-0.5, 0.5, P)
    plaintext, encrypted, projections = round_factors(
        [11, 12, 13], [40, 60, 100], delta_scale=delta_scale)

    expected = theta + fipa.preconditioned_sum(plaintext)

    produced = fipa_encrypted.fipa_aggregate_encrypted(
        model=encrypt_model(theta),
        operators=fipa.preconditioners(encrypted),
        projections=[fipa_encrypted.encrypt_projection(PUBLIC_KEY, z)
                     for z in projections],
        model_denominator=1.0,
    )

    np.testing.assert_allclose(decrypt(produced), expected, rtol=1e-9, atol=1e-15)


def test_the_warmup_boundary_rescales_the_model_once():
    """Where the warmup hands over, the server holds `Enc(N * theta)`.

    The clients divide by `N` themselves after decrypting; the server cannot -
    it has no private key - so it multiplies homomorphically by `1/N`, encoded
    so the product lands on the model exponent. Get this wrong and every client
    trains from a model scaled by ~20000, which raises nothing and simply does
    not converge.
    """
    theta = np.linspace(-0.5, 0.5, P)
    total = 2250  # the smoke test's real N: 150 + 300 + 420 + 1380
    plaintext, encrypted, projections = round_factors([21, 22], [900, 1350])

    expected = theta + fipa.preconditioned_sum(plaintext)

    produced = fipa_encrypted.fipa_aggregate_encrypted(
        # What a FedAvg round leaves behind: the weighted sum, never averaged.
        model=encrypt_model(theta * total),
        operators=fipa.preconditioners(encrypted),
        projections=[fipa_encrypted.encrypt_projection(PUBLIC_KEY, z)
                     for z in projections],
        model_denominator=float(total),
    )

    np.testing.assert_allclose(decrypt(produced), expected, rtol=1e-8, atol=1e-13)


def test_a_plaintext_model_takes_an_encrypted_increment():
    """With no warmup at all, round 0 broadcasts the initial parameters in the clear.

    `Aggregator._initialize_weights` builds them from a throwaway model and
    never encrypts them, so the first encrypted FIPA aggregation has to add
    ciphertexts to plain floats. `phe` allows it; encoding the plaintext side on
    the grid is what keeps the result where the next round expects it.
    """
    theta = np.linspace(-0.5, 0.5, P).astype(np.float32)
    plaintext, encrypted, projections = round_factors([31, 32], [40, 60])

    expected = theta + fipa.preconditioned_sum(plaintext)

    produced = fipa_encrypted.fipa_aggregate_encrypted(
        model=[theta],  # plain numpy, exactly as round 0 broadcasts it
        operators=fipa.preconditioners(encrypted),
        projections=[fipa_encrypted.encrypt_projection(PUBLIC_KEY, z)
                     for z in projections],
        model_denominator=0,  # round 0: the payload was not scaled at all
    )

    np.testing.assert_allclose(decrypt(produced), expected, rtol=1e-6, atol=1e-9)
    values, _ = fipa_encrypted.flatten_model(produced)
    assert {c.exponent for c in values} == {fipa_encrypted.E_MODEL}


def test_the_exponent_does_not_drift_across_rounds():
    """Three consecutive FIPA rounds, and the representation never moves.

    The reason this is the test that matters: a run breaks late, when the deltas
    get small near a minimum. A single round proves nothing about that; what
    proves it is that the exponent and the budget usage are the same on round 3
    as on round 1, with the deltas shrinking by two orders each time.
    """
    theta = np.linspace(-0.5, 0.5, P)
    model = encrypt_model(theta)
    expected = theta.copy()

    for round_number, delta_scale in enumerate((1e-2, 1e-4, 1e-6)):
        plaintext, encrypted, projections = round_factors(
            [41 + round_number, 51 + round_number], [40, 60], delta_scale=delta_scale)
        expected = expected + fipa.preconditioned_sum(plaintext)

        model = fipa_encrypted.fipa_aggregate_encrypted(
            model=model,
            operators=fipa.preconditioners(encrypted),
            projections=[fipa_encrypted.encrypt_projection(PUBLIC_KEY, z)
                         for z in projections],
            # 1.0 from the second round on: the previous aggregation was FIPA,
            # so what it left behind is already the finished model.
            model_denominator=1.0 if round_number else 1.0,
        )

        values, _ = fipa_encrypted.flatten_model(model)
        assert {c.exponent for c in values} == {fipa_encrypted.E_MODEL}

    np.testing.assert_allclose(decrypt(model), expected, rtol=1e-9, atol=1e-15)


# ---------------------------------------------------------------------------
# The wire format, and the lane that needs no Paillier
# ---------------------------------------------------------------------------


def test_the_parameter_shapes_survive_the_round_trip():
    # `decrypt_weights` reshapes by these, so losing them gives a model that
    # loads, trains, and is wrong.
    shapes = [(2, 3), (3,), (4, 1)]
    model = [{'shape': shape, 'values': [PUBLIC_KEY.encrypt(0.1)] * int(np.prod(shape))}
             for shape in shapes]

    values, recovered = fipa_encrypted.flatten_model(model)
    rebuilt = fipa_encrypted.unflatten_model(values, recovered)

    assert [tuple(w['shape']) for w in rebuilt] == shapes
    assert sum(len(w['values']) for w in rebuilt) == 13


def test_a_size_mismatch_is_refused_rather_than_reshaped():
    with pytest.raises(ValueError, match="values"):
        fipa_encrypted.unflatten_model([0.0] * 5, [(2, 3)])


def test_the_plaintext_lane_computes_the_same_thing_without_paillier():
    """Duck-typed on the projection, so the server algebra is testable at speed.

    Same arithmetic, in numpy, when what arrives is numbers rather than
    ciphertexts - which is what the `simulation` encryption mode passes through.
    """
    _, encrypted, projections = round_factors([61, 62], [40, 60])
    operators = list(fipa.preconditioners(encrypted))

    increment = None
    for operator, projection in zip(operators, projections):
        increment = fipa_encrypted.accumulate_increment(operator, projection, increment)

    expected = sum(operator @ projection
                   for operator, projection in zip(operators, projections))
    np.testing.assert_allclose(np.array(increment), expected, atol=1e-12)


def test_pairing_a_projection_with_the_wrong_operator_is_refused():
    # Not a hypothetical: `P_m` and `z_m` are matched by list position, and a
    # mismatch preconditions every client's movement with someone else's
    # curvature - no error, just a run that does not converge.
    _, encrypted, projections = round_factors([71, 72], [40, 60])
    operator = next(fipa.preconditioners(encrypted))

    with pytest.raises(ValueError, match="columns"):
        fipa_encrypted.accumulate_increment(operator, list(projections[0]) + [0.0])