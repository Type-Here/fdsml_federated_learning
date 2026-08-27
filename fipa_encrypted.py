"""FIPA over Paillier ciphertexts: the fixed-point discipline, and the one multiplication.

The linear algebra of FIPA is in `fipa.py` and does not change under encryption:
the server still builds the consensus curvature `H` and its pseudo-inverse `H^+`
from the clients' curvature factors, which travel in the clear because a QR
decomposition of a ciphertext does not exist. What changes is *what* gets
multiplied by what, and in which number representation. That is this module.

Two facts about Paillier decide everything here, and neither is about algebra:

  1. **Paillier encrypts integers, not floats.** `phe` writes a float as
     `int_rep * 16^exponent` and encrypts only `int_rep`; the exponent travels
     next to the ciphertext in the clear (`phe/paillier.py`, `EncryptedNumber`).
     `int_rep` must stay below `n/3` - about 7e37 with the 128-bit modulus this
     project's Trusted Authority generates - where `n` is the Paillier modulus.
     Above that the number wraps around and decrypts to a *plausible* wrong
     value: sometimes negative, sometimes positive and of the right order of
     magnitude. Only the middle third of `[0, n)` raises `OverflowError`, so
     crossing the ceiling is not reliably an error.

  2. **Multiplication adds exponents; addition aligns to the smaller one.**
     Multiplying a ciphertext by a plaintext float sums the two exponents and
     multiplies the two integers. Adding two ciphertexts requires equal
     exponents, and the only legal way to equalise them is to drag the larger
     one *down* to the smaller, multiplying its integer by `16^difference` -
     down and not up, because dividing a ciphertext without the private key is
     impossible. So in a sum **the single smallest term dictates the
     representation of every other term**, and it can inflate them by many
     orders of magnitude without anything signalling it.

Which is why the inherited encrypted path has never had a problem: it multiplies
by `train_size`, an *integer*, and `phe` encodes an integer with exponent 0, so
the exponent does not move at all (`aggregator_ext._encrypted_sum_weighted_by_size`).
The moment a float multiplier appears, the budget usage jumps by fourteen orders
of magnitude - still safe once, not safe twice in a row.

FIPA done the natural way would multiply the same ciphertext three times in
cascade (project onto the joint subspace, apply `H^+`, come back up to R^p), and
the third product lands past the ceiling. It would also sum terms whose sizes
differ by orders of magnitude, so the smallest `z` entry of the round would
inflate everyone else's integer. Both failures are silent, both depend on the
data, and both appear late - once the deltas get small near a minimum, which is
exactly the phase FIPA exists for.

The two design decisions that follow, and this module is nothing but their
implementation:

    fuse the three server steps into ONE plaintext matrix per client
        -> `fipa.preconditioners` gives `P_m`, shape (p, r);
           every ciphertext is multiplied exactly once.

    pin the exponent on BOTH sides of that multiplication
        -> `z_m` encrypted at 16^-13, `P_m` encoded at 16^-13,
           so every product lands on 16^-26, always, and every sum is an
           exact integer addition with no alignment and no drift.

    client              wire                    server
      z_m               Enc(z_m) at 16^-13       P_m encoded at 16^-13
      = U_m^T Delta_m   5 ciphertexts            product -> 16^-26
                        U_m, L_m, N_m clear      model rescaled to 16^-26
                                                 sum: exact, no realignment

Resolution 16^-13 = 2^-52 = 2.2e-16 **in absolute value**, six orders finer than
the float32 noise `U_m` travels in anyway, so it costs no real accuracy. The
residual error against the plaintext route is ~1e-13, and - this is the point -
it no longer depends either on the data or on which key the Trusted Authority
happened to generate.

Absolute, not relative, and that distinction is the whole nature of a fixed
point: 3e-8 comes back with a relative error of ~4e-9, because the grid spacing
does not shrink with the value. It does not matter here because what has to be
accurate is the *increment added to the parameters*, whose own magnitude the
grid measures against - not each intermediate in isolation.

The declared limit: a value whose magnitude is below 16^-13 rounds to zero. For
`z_m` that is a client that did not move; for `P_m` it would be a whole
preconditioner falling off the grid, which `_fixed_point_matrix` refuses rather
than aggregating an increment of zeros.

What this module does NOT fix. `U_m` still travels in the clear, because the
server has to run a QR on it. `U_m` is a compressed summary of the client's
gradients, so encrypted FIPA hides the parameters and the updates but not the
curvature - a weaker threat model than encrypted FedAvg, and one to state rather
than paper over.

No torch here, and no `Aggregator`: everything below is `phe` and numpy, so it
runs and is unit tested on a machine without a GPU stack.
"""

import logging
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from phe import EncodedNumber
from phe.paillier import EncryptedNumber

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

# The fixed-point grid, as powers of `phe`'s encoding base (16).
#
#   E_PROJECTION  the exponent the client encrypts `z_m` at.
#   E_OPERATOR    the exponent the server encodes `P_m` at.
#   E_MODEL       where their product lands, and therefore where the model has
#                 to sit for the addition to need no realignment.
#
# Both sides are pinned, not just one: pinning only the client would still let
# `P_m`'s value-dependent exponent move the product around from round to round.
E_PROJECTION = -13
E_OPERATOR = -13
E_MODEL = E_PROJECTION + E_OPERATOR

BASE = EncodedNumber.BASE

# `phe`'s base is 16 = 2^4, so putting the point at 16^e means multiplying by
# 2^(-4e) - a pure binary exponent shift, which float64 performs exactly at any
# magnitude. That is what lets `_fixed_point_matrix` compute the integers in
# numpy and still match `phe`'s own exact `Fraction` arithmetic; see its
# docstring. `np.ldexp` is used rather than a power so no rounded `16.0 ** -13`
# ever enters the arithmetic.
LOG2_BASE = 4


# ---------------------------------------------------------------------------
# The grid: floats in, exact integers out
# ---------------------------------------------------------------------------


def _encode_fixed(public_key, value: float, exponent: int) -> EncodedNumber:
    """Encode one plaintext number at an exponent we choose, not one `phe` picks.

    `EncodedNumber.encode` left to itself reads the exponent off the value, which
    is what makes a sum realign. It does accept a `precision` argument, but it
    turns it into an exponent through `floor(log(precision, 16))` - a float
    logarithm, one ulp away from picking the neighbouring exponent. Here the
    exponent is passed as an integer and used as such, so the grid is exact by
    construction.

    `Fraction` rather than float arithmetic for the same reason `phe` uses it:
    `value * 16^-exponent` in floating point would round before `round` does.

    Args:
        public_key: the Paillier public key the encoding is for. Needed because
            negative integers are represented as `int_rep + n`.
        value: the plaintext number.
        exponent: the power of 16 to put the point at.

    Returns:
        An `EncodedNumber` whose `exponent` is exactly `exponent`.

    Raises:
        ValueError: if the value rounds to integer zero at this exponent while
            not being zero itself - i.e. it fell through the grid - or if it
            does not fit under the Paillier ceiling.
    """
    int_rep = round(Fraction(value) * Fraction(BASE) ** -exponent)

    if int_rep == 0 and value != 0:
        raise ValueError(
            f"{value!r} rounds to zero on the fixed-point grid 16^{exponent} "
            f"(resolution {float(BASE) ** exponent:.3e}). Encoding it would "
            f"silently contribute nothing to the aggregation."
        )
    if abs(int_rep) > public_key.max_int:
        raise ValueError(
            f"{value!r} at 16^{exponent} needs the integer {int_rep}, past the "
            f"Paillier ceiling {public_key.max_int}. Past it a value does not "
            f"raise on decryption, it comes back plausible and wrong."
        )
    return EncodedNumber(public_key, int_rep % public_key.n, exponent)


def _fixed_point_matrix(public_key, values: np.ndarray, exponent: int) -> np.ndarray:
    """The integer representations of a whole array, on the grid, in one numpy pass.

    The bulk counterpart of `_encode_fixed`, and it exists for speed alone: the
    server encodes `p * r` entries of `P_m` per client per round - 712k with the
    ResNet18 head at rank 5 - and going through `Fraction` for each costs about
    9 us against 0.6 us for building the `EncodedNumber` from an integer already
    computed. On a round that is the difference between paying a fifth more and
    paying nothing.

    It is exact at every magnitude, and that is a fact about binary floating
    point rather than a tolerance being accepted. `phe`'s base is 16 = 2^4, so
    putting the point at 16^e is multiplying by 2^(-4e); `np.ldexp` does exactly
    that, changing the binary exponent and leaving the 53-bit mantissa
    untouched. The scaled value therefore carries precisely the information the
    input float had, and `np.rint` rounds half to even exactly as `round` does
    on a `Fraction`. Verified against `phe`'s own encoding over five magnitude
    regimes in `tests/test_fipa_encrypted.py`, including integers past 2^53 -
    where a *value* would lose precision but this shift does not, because the
    input had no more bits to begin with.

    What is checked is what can actually go wrong: an array that lands entirely
    on zero, and the Paillier ceiling.

    Args:
        public_key: needed for the ceiling check only.
        values: any float array.
        exponent: the power of 16 to put the point at.

    Returns:
        A float64 array holding integer values - float64 rather than int64
        because int64 would overflow silently at 9.2e18, well inside the range
        the Paillier ceiling still allows.

    Raises:
        ValueError: if every entry rounds to zero - the whole array fell through
            the grid - or if the largest one crosses the Paillier ceiling, past
            which a value does not raise on decryption but comes back plausible
            and wrong.
    """
    values = np.asarray(values, dtype=np.float64)
    int_reps = np.rint(np.ldexp(values, -LOG2_BASE * exponent))

    largest = float(np.abs(int_reps).max()) if int_reps.size else 0.0
    if largest == 0.0 and np.any(values != 0):
        raise ValueError(
            f"every entry rounds to zero on the fixed-point grid 16^{exponent} "
            f"(resolution {np.ldexp(1.0, LOG2_BASE * exponent):.3e}); the "
            f"largest input was {float(np.abs(values).max()):.3e}. Aggregating "
            f"this would add an increment of exactly zero and look like a plateau."
        )
    if largest > public_key.max_int:
        raise ValueError(
            f"the fixed-point grid 16^{exponent} needs integers up to "
            f"{largest:.3e}, past the Paillier ceiling {public_key.max_int}."
        )
    return int_reps


# ---------------------------------------------------------------------------
# Client side: encrypt r numbers instead of p
# ---------------------------------------------------------------------------


def encrypt_projection(public_key, projection: Sequence[float]) -> List[EncryptedNumber]:
    """Encrypt `z_m = U_m^T Delta_m`, the r numbers that replace the whole delta.

    This is the entire client-side cost of encrypted FIPA: five encryptions
    where encrypted FedAvg pays 142379. The reason it is legitimate to send so
    little is in `fipa.project_delta` - every appearance of `Delta_m` in the
    update rule sits behind `U_m^T`, so the rest of the delta is multiplied by
    zero whatever the other clients send.

    Args:
        public_key: this client's Paillier public key, from the Trusted
            Authority.
        projection: `z_m`, r numbers.

    Returns:
        r `EncryptedNumber`s, every one of them at exponent `E_PROJECTION` -
        which is what lets the server sum the products without realigning
        anything.
    """
    return [
        public_key.encrypt_encoded(
            _encode_fixed(public_key, float(value), E_PROJECTION), r_value=None
        )
        for value in projection
    ]


# ---------------------------------------------------------------------------
# Between the wire format and a flat list of numbers
# ---------------------------------------------------------------------------
# On the encrypted path `current_weights` is a list of `{'shape', 'values'}`
# dicts, one per parameter tensor, with `values` a flat list of ciphertexts -
# the format `utils.encrypt_weights` produces and `utils.decrypt_weights`
# consumes. FIPA's maths speaks one flat vector of R^p. These two are the
# translation, the encrypted twin of `fipa.flatten_weights`.


def flatten_model(weights: Sequence[Any]) -> Tuple[List[Any], List[Tuple[int, ...]]]:
    """Flatten the server's model into one list of p values, plus the shapes.

    Accepts both representations `current_weights` can be in, because the server
    genuinely holds both at different moments:

      - `{'shape', 'values'}` dicts, after any encrypted aggregation;
      - plain numpy arrays, before the first one - `Aggregator._initialize_weights`
        builds the initial parameters from a throwaway model, in the clear, and
        with no warmup at all round 0 broadcasts exactly those.

    Returns:
        `(values, shapes)`. `values` holds ciphertexts or plain floats depending
        on which of the two came in; the caller has to cope with either, and
        `add_increment` does.
    """
    values: List[Any] = []
    shapes: List[Tuple[int, ...]] = []
    for tensor in weights:
        if isinstance(tensor, dict):
            shapes.append(tuple(tensor['shape']))
            values.extend(tensor['values'])
        else:
            array = np.asarray(tensor)
            shapes.append(tuple(array.shape))
            values.extend(array.ravel().tolist())
    return values, shapes


def unflatten_model(values: Sequence[Any],
                    shapes: Sequence[Tuple[int, ...]]) -> List[Dict[str, Any]]:
    """Rebuild the `{'shape', 'values'}` list the client knows how to decrypt.

    Always emits the dict form, including when the values are plain floats: the
    round after an encrypted FIPA aggregation must look encrypted to
    `federated_client._process_server_weights`, which dispatches on whether the
    first element is a dict.

    Raises:
        ValueError: if the values do not fill the shapes exactly. A silent size
            mismatch here would produce a model that loads, trains and is wrong.
    """
    sizes = [int(np.prod(shape)) for shape in shapes]
    if len(values) != sum(sizes):
        raise ValueError(
            f"{len(values)} values but the shapes hold {sum(sizes)}"
        )
    rebuilt = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        rebuilt.append({'shape': shape, 'values': list(values[offset:offset + size])})
        offset += size
    return rebuilt


# ---------------------------------------------------------------------------
# Server side: put the model on the grid, then add the increment
# ---------------------------------------------------------------------------


def rescale_to_grid(values: Sequence[Any], factor: float) -> List[Any]:
    """Multiply every ciphertext by `factor` and land it exactly on `E_MODEL`.

    Needed once per run, at the round where the warmup hands over to FIPA. Up to
    that round the server has been running plain FedAvg, so `current_weights`
    holds `sum_k n_k W_k`, that is `N * theta` - the weighted sum the clients
    divide by `N` themselves after decrypting. FIPA's server does not get to
    delegate that division: it has to add its increment to `theta`, not to
    `N * theta`, and it has no private key. So it scales the ciphertexts by
    `1/N` homomorphically, which is allowed - multiplying a ciphertext by a
    plaintext scalar is one of Paillier's two operations.

    The scalar's own exponent is chosen per ciphertext, as `E_MODEL - exponent`
    of that ciphertext, so that the product lands on `E_MODEL` whatever the
    encrypted sum left behind. Per ciphertext and not once for the whole model
    because the entries of an encrypted sum do not share an exponent: each one
    was aligned against its own neighbours.

    A note to keep in the report rather than hide: the server's `theta` and the
    clients' `theta` are now computed differently - the clients divide in
    float64, the server multiplies by a fixed-point `1/N`. They agree to about
    1e-10 relative, three orders below the float32 the weights travel in, so the
    increment is added to the same starting point for every practical purpose.

    Args:
        values: the flat model, ciphertexts or plain floats.
        factor: `1/N`.

    Returns:
        A new list; the input is not modified.
    """
    rescaled: List[Any] = []
    for value in values:
        if isinstance(value, EncryptedNumber):
            scale = _encode_fixed(value.public_key, factor, E_MODEL - value.exponent)
            rescaled.append(value * scale)
        else:
            # Plaintext model: nothing to keep on a grid, and no encoding to
            # consume. This is the simulation lane and the unit tests.
            rescaled.append(value * factor)
    return rescaled


def accumulate_increment(operator: np.ndarray, projection: Sequence[Any],
                         into: Optional[List[Any]] = None) -> List[Any]:
    """`into[i] += sum_j P_m[i, j] * Enc(z_m[j])`, one client's whole contribution.

    The only place a ciphertext is touched, and it is touched exactly once each:
    `P_m` already has the projection onto the joint subspace, the pseudo-inverse
    of the consensus curvature and the return to R^p fused into it by
    `fipa.preconditioners`. Doing those three as three multiplications on the
    same ciphertext is what the fixed-point analysis rules out.

    Cost, and it is the honest number for the whole feature: `p * r` ciphertext
    multiplications per client, so `p * r * M` per round - 5.7 million with the
    ResNet18 head, rank 5 and 8 clients, at tens of microseconds each. That is
    around 1.3 times the cost of an encrypted FedAvg round, not five times,
    because what the server pays extra the clients stop paying in encryption.

    Duck-typed on purpose. If `projection` holds plain numbers rather than
    ciphertexts the whole thing collapses to `operator @ projection` in numpy,
    which is the same arithmetic without Paillier - useful for testing the
    server-side algebra at full size in milliseconds, and it is what the
    `simulation` encryption mode would exercise.

    Args:
        operator: `P_m`, shape (p, r).
        projection: `Enc(z_m)`, r ciphertexts, or r plain numbers.
        into: the running total over clients, length p, or None to start one.
            Modified in place and returned, so a round accumulates into a single
            list instead of building M of them.

    Returns:
        The accumulator.

    Raises:
        ValueError: on a shape mismatch, or through `_fixed_point_matrix` if
            `P_m` does not fit the grid.
    """
    operator = np.asarray(operator, dtype=np.float64)
    if operator.ndim != 2:
        raise ValueError(f"the operator is not 2-D: shape {operator.shape}")
    if operator.shape[1] != len(projection):
        raise ValueError(
            f"the operator has {operator.shape[1]} columns but the projection "
            f"holds {len(projection)} values"
        )
    if into is not None and len(into) != operator.shape[0]:
        raise ValueError(
            f"the accumulator holds {len(into)} values but the operator maps "
            f"into R^{operator.shape[0]}"
        )

    # `len` and not truthiness: a projection can arrive as a numpy array, and
    # `not array` raises on anything with more than one element.
    if len(projection) == 0:
        raise ValueError("empty projection: the client sent no curvature directions")

    if not isinstance(projection[0], EncryptedNumber):
        # Plaintext lane: the same sum, in numpy.
        contribution = operator @ np.asarray(projection, dtype=np.float64)
        if into is None:
            return list(contribution)
        return [total + term for total, term in zip(into, contribution)]

    public_key = projection[0].public_key
    modulus = public_key.n
    int_reps = _fixed_point_matrix(public_key, operator, E_OPERATOR)

    total = into if into is not None else [None] * operator.shape[0]
    for i, row in enumerate(int_reps):
        term = projection[0] * EncodedNumber(public_key, int(row[0]) % modulus, E_OPERATOR)
        for j in range(1, len(projection)):
            term = term + projection[j] * EncodedNumber(
                public_key, int(row[j]) % modulus, E_OPERATOR)
        total[i] = term if total[i] is None else total[i] + term
    return total


def add_increment(model_values: Sequence[Any], increment: Sequence[Any]) -> List[Any]:
    """`theta <- theta + increment`, whichever of the two is encrypted.

    Three cases, and the middle one is the one that is easy to forget:

      - model encrypted, increment encrypted: an ordinary homomorphic addition.
        Both sit on `E_MODEL`, so no realignment happens and the integers do not
        grow.
      - **model in the clear, increment encrypted**: happens when FIPA runs from
        round 0 with no warmup, because round 0 broadcasts the initial
        parameters as plain arrays even in encrypted mode. `phe` can add a
        plaintext to a ciphertext, and encoding that plaintext at `E_MODEL`
        keeps the result on the grid instead of letting the model's own
        magnitude choose an exponent.
      - neither encrypted: the plaintext lane of `accumulate_increment`.

    Raises:
        ValueError: if the two do not have the same length - that would mean the
            operator and the model disagree about p.
    """
    if len(model_values) != len(increment):
        raise ValueError(
            f"the model holds {len(model_values)} values and the increment "
            f"{len(increment)}; they must both live in R^p"
        )

    updated: List[Any] = []
    for base, term in zip(model_values, increment):
        if isinstance(base, EncryptedNumber):
            updated.append(base + term)
        elif isinstance(term, EncryptedNumber):
            updated.append(term + _encode_fixed(term.public_key, float(base), E_MODEL))
        else:
            updated.append(base + term)
    return updated


def fipa_aggregate_encrypted(model: Sequence[Any],
                             operators: Iterable[np.ndarray],
                             projections: Sequence[Sequence[Any]],
                             model_denominator: float = 1.0,
                             logger: logging.Logger = log) -> List[Dict[str, Any]]:
    """The whole server-side encrypted FIPA round, in the wire format.

    Composition only - every piece it calls is documented above. In order:

        model, shapes   <- flatten_model(current_weights)
        model           <- rescale_to_grid(model, 1/N)      once, at the boundary
        increment       <- sum_m accumulate_increment(P_m, Enc(z_m))
        current_weights <- unflatten_model(add_increment(model, increment), shapes)

    Args:
        model: `current_weights` as the aggregator holds it - the dict form
            after an encrypted aggregation, plain arrays before the first one.
        operators: `P_m` per client, in the same order as `projections`. Takes
            an iterable so `fipa.preconditioners` can stay a generator and only
            one 5.7 MB matrix exists at a time.
        projections: `Enc(z_m)` per client. **Order matters**: `P_m` and `z_m`
            have to come from the same client, or every client's movement is
            preconditioned with someone else's curvature.
        model_denominator: what the clients were told to divide this round's
            payload by, i.e. what `model` is currently scaled by. `N` at the
            round where the warmup hands over, `1.0` afterwards, `0` at round 0
            where the payload is the initial parameters and was not scaled at
            all. Anything other than 1 (or 0) triggers the one-off rescale.
        logger: where to report the rescale. Passed in rather than taken from
            this module, so the line lands in the server's own log next to the
            round it belongs to - the module logger carries a `NullHandler` and
            would swallow it.

    Returns:
        The new `current_weights`, in `{'shape', 'values'}` form.

    Raises:
        ValueError: if the two sequences have different lengths, or through any
            of the pieces.
    """
    values, shapes = flatten_model(model)

    if model_denominator not in (0, 1.0):
        logger.info("Rescaling the encrypted model by 1/%s onto the fixed-point grid.",
                    model_denominator)
        values = rescale_to_grid(values, 1.0 / float(model_denominator))

    increment: Optional[List[Any]] = None
    clients = 0
    for operator, projection in zip(operators, projections):
        increment = accumulate_increment(operator, projection, increment)
        clients += 1

    if clients != len(projections):
        raise ValueError(
            f"{clients} operators for {len(projections)} clients: every client's "
            f"projection must be paired with its own preconditioner"
        )
    if increment is None:
        raise ValueError("no clients to aggregate")

    return unflatten_model(add_increment(values, increment), shapes)