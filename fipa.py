"""FIPA: Fisher-Information-Preconditioned Aggregation. The maths, torch-free.

FedAvg weighs a client by how much data it has: one number, `n_k / N`, applied
to every parameter alike. FIPA changes the granularity - the weight becomes
*per direction of parameter space*, so a client can be authoritative about one
part of the model and ignorant about another. What measures "authoritative" is
the curvature of that client's loss: if perturbing the parameters along a
direction `v` changes client m's loss a lot, m's data constrain `v` strongly.

Why this module has no torch import. Everything below is linear algebra over
numpy arrays. Keeping the maths here means the part that is easy to get
subtly wrong - and the part that has to be defended at the exam - is unit
testable on the machine where it is written. Torch is only needed to *produce*
the gradient matrix `G`, which happens in `model_manager_ext.py`.

The vocabulary, once:

    theta       the trainable parameters, flattened into one vector of R^p.
                With a frozen backbone (`num_custom_layers > 0`) this is the
                custom head only: p = 142379 for ResNet18 + 43 classes.
    g_i         the gradient of the loss on sample (here: mini-batch) i. Which
                way those data would push the parameters.
    G           all the collected gradients stacked, shape (n, p).
    H_m         the empirical Fisher Information Matrix of client m,
                H_m = (1/n) G^T G. A p x p matrix saying how sensitive the loss
                is to every pair of directions.
    U_m, L_m    the low-rank truncation of H_m: its top-r eigenvectors (p x r)
                and eigenvalues (r). H_m ~= U_m diag(L_m) U_m^T.
    Delta_m     theta_m - theta_global, how far client m moved during its local
                training.
    N_m, N      client m's sample count (`train_size`), and their sum.
    a_m         N_m / N.

THE NUMBER THAT DECIDES EVERYTHING: H_m is p x p with p = 142379, i.e. 2.03e10
floats, 81 GB per client per round. This is not an optimization detail - it is
the reason FIPA exists in low-rank form. Truncated to r = 5 the same information
takes 2.85 MB. **No function in this module may ever materialize a p x p
matrix.** Everything stays in factored form.

The server-side rule is:

    H     = sum_m a_m H_m                 the consensus curvature of the round
    B_m   = a_m H^+ H_m                   client m's preconditioner
    theta = theta + sum_m B_m Delta_m     the update

`H^+` is the Moore-Penrose pseudo-inverse, needed because H is singular by
construction: its rank is at most M*r (40 with 8 clients and r=5) out of
p = 142379, so it cannot be inverted.

Read B_m like this: in directions where m is the only one with curvature,
H^+ H_m is close to the identity and m passes through almost whole; in
directions where m is flat, H_m ~= 0 and m's contribution is zeroed instead of
entering as noise. If every client has the *same* curvature, B_m collapses to
a_m times a projector and the whole thing degenerates to FedAvg restricted to
the common subspace - FIPA is not a different algorithm, it is FedAvg that
notices when clients are not interchangeable.

Two facts make the implementation tractable, and both are worth knowing before
writing a line:

  1. `H^+` does not depend on m, so it factors out of the sum:

         sum_m B_m Delta_m = H^+ ( sum_m a_m H_m Delta_m )

     One pseudo-inverse applied to one vector, instead of M matrices.

  2. `H_m Delta_m` is computed right-to-left from the factors:

         H_m Delta_m = U_m ( L_m * (U_m^T Delta_m) )

     r dot products, r scalings, one recombination. The result lives in R^p but
     no p x p matrix ever existed.

Contents:
  - flatten_weights / unflatten_weights : between the framework's list-of-arrays
    representation and the single vector the maths needs.
  - top_r_factors                       : WP3.1, client side. G -> (U_m, L_m).
  - preconditioned_sum                  : WP3.2, server side. The core.
  - fipa_aggregate                      : the thin wrapper the aggregator calls.
"""

from typing import List, NamedTuple, Sequence, Tuple
import sklearn.utils.extmath
import numpy as np

# Relative threshold for the pseudo-inverse. Eigenvalues of the consensus
# curvature below `rtol * largest_eigenvalue` are treated as exact zeros and
# their directions are dropped instead of inverted.
#
# Why this is not optional: H is singular by construction, so its spectrum
# always contains numerical noise around zero. Inverting a 1e-18 eigenvalue
# turns a direction nobody has any information about into a 1e18 amplification
# of round-off. The truncation is what keeps the update finite.
DEFAULT_PINV_RTOL = 1e-8


class ClientFactors(NamedTuple):
    """One client's contribution to a FIPA round, in flattened space.

    Deliberately independent of the wire format: this module knows nothing about
    pickles, sockets or the shape of `client_update`. Building these from the
    round's updates is `aggregator_ext.py`'s job.

    Fields:
        delta:      Delta_m = theta_m - theta_global, shape (p,).
        directions: U_m, the top-r curvature directions, shape (p, r),
                    orthonormal columns.
        curvature:  L_m, the matching eigenvalues, shape (r,), non-negative and
                    in decreasing order.
        n_samples:  N_m, client m's `train_size`.
    """

    delta: np.ndarray
    directions: np.ndarray
    curvature: np.ndarray
    n_samples: float


# ---------------------------------------------------------------------------
# Between the framework's representation and the maths' representation
# ---------------------------------------------------------------------------
# `get_weights` / `set_weights` (model_manager.py:213-219) speak a list of numpy
# arrays, one per parameter tensor, in the order `_get_trainable_parameters`
# yields them. FIPA's maths lives on a single vector of R^p. These two functions
# are the translation, and they must be exact inverses: `set_weights` copies by
# zip order, so an unflatten that reorders or reshapes anything silently scrambles
# the model instead of failing.


def flatten_weights(weights: List[np.ndarray]) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Concatenate a list of parameter tensors into one vector.

    Args:
        weights: parameter tensors in `_get_trainable_parameters` order.

    Returns:
        `(flat, shapes)` where `flat` has shape (p,) with p the total number of
        parameters, and `shapes` records each tensor's shape so the operation
        can be undone. Use float64 for `flat`: the aggregation does a QR and an
        eigendecomposition, and float32 there costs real precision for no
        meaningful memory saving at p ~ 1.4e5.
    """
    flat = np.concatenate([w.ravel() for w in weights]).astype(np.float64)
    shapes = [w.shape for w in weights]
    return flat, shapes


def unflatten_weights(flat: np.ndarray, shapes: Sequence[Tuple[int, ...]]) -> List[np.ndarray]:
    """Split a vector of R^p back into the list of parameter tensors.

    The inverse of `flatten_weights`. The result feeds `set_weights`, which
    copies tensor by tensor in order, so both the order and the dtype matter:
    return float32 arrays, matching what `get_weights` produces.

    Args:
        flat: shape (p,).
        shapes: the shapes returned by `flatten_weights`.

    Returns:
        The list of tensors.

    Raises:
        ValueError: if `flat` does not hold exactly `sum(prod(shape))` values.
            Failing loudly matters here - a silent size mismatch would produce a
            model that loads and trains but is wrong.
    """
    # `int(...)`: np.prod of an empty shape () returns 1.0, a float, and a float
    # slice index raises TypeError instead of the ValueError documented above.
    sizes = [int(np.prod(shape)) for shape in shapes]
    if flat.size != sum(sizes):
        raise ValueError(f"flat has {flat.size} values but shapes sum to {sum(sizes)}")
    arrays = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        arrays.append(flat[offset : offset + size].reshape(shape).astype(np.float32))
        offset += size
    return arrays


# ---------------------------------------------------------------------------
# Client side: compress the curvature
# ---------------------------------------------------------------------------


def top_r_factors(G: np.ndarray, rank: int, random_state: int = 42
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """The top-r eigenpairs of the empirical Fisher matrix, without building it.

    What this computes, in words: take the gradients this client collected, find
    the few directions along which they are most consistently large, and report
    those directions plus how large.

    The maths:

        H_m = (1/n) G^T G                       the empirical FIM, p x p
        G   = W diag(s) V^T                     the SVD of G
        =>  H_m = V diag(s^2 / n) V^T           by substitution

    So the eigenvectors of H_m are the *right* singular vectors of G, and its
    eigenvalues are the squared singular values divided by n. That identity is
    the whole trick: an SVD of G (n x p, with n the number of collected
    gradients - a few dozen) gives the eigendecomposition of a p x p matrix that
    is never formed.

    Symbols:
        G      : the collected gradients stacked, shape (n, p). One row per
                 mini-batch (see the honesty note below).
        n      : G.shape[0].
        rank   : r, how many directions to keep. Config key `fipa_rank`,
                 default 5.
        s      : the singular values of G, decreasing.
        V      : the right singular vectors, one per column.

    Use `sklearn.utils.extmath.randomized_svd(G, n_components=rank,
    random_state=random_state)` - a randomized SVD computes only the leading
    factors, costs O(n*p*r), and is already a dependency (scikit-learn 1.5.0).
    A full `np.linalg.svd` on an (n, 142379) matrix is not an option.

    Why truncating is legitimate: the FIM spectrum of a network decays sharply -
    a few dominant eigenvalues and a long, nearly flat tail. Dropping the tail
    costs little. Whether it is true *here* is exactly what the WP3.1 acceptance
    criterion measures.

    Honesty note to carry into the report: the formula wants per-*sample*
    gradients, but PyTorch returns the mean gradient of a mini-batch. We collect
    per-mini-batch gradients, so `n` counts batches, not samples. With large
    batches the averaged gradients differ less from each other and the estimated
    spectrum flattens.

    Args:
        G: collected gradients, shape (n, p).
        rank: r. Clamped to `min(n, p)` if larger - asking for more directions
            than there are gradients is a config mistake, not a crash.
        random_state: seed for the randomized SVD, so a run is reproducible.

    Returns:
        `(U, lam)` with `U` of shape (p, r), orthonormal columns, and `lam` of
        shape (r,), non-negative and decreasing.

    Raises:
        ValueError: if `G` is not 2-D, is empty, or `rank < 1`.
    """
    # returns already in decreasing order
    # float64: `G` arrives from torch as float32 (`.cpu().numpy()` on float32
    # tensors), and randomized_svd works in whatever dtype it is handed. The
    # factors computed here feed a QR and an eigendecomposition downstream,
    # where single precision costs real accuracy on a badly conditioned
    # spectrum. `asarray` also turns a list into an array, so the checks below
    # raise the documented ValueError instead of AttributeError.
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError('G is not 2-D')
    if G.size == 0:
        raise ValueError('G is empty')
    if rank < 1:
        raise ValueError('rank must be at least 1')

    # Get rank minimum from required shape and real shape
    rank = min(rank, *G.shape)
    _, s, Vh = sklearn.utils.extmath.randomized_svd(G, n_components=rank,
                                                     random_state=random_state)
    return Vh.T, s**2 / G.shape[0] # Vh.T Transposed randomized_svd returns V^T and we want V


def explained_variance_ratio(G: np.ndarray, curvature: np.ndarray) -> float:
    """How much of the gradients' variance the `r` kept directions account for.

    The second half of the WP3.1 acceptance criterion, and the number that
    justifies `fipa_rank` in the report. Without it, r = 5 is a value we took
    from the plan; with it, we can write "r = 5 captures 87% of the curvature on
    GTSRB at alpha = 0.5" - or discover that it captures 40% and raise it.

    The maths:

        trace(H_m) = trace((1/n) G^T G) = ||G||_F^2 / n = sum of ALL eigenvalues
        ratio      = sum(L_m) / trace(H_m)              = the share kept

    Symbols:
        ||G||_F   : the Frobenius norm, i.e. the square root of the sum of every
                    squared entry of G.
        trace     : the sum of the diagonal, which for a symmetric matrix equals
                    the sum of its eigenvalues - that identity is why the total
                    can be read off G directly, without ever computing the
                    p - r eigenvalues we threw away.
        L_m       : the `curvature` returned by `top_r_factors`.

    Practical reading: 1.0 means the gradients live exactly in the r directions
    we kept and the truncation is free; a low value means the FIM spectrum is
    not decaying and FIPA is discarding signal, not tail. This is what makes the
    low-rank approximation defensible or not, so it belongs in the logs of every
    FIPA round, not only in a unit test.

    Args:
        G: the same gradient matrix passed to `top_r_factors`, shape (n, p).
        curvature: the eigenvalues it returned, shape (r,).

    Returns:
        A number in [0, 1]. Exactly 0.0 when the gradients are all zero - there
        is no variance to explain, and reporting 0 is more honest than 0/0.

    Raises:
        ValueError: if `G` is not a non-empty 2-D array.
    """
    G = np.asarray(G, dtype=np.float64)
    if G.ndim != 2 or G.size == 0:
        raise ValueError("G must be a non-empty 2-D array")

    total_variance = float((G ** 2).sum()) / G.shape[0]
    if total_variance <= 0:
        return 0.0

    kept = float(np.maximum(np.asarray(curvature, dtype=np.float64), 0.0).sum())
    return kept / total_variance


# ---------------------------------------------------------------------------
# Server side: from many curvatures to one update
# ---------------------------------------------------------------------------

# Preconditioning: Multiplying a problem times a chosen matrix in order to
#                  change the geometry where the problem will be solved
# What we want: result = H+ ( sum_m * a_m * H_m * Delta_m ) = H+ * v
# Decomposing:
# v = sum_m a_m H_m Delta_m : sum of "information * shift". (Delta_m high where info is high -> shift is high)
# H = sum_m a_m H_m : sum of info, w/o shifts.
#
# SO:
# result = (sum of info)+ * (sum of info * shift)

def preconditioned_sum(clients: Sequence[ClientFactors],
                       rtol: float = DEFAULT_PINV_RTOL) -> np.ndarray:
    """`sum_m B_m Delta_m`, the FIPA increment.

    Computes, without ever forming a p x p matrix:

        a_m   = N_m / N
        H     = sum_m a_m U_m diag(L_m) U_m^T
        B_m   = a_m H^+ H_m
        result = sum_m B_m Delta_m  =  H^+ ( sum_m a_m H_m Delta_m )

    The suggested procedure, five steps:

      1. The weights.  a_m = N_m / sum_m N_m. If the total is 0 there is nothing
         to aggregate - return a zero vector rather than dividing by zero.

      2. The right-hand side.  v = sum_m a_m * U_m @ (L_m * (U_m.T @ delta_m)).
         Read the inner expression right to left: project the delta onto the r
         directions (r dot products), scale each by its curvature, recombine.
         `v` is the only vector of R^p built in this function besides the result.

      3. The joint subspace.  Stack every client's directions, each column
         pre-scaled so that the stack squared *is* H:

             C = [ U_1 * sqrt(a_1 * L_1) | ... | U_M * sqrt(a_M * L_M) ]

         with shape (p, s), s = sum_m r_m <= M*r. Then H = C C^T exactly -
         legitimate because a_m and L_m are non-negative, so the square roots
         are real. Clamp L_m at 0 first: a randomized SVD can return a tiny
         negative eigenvalue where the true one is 0.

      4. Down to s dimensions.  A thin QR, `Q, R = np.linalg.qr(C)`, gives an
         orthonormal basis Q (p, s) of the subspace containing all the clients'
         directions, and H = Q (R R^T) Q^T. Eigendecompose the *small* matrix
         S = R R^T with `np.linalg.eigh` - it is s x s, i.e. 40 x 40 with 8
         clients at r = 5, and symmetric PSD.

         This is Rayleigh-Ritz: solve a huge problem inside a small subspace
         known to contain the answer. Cost O(p*r) instead of O(p^2).

      5. The pseudo-inverse, and back up.  With S = V diag(sigma) V^T:

             H^+ = Q V diag(sigma^+) V^T Q^T
             sigma^+_i = 1/sigma_i  if sigma_i > rtol * max(sigma), else 0

         Apply it to `v`: project with `Q.T @ v`, do the small work in R^s,
         then map back with `Q @ ...`. Note `v` lies in the span of Q by
         construction, so nothing is lost by going through Q.

    What breaks if done naively:
      - forming H_m or H explicitly: 81 GB per client;
      - inverting sigma without the `rtol` cut: H is singular by construction,
        so directions nobody has information about get amplified by 1/epsilon;
      - forgetting the clamp in step 3: `sqrt` of a small negative number is nan,
        and a single nan poisons the whole update silently.

    Sanity check to keep in mind while writing it: if every client sends the
    same (U, L), then `v = H_1 @ (sum_m a_m delta_m)` and `H^+ H_1` is the
    orthogonal projector onto the common subspace, so the result is the FedAvg
    average of the deltas, projected. `tests/test_fipa.py` asserts exactly this.

    Args:
        clients: one `ClientFactors` per client that contributed to the round.
            All `delta` and `directions` must share the same p; `directions` may
            have a different r per client.
        rtol: relative cut for the pseudo-inverse; see `DEFAULT_PINV_RTOL`.

    Returns:
        The increment, shape (p,). Add it to theta - it is not an average and
        must not be divided by anything (this is why FIPA lives in
        `aggregation_policy.SERVER_RETURNS_FINAL_MODEL`, denominator 1.0).

    Raises:
        ValueError: on an empty `clients`, or on mismatched p between clients.
    """
    if not clients:
        raise ValueError("no clients to aggregate")
    p = clients[0].delta.size # p = dimension of the parameter space
    for m, c in enumerate(clients):
        if c.delta.size != p: # delta_m number should be 1 per parameter
            raise ValueError(f"client {m} has delta of size {c.delta.size}, expected {p}")
        if c.directions.shape[0] != p:
            raise ValueError(f"client {m} has directions of shape {c.directions.shape}, expected first dim {p}")
    N = sum(c.n_samples for c in clients)
    if N == 0:
        return np.zeros(p, dtype=np.float64)
    # 1.
    weights = [c.n_samples / N for c in clients]

    # 2.
    # v = clients shifts, each weighted by how much info they have about that direction. Dimension: R^p
    # v = sum_m a_m H_m Delta_m
    v = np.zeros(p, dtype=np.float64)
    blocks = []
    for c, a_m in zip(clients, weights):

        # float64 at the door. These three arrive over the socket as whatever
        # the client pickled, and with a frozen backbone that is float32
        # (`get_weights` -> `.cpu().numpy()` on float32 tensors). The QR and the
        # eigendecomposition below are where single precision actually hurts:
        # the pseudo-inverse amplifies exactly the small eigenvalues, so their
        # relative error is what ends up in the update.
        directions = np.asarray(c.directions, dtype=np.float64)
        delta = np.asarray(c.delta, dtype=np.float64)
        # Clamped once, here, and used by both step 2 and step 3: a randomized
        # SVD can return a tiny negative eigenvalue where the true one is 0, and
        # `v` and `C` must describe the same H or the pseudo-inverse is applied
        # to a vector that is not in its range.
        curvature = np.maximum(np.asarray(c.curvature, dtype=np.float64), 0.0)

        v += directions @ ((a_m * curvature) * (directions.T @ delta))
    # 3.
        # factor of Curvature of client m: C_m @ C_m.T == a_m H_m == curvature
        C_m = directions * np.sqrt(a_m * curvature)
        blocks.append(C_m)
    # C = total info available C @ C^T
    C = np.hstack(blocks)  # shape (p, s), s = sum_m r_m <= M*r

    # 4.
    Q, R = np.linalg.qr(C)
    S = R @ R.T
    sigma, V = np.linalg.eigh(S)

    # 5.
    #sigma ^ +_i = 1 / sigma_i if sigma_i > rtol * max(sigma), else 0
    sigma_plus = np.zeros_like(sigma)
    keep = sigma > rtol * max(sigma.max(), 0.0) # sigma.max could be negative if rounded about 0
    sigma_plus[keep] = 1.0 / sigma[keep]

    return Q @ (V @ (sigma_plus * (V.T @ (Q.T @ v)))) # Pseudo-inverse of Moore-Penrose

    # It simplifies H+ @ v which, in turn, simplifies a division v / H,
    # which is the same as multiplying by the inverse of H.
    # The pseudo-inverse is used because H is singular by construction,
    # so directions nobody has information about get amplified by 1/epsilon.
    # The result is the increment to be added to theta.

# ---------------------------------------------------------------------------
# The wrapper the aggregator calls
# ---------------------------------------------------------------------------


def fipa_aggregate(global_weights: List[np.ndarray],
                   clients: Sequence[ClientFactors],
                   rtol: float = DEFAULT_PINV_RTOL) -> List[np.ndarray]:
    """`theta <- theta + sum_m B_m Delta_m`, in the framework's representation.

    The only function in this module the aggregator needs to know about. It is
    deliberately trivial: all it does is move between the list-of-arrays the
    framework speaks and the vector the maths speaks.

    Args:
        global_weights: theta, the parameters the clients started this round
            from, as `get_weights` returns them.
        clients: this round's contributions.
        rtol: relative cut for the pseudo-inverse.

    Returns:
        The new parameters, same structure as `global_weights`. Already the
        finished model: the client must not rescale it.
    """
    flat, shapes = flatten_weights(global_weights)
    return unflatten_weights(flat + preconditioned_sum(clients, rtol), shapes)