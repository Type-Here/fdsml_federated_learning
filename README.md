# Federated Learning and Homomorphic Encryption Project
**Note**: Codebase Received From Lab 8 at Unisa, as per initial commit.  
Our work is meant to test new Aggregation Methods.

---

## FIPA - Fisher-Information-Preconditioned Aggregation

FedAvg weighs a client by **one number**, how much data it has, applied
identically to every parameter. FIPA makes the weight **per direction of
parameter space**, so a client can be authoritative about one part of the model
and ignorant about another. What measures "authoritative" is the curvature of
that client's loss: if moving the parameters along a direction changes its loss a
lot, its data constrain that direction strongly.

```
H     = sum_m a_m H_m                 the round's consensus curvature
B_m   = a_m H^+ H_m                   client m's preconditioner
theta = theta + sum_m B_m Delta_m     the update
```

`H_m` is client m's empirical Fisher information matrix, `Delta_m` how far it
moved during local training, `a_m = N_m / N` its share of the round's data, `H^+`
a pseudo-inverse (`H` is singular by construction).

If every client has the *same* curvature the rule collapses to FedAvg restricted
to the common subspace - FIPA is not a different algorithm, it is FedAvg that
notices when clients are not interchangeable.

### What was implemented

- **`fipa.py`** - all the linear algebra, **torch-free** so it is unit tested
  without a GPU stack. `H_m` written out is `p x p`, i.e. 81 GB per client per
  round for a model with `p` = 142379 trainable parameters, so no function in
  that module ever materializes it: everything stays in the
  low-rank form `U_m diag(L_m) U_m^T` with `r = 5` directions, obtained from a
  randomized SVD of the client's collected gradients.
- **`model_manager_ext.py`** - the one piece that needs torch: a forward/backward
  sweep over the local data, under `eval()` and with no optimizer step, that
  produces the gradient matrix the SVD works on.
- **`aggregator_ext.py`** - the server-side branch, plus the bookkeeping that
  tells each client what to divide the round payload by (FIPA's output is already
  the finished model, so the answer is "nothing").
- **A warmup.** A low-rank curvature estimate only says something useful near a
  minimum, so the first `fipa_warmup_rounds` rounds run as plain FedAvg.
- **`explained_variance_ratio`**, reported per round in the metrics CSV: the
  share of the gradients' variance the kept directions account for, i.e. whether
  `fipa_rank` is large enough on this model and this data.

### Under Paillier encryption

Supported, and it makes the client side *cheaper* rather than more expensive.

- The client sends `z_m = U_m^T Delta_m` - **`r` ciphertexts instead of `p`** -
  because every appearance of `Delta_m` in the update rule sits behind `U_m^T`.
  What is left at home is exactly what the algorithm multiplies by zero, so the
  encrypted result equals the plaintext one.
- The server fuses its three steps into **one plaintext matrix per client**
  (`fipa.preconditioners`), so each ciphertext is multiplied exactly once.
- **`fipa_encrypted.py`** pins the fixed-point exponent on both sides of that
  multiplication. This is a correctness requirement, not a tuning knob: `phe`
  encodes a float as an integer times a power of 16, that integer must stay under
  a ceiling, and crossing the ceiling does not raise - it decrypts to a plausible
  wrong number. The module docstring has the full argument.
- `U_m` and `L_m` travel **in the clear**, because the server runs a QR
  decomposition on them and a QR of a ciphertext does not exist. Encrypted FIPA
  therefore hides the parameters and the updates, **not the curvature** - a
  weaker guarantee than encrypted FedAvg, and one worth stating.

Net cost per round is roughly **1.3x encrypted FedAvg**: what the server pays
extra, the clients stop paying in encryption.

### Running it

Two grids are provided rather than one, because accuracy and encryption cost are
different questions - the encrypted path is designed to reproduce the plaintext
one exactly, so running the whole comparison twice would measure only the clock:

| file | what it answers | runs |
|---|---|---|
| `grid_search_config.json` | do the aggregation rules differ under label skew? Dirichlet at three alphas, plaintext | 45 |
| `grid_search_config_encryption_cost.json` | what does Paillier cost, and does it change the result? Both modes, 5 rounds | 8 |

```bash
python federated_grid_search.py grid_search_config.json
```

A sanity run first, on a subsample, with the acceptance criteria to check:
`notebooks/SMOKE_TEST_FIPA.md`.

### Configuring FIPA

The keys go in `common_search_space` **as lists**, even when single-valued: they
are search axes, and the grid deduplicates against the results CSV by a
fingerprint built from those keys. Declared as fixed parameters instead, two runs
differing only in `fipa_rank` would look identical and the second would be
skipped. This is what `grid_search_config.json` sets:

```json
"aggregation_algorithm": ["FedAvg", "FedProx", "FIPA"],
"global_epoch":          [30],
"fipa_warmup_rounds":    [3],
"fipa_rank":             [5],
"fipa_grad_batches":     [64],
"fipa_pinv_rtol":        [1e-8]
```

| key | default | what it does |
|---|---|---|
| `fipa_warmup_rounds` | 0 | rounds of plain FedAvg before FIPA takes over |
| `fipa_rank` | 5 | `r`, curvature directions kept per client |
| `fipa_grad_batches` | all | mini-batches used to estimate the curvature |
| `fipa_pinv_rtol` | 1e-8 | relative cut below which a consensus-curvature eigenvalue is dropped from the pseudo-inverse |

Three things to get right:

- **`fipa_warmup_rounds` must be smaller than `global_epoch`**, or every round is
  FedAvg while the results row still says FIPA.
- **`fipa_grad_batches` must be at least `fipa_rank`** - asking for more
  directions than there are gradients is not a crash (the rank is clamped) but it
  is a meaningless estimate.
- **`fipa_rank` is the direct multiplier of the encrypted server cost**
  (`p x r x M` ciphertext operations per round), so widening it widens the most
  expensive axis of the grid.

