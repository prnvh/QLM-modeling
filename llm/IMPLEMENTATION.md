# Implementation Notes

This document explains how the LMF stack works today: what each stage does, how to
run it from the CLI, and which tests guard it. Sections follow the real data path
from text in to field state out.

## Pipeline overview

```text
text
  → tokenizer + vocabulary
  → cue encoder (CuePacket)
  → trace bank + sparse router (ActiveRegion, with C1 provenance)
  → field loop (K settling steps):
        context pressure
        binding layer (C2 pair scorer → sparse edges)
        binding forces (trace forces + basin forces)
        interference
        settling on trace amps and basin pressures
  → (future) lucidity, decoder
```

Binding **training** (`BindingStack`) runs cue → router → context → binding only.
It does not run the full field loop or basin settling yet.

| Stage | Main code | Test file |
|-------|-----------|-----------|
| Tokenizer | `lmf/core/input/tokenizer.py` | `tests/test_tokenizer.py` |
| Cue encoder | `lmf/core/input/cue_encoder.py` | `tests/test_cue_encoder.py` |
| Trace bank / router | `lmf/core/dmf/` | `tests/test_trace_bank.py`, `tests/test_trace_router.py` |
| Cue provenance | `lmf/core/dmf/cue_provenance.py` | `tests/test_cue_provenance.py` |
| Field loop | `lmf/core/field/loop.py` | `tests/test_field_loop.py`, `tests/test_field_loop_registration.py` |
| Basins | `lmf/core/basin/` | `tests/test_basin_contract.py`, `tests/test_pair_basin_support.py`, `tests/test_binding_forces_d1.py` |
| Binding training | `lmf/training/binding_stack.py` | `tests/test_binding_edge_evaluator.py`, `tests/test_binding_overfit.py` |

Recommended end-to-end inspection:

```powershell
py lmf/infra/scripts/inspect_pipeline.py text "Help me withdraw money from the bank"
```

Open engineering gaps (e.g. interference not yet driving basins): see `llm/KNOWN_ISSUES.md`.

---

## Tokenizer, vocabulary, and inspection

The tokenizer is the first input step. It does not decide what a sentence means.
It only turns raw text into visible pieces that later layers can use.

Split rule (regex):

```text
\w+(?:['-]\w+)*|[^\w\s]
```

Word-like chunks stay together; punctuation is its own token; whitespace is not
kept. Default mode lowercases (`Bank` and `bank` match). The vocabulary maps
tokens to ids with specials first: `<pad>`, `<unk>`, `<mask>`, `<bos>`, `<eos>`,
then frequency order.

```powershell
py lmf/core/input/tokenizer.py text "Help bank!"
py lmf/infra/scripts/build_vocab.py --help
py lmf/infra/scripts/inspect_tokenizer.py text "Help bank!"
```

---

## Cue encoder

Token ids are labels, not forces. The cue encoder turns ids into small vectors
(**cues**) the trace router can score against memory keys.

Design: token embedding → 1D convolution over a local window (no transformer, no
self-attention). Output is a `CuePacket`: `cues` `[batch, seq, cue_dim]`, `mask`,
`positions`, `pooled`, and (for provenance) `token_ids`.

Strict validation on bad shapes, OOV ids, and config mistakes so training data
errors surface early.

```powershell
py lmf/core/input/cue_encoder.py text "Help bank!" --cue-dim 6 --trace
```

---

## Trace bank, router, and active region

**Trace bank** (`lmf/core/dmf/trace_bank.py`): fixed learnable slots. Each slot
has a **key** (routing), **content** (what gets passed on), **threshold**, and
**decay**. Nothing is hand-labeled (“slot 3 = bank”).

**Trace router** (`lmf/core/dmf/trace_router.py`): scores cues vs keys, applies
thresholds, keeps top-k only (`max_token` or `pooled` mode). This is sparse
lookup, not full attention.

**Active region** (`lmf/core/dmf/active_region.py`): the working set — trace ids,
content, amplitudes, cue drive — gathered from the bank for this step.

Stage 1 defaults: `lmf/infra/config/stage1_local.yaml`.

```powershell
py lmf/core/dmf/trace_bank.py inspect --num-traces 16 --key-dim 8 --content-dim 8 --trace
py lmf/core/dmf/trace_router.py text "Help bank!" --top-k 5 --trace
```

Use `--routing-only` on the pipeline script to stop after routing.

---

## Cue provenance (Commit C1)

Routing used to say *which* traces woke up, not *which token* drove each trace.
Binding supervision needs that link.

C1 records, per active trace:

- `source_cue_id` — sequence index of the winning cue (`-1` if pooled)
- `source_token_id` — vocab id of that token
- `source_span` — inclusive `[start, end]` in the sequence
- `cue_type` — token / bos / eos / pooled / …
- `normalized_cue_ids` — stable id for binding (pooled uses `-1`)

`CuePacket` must carry `token_ids`; routing fails loudly if they are missing.
Helpers: `lmf/core/dmf/cue_provenance.py`.

```powershell
py -m pytest tests/test_cue_provenance.py tests/test_trace_router.py -q
```

---

## Context-sensitive binding (Commit C2)

A plain content similarity cannot separate “bank” near “money” from “bank” near
“river”. C2 adds `BindingPairScorer` (`lmf/core/field/binding_pair_scorer.py`):
for each trace pair and relation channel it sees both contents, product, abs-diff,
pooled **context summary**, span distance/order, and cue-type embeddings. Output:
sigmoid strengths `[batch, traces, traces, relations]`.

`BindingLayer` (`lmf/core/field/binding.py`) keeps top-k edges per trace, stores
`relation_strength` and **`relation_index`** (argmax relation per edge).
`ContextOp` supplies `context_summary` on `ContextPressure`.

```powershell
py -m pytest tests/test_binding_edge_evaluator.py::test_binding_pair_scorer_changes_with_span_distance -q
```

---

## Field loop (Commits B1 and B2)

After routing, the **field loop** (`lmf/core/field/loop.py`) updates the active
trace set and **basin pressures** over several **settling steps** (default 3).
This is local dynamics on a small active set, not global sentence attention.

### Registered children

| Module | Role |
|--------|------|
| `context_op` | Pooled cue → trace drive, **basin drive**, threshold shift, context summary |
| `binding_layer` | C2 scorer → sparse `BindingState` |
| `binding_forces` | Edge scatter forces on traces; **basin force** via `BasinForceComposer` |
| `basin_bank` | Learnable attractor vectors for all basin slots |
| `interference_layer` | Binding-gated compatibility / conflict (mostly trace drive today) |
| `settling` | Damped update: bounded blend of current state and force |

B1 requires every piece to be `self.<name>` so weights appear in `parameters()`
and `state_dict()`. B2 tests in `tests/test_field_loop_registration.py` catch
detached submodules and checkpoint gaps.

### One settling step

```text
context      = context_op(cue_packet, active_region)
binding      = binding_layer(active_region, basin_state, context)
forces       = binding_forces(active_region, basin_state, binding, context)
interference = interference_layer(active_region, binding, basin_state)

trace_amp       = settling(trace_amp, trace_forces + context.trace_drive + …)
basin_pressures = settling(basin_pressures, forces.basin_force + context.basin_drive)
```

### Basin state for the loop

Always build basin state from the loop’s bank:

```python
basin_state = field_loop.make_basin_state(batch_size)
```

That sets `pressures` to zero and `vectors` to `field_loop.basin_bank.vectors`.
Passing mismatched vectors raises an error.

```powershell
py lmf/core/field/loop.py text "Help bank!" --top-k 8 --cue-dim 16 --settling-steps 3 --trace
py -m pytest tests/test_field_loop_registration.py tests/test_field_loop.py -q
```

Basin behavior is documented in the next section.

---

## Basins (Commits D1, D2, and basin package)

Basins are **learnable attractor slots** (default 32). They are meant to hold
**relationship patterns** built from bound trace pairs, not just “how loud” each
trace is in isolation.

### Two tensors per step

| Field | Shape | What it is |
|-------|-------|------------|
| `pressures` | `[batch, num_basins]` | How active each basin is **for this sentence**; updated every settling step |
| `vectors` | `[num_basins, basin_dim]` | Learnable directions in `BasinBank`; stored and checkpointed; **not** used inside the force formulas yet |

`BasinState` lives in `lmf/core/state/types.py`. Creation and checks:
`lmf/core/basin/basin_state.py` (`validate_basin_state`, `make_basin_state`).

### How basin force is computed

`BindingForcesModule` delegates to `BasinForceComposer` (`lmf/core/basin/basin_forces.py`):

```text
basin_force =
  0.2 × direct_trace_basin_force
+ 1.0 × bound_pair_basin_force
```

Scales are defaults in `lmf/infra/config/stage1_local.yaml`.

**Direct path (weak):** mask-aware mean of trace **amplitudes**, times a learnable
weight per basin. Down-weighted so it cannot dominate.

**Bound-pair path (primary):** for each sparse binding edge `(i, j)`:

1. `gather_binding_edge_batch` validates indices and gathers trace contents
   (`lmf/core/basin/binding_edges.py`).
2. Build pair features:

   ```text
   pair_ijr = [trace_i, trace_j, trace_i ⊙ trace_j, embed(relation_r), strength]
   ```

3. `PairDerivedBasinSupport` MLP (`lmf/core/basin/pair_basin_support.py`) maps each
   edge to a vector over basins; strengths weight and sum to `[batch, num_basins]`.

`relation_index` comes from `BindingLayer` when it sparsifies C2 scores. Out-of-range
relation ids **raise** (no silent clamp).

`BindingForces` exposes `direct_basin_force` and `bound_pair_basin_force` for tests
and logging. After composition, `FieldLoop` adds `context.basin_drive` and runs
`settling` on pressures.

### Package layout

```text
lmf/core/basin/
  basin_state.py      # contracts + make_basin_state
  basin_bank.py       # learnable vectors, batch_state(), CLI inspect
  binding_edges.py    # validate_binding_state, gather_binding_edge_batch
  pair_basin_support.py
  basin_forces.py     # BasinForceComposer (D1 formula + D2 module)
  emergence.py        # stub (future attractor dynamics)
```

### What basins do not do yet

- No decoder readout from `basin_pressures` or `vectors`
- `basin_bank.vectors` are not read in the force path (pressures carry the dynamics today)
- No D3 basin-family contrastive loss
- `BindingStack` training does not run basin settling

### Verify

```powershell
py lmf/core/basin/basin_bank.py inspect --num-basins 12 --trace
py lmf/core/basin/pair_basin_support.py inspect --content-dim 8 --num-basins 12 --trace
py -m pytest tests/test_basin_contract.py tests/test_pair_basin_support.py tests/test_binding_forces_d1.py -q
```

---

## Module diagnostics (Commit B3)

After backward, log gradient norms and parameter counts per block so “training
runs” cannot hide a module that never learns.

`lmf/training/module_diagnostics.py` — report fields include `binding_layer_grad_norm`,
`binding_forces_grad_norm`, `basin_grad_norm` (looks for `basin_bank` on the model),
`field_loop_param_count`, etc. Missing modules log as `n/a`.

```powershell
py lmf/training/module_diagnostics.py text "Help bank!" --top-k 8 --cue-dim 16 --trace
py -m pytest tests/test_module_diagnostics.py -q
```

---

## Binding edge training and evaluation (Commits C3)

### Goal

Teach whether two **words in the same sentence** should bind, using the real
routing + provenance path — not post-hoc string matching.

Example sentence: *"Help me withdraw money from the bank"*

- `withdraw` + `money` → should connect (label 1)
- `help` + `bank` → should not (label 0)

### Answer key

`data/stage1/binding_edges.jsonl` — one JSON object per line with `text` and `edges`
(`cue_a`, `cue_b`, `label`). Both cues must appear in the text.

### Stack

`lmf/training/binding_stack.py`:

1. CueEncoder  
2. TraceBank + TraceRouter (with C1 provenance)  
3. ContextOp  
4. BindingLayer (C2)  

`resolve_example_edges()` maps words to **sequence positions**, then matches traces
by `source_cue_id`. Loss: BCE on covered edges only.

```powershell
py lmf/training/train_binding_edges.py --steps 200 --trace
py lmf/training/binding_edge_evaluator.py text "Help me withdraw money from the bank" --top-k 16
py -m pytest tests/test_binding_edge_evaluator.py -q
```

### Reading the eval table

| Column | Meaning |
|--------|---------|
| `cue_a` / `cue_b` | Word pair tested |
| `label` | 1 = should connect, 0 = should not |
| `mass` | C2 binding strength (0–1) |
| `pos_a` / `pos_b` | Sequence positions (provenance) |
| `ok` | `yes` / `no` / `miss` (miss = trace never woke; try larger `--top-k`) |

### Limits

Does not yet cover double negatives, compositional negation, or full field/basin
training — only the binding ruler on labeled pairs.

---

## Binding overfit sanity (Commit C5)

Controlled tiny dataset; trains **full** vs **frozen binding**. Gates: accuracy ≥
0.90, positive mass ≥ 0.70, negative mass ≤ 0.20, full beats no-binding. Untrained
stack must fail the same gates (adversarial guard).

```powershell
py lmf/training/binding_overfit.py --trace
py -m pytest tests/test_binding_overfit.py -q
```

---

## Quick test commands

```powershell
# Full unit suite
py -m pytest -q

# Field + basins + binding
py -m pytest tests/test_field_loop.py tests/test_basin_contract.py tests/test_binding_edge_evaluator.py -q
```
