# Implementation Notes

## Tokenizer, Vocabulary, and Inspection

The tokenizer is the first input step of the LMF system. It does not decide what a sentence means. It only turns raw text into visible pieces that later model layers can use.

The split rule is regex based:

```text
\w+(?:['-]\w+)*|[^\w\s]
```

In simple words, it keeps word-like chunks together and keeps punctuation visible. `\w+` captures a word body. The `(?:['-]\w+)*` part lets apostrophes and hyphens stay inside a word, so `can't` and `re-enter` remain one token each. The `[^\w\s]` part captures punctuation or symbols, so `!`, `.`, `?`, and similar characters are separate tokens. Whitespace only separates tokens. It is not kept as a token. By default the tokenizer lowercases text, so `Bank` and `bank` become the same token. Case-sensitive mode can turn that off.

After tokenization, the vocabulary maps tokens to ids. Special tokens always come first: `<pad>`, `<unk>`, `<mask>`, `<bos>`, and `<eos>`. Normal tokens are ordered by frequency, then alphabetically when tied. This makes the ids repeatable. Unknown tokens become `<unk>`.

The build-vocab script reads files and writes a vocab JSON. The inspection commands show the exact input, tokens, ids, token/id rows, and decoded text. Tests cover normal text, punctuation, contractions, hyphenated words, unicode, unknown words, padding, bad vocab data, CLI output, and logs.

Simple run command:

```powershell
py lmf/core/input/tokenizer.py text "Help bank!"
```

## Cue Encoder

The cue encoder is the second input step. The tokenizer gives us visible token
ids, but the memory field cannot use ids directly. An id is only a label. It
does not contain a direction, strength, or shape that can push on traces. The cue
encoder turns those ids into small vectors called cues. These cues are still not
"meaning" in the full system. They are the first learned signals that later
components can compare against trace keys, route into sparse active regions, and
use as pressure in the field loop.

The implementation is intentionally local and small. Each token id first passes
through a token embedding table. That gives every token a learned vector. Then
the sequence goes through a one-dimensional convolution. A convolution looks at a
fixed local window around each token, such as the token itself plus its immediate
neighbors. This lets "bank" near "money" produce a different local cue than
"bank" near "river" once training has shaped the weights, but it does not build a
global token-token attention map. The design follows the plan: token embedding,
local window features, then cue vector. There is no transformer encoder and no
self-attention stack.

The output is a `CuePacket`. Its main field, `cues`, has shape `[batch,
sequence, cue_dim]`. The packet also carries a boolean `mask`, integer
`positions`, and a pooled cue summary. Padding ids are masked before the
convolution and zeroed after projection, so padded positions do not become active
cue signals. The pooled cue is a masked average, which gives later modules a
compact summary without losing access to the per-token cues.

The module has strict input checks because this part sits close to the training
data. It rejects empty tensors, non-integer ids, ids outside the vocabulary,
masks with the wrong shape, positions with the wrong shape, even convolution
windows, invalid dropout values, and bad dimensions. These checks are meant to
fail loudly instead of silently producing damaged training examples.

The cue encoder can also be inspected from the command line. It builds a
temporary vocabulary from the input text, encodes the text with `<bos>` and
`<eos>`, runs the cue encoder with a reproducible seed, and prints the token ids,
cue shape, mask, positions, first cue, and pooled cue. Add `--trace` to emit a
human-readable log line showing the input shape, output shape, active token
count, id preview, and pooled cue preview.

Simple run command:

```powershell
py lmf/core/input/cue_encoder.py text "Help bank!" --cue-dim 6 --trace
```

## Trace bank, router, and active region

After cues are built, the memory layer starts. A **trace bank** is a fixed set of
learnable memory slots. Nothing is labeled by hand (no “slot 3 means bank”).
Training is meant to shape what each slot stores.

Each slot has four learned parts: a **key** (does this cue match me?), **content**
(what this slot holds), **threshold** (how strong the match must be), and
**decay** (how fast the slot fades). Keys are compared to cues; content is what
gets passed on when the slot is selected.

The bank lives in `lmf/core/dmf/trace_bank.py`. You can inspect it alone with a
fixed seed. `key_dim` should match the cue encoder width. Stage 1 defaults live
in `lmf/infra/config/stage1_local.yaml`.

```powershell
py lmf/core/dmf/trace_bank.py inspect --num-traces 16 --key-dim 8 --content-dim 8 --trace
```

The system must not wake up the whole bank on every step. The **trace router**
(`lmf/core/dmf/trace_router.py`) scores cues against trace keys, applies each
trace’s threshold, and keeps only the best few matches. That is memory lookup,
not attention over every token pair.

You can route from the strongest token cue in the sentence (`max_token`) or from
one pooled sentence cue (`pooled`). Similarity can be dot product or cosine.
`lmf/core/dmf/sparsity.py` holds the small top-k helper and a simple report of
how much of the bank stayed inactive.

The **active region** (`lmf/core/dmf/active_region.py`) is the working set: which
trace ids were picked, their content, how active they are, and how hard the cue
is pushing them. Only those rows are gathered from the bank; the rest stay idle.

For **readable manual inspection** (recommended), use the pipeline report. It
prints tokens, a ranked table of which trace slots were selected, binding links,
and how activations changed after settling — not raw tensor dumps.

```powershell
py lmf/infra/scripts/inspect_pipeline.py text "Help me withdraw money from the bank"
```

Use `--routing-only` to stop after sparse routing. Use `--top-k 6 --num-traces 32`
to control how many slots wake up.

Lower-level CLIs still exist (`trace_router.py`, `loop.py`). Add `--numbers` on
`trace_router.py` if you want the old numeric output. Add `--trace` on any script
for debug logs on stderr.

## Field loop (Commits B1 and B2)

After the active region is built, the **field loop** is where the small active
memory set is allowed to change over a few settling steps. This is not the same
as running attention over the whole sentence. The loop applies pressures and
relation constraints, then updates trace strengths and basin pressures a little
at a time until the local field state settles.

Commit B1 wrapped this in one PyTorch module called `FieldLoop`. Before B1, field pieces could exist as plain Python objects sitting next to the model. They would run during`forward`, but their weights would not show up in `model.parameters()`, so training would skip them and checkpoints would silently drop them. B1 fixes that by attaching every field piece as a named child of `FieldLoop`.

Those children live in `lmf/core/field/loop.py` and are assigned on `self` so
their names are stable:

- `context_op` in `context_op.py` — turns the pooled cue into drive on active
  traces and basins, plus a small threshold shift. It does not mix token vectors.
- `binding_layer` in `binding.py` — scores soft edges between active traces using
  the context-sensitive pair scorer (C2). Returns a sparse binding state. Edges
  are constraints, not copied content.
- `binding_forces` in `binding_forces.py` — converts binding edges into forces on
  traces and basins. Again, no value mixing like attention.
- `interference_layer` in `interference.py` — adds binding-gated compatibility
  and conflict terms that feed the same force picture.
- `settling` in `settling.py` — applies the damped update: propose a new strength
  from the current state plus force, then blend old and new so the step stays
  bounded.

Each step of the loop runs context pressure, binding, binding forces,
interference, and settling. Only the active traces from routing are touched; the
rest of the trace bank stays idle. This is still an early Stage 1 version of the
field: binding and interference will grow richer in later plan commits, but the
wiring and training path are real.

You can inspect the full text-to-field path from the command line. The script
builds cues, routes into an active region, runs settling, and prints shapes.
Add `--trace` to see per-step logs on stderr.

```powershell
py lmf/core/field/loop.py text "Help bank!" --top-k 8 --cue-dim 16 --settling-steps 3 --trace
```

Commit B2 adds tests that make the B1 wiring hard to break by accident. The
problem B2 guards against is subtle: code can look correct in a forward pass while
still failing as a trainable model. If `binding_layer` were constructed but never
assigned to `self.binding_layer`, PyTorch would not list its weights under
`FieldLoop`. Training would appear to run, but binding would never learn. The same
failure happens if weights are not written into `state_dict` when you save a
checkpoint.

The B2 tests live in `tests/test_field_loop_registration.py`. They require that
`binding_layer`, `binding_forces`, and `interference_layer` each contribute at
least one entry to `named_parameters` and `state_dict` with the expected name
prefix. They also save the full `state_dict` to an in-memory checkpoint, reload
it into a fresh `FieldLoop`, and assert every tensor matches. A separate test
puts `FieldLoop` inside a parent `nn.Module`, because the real trainer will own
field code as `model.field_loop`, not as a loose global. That test checks
checkpoint keys like `field_loop.binding_forces.*` survive the round trip.

General field behavior tests (forward pass, gradients, CLI) stay in
`tests/test_field_loop.py`. Run the registration suite when you change how
modules are constructed or nested.

```powershell
py -m pytest tests/test_field_loop_registration.py -q
py -m pytest tests/test_field_loop.py -q
```

## Module diagnostics (Commit B3)

B1 made field modules trainable. B2 proved they are registered and checkpointed.
B3 adds **visibility during training**: after a backward pass, log whether each major
block actually received gradients and how many learnable weights it owns.

Without this, a silent wiring bug looks like “training runs” while binding or
interference never updates. Gradient norm zero (or missing) is an early warning.
Parameter counts help catch empty submodules or accidental duplication.

`lmf/training/module_diagnostics.py` collects a `ModuleDiagnosticReport` with the
plan’s fields: `trace_bank_grad_norm`, `binding_layer_grad_norm`,
`binding_forces_grad_norm`, `interference_grad_norm`, `basin_grad_norm`,
`decoder_grad_norm`, plus `field_loop_param_count`, `binding_param_count`, and
`interference_param_count`. Modules are found by dotted paths on the model
(for example `field_loop.binding_layer`), so the same helper works when
`FieldLoop` is nested under a parent `nn.Module`. Missing modules log as `n/a`.

`log_module_diagnostics(model, step=...)` writes one human-readable INFO line.
`format_module_diagnostics(report)` returns the same text for stdout. The CLI
runs a tiny forward/backward on text, then prints the report so you can verify
logging without a full trainer run.

```powershell
py lmf/training/module_diagnostics.py text "Help bank!" --top-k 8 --cue-dim 16 --trace
py -m pytest tests/test_module_diagnostics.py -q
```

## Cue provenance through routing (Commit C1)

Before C1, sparse routing only told you *which* trace slots woke up. It did
not record *why* — which input cue or token drove each selection. That is a
problem for binding supervision: to learn whether two active traces should
link, the system needs to know their source tokens, positions, and cue types.

C1 attaches provenance at routing time and copies it onto each active trace.
Five fields ride along on `RoutingResult` and `ActiveRegion`:

- `source_cue_id` — sequence index of the winning cue (`-1` for pooled mode)
- `source_token_id` — vocabulary id of that cue's token (`-1` when pooled)
- `source_span` — inclusive token span `[start, end]` for the source cue
- `cue_type` — integer code (`token`, `bos`, `eos`, `pooled`, etc.)
- `normalized_cue_ids` — binding-stable cue id (content and special tokens
  keep their vocab id; pooled uses `-1`)

In **max_token** mode the router already scores every token cue against every
trace key. C1 keeps the argmax position per trace before top-k selection, then
gathers token ids, cue types, and spans for the traces that survive. In
**pooled** mode the whole sentence cue drives all selections, so provenance
marks `cue_type=pooled` and span covers the active token range.

The cue encoder now puts `token_ids` on every `CuePacket`. The text helper
also stores `special_token_ids` from the vocabulary so cue types classify
correctly. Routing fails loudly if `token_ids` are missing — no silent empty
provenance.

Helpers live in `lmf/core/dmf/cue_provenance.py`. Human-readable reports and
the trace-router CLI show provenance columns (Src cue, Token id, Span, etc.).
Add `--trace` for routing logs that include a provenance preview on stderr.

Simple verification:

```powershell
py lmf/core/dmf/trace_router.py text "Help bank!" --top-k 5 --trace
py -m pytest tests/test_cue_provenance.py tests/test_trace_router.py -q
```

## Context-sensitive binding scorer (Commit C2)

Before C2, binding strength came from a simple trace-content similarity. That
could not tell apart cases where the same two words mean different things in
different sentences — "bank" near "money" vs "bank" near "river".

C2 adds a learned **pair scorer** (`lmf/core/field/binding_pair_scorer.py`) that
scores every active trace pair under every relation type. For each pair it sees:

- both trace contents and their element-wise product and absolute difference
- a **context summary** vector from the pooled cue (via `ContextPressure`)
- span distance and cue order from C1 provenance
- cue type embeddings for both source cues

The scorer outputs sigmoid strengths with shape `[batch, traces, traces,
relations]`. `BindingLayer` (`binding.py`) calls it, keeps only the top-k edges
per trace, and passes the sparse result to binding forces and interference.

`ContextOp` now writes `context_summary` onto `ContextPressure` so binding can
use sentence-level context without mixing token vectors like attention.

This is wired into the field loop and into the end-to-end training stack below.
Binding is no longer a standalone eval shortcut — it runs on the same active
region and provenance that routing produces during training.

```powershell
py -m pytest tests/test_binding_edge_evaluator.py::test_binding_pair_scorer_changes_with_span_distance -q
py -m pytest tests/test_field_loop.py -q
```

## Binding edge training and evaluation (Commits C2 + C3)

### What we're trying to teach the model

Read this sentence:

**"Help me withdraw money from the bank"**

Some word pairs belong together in meaning. Others do not.

- **withdraw** and **money** go together (you withdraw money).
- **help** and **bank** do not really go together in this sentence.

C3 is how we **train and check** whether the model's internal "these two ideas
are connected" score matches what we know is true — using the full pipeline, not
a brittle string-matching harness.

### The answer key file

`data/stage1/binding_edges.jsonl`

Each line is one sentence and a list of word pairs we already judged:

```json
{
  "text": "Help me withdraw money from the bank",
  "edges": [
    {"cue_a": "withdraw", "cue_b": "money", "label": 1},
    {"cue_a": "help", "cue_b": "bank", "label": 0}
  ]
}
```

- **cue_a** and **cue_b** are two words from that sentence.
- **label 1** means "these two should connect."
- **label 0** means "these two should not connect."

Both words must actually appear in the sentence. We do not use pairs like
`help` + `river` when the sentence never mentions river.

### End-to-end path (not a harness)

The old approach scored pairs with a standalone binding layer and matched words
by string after the fact. That was brittle: it could pass tests without the real
routing and provenance path learning anything.

The current stack (`lmf/training/binding_stack.py`) runs the full Stage 1 path:

1. **CueEncoder** — token ids to cue vectors
2. **TraceBank + TraceRouter** — sparse wake-up with C1 provenance
3. **ContextOp** — context summary for the pair scorer
4. **BindingLayer** — C2 context-sensitive pair strengths

Training examples are resolved by **sequence position**, not post-hoc string
labels. `resolve_example_edges()` maps `cue_a` / `cue_b` to token positions in
the encoded sequence, then matches active traces whose `source_cue_id` equals
those positions. Loss is binary cross-entropy on covered edges only.

`BindingStack.from_examples()` builds one shared vocabulary from all training
sentences so every example uses the same token ids.

### Training

```powershell
py lmf/training/train_binding_edges.py --steps 200 --trace
```

This runs Adam over all examples in the answer key file, backpropagating through
cue encoder, trace bank, router, context op, and binding layer. After training
it prints the same eval table as the evaluator CLI.

Untrained scores are expected to be poor. After a few hundred steps you should
see loss drop and accuracy rise on covered pairs — proof the full stack is
learning, not just a pattern-matching wrapper.

### Reading the printed table

Run:

```powershell
py lmf/training/binding_edge_evaluator.py text "Help me withdraw money from the bank" --top-k 16
```

You might see:

```text
cue_a      cue_b      label   mass   topk   pos_a  pos_b  ok
withdraw   money          1  0.113  0.722  3      4      no
money      bank           1  0.657  0.657  4      7     yes
help       bank           0  0.066  0.066  1      7     yes
```

Read it like a report card for each word pair:

- **cue_a / cue_b** — the two words being tested.
- **label** — what we expect: 1 = should connect, 0 = should not.
- **mass** — binding strength from the C2 scorer (0 = not at all, 1 = very strong).
- **pos_a / pos_b** — sequence positions used to match traces (provenance-based).
- **ok** —
  - **yes** = model agrees with the answer key.
  - **no** = model disagrees.
  - **miss** = one or both words never woke a trace. Try larger `--top-k`.

The summary line (loss, acc, pos_mass, neg_mass, pos_cov, neg_cov) aggregates
across all pairs in the run.

### What this does not solve yet

This stack trains binding on labeled pairs in context. It does **not** yet handle
double negatives, compositional negation, or interference between competing
meanings — those need richer data and later field commits. C3 gives an honest
end-to-end ruler; C2 gives the learnable scorer that ruler measures.

### Commands

```powershell
# Train on the full answer key
py lmf/training/train_binding_edges.py --steps 200 --trace

# Evaluate every sentence in the answer key file
py lmf/training/binding_edge_evaluator.py

# Evaluate one sentence
py lmf/training/binding_edge_evaluator.py text "Help me withdraw money from the bank" --top-k 16

# Automated tests (scorer, stack, evaluator, overfit)
py -m pytest tests/test_binding_edge_evaluator.py tests/test_binding_overfit.py -q
```

Code:

- `lmf/training/binding_stack.py` — full trainable forward + loss
- `lmf/training/binding_edge_evaluator.py` — eval via the same stack
- `lmf/training/train_binding_edges.py` — training CLI
- `lmf/training/binding_edges.py` — answer key loader + position resolution
- `lmf/core/field/binding_pair_scorer.py` — C2 pair scorer

## Binding overfit sanity (Commit C5)

Before trusting binding training on real sentences, we need proof the stack can
actually learn labeled edges end-to-end — not just print tables on random weights.

C5 runs a **controlled overfit experiment** on a tiny synthetic dataset (two
short sentences, present-present positive and negative pairs). It trains the full
binding stack twice:

1. **Full** — binding layer learns along with cues, trace bank, router, and context.
2. **No binding** — binding layer stays frozen at init; everything else still trains.

After training, four gates must pass:

- `binding_edge_accuracy >= 0.90`
- `positive_binding_mass_mean >= 0.70`
- `negative_binding_mass_mean <= 0.20`
- full accuracy **beats** no-binding accuracy

If full passes and no-binding fails, binding supervision is real — the learned
pair scorer is doing work, not accidental routing noise.

An adversarial test also checks that an **untrained** stack fails the same gates,
so the test cannot pass by default.

Run manually:

```powershell
py lmf/training/binding_overfit.py --trace
py -m pytest tests/test_binding_overfit.py -q
```

Add `--trace` to see step-by-step training logs on stderr. Exit code 1 means a
gate failed and the printed report lists which one.

Code: `lmf/training/binding_overfit.py`, tests in `tests/test_binding_overfit.py`.
