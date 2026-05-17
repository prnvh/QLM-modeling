# QLM-modeling

LMF (lucid memory field) research code. The repo is currently in Stage 1: a
small, inspectable tokenizer/vocabulary path plus shared tensor contracts and
the planned package structure for the rest of the model.

## Current Structure

- `lmf/core/input/tokenizer.py` - regex tokenizer, vocabulary, encode/decode, and optional human-readable trace logging.
- `lmf/infra/scripts/build_vocab.py` - CLI for building a vocabulary JSON from local text files.
- `lmf/infra/scripts/inspect_tokenizer.py` - CLI for controlling tokenizer input/options and viewing tokens, ids, decoded text, and JSON reports.
- `lmf/infra/config/stage1_local.yaml` - local Stage 1 smoke/dev config.
- `lmf/core/state/types.py` - shared dataclasses passed through the LMF stack.
- `tests/` - adversarial and smoke tests for the implemented tokenizer/vocab path.
- `llm/` - local planning and implementation notes.

Root-level `configs/` and `scripts/` are intentionally not used; project-owned
configs and scripts live inside `lmf/infra/`.

## Implemented Usage

Quickly inspect one text input:

```powershell
py lmf/core/input/tokenizer.py text "Help bank!"
```

Build a vocabulary from a text file:

```powershell
py lmf/infra/scripts/build_vocab.py README.md --output artifacts/vocab.json --preview-tokens 20
```

Write a human-readable build log:

```powershell
py lmf/infra/scripts/build_vocab.py README.md --output artifacts/vocab.json --log-file artifacts/build_vocab.log
```

Inspect an exact tokenizer input and output:

```powershell
py lmf/infra/scripts/inspect_tokenizer.py --text "Help me withdraw money from bank." --build-vocab-from-input --add-bos --add-eos --skip-special-decode
```

Write the inspection output as JSON and a separate trace log:

```powershell
py lmf/infra/scripts/inspect_tokenizer.py --text "Help bank!" --build-vocab-from-input --format json --output artifacts/tokenizer_report.json --log-file artifacts/tokenizer_trace.log
Get-Content artifacts/tokenizer_report.json
Get-Content artifacts/tokenizer_trace.log
```

Run tests:

```powershell
py -m pytest -q
```

## Planned Pipeline

Tokenizer/chunker: turns text into small units.

Cue encoder: turns units into cue signals.

TraceBank/DMF: stores learnable memory traces.

Sparse router: picks the small active trace region.

Binding: learns which active traces influence which others.

ContextOp: applies prompt-level pressure to the active field.

Interference: makes compatible traces reinforce and conflicting traces weaken.

Basins: stable learned attractor patterns formed from active traces.

Loop: repeats updates until the field stabilizes.

Lucidity: estimates readiness, ambiguity, contradiction, instability.

Decoder: turns the final relational state into text/output.
