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
