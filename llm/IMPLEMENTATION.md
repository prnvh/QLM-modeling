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
