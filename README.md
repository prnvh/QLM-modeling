# QLM-modeling



----

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