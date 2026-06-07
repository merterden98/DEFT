# DEFT

DEFT classifies enzymes by combining a protein language model with structural similarity search. Given a protein structure it assigns an EC number, then uses structural alignments to a labelled database to refine and filter those assignments.

## Language

**EC number**:
An Enzyme Commission number — the hierarchical enzyme classification (e.g. `3.4.11.2`). DEFT predicts to the first two levels by default (`ec_level=2`).
_Avoid_: class, label, category.

**EC prediction**:
Running a structure (or chain) through the model to produce a predicted EC number per ID. The shared core behind `predict`, `search`, `annotate`, and `evaluate`. Exposed as `predict_ec(model, tokenizer, dataset) -> [(id, ec)]`.
_Avoid_: inference, classification, scoring.

**3Di**:
Foldseek's structural alphabet — a per-residue token sequence encoding local structure. DEFT interleaves the 3Di string with the amino-acid sequence as model input.
_Avoid_: structure tokens, struct sequence.

**Alignment**:
A foldseek structural match between a query and a database structure, read from a `.m8` file. The downstream filter keeps alignments whose EC prediction agrees with the target's EC.
_Avoid_: hit, match, m8 row.
