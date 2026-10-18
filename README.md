# spacy-historic-models
spaCy models for texts of the 16th to 20th centuries. Both do part-of-speech tagging (coarse `pos_` and fine-grained `tag_`) and lemmatization (`lemma_`).

| Model | Language | Training data | Download |
|-------|----------|---------------|----------|
| [historic-french-model](fr/) | French | 50,000 sentences from ARTFL-Frantext | [fr-v7.0](https://github.com/ARTFL-Project/spacy-historic-models/releases/tag/fr-v7.0) |
| [historic-english-model](en/) | British English | 65,000 sentences from EEBO-TCP, ECCO-TCP and Project Gutenberg | [en-v2.0](https://github.com/ARTFL-Project/spacy-historic-models/releases/tag/en-v2.0) |

Each model's README gives its scores, installation and rebuilding steps.

## Layout

- `fr/`, `en/`: one directory per model, with its training data, gold sets, tokenizer, annotation guidelines and scripts. Run a model's scripts from its directory.
- `shared/package_model.py`: makes a trained pipeline load without this repository's code.

## Releases

Release tags start with the model's language (`fr-v7.0`, `en-v1.0`). The tags without a prefix, `v1.0.0` to `v6.0`, are French releases from before the English model was added.

This repository was previously named `spacy-historic-french-model`; old links redirect here.
