import gc
import sys
from pathlib import Path

import spacy
from conllu import parse_incr
from spacy.tokens import DocBin
from spacy.training import Example
from tqdm import tqdm

spacy.require_gpu()

# Models to compare (pass names on the command line to pick some); missing model directories are skipped
MODELS = {
    "fr_dep_news_trf": "fr_dep_news_trf",
    "historic-french-model": "historic-french-model",
    "v7 release": "training_space/release_v7/stage/historic-french-model",
}
DEV_DATA_PATH = "training_space/data/test.spacy"
VALIDATED_TEST_PATH = "training_space/data/validated_test.spacy"
VALIDATED_TEST_2_PATH = "training_space/data/validated_test_2.spacy"
SEQUOIA_TEST_PATH = "fr_sequoia-ud-test.conllu"

# Define century test paths
CENTURY_TEST_PATHS = {
    "1500-1600": "training_space/data/1500-1600_test.spacy",
    "1600-1700": "training_space/data/1600-1700_test.spacy",
    "1700-1800": "training_space/data/1700-1800_test.spacy",
    "1800-1900": "training_space/data/1800-1900_test.spacy",
    "1900-2000": "training_space/data/1900-2000_test.spacy"
}

def load_evaluation_data(vocab, path):
    """Load evaluation data from a .spacy file."""
    doc_bin = DocBin().from_disk(path)
    docs = list(doc_bin.get_docs(vocab))
    print(len(docs), "docs loaded")
    return docs

def surface_tokens(sentence):
    """Tokens as they appear in the text: a contraction such as "du" is listed as a
    multiword token followed by its words ("de", "le"); keep only the former."""
    tokens = []
    covered = set()
    for token in sentence:
        if isinstance(token["id"], tuple):
            if token["id"][1] == "-":
                covered.update(range(token["id"][0], token["id"][2] + 1))
                tokens.append(token)
            continue
        if token["id"] not in covered:
            tokens.append(token)
    return tokens

def load_conllu_data(nlp_model, path):
    """Load CoNLL-U data and convert it to spaCy Example objects."""
    examples = []

    with open(path, "r", encoding="utf-8") as file:
        sentences = list(parse_incr(file))

    for sentence in tqdm(sentences, desc="Converting Sequoia sentences to spaCy Examples", unit="sent", leave=False):
        tokens = surface_tokens(sentence)
        words = [token["form"] for token in tokens]
        spaces = []
        for i, token in enumerate(tokens):
            has_space = True
            if "misc" in token and token["misc"] is not None:
                if "SpaceAfter" in token["misc"] and token["misc"]["SpaceAfter"] == "No":
                    has_space = False
            # For the last token, never add space
            if i == len(tokens) - 1:
                has_space = False
            spaces.append(has_space)

        doc = spacy.tokens.Doc(nlp_model.vocab, words=words, spaces=spaces)

        # Multiword tokens have no annotation of their own and are not scored
        for i, token in enumerate(tokens):
            if "upos" in token and token["upos"] and token["upos"] != "_":
                doc[i].pos_ = token["upos"]

            if "lemma" in token and token["lemma"] and token["lemma"] != "_":
                doc[i].lemma_ = token["lemma"]

        predicted_doc = nlp_model.make_doc(doc.text)
        example = Example(predicted_doc, doc)
        examples.append(example)

    return examples

def evaluate_on_docs(nlp, docs):
    """Evaluate a model directly on Doc objects."""
    examples = []
    for doc in tqdm(docs, desc="Evaluating model on test data", unit="doc", leave=False):
        pred = nlp(doc.text)
        examples.append(Example(pred, doc))
    return nlp.evaluate(examples)

def print_scores_table(results):
    """Display evaluation results as a formatted table with borders."""
    rows = []
    for dataset, model_scores in results.items():
        for model, scores in model_scores.items():
            # Format scores as percentages
            pos = f"{scores.get('pos_acc', 0)*100:.2f}%" if scores.get('pos_acc') is not None else "N/A"
            tag = f"{scores.get('tag_acc', 0)*100:.2f}%" if scores.get('tag_acc') is not None else "N/A"
            lemma = f"{scores.get('lemma_acc', 0)*100:.2f}%" if scores.get('lemma_acc') is not None else "N/A"

            rows.append([dataset, model, pos, tag, lemma])

    # Define column headers and widths
    headers = ["Dataset", "Model", "POS", "TAG", "Lemma"]
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(5)]

    # Print table with borders
    print("\n=== MODEL EVALUATION RESULTS ===")
    print("(fr_dep_news_trf uses a different TAG set, so its TAG scores are not comparable)")

    # Top border
    print("+" + "+".join("-" * (w + 2) for w in col_widths) + "+")

    # Header row
    header_row = "| " + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers)) + " |"
    print(header_row)

    # Header-data separator
    print("+" + "+".join("=" * (w + 2) for w in col_widths) + "+")

    # Data rows
    for row in rows:
        data_row = "| " + " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row)) + " |"
        print(data_row)
        print("+" + "+".join("-" * (w + 2) for w in col_widths) + "+")

    print()

def main():
    # Load evaluation data
    print("Loading evaluation data...")
    vocab = spacy.blank("fr").vocab
    datasets = {"Hand-validated": VALIDATED_TEST_PATH, "Gold set 2": VALIDATED_TEST_2_PATH, "1500-2000": DEV_DATA_PATH, **CENTURY_TEST_PATHS}
    datasets = {name: path for name, path in datasets.items() if Path(path).exists()}
    docs = {name: load_evaluation_data(vocab, path) for name, path in datasets.items()}

    # Store all results in a dictionary for table display
    results = {"Sequoia": {}, **{name: {} for name in datasets}}

    selected = sys.argv[1:] or list(MODELS)
    for model_name, model_path in MODELS.items():
        if model_name not in selected:
            continue
        if model_name != model_path and not Path(model_path).exists():
            print(f"Warning: {model_path} not found, skipping {model_name}")
            continue
        print(f"\nEvaluating {model_name} ({model_path})...")
        nlp = spacy.load(model_path)
        results["Sequoia"][model_name] = nlp.evaluate(load_conllu_data(nlp, SEQUOIA_TEST_PATH))
        for name, dataset_docs in docs.items():
            results[name][model_name] = evaluate_on_docs(nlp, dataset_docs)
        del nlp
        gc.collect()

    # Display results as a table
    print_scores_table(results)


if __name__ == "__main__":
    main()
