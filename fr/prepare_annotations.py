#!/usr/bin/env python3
"""
Rebuild the corrected annotation files from the original GPT-4 annotations.

The originals are read from git (ORIGINAL_COMMIT) so that every run starts from the
same untouched data. Two passes are applied to each sentence:

1. Retokenization: tokens that the spaCy French tokenizer splits (elisions such as
   "c'est" or "qu'il", inverted clitics such as "dit-il") are split into the same
   pieces, keeping the surface form of each piece. Fixed expressions whose lemma is
   a single unit ("d'abord", "peut-être", "aujourd'hui") are left whole.
2. Label fixes. Each rule was scored against the hand-validated sentences in
   validated_test_data/validated_jsonl; rules that made the data worse are disabled.

Writes training_data/annotations_*.jsonl and test_data/annotations_*_test_set.jsonl.
"""

import json
import re
import subprocess
from collections import Counter, defaultdict

import spacy
from spacy.tokens import Doc
from spacy.training import Alignment

from historic_tokenizer import make_nlp

ORIGINAL_COMMIT = "6df8d70"
PERIODS = ["1500-1600", "1600-1700", "1700-1800", "1800-1900", "1900-2000"]

# Contracted preposition + article: lemmatize as the preposition, like fr_dep_news_trf
CONTRACTED_LEMMAS = {
    "de le": "de",
    "de les": "de",
    "de+le": "de",
    "de+les": "de",
    "à le": "à",
    "à les": "à",
    "à+le": "à",
    "à+les": "à",
    "en les": "en",
}

# Rules tried and disabled after scoring them against the hand-validated sentences
# (fixes / breaks):
#   avoir VERB -> AUX everywhere:           main-verb avoir is VERB ("j'ai faim")
#   que SCONJ -> PRON before a pronoun:     15 / 24  (mostly real conjunctions)
#   pas NOUN -> ADV:                        all real nouns ("un faux pas", "à pas lents")
#   tout in "tout ce que" -> PRON:          1 / 3
#   P+D tag -> ADP:                         3 / 3    (GPT-4's P+D tag is unreliable)
AUX_ONLY_WHEN_AUX_DEP = True

# Pieces other than the head of a split token must be function words
CLOSED_CLASS = {"PRON", "DET", "ADP", "SCONJ", "CCONJ", "AUX", "PART"}


def read_original(path):
    output = subprocess.run(
        ["git", "show", f"{ORIGINAL_COMMIT}:{path}"], capture_output=True, text=True, check=True
    ).stdout
    return [json.loads(line) for line in output.splitlines()]


def should_have_space_after(text: str, next_text: str = None) -> bool:
    """Determine if a token should have a space after it in French."""
    if next_text is None:
        return False

    # No space before these characters
    if next_text in [",", ".", ")", "]", "}", "»", ";", ":", "!", "?"]:
        return False

    # No space after these characters
    if text in ["(", "[", "{", "«"]:
        return False

    # No space around apostrophes
    if text.endswith(("'", "’")) or next_text.startswith(("'", "’")):
        return False

    # No space around hyphens
    if text.endswith("-") or next_text.startswith("-"):
        return False

    # Default: include space
    return True


def token_spaces(sentence_text, words):
    """Whether each token is followed by a space: read from the sentence text when the
    tokens match it, otherwise guessed with French spacing rules."""
    spaces = []
    position = 0
    for word in words:
        while position < len(sentence_text) and sentence_text[position].isspace():
            position += 1
        if not sentence_text.startswith(word, position):
            break
        position += len(word)
        spaces.append(position < len(sentence_text) and sentence_text[position].isspace())
    else:
        return spaces
    return [should_have_space_after(word, words[i + 1] if i + 1 < len(words) else None) for i, word in enumerate(words)]


def norm(text):
    return text.lower().replace("’", "'")


def build_label_table(sentences):
    """Label counts of each token form, used to label the pieces of split tokens."""
    by_text_lemma = defaultdict(Counter)
    by_text = defaultdict(Counter)
    for sentence in sentences:
        for token in sentence["annotations"]["tokens"]:
            if "text" not in token:
                continue
            text = norm(token["text"])
            by_text_lemma[(text, token["lemma"].lower())][(token["pos"], token["tag"], token["dep"])] += 1
            by_text[text][(token["pos"], token["tag"], token["lemma"], token["dep"])] += 1
    return by_text_lemma, by_text


def lookup_keys(piece):
    # "-t" is usually annotated as "-t-", and "-il" as "il"
    text = norm(piece)
    return (text, text + "-", text.strip("-"))


def most_common(counts, prefer_pronoun):
    if prefer_pronoun:
        pronouns = Counter({labels: n for labels, n in counts.items() if labels[0] == "PRON"})
        counts = pronouns or counts
    return counts.most_common(1)[0][0]


def prefers_pronoun(index, piece):
    # Clitics after the verb or after an elided pronoun: "dites-le", "s'en", "n'y"
    return index > 0 and ((piece.startswith("-") and len(piece) > 1) or norm(piece) in ("en", "y"))


def piece_labels(piece, table, lemma=None, prefer_pronoun=False):
    by_text_lemma, by_text = table
    for key in lookup_keys(piece):
        if lemma is not None and (key, lemma.lower()) in by_text_lemma:
            pos, tag, dep = most_common(by_text_lemma[(key, lemma.lower())], prefer_pronoun)
            return {"pos": pos, "tag": tag, "lemma": lemma, "dep": dep}
        if key in by_text:
            pos, tag, usual_lemma, dep = most_common(by_text[key], prefer_pronoun)
            return {"pos": pos, "tag": tag, "lemma": lemma if lemma is not None else usual_lemma, "dep": dep}
    return None


def label_share(piece, lemma, token, table):
    """How often this piece, with this lemma, carries the token's POS and tag."""
    by_text_lemma, _ = table
    for key in lookup_keys(piece):
        counts = by_text_lemma.get((key, lemma.lower()))
        if counts:
            matching = sum(n for (pos, tag, _), n in counts.items() if (pos, tag) == (token["pos"], token["tag"]))
            return matching / sum(counts.values())
    return 0.0


def split_token(token, pieces, table):
    """Return the tokens replacing `token`, or None to keep it whole."""
    lemma_parts = [part for part in re.split(r"\s+|\+", token["lemma"].strip()) if part]

    if len(lemma_parts) == len(pieces):
        # The lemma names each word ("ce être"): give each piece its lemma. The token's
        # own labels go to the piece that most often carries them ("qu'" in a relative
        # "qu'elle"); the other pieces get their usual labels.
        shares = [label_share(piece, lemma, token, table) for piece, lemma in zip(pieces, lemma_parts)]
        head = shares.index(max(shares)) if max(shares) > 0 else None
        new_tokens = []
        for i, (piece, lemma) in enumerate(zip(pieces, lemma_parts)):
            guess = piece_labels(piece, table, lemma, prefers_pronoun(i, piece))
            if i == head or guess is None:
                new_tokens.append({**token, "text": piece, "lemma": lemma})
            else:
                new_tokens.append({**token, **guess, "text": piece})
        return new_tokens, "lemma_per_piece"

    # The lemma is that of one piece ("dit-il" -> "dire"): that piece keeps the token's
    # labels, as long as the other pieces are function words.
    guesses = [piece_labels(piece, table, prefer_pronoun=prefers_pronoun(i, piece)) for i, piece in enumerate(pieces)]
    heads = [i for i, guess in enumerate(guesses) if guess and guess["lemma"].lower() == token["lemma"].lower()]
    if len(heads) != 1:
        # a rare verb followed by hyphen clitics ("repliqua-t'il"): the verb is the head
        heads = [i for i, piece in enumerate(pieces) if not piece.startswith(("-", "–")) and not (i and pieces[i - 1].lower().startswith(("-t'", "-t’")))]
        if len(heads) != 1 or heads[0] != 0 or len(pieces) < 2:
            return None, None
    head = heads[0]
    if any(guess is None or guess["pos"] not in CLOSED_CLASS for i, guess in enumerate(guesses) if i != head):
        return None, None
    new_tokens = [
        {**token, "text": piece} if i == head else {**token, **guess, "text": piece}
        for i, (piece, guess) in enumerate(zip(pieces, guesses))
    ]
    return new_tokens, "head_piece"


def merge_tokens(parts, text, table):
    """One token for annotation tokens the tokenizer keeps together ("-" + "tu", "moi" + "-" + "même")."""
    words = [t for t in parts if any(c.isalpha() for c in t["text"])]
    if len(words) == 1:
        return {**words[0], "text": text}
    labels = piece_labels(text, table)
    return {**(words or parts)[0], **(labels or {}), "text": text}


def make_tokenizer():
    return make_nlp().tokenizer


def retokenize(sentence, tokenizer, table, stats):
    """Split and merge tokens the way the tokenizer does in the sentence, as during training."""
    tokens = sentence["annotations"]["tokens"]
    words = [t["text"] for t in tokens]
    doc = Doc(tokenizer.vocab, words=words, spaces=token_spaces(sentence["text"], words))
    predicted = [t.text for t in tokenizer(doc.text)]
    alignment = Alignment.from_strings([t.text for t in doc], predicted)
    merges = {}
    for j in range(len(predicted)):
        sources = [int(i) for i in alignment.y2x[j]]
        if len(sources) > 1 and all([int(k) for k in alignment.x2y[i]] == [j] for i in sources):
            merges[sources[0]] = (sources, predicted[j])
    new_tokens, merged = [], set()
    for i, token in enumerate(tokens):
        if i in merged:
            continue
        if i in merges:
            sources, text = merges[i]
            new_tokens.append(merge_tokens([tokens[k] for k in sources], text, table))
            merged.update(sources)
            stats["merged"] += 1
            stats[f"merged: {norm(text)}"] += 1
            continue
        targets = [int(j) for j in alignment.x2y[i]]
        if len(targets) > 1 and all(list(alignment.y2x[j]) == [i] for j in targets):
            pieces = [predicted[j] for j in targets]
            replacement, how = split_token(token, pieces, table)
            if replacement is not None:
                new_tokens.extend(replacement)
                stats[f"split ({how})"] += 1
                stats[f"split: {norm(token['text'])}"] += 1
                continue
            stats["kept whole"] += 1
        new_tokens.append(token)
    return new_tokens


def is_aux_dep(dep):
    dep = dep.lower()
    return dep.startswith("aux") or dep == "cop"


def fix_labels(token, stats):
    token = dict(token)
    if token["lemma"] in CONTRACTED_LEMMAS:
        token["lemma"] = CONTRACTED_LEMMAS[token["lemma"]]
        stats["lemma: contracted article -> preposition"] += 1
    if token["lemma"] == "pas" and token["pos"] == "PART":
        token["pos"] = "ADV"
        stats["pos: pas PART -> ADV"] += 1
    if token["lemma"] in ("être", "avoir") and token["pos"] == "VERB":
        if is_aux_dep(token["dep"]) or not AUX_ONLY_WHEN_AUX_DEP:
            token["pos"] = "AUX"
            stats["pos: être/avoir VERB -> AUX (aux/cop dep)"] += 1
    return token


# Rules that look at neighbouring tokens. Each was scored against the hand-validated
# sentences and spot-checked on random training sentences before being added.
NEGATION = {"pas", "point", "plus", "jamais", "rien", "guère", "guere", "guères", "gueres", "personne", "aucun", "aucune",
            "nul", "nulle", "ni", "ny", "goutte", "mie"}
# "ne ... tant/autant/plus ... que" is a comparison, not restrictive
COMPARATIVE = {"tant", "autant", "aussi", "aussy", "si", "moins", "mieux", "pis", "tel", "telle", "tels", "telles",
               "même", "mesme", "autre", "autres", "aultre", "aultres", "autrement", "aultrement", "plutôt", "plustost",
               "davantage"}
CLAUSE_BOUNDARY = {"que", "qu'", "qui", "et", "ou", "mais", "car", "si", "s'", "comme", "quand",
                   ",", ";", ":", ".", "?", "!", "«", "»", '"'}
OBJECT_START = {"du", "des", "de", "d'", "que", "qu'"}
PARTITIVE_BEFORE = {"un", "une", "l'un", "l'une", "aucun", "aucune", "chacun", "chacune", "plusieurs",
                    "beaucoup", "peu", "plus", "moins", "tant", "trop", "assez", "nombre", "plupart"}
REFLEXIVE_LEMMAS = {"me": "me", "m'": "me", "te": "te", "t'": "te", "nous": "nous", "vous": "vous"}
SUBJECT_PRONOUNS = {"il", "ils", "elle", "elles", "je", "j'", "tu", "nous", "vous", "on", "l'on"}
# "je ne puis", "je ne sais" without "pas": negation, not "ne ... que"
LITERARY_NE = {"pouvoir", "savoir", "oser", "cesser", "bouger"}
# future, conditional and imperfect endings are never subjunctive
INDICATIVE_ENDINGS = ("rai", "ray", "ras", "ra", "rons", "rez", "ront", "rais", "rois", "rait", "roit", "raient", "roient",
                      "ait", "oit", "aient", "oient", "ais", "ois")
SUBJUNCTIVE_FORMS = {"ait", "aient", "soit", "soient", "sois", "aies", "ayt", "voie", "croie"}
CLITICS = {"ne", "n'", "me", "m'", "te", "t'", "se", "s'", "lui", "luy", "leur", "nous", "vous", "en", "y", "le", "la", "les", "l'"}
TITLES = {"roy": "roi", "roi": "roi", "royne": "reine", "reine": "reine", "madame": "madame", "monsieur": "monsieur",
          "mademoiselle": "mademoiselle", "messieurs": "monsieur", "mesdames": "madame", "sire": "sire", "seigneur": "seigneur",
          "monseigneur": "monseigneur", "duc": "duc", "duchesse": "duchesse", "comte": "comte", "comtesse": "comtesse",
          "prince": "prince", "princesse": "princesse", "marquis": "marquis", "marquise": "marquise", "baron": "baron",
          "pape": "pape", "empereur": "empereur", "dame": "dame", "maistre": "maître", "maître": "maître", "docteur": "docteur",
          "abbé": "abbé", "capitaine": "capitaine", "cardinal": "cardinal", "évêque": "évêque"}
# closed-class lemma variants -> the majority lemma in the training data
LEMMA_GROUPS = {
    "DET": {"le": {"le", "la", "les", "l'"}, "un": {"un", "une", "vne"}, "ce": {"ce", "cet", "cette", "ceste", "cest", "ces"},
            "mon": {"mon", "ma", "mes"}, "ton": {"ton", "ta", "tes"}, "son": {"son", "sa", "ses"},
            "notre": {"notre", "nos", "nostre"}, "votre": {"votre", "vos", "vostre"}, "leur": {"leur", "leurs"},
            "tout": {"tout", "toute", "toutes", "tous"}, "aucun": {"aucun", "aucune"}, "nul": {"nul", "nulle"},
            "quel": {"quel", "quelle", "quels", "quelles"}},
    "PRON": {"le": {"le", "la", "les", "l'"}, "il": {"il", "ils"}, "elle": {"elle", "elles"},
             "celui": {"celui", "ceux", "celle", "celles"}, "un": {"un", "une"}, "tout": {"tout", "tous", "toute", "toutes"}},
}
PRONOMINAL_BEFORE = {"l'", "les", "un", "une", "des", "aux", "quelque", "quelques", "nul", "aucun", "aucune", "cet", "cette", "ces"}
INFINITIVE_ENDINGS = ("er", "ir", "re", "oir", "ïr")
# lemmas are written without ligatures, the long s or abbreviation marks, so that they can be searched as typed
PLAIN_LEMMA = str.maketrans({"œ": "oe", "Œ": "Oe", "æ": "ae", "Æ": "Ae", "ſ": "s", "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
                             "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st", "ꝑ": "per", "ꝓ": "pro"})
# past participle form -> infinitive, infinitive -> its participle forms and masculine singular form,
# filled from the training data by build_participle_lemmas()
PARTICIPLE_LEMMAS = {}
PARTICIPLE_FORMS = {}
MASCULINE_PARTICIPLE = {}


def build_participle_lemmas(sentences):
    counts = defaultdict(Counter)
    for sentence in sentences:
        for token in sentence["annotations"]["tokens"]:
            if (token.get("tag") == "VPP" and token.get("pos") in ("VERB", "AUX") and "text" in token
                    and token["lemma"].endswith(INFINITIVE_ENDINGS)):
                counts[norm(token["text"])][token["lemma"]] += 1
    PARTICIPLE_LEMMAS.clear()
    PARTICIPLE_FORMS.clear()
    MASCULINE_PARTICIPLE.clear()
    forms_by_lemma = defaultdict(Counter)
    for form, lemmas in counts.items():
        lemma, n = lemmas.most_common(1)[0]
        if n >= 2 and n / sum(lemmas.values()) >= 0.8:
            PARTICIPLE_LEMMAS[form] = lemma
            forms_by_lemma[lemma][form] += n
    for lemma, forms in forms_by_lemma.items():
        PARTICIPLE_FORMS[lemma] = set(forms)
        if lemma.endswith("er"):
            MASCULINE_PARTICIPLE[lemma] = lemma[:-2] + "é"
        else:
            # the masculine singular is the shortest form: mis/mise, fait/faite, couvert/couverte
            MASCULINE_PARTICIPLE[lemma] = min(forms, key=lambda f: (len(f), -forms[f]))


def word(token):
    return norm(token["text"])


def neighbour(tokens, i, offset):
    j = i + offset
    return tokens[j] if 0 <= j < len(tokens) else None


def followed_by_participle(tokens, i, window=6):
    for j in range(i + 1, min(len(tokens), i + 1 + window)):
        token = tokens[j]
        if token["tag"] == "VPP":
            return True
        if token["pos"] not in ("ADV", "PART", "PRON") and word(token) not in ("-t", "-t-"):
            return False
    return False


def etre_aux(tokens, i):
    if tokens[i]["lemma"] == "être" and tokens[i]["pos"] == "VERB":
        return {"pos": "AUX"}


def avoir_aux(tokens, i):
    if tokens[i]["lemma"] == "avoir" and tokens[i]["pos"] == "VERB" and followed_by_participle(tokens, i):
        return {"pos": "AUX"}


def avoir_main_verb(tokens, i):
    # possession: the next word (after adverbs and clitics) starts an object ("j'ai faim", "il y a des")
    if tokens[i]["lemma"] != "avoir" or tokens[i]["pos"] != "AUX":
        return None
    for j in range(i + 1, min(len(tokens), i + 6)):
        token = tokens[j]
        if token["pos"] in ("ADV", "PART", "PRON") or word(token) in ("-t", "-t-"):
            continue
        if (token["pos"] in ("DET", "NOUN", "PROPN", "NUM") or word(token) in OBJECT_START
                or (token["pos"] == "ADJ" and token["tag"] != "VPP")):
            # "a seule eu", "ai de plus envoyé": a participle shortly after still means auxiliary
            for k in range(j + 1, min(len(tokens), j + 4)):
                if tokens[k]["pos"] == "PUNCT":
                    break
                if tokens[k]["tag"] == "VPP":
                    return None
            return {"pos": "VERB"}
        return None
    return None


def modal_verb(tokens, i):
    if tokens[i]["pos"] == "AUX" and tokens[i]["lemma"] not in ("être", "avoir"):
        return {"pos": "VERB"}


def elided_si(tokens, i):
    following = neighbour(tokens, i, 1)
    if word(tokens[i]) == "s'" and following is not None and word(following) in ("il", "ils"):
        return {"pos": "SCONJ", "tag": "CS", "lemma": "si"}


def restrictive_que(tokens, i):
    # "il n'est qu'un effet", "on ne voyait que coiffes"
    if word(tokens[i]) not in ("que", "qu'") or tokens[i]["pos"] != "SCONJ":
        return None
    j = i - 1
    while j >= 0 and tokens[j]["pos"] in ("ADV", "PART", "PRON") and word(tokens[j]) not in ("ne", "n'"):
        j -= 1
    if j < 0 or tokens[j]["pos"] not in ("VERB", "AUX"):
        return None
    saw_verb = False
    for j in range(i - 1, max(-1, i - 9), -1):
        w = word(tokens[j])
        if w in NEGATION or w in COMPARATIVE or w in CLAUSE_BOUNDARY:
            return None
        if tokens[j]["pos"] in ("VERB", "AUX"):
            saw_verb = True
        if w in ("ne", "n'"):
            return {"pos": "ADV", "tag": "ADV"} if saw_verb else None
    return None


def relative_ce_que(tokens, i):
    # after a preposition ("à ce que", "de ce que") it is often a complementizer: leave those
    previous, before = neighbour(tokens, i, -1), neighbour(tokens, i, -2)
    if word(tokens[i]) in ("que", "qu'") and tokens[i]["pos"] == "SCONJ" and previous is not None and word(previous) == "ce":
        if before is not None and before["pos"] == "ADP":
            return None
        return {"pos": "PRON", "tag": "PROREL"}


def ce_pronoun(tokens, i):
    following = neighbour(tokens, i, 1)
    if (word(tokens[i]) == "ce" and tokens[i]["pos"] == "DET" and following is not None
            and word(following) in ("que", "qu'", "qui", "dont", "quoi")):
        return {"pos": "PRON", "tag": "PRO"}


def object_clitic(tokens, i):
    following = neighbour(tokens, i, 1)
    if (word(tokens[i]) in ("le", "la", "les", "l'") and tokens[i]["pos"] == "DET"
            and following is not None and following["pos"] in ("VERB", "AUX")):
        return {"pos": "PRON", "tag": "CLO", "lemma": "le"}


def un_article(tokens, i):
    previous, following = neighbour(tokens, i, -1), neighbour(tokens, i, 1)
    if word(tokens[i]) not in ("un", "une") or tokens[i]["pos"] != "NUM":
        return None
    # counting: "un, deux, trois"
    if (following is not None and (following["pos"] == "NUM" or word(following) == ",")) or (
            previous is not None and previous["pos"] == "NUM"):
        return None
    if previous is not None and word(previous) == "l'":
        return {"pos": "PRON", "tag": "PRO"}
    return {"pos": "DET", "tag": "DET"}


def indefinite_des(tokens, i):
    # "des travaux" -> un, but not "un des motifs", "la forme des Hermites" or "des plus" (= de + les)
    previous, following = neighbour(tokens, i, -1), neighbour(tokens, i, 1)
    token = tokens[i]
    if word(token) != "des" or token["pos"] != "DET" or token["tag"] != "DET" or token["lemma"] == "un":
        return None
    if previous is not None and (word(previous) in PARTITIVE_BEFORE or previous["pos"] in ("NUM", "NOUN", "PROPN")):
        return None
    if following is not None and word(following) in ("plus", "moins", "mieux"):
        return None
    return {"lemma": "un"}


def celui_lemma(tokens, i):
    if word(tokens[i]) in ("ceux", "celle", "celles", "celuy", "celui") and tokens[i]["lemma"] == "ce":
        return {"lemma": "celui"}


def reflexive_lemma(tokens, i):
    if tokens[i]["lemma"] == "se" and word(tokens[i]) in REFLEXIVE_LEMMAS:
        return {"lemma": REFLEXIVE_LEMMAS[word(tokens[i])]}


def euphonic_l(tokens, i):
    following = neighbour(tokens, i, 1)
    # lemma "l'", as in the owner's own correction
    if word(tokens[i]) == "l'" and following is not None and word(following) == "on":
        return {"pos": "PART", "tag": "ADV", "lemma": "l'"}


def participle_after_etre(tokens, i):
    if tokens[i]["pos"] != "ADJ" or tokens[i]["tag"] != "VPP":
        return None
    for j in range(i - 1, max(-1, i - 4), -1):
        if tokens[j]["lemma"] == "être":
            return {"pos": "VERB"}
        if tokens[j]["pos"] not in ("ADV", "PART"):
            return None
    return None


def introduces_clause(tokens, i):
    # a finite verb shortly after "que" (skipping adverbs and clitics): relative or completive clause
    for j in range(i + 1, min(len(tokens), i + 4)):
        token = tokens[j]
        if token["pos"] in ("VERB", "AUX") and token["tag"] in ("V", "VS", "VIMP"):
            return True
        if token["pos"] not in ("ADV", "PART", "PRON"):
            return False
    return False


def restrictive_que_loose(tokens, i):
    # "ne ... que" with words other than a verb before "que": "n'est plaine que d'amour"
    following = neighbour(tokens, i, 1)
    if word(tokens[i]) not in ("que", "qu'") or tokens[i]["pos"] != "SCONJ":
        return None
    if (following is not None and word(following) in SUBJECT_PRONOUNS) or introduces_clause(tokens, i):
        return None
    saw_verb = False
    for j in range(i - 1, max(-1, i - 10), -1):
        w = word(tokens[j])
        if w in NEGATION or w in COMPARATIVE or w in CLAUSE_BOUNDARY:
            return None
        if tokens[j]["pos"] in ("VERB", "AUX"):
            saw_verb = True
            if tokens[j]["lemma"] in LITERARY_NE:
                return None
        if w in ("ne", "n'"):
            return {"pos": "ADV", "tag": "ADV"} if saw_verb else None
    return None


def indicative_mood(tokens, i):
    w = word(tokens[i])
    if tokens[i]["tag"] == "VS" and w not in SUBJUNCTIVE_FORMS and len(w) > 3 and w.endswith(INDICATIVE_ENDINGS):
        return {"tag": "V"}


def relative_ou(tokens, i):
    previous = neighbour(tokens, i, -1)
    if (word(tokens[i]) in ("où", "oû") and tokens[i]["pos"] in ("ADV", "SCONJ")
            and previous is not None and previous["pos"] in ("NOUN", "PROPN")):
        return {"pos": "PRON", "tag": "PROREL"}


def coordinating_conjunction(tokens, i):
    if word(tokens[i]) in ("car", "ou") and tokens[i]["pos"] == "SCONJ":
        return {"pos": "CCONJ", "tag": "CC"}


def clitic_y_en_tag(tokens, i):
    if word(tokens[i]) in ("y", "en", "-y", "-en") and tokens[i]["pos"] == "PRON" and tokens[i]["tag"] != "CLO":
        return {"tag": "CLO"}


def en_pronoun(tokens, i):
    # "s'en aller", "en profiter"; before a gerund ("en faisant") it stays a preposition
    following = neighbour(tokens, i, 1)
    if (word(tokens[i]) == "en" and tokens[i]["pos"] == "ADP" and following is not None
            and following["pos"] in ("VERB", "AUX") and following["tag"] != "VPR"):
        return {"pos": "PRON", "tag": "CLO", "lemma": "en"}


def ce_subject(tokens, i):
    # "ce n'est", "c'était": subject clitic of être
    if word(tokens[i]) not in ("ce", "c'", "ç'") or tokens[i]["tag"] == "CLS":
        return None
    for j in range(i + 1, min(len(tokens), i + 4)):
        if tokens[j]["lemma"] == "être":
            return {"pos": "PRON", "tag": "CLS"}
        if word(tokens[j]) not in CLITICS:
            return None
    return None


def object_clitic_tag(tokens, i):
    if word(tokens[i]) not in ("le", "la", "les", "l'", "lui", "luy", "leur") or tokens[i]["pos"] != "PRON" or tokens[i]["tag"] != "PRO":
        return None
    j = i + 1
    while j < len(tokens) and word(tokens[j]) in ("y", "en", "le", "la", "les", "l'", "lui", "leur"):
        j += 1
    if j < len(tokens) and tokens[j]["pos"] in ("VERB", "AUX"):
        return {"tag": "CLO"}


def title_noun(tokens, i):
    if tokens[i]["pos"] == "PROPN" and word(tokens[i]) in TITLES:
        return {"pos": "NOUN", "tag": "NC", "lemma": TITLES[word(tokens[i])]}


def closed_class_lemma(tokens, i):
    w, lemma = word(tokens[i]), tokens[i]["lemma"]
    for target, forms in LEMMA_GROUPS.get(tokens[i]["pos"], {}).items():
        if w in forms and lemma.lower() in forms and lemma != target:
            return {"lemma": target}
    return None


def pronominal_autre(tokens, i):
    # "l'autre", "les autres" with no noun after
    previous, following = neighbour(tokens, i, -1), neighbour(tokens, i, 1)
    if (word(tokens[i]) not in ("autre", "autres") or tokens[i]["pos"] not in ("ADJ", "NOUN")
            or previous is None or word(previous) not in PRONOMINAL_BEFORE):
        return None
    if following is not None and following["pos"] in ("NOUN", "ADJ", "PROPN", "NUM"):
        return None
    return {"pos": "PRON", "tag": "PRO"}


def euphonic_t(tokens, i):
    # also the historic "a-t'il"; in "va-t'en" -t' is the pronoun te
    following = neighbour(tokens, i, 1)
    if word(tokens[i]) in ("-t", "-t-") or (
            word(tokens[i]) == "-t'" and following is not None and word(following) in ("il", "elle", "on", "ils", "elles")):
        return {"pos": "X", "tag": "X", "lemma": "t"}


def participle_lemma(tokens, i):
    token = tokens[i]
    if token["tag"] != "VPP" or token["pos"] != "VERB" or token["lemma"].endswith(INFINITIVE_ENDINGS):
        return None
    lemma = PARTICIPLE_LEMMAS.get(word(token))
    if lemma:
        return {"lemma": lemma}


ER_PARTICIPLE_ENDINGS = ("é", "ée", "és", "ées", "ez", "ee", "ees")
IR_PARTICIPLE_ENDINGS = ("i", "ie", "is", "ies")


def adjective_participle(tokens, i):
    # UD: a participle used as an adjective takes the adjective as lemma ("glacés" -> glacé, not glacer)
    token = tokens[i]
    if token["pos"] != "ADJ":
        return None
    lemma, form = token["lemma"].lower(), word(token)
    if lemma in MASCULINE_PARTICIPLE and form in PARTICIPLE_FORMS[lemma]:
        return {"lemma": MASCULINE_PARTICIPLE[lemma]}
    # regular participles, even when this spelling isn't attested ("establies", "glacés")
    if lemma.endswith("er") and form.endswith(ER_PARTICIPLE_ENDINGS):
        return {"lemma": lemma[:-2] + "é"}
    if lemma.endswith("ir") and form.endswith(IR_PARTICIPLE_ENDINGS) and lemma in MASCULINE_PARTICIPLE:
        return {"lemma": lemma[:-1]} if MASCULINE_PARTICIPLE[lemma] == lemma[:-1] else None
    return None


def adjective_tag(tokens, i):
    # French Treebank: participles used as adjectives are tagged ADJ
    if tokens[i]["pos"] == "ADJ" and tokens[i]["tag"] == "VPP":
        return {"tag": "ADJ"}


def adverb_y(tokens, i):
    if word(tokens[i]) in ("y", "-y") and tokens[i]["pos"] == "ADV":
        return {"pos": "PRON", "tag": "CLO", "lemma": "y"}


def multiword_lemma_piece(tokens, i):
    # UD: each word of "parce que", "afin que" has its own lemma
    token = tokens[i]
    words = token["lemma"].lower().split()
    if len(words) > 1 and token["pos"] != "PROPN" and word(token) in words:
        return {"lemma": word(token)}


def elided_de_tag(tokens, i):
    if word(tokens[i]) == "d'" and tokens[i]["tag"] in ("PREF", "PONCT", "X", "P+D"):
        return {"pos": "ADP", "tag": "P"}


def plain_lemma(tokens, i):
    lemma = tokens[i]["lemma"].translate(PLAIN_LEMMA)
    if lemma != tokens[i]["lemma"]:
        return {"lemma": lemma}


def ne_tag(tokens, i):
    if word(tokens[i]) in ("ne", "n'") and tokens[i]["pos"] == "PART" and tokens[i]["tag"] != "ADV":
        return {"tag": "ADV"}


CONTEXT_RULES = [
    ("pos: être VERB -> AUX", etre_aux),
    ("pos: avoir VERB -> AUX before a participle", avoir_aux),
    ("pos: avoir AUX -> VERB before an object", avoir_main_verb),
    ("pos: modal AUX -> VERB", modal_verb),
    ("s' before il -> si", elided_si),
    ("restrictive ne ... que -> ADV", restrictive_que),
    ("relative ce que -> PRON", relative_ce_que),
    ("ce before que/qui -> PRON", ce_pronoun),
    ("le/la/les before a verb -> PRON", object_clitic),
    ("un NUM -> DET/PRON", un_article),
    ("lemma: indefinite des -> un", indefinite_des),
    ("lemma: ceux/celle -> celui", celui_lemma),
    ("lemma: reflexive me/te/nous/vous", reflexive_lemma),
    ("l' in l'on -> PART", euphonic_l),
    ("pos: participle after être -> VERB", participle_after_etre),
    ("tag: ne -> ADV", ne_tag),
    ("tag: d' -> P", elided_de_tag),
    ("restrictive ne ... que (other words before que) -> ADV", restrictive_que_loose),
    ("tag: future/conditional/imperfect VS -> V", indicative_mood),
    ("relative où after a noun -> PRON", relative_ou),
    ("car/ou SCONJ -> CCONJ", coordinating_conjunction),
    ("tag: pronoun y/en -> CLO", clitic_y_en_tag),
    ("en before a verb -> PRON", en_pronoun),
    ("tag: ce/c' before être -> CLS", ce_subject),
    ("tag: le/la/les/lui/leur before a verb -> CLO", object_clitic_tag),
    ("titles PROPN -> NOUN", title_noun),
    ("lemma: articles and pronouns -> majority lemma", closed_class_lemma),
    ("autre(s) with no noun -> PRON", pronominal_autre),
    ("euphonic -t -> X", euphonic_t),
    ("lemma: participle -> infinitive", participle_lemma),
    ("lemma: participle used as adjective -> adjective", adjective_participle),
    ("tag: participle used as adjective -> ADJ", adjective_tag),
    ("y ADV -> PRON", adverb_y),
    ("lemma: piece of a multi-word lemma -> its own word", multiword_lemma_piece),
    ("lemma: no ligatures or long s (œ -> oe, æ -> ae, ſ -> s)", plain_lemma),
]


def apply_context_rules(tokens, stats):
    tokens = [dict(t) for t in tokens]
    for name, rule in CONTEXT_RULES:
        for i in range(len(tokens)):
            change = rule(tokens, i)
            if change and any(tokens[i][field] != value for field, value in change.items()):
                tokens[i].update(change)
                stats[name] += 1
    return tokens


def process_sentence(sentence, tokenizer, table, stats):
    tokens = sentence["annotations"]["tokens"]
    if not tokens or not all("text" in t and "lemma" in t for t in tokens):
        stats["malformed sentences left as-is"] += 1
        return sentence
    tokens = retokenize(sentence, tokenizer, table, stats)
    tokens = [fix_labels(t, stats) for t in tokens]
    tokens = apply_context_rules(tokens, stats)
    return {**sentence, "annotations": {**sentence["annotations"], "tokens": tokens}}


def main():
    tokenizer = make_tokenizer()
    files = [(f"training_data/annotations_{p}.jsonl", read_original(f"training_data/annotations_{p}.jsonl")) for p in PERIODS]
    table = build_label_table(sentence for _, sentences in files for sentence in sentences)
    build_participle_lemmas(sentence for _, sentences in files for sentence in sentences)
    files += [
        (f"test_data/annotations_{p}_test_set.jsonl", read_original(f"test_data/annotations_{p}_test_set.jsonl"))
        for p in PERIODS
    ]

    total = Counter()
    for path, sentences in files:
        stats = Counter()
        fixed = [process_sentence(sentence, tokenizer, table, stats) for sentence in sentences]
        with open(path, "w", encoding="utf-8") as f:
            for sentence in fixed:
                f.write(json.dumps(sentence, ensure_ascii=False) + "\n")
        summary = {k: v for k, v in stats.items() if not k.startswith(("split: ", "merged: "))}
        print(f"{path}: {len(fixed)} sentences, {summary}")
        total.update(stats)

    for kind in ("split", "merged"):
        print(f"\nMost frequent ({kind}):")
        for key, count in Counter({k: v for k, v in total.items() if k.startswith(f"{kind}: ")}).most_common(25):
            print(f"  {key[len(kind) + 2:]:20s} {count}")


if __name__ == "__main__":
    main()
