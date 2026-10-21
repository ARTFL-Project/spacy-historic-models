"""
spaCy's French tokenizer, adapted to historic printing.

The default tokenizer splits elisions (c'est -> c' + est) and inverted clitics (dit-il -> dit + -il),
and keeps a short list of hyphenated words whole. Historic texts break it in a few ways:

    ---Mais      -> one token                  now "---" + "Mais": dashes printed as -- or ---, glued
                                               to words or to punctuation (dit.--Mais)
    -Mais        -> one token                  now "-" + "Mais": a single hyphen used as a dialogue
                                               dash; "dit -il" keeps its clitic
    Est-il       -> one token                  now "Est" + "-il", like "est-il"
    Peut-être    -> "Peut" + "-" + "être"      now one token, like "peut-être": spaCy's exceptions
                                               are lowercase only
    peut-estre   -> "peut" + "-" + "estre"     now one token, like "peut-être"; also aujourd'huy,
                                               luy-mesme, celuy-cy and long s spellings
    moi-même     -> "moi" + "-" + "même"       now one token, like "lui-même"; also ceux-ci, like
                                               "celui-ci"
    as-tu        -> "as" + "-" + "tu"          now "as" + "-tu", like "dit" + "-il": the clitics
                                               missing from spaCy's suffixes (tu, elles, le, lui, y...)
    a-t'il       -> "a" + "-" + "t'" + "il"    now "a" + "-t'" + "il", like "a" + "-t" + "-il";
                                               also va-t'en, donne-m'en
"""

import itertools

import spacy
from spacy.attrs import ORTH
from spacy.lang.char_classes import ALPHA
from spacy.lang.fr.punctuation import HYPHENS
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex

APOSTROPHES = ("'", "’")
DASH = r"(?:—+|-{2,})"
# clitics after a hyphen that spaCy's French suffixes don't split off
CLITICS = ["tu", "elles", "le", "la", "les", "lui", "luy", "leur", "y", "toi", "toy", "moy"]
# a hyphen before these, after a space, is a spaced-out clitic ("dit -il"), not a dialogue dash
SPACED_CLITICS = CLITICS + ["ce", "en", "elle", "il", "ils", "je", "là", "moi", "nous", "on", "t", "vous", "ci"]
# historic spellings of spaCy's whole-word exceptions
HISTORIC_WHOLE = ["peut-estre", "peut-eſtre", "aujourd'huy", "auiourd'huy", "auiourd'hui", "celuy-cy", "celle-cy",
                  "celles-cy", "ceux-ci", "ceux-cy"]
# compounds the annotations keep whole but the tokenizer would split ("chef-d'œuvre" -> chef - d' œuvre)
COMPOUNDS = ["chef-d'oeuvre", "chefs-d'oeuvre", "main-d'oeuvre", "crève-coeur"]
PRONOUNS = ["moi", "toi", "soi", "lui", "elle", "nous", "vous", "eux", "elles", "moy", "toy", "soy", "luy"]
MEME = ["même", "mêmes", "mesme", "mesmes", "meſme", "meſmes"]


def case_variants(word):
    return {word, word[0].upper() + word[1:], word[0].lower() + word[1:]}


def add_special_cases(tokenizer):
    # spaCy's exceptions are case-sensitive: "Peut-être" and "Cf." were split, "Est-il" kept whole
    for text, tokens in list(tokenizer.rules.items()):
        if len(text) < 3 or not any(c in text for c in "-'’.") or not text[0].isalpha():
            continue
        if len(tokens) == 1:
            for form in case_variants(text):
                if form not in tokenizer.rules:
                    tokenizer.add_special_case(form, [{ORTH: form}])
        elif text[0].islower() and text[0].upper() + text[1:] not in tokenizer.rules:
            first = tokens[0][ORTH]
            tokenizer.add_special_case(text[0].upper() + text[1:],
                                       [{ORTH: first[0].upper() + first[1:]}] + [{ORTH: t[ORTH]} for t in tokens[1:]])
    words = HISTORIC_WHOLE + [f"{pronoun}-{meme}" for pronoun, meme in itertools.product(PRONOUNS, MEME)]
    words += COMPOUNDS + [word.replace("oe", "œ") for word in COMPOUNDS]
    for word in words:
        for apostrophe in APOSTROPHES:
            for form in case_variants(word.replace("'", apostrophe)):
                tokenizer.add_special_case(form, [{ORTH: form}])


def add_punctuation_rules(nlp):
    hyphen_infix = next(infix for infix in nlp.Defaults.infixes if "(?:-|–|—|--|---|——|~)" in infix)
    infixes = [infix for infix in nlp.Defaults.infixes if infix != hyphen_infix] + [
        # a hyphen before an elided clitic stays with it: a-t'il -> a + -t' + il, va-t'en, donne-m'en
        hyphen_infix.replace("(?:-|", rf"(?:-(?![tmlsTMLS]['’][{ALPHA}])|"),
        rf"(?<=[{ALPHA}])(?=-[tmlsTMLS]['’][{ALPHA}])",
        # punctuation and a dash between two words: "monde.--je", "contez,---dit"
        rf"(?<=[{ALPHA}])[!?;:,.](?={DASH})", rf"(?<=[!?;:,.]){DASH}(?=[{ALPHA}])",
    ]
    spaced = "|".join(SPACED_CLITICS)
    prefixes = list(nlp.Defaults.prefixes) + [DASH, rf"[-–](?=[{ALPHA}])(?!(?:{spaced})(?![{ALPHA}]))"]
    clitics = "|".join(CLITICS + [clitic.upper() for clitic in CLITICS])
    suffixes = list(nlp.Defaults.suffixes) + [rf"(?<=[{ALPHA}!?;:,.»]){DASH}",
                                             rf"(?<=[{ALPHA}])[{HYPHENS}](?:{clitics})"]
    nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer
    nlp.tokenizer.prefix_search = compile_prefix_regex(prefixes).search
    nlp.tokenizer.suffix_search = compile_suffix_regex(suffixes).search


@spacy.registry.callbacks("historic_french_tokenizer.v1")
def create_tokenizer_callback():
    """For training configs: [nlp] after_creation = {"@callbacks":"historic_french_tokenizer.v1"}.
    The rules are saved with the model's tokenizer, so a trained model loads without this code."""
    def customize(nlp):
        add_punctuation_rules(nlp)
        add_special_cases(nlp.tokenizer)
        return nlp
    return customize


def make_nlp():
    nlp = spacy.blank("fr")
    add_punctuation_rules(nlp)
    add_special_cases(nlp.tokenizer)
    return nlp


if __name__ == "__main__":
    nlp = make_nlp()
    for text in [
        "---Oui, dit don Garcia; contez,---dit Mlle Sophie. Du monde.--je ne le dis pas, dit-il---",
        "-Mais, dit-il, -C'est vrai. -Oui. Est-il venu? Est-ce lui? Dit -il, ce -t-il.",
        "Peut-être, peut-estre et peut-eſtre; aujourd'huy, auiourd’huy, aujourd'hui; Celui-ci, celuy-cy, ceux-ci, ceux-là.",
        "Moi-même, luy-mesme, eux-mêmes, soy-meſme, lui-même; as-tu, sont-elles, dites-lui, allez-y, donnez-le-moi, dis-moy.",
        "A-t'il dit? repliqua-t'il; va-t'en, donne-m'en; a-t-il, dit-il, c'est, qu'il, l'on, d'abord, jusqu'à, entr'eux.",
        "Il est tres-humble, c'est-à-dire, là-dessus, chez–vous, poursuiuit–il, Cf. Id. p. 4, M. de, St. Paul.",
    ]:
        print([t.text for t in nlp(text)])
