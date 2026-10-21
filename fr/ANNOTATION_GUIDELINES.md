# Annotation guidelines: historic French (1500–2000)

These are the conventions used in the training data, the test sets and both gold sets. Each token has three labels: **POS** (Universal Dependencies UPOS), **TAG** (French Treebank tagset) and **LEMMA**. POS and lemmas follow UD French; TAG follows the French Treebank. Where the project owner's own corrections set a convention, those win.

DEP labels come from the GPT-4 annotation and aren't used for training; leave them as they are. Tokenization is fixed: never split, merge or edit token text. Spellings are historic (*estoit*, *luy*, *mesme*, *tousjours*) and may contain OCR errors (long s read as f: *fçu* = *sçu*).

`prepare_annotations.py` applies most of these conventions to the GPT-4 annotations automatically; the rest need a person.

## Tokenization
Tokens follow `historic_tokenizer.py`, so the labels line up with what the model sees:
- Elisions are split (*c'* + *est*, *qu'* + *il*), and so are inverted and imperative clitics, keeping their hyphen: *dit* + *-il*, *as* + *-tu*, *sont* + *-elles*, *dites* + *-lui*, *a* + *-t'* + *il*, *va* + *-t'* + *en*.
- Dashes are separate tokens, including a single hyphen used as a dialogue dash (*-* + *Mais*) and `--` or `---` glued to a word.
- Kept whole: *peut-être*, *aujourd'hui* and their historic spellings (*peut-estre*, *aujourd'huy*), pronouns with *même* (*moi-même*, *luy-mesme*), *celui-ci*, *ceux-ci*, *celuy-cy*, and other hyphenated words on spaCy's list (*au-dessus*, *rendez-vous*), in any capitalization.
- Other hyphenated words are split around a separate hyphen token (*long* + *-* + *temps*, *c'* + *est* + *-* + *à* + *-* + *dire*); every piece keeps the whole word's labels.

## Tagset: TAG and the POS it goes with
| TAG | POS | Use |
|---|---|---|
| NC | NOUN | common noun |
| NPP | PROPN | proper noun |
| V | VERB/AUX | finite indicative or conditional verb |
| VS | VERB/AUX | subjunctive |
| VIMP | VERB/AUX | imperative |
| VINF | VERB/AUX | infinitive |
| VPP | VERB/AUX | past participle used as a verb |
| VPR | VERB/AUX | present participle, gerund |
| ADJ | ADJ | adjective, including participles used as adjectives |
| ADV | ADV, PART | adverb; also *ne* (POS PART), *pas*/*point*/*plus* of negation (POS ADV) |
| DET | DET | determiner, article |
| DETWH | DET | *quel* determiner |
| P | ADP | preposition |
| P+D | ADP | contracted preposition and article: *au, aux, du, des* (when = de + le/les) |
| P+PRO | ADP | *duquel, auquel, desquels* |
| CS | SCONJ | subordinating conjunction |
| CC | CCONJ | coordinating conjunction |
| CLS | PRON | subject clitic (*je, tu, il, on*, and *ce* before être) |
| CLO | PRON | object clitic (*le, la, les, lui, leur, y, en, me, te*) |
| CLR | PRON | reflexive clitic (*se*, and *me/te/nous/vous* used reflexively) |
| PRO | PRON | other pronoun (*cela, celui, ceux, eux, moi, chacun, l'un, l'autre*) |
| PROREL | PRON | relative pronoun (*qui, que, dont, où, lequel* in relative clauses) |
| PROWH | PRON | interrogative pronoun |
| ADVWH | ADV | interrogative adverb |
| I | INTJ | interjection (*oui*, *ah*) |
| PONCT | PUNCT | punctuation |
| ET | X | foreign word |
| X | X | other, including euphonic *-t* |

Numbers have POS NUM and TAG DET before a noun (*mille hommes*), ADJ after an article (*les neuf Muses*), PRO when used as a pronoun (*tous deux*). TAG must always be consistent with POS.

## Verbs
- **être**: AUX as tense auxiliary (*il est venu*), in the passive (*il a esté importuné*, *estant caché*) and as copula, including place (*il est grand*, *c'est un homme*, *est entre ses mains*). VERB only when existential (*s'il est un heur*, *si cela n'est pas*).
- **avoir**: AUX only as a tense auxiliary, followed by a past participle (*il a dit*, *ils ont été*). Otherwise VERB: possession (*j'ai faim*, *vous avez raison*) and *il y a* (*il n'y en a qu'un*).
- **Modals and other verbs** (*pouvoir, devoir, falloir, vouloir, savoir, aller, venir, faire*): always VERB, never AUX.
- **Past participles**: VERB (TAG VPP, infinitive lemma) in compound tenses, in the passive (*sont obligées*, *fut ravi*) and when the participle has its own complement (*animé d'un zèle*, *renfermées dans*). Used as an adjective, it is ADJ, TAG ADJ, with the masculine singular participle as lemma (*glacés* → glacé, *guérie* → guéri).
- **Mood**: conditional, future and imperfect forms are TAG V, never VS (*seroit*, *pourrait*, *estoit*).
- **Lemma**: the modern infinitive (*estoit* → être, *connoistre* → connaître, *fçu* → savoir).
- **Euphonic *-t*** (*a-t-il*): X, TAG X, lemma *t*.

## que, qu', où, dont, s', conjunctions
- Relative pronoun, with an antecedent noun or pronoun, including *ce que* (*la douleur que je souffre*, *ce qu'il demandoit*): PRON, TAG PROREL.
- Complementizer (*je crois que*, *afin que*, *avant qu'il*) and comparative (*plus grand que*, *un autre objet que vous*): SCONJ, TAG CS.
- Restrictive *ne … que* meaning "only" (*il n'est qu'un effet*, *on ne voyait que coiffes*): ADV, TAG ADV.
- *où* relative (*l'heure où*): PRON, TAG PROREL. *là où*: ADV. *dont*: PRON, TAG PROREL.
- *s'* before *il*/*ils* is *si*: SCONJ, TAG CS, lemma *si*. Reflexive *s'*: PRON, TAG CLR, lemma *se*.
- *car*, *ou*, *or* (discourse "now"): CCONJ, TAG CC. Old French *ne* meaning *ni*: CCONJ, lemma *ni*.

## Pronouns or determiners
- *le, la, les, l'* directly before a verb (object): PRON, TAG CLO, lemma *le* (*pour les chasser*, *et les conduire*).
- *ce* before *que, qui, dont*: PRON, TAG PRO. *ce*/*c'* before être: PRON, TAG CLS. *ce* before a noun: DET.
- *tout* in *tout ce que*: PRON.
- *y*: PRON, TAG CLO, including *il y a*. *en* as a pronoun (*j'en ai*, *s'en aller*): PRON, TAG CLO; as a preposition (*en France*, *en lui prouvant*): ADP, TAG P.
- *l'un, l'autre, les autres* with no noun: PRON, TAG PRO. *un/une* as an article (*un tel amour*, *qu'un seul*): DET, not NUM.
- Partitive *du*, *de la* (*du pain*) and the reduced article *de* before an adjective and a noun (*de simples jeux*): DET, TAG DET, lemma *de*.
- *ne*: PART, TAG ADV. *pas, point*: ADV, TAG ADV.
- *l'* in *l'on*: PART, TAG ADV, lemma *l'*.
- Titles (*Madame, Monsieur, M., Mme, Roy*), *saint* before a name, and capitalized common nouns (*Muses, Nature, Université, Jésuites*): NOUN, TAG NC, lowercase lemma. Names are PROPN even when written in lowercase (*la chevrette*).

## Lemmas
- Modern standard spelling: *enfans* → enfant, *medecins* → médecin, *luy* → lui, *mesme* → même, *tousjours* → toujours.
- No ligatures, long s or abbreviation marks, so that lemmas can be searched as typed: œ → oe, æ → ae, ſ → s (*cœur*, *coeur* → coeur; *œuvres* → oeuvre; *eſt* → être; *Œdipe* → Oedipe).
- Nouns singular, adjectives masculine singular, verbs infinitive.
- Contracted *au, aux* → à; *du, des* (= de + le/les) → de, with POS ADP, TAG P+D.
- Indefinite article *des* (*des travaux*, *il y a des objections*) → un, with POS DET, TAG DET.
- Articles and pronouns reduce to one lemma: *la, les* → le; *une* → un; *ces, cette* → ce; *sa, ses* → son; *ils* → il; *elles* → elle; *ceux, celle, celles* → celui.
- Pronouns: *me/m'* → me, *te/t'* → te, *se/s'* → se, *nous* → nous, *vous* → vous, *luy/lui* → lui, *eux* → eux.
- Elided forms take the full lemma: *l'* → le, *qu'* → que, *j'* → je, *d'* → de, *n'* → ne, *c'* → ce.
- Proper nouns: modern standard form, capitalized (*Julian* → Julien, *Remy* → Rémy), unless there is no settled modern form (*Simontault*).
- Each word of a multi-word expression written separately has its own lemma (*parce que* → parce, que). A lemma never contains a space.
- Fixed expressions that the tokenizer splits (*d'abord, aujourd'hui, peut-être, jusqu'à, c'est-à-dire*): every piece keeps the whole expression's lemma and POS.
- A token that merges two words takes the main word's lemma (*trescontent* → content, *s'effaraient* → effarer, *jour-là* → jour).
