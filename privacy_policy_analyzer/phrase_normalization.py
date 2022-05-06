import importlib.resources as pkg_resources
import json
import re

import regex
import spacy
import tldextract
import yaml

import privacy_policy_analyzer


class RuleBasedPhraseNormalizer:
    def __init__(self):
        with pkg_resources.open_text(privacy_policy_analyzer, "phrase_normalization.yml") as fin:
            config = yaml.safe_load(fin)

            self.stop_words = frozenset(config["stop_words"])
            self.token_map = config["token_map"]
            self.normalization_rules = dict()

            for key, regex_list in config["normalization_rules"].items():
                self.normalization_rules[key] = rules = []

                for r in (regex_list or []):
                    if r[0] != "^":
                        r = r'\b' + r
                    if r[-1] != "$":
                        r = r + r'\b'
                    r.replace(' ', r'\s+')
                    rules.append(re.compile(r, re.I))

    def normalize(self, ent):
        def dfs(token):
            nonlocal ent
            yield token.i

            if token.dep_ == "amod":
                accepted_dep = {"advmod", "npadvmod"}
            elif token.pos_ in ["NOUN", "PROPN"]:
                accepted_dep = {"amod", "nmod", "compound", "poss"}
            else:
                accepted_dep = {}

            for c in token.children:
                if c.dep_ in accepted_dep and c.lemma_ not in self.stop_words and c in ent:
                    yield from dfs(c)

        doc = ent.doc
        normalized_string = ""
        prev_i = None

        for i in sorted(dfs(ent.root)):
            lemma = doc[i].lemma_.lower()
            norm = self.token_map.get(lemma, lemma)

            if prev_i is not None:
                if i != prev_i + 1 or doc[prev_i].whitespace_:
                    normalized_string += " "

            prev_i = i
            normalized_string += norm

        for key, regex_list in self.normalization_rules.items():
            for r in regex_list:
                if r.search(normalized_string):
                    return key

        return normalized_string



class EntityMatcher:
    def __init__(self, entity_info_file):
        with open(entity_info_file, "r", encoding="utf-8") as fin:
            entity_info = json.load(fin)

        # Use en_core_web_sm because we want a smaller English word list
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        common_english_words = frozenset(nlp.vocab.strings)

        # Gather all entity names for NLP processing
        self.entity_names = {e: set(i["aliases"]) for e, i in entity_info.items()}
        name_to_doc = dict()
        all_names = set.union(*[s for s in self.entity_names.values()])

        for doc in nlp.pipe(all_names):
            name_to_doc[doc.text] = doc

        # Find ngrams for fuzzy matching
        self.domain_mapping = dict()
        self.ngram_mapping = dict()

        for entity, info in entity_info.items():
            ngrams = set()

            for name in self.entity_names[entity]:
                doc = name_to_doc[name]

                for i in range(len(doc)):
                    propn_flag = False
                    oov_flag = False

                    for j in range(i, len(doc)):
                        propn_flag |= doc[j].pos_ == "PROPN"
                        oov_flag |= doc[j].norm_ not in common_english_words

                        if propn_flag:
                            ngrams.add((doc[i:j+1].text.lower(), oov_flag))

            # Also use domain names as keywords
            for full_domain in info["domains"]:
                self.domain_mapping[full_domain] = entity
                domain = tldextract.extract(full_domain).domain
                ngrams.add((domain, domain not in common_english_words))

            for s, oov_flag in ngrams:
                if s not in self.ngram_mapping:
                    self.ngram_mapping[s] = (entity, oov_flag)
                elif self.ngram_mapping[s] is not None:
                    self.ngram_mapping[s] = None

        # ngrams that uniquely identify entities
        self.ngram_mapping = {k: v for k, v in self.ngram_mapping.items() if v}

        # Main entity name should always be in the ngram_mapping
        for entity_name in self.entity_names:
            if (name := entity_name.lower()) not in self.ngram_mapping:
                self.ngram_mapping[name] = (entity_name, False)

        self.keyword_matching_regex = regex.compile(
            r"\b(?:\L<keywords>)\b",
            keywords=self.ngram_mapping.keys(),
            flags=regex.I
        )

    def match_name(self, name):
        if name.lower() in self.domain_mapping:
            return self.domain_mapping[name.lower()]

        for m in self.keyword_matching_regex.finditer(name):
            entity, oov_flag = self.ngram_mapping[m[0].lower()]

            if oov_flag:
                return entity

            r = regex.compile(r"\b(?:\L<keywords>)\b", keywords=[m[0]])
            for full_name in self.entity_names[entity]:
                if r.search(full_name):
                    return entity

        return None
