import json
import re

import regex


class RuleBasedPhraseNormalizer:
    def __init__(self, phrase_map_rules):
        self.positive_rules = dict()
        self.negative_rules = dict()

        for norm_name, regex_list in phrase_map_rules.items():
            self.negative_rules[norm_name] = []
            self.positive_rules[norm_name] = []

            for r in (regex_list or []):
                if r[0] == "!":
                    r = r[1:]
                    l = self.negative_rules[norm_name]
                else:
                    l = self.positive_rules[norm_name]

                # By default case insensitive. Prefixing "=" to override
                if r[0] == "=":
                    r = r[1:]
                    flags = 0
                else:
                    flags = re.I

                if r[0] != "^":
                    r = r'\b' + r
                if r[-1] != "$":
                    r = r + r'\b'
                r.replace(' ', r'\s+')

                l.append(re.compile(r, flags=flags))

    def normalize(self, phrase):
        for norm_name in self.positive_rules:
            rejected = False

            for r in self.negative_rules[norm_name]:
                if r.search(phrase.lemma_) or r.search(phrase.text):
                    rejected = True
                    break

            if rejected:
                continue

            for r in self.positive_rules[norm_name]:
                if r.search(phrase.lemma_) or r.search(phrase.text):
                    yield norm_name
                    break


class EntityMatcher:
    def __init__(self, entity_info_file):
        with open(entity_info_file, "r", encoding="utf-8") as fin:
            entity_info = json.load(fin)

        self.entity_names = dict()
        self.ngram_mapping = dict()
        self.domain_mapping = dict()

        for entity, info in entity_info.items():
            self.entity_names[entity] = info["aliases"]

            for domain in info["domains"]:
                self.domain_mapping[domain] = entity

            for ngram, oov_flag in info["ngrams"].items():
                self.ngram_mapping[ngram] = (entity, oov_flag)

        self.keyword_matching_regex = regex.compile(
            r"\b(?:\L<keywords>)\b",
            keywords=self.ngram_mapping.keys(),
            flags=regex.I
        )

    def match_name(self, name):
        if name.lower() in self.domain_mapping:
            yield self.domain_mapping[name.lower()]
            return

        for m in self.keyword_matching_regex.finditer(name):
            entity, oov_flag = self.ngram_mapping[m[0].lower()]

            if oov_flag:
                yield entity
                continue

            r = regex.compile(r"\b(?:\L<keywords>)\b", keywords=[m[0]])
            for full_name in self.entity_names[entity]:
                if r.search(full_name):
                    yield entity
                    break
