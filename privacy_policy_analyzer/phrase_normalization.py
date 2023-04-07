"""Module of phrase normalization"""

import json
import re
import string

import regex

from privacy_policy_analyzer.utils import TRIVIAL_WORDS


def trim_phrase(phrase):
    def dfs(token, state):
        match state:
            case "compound" | "nmod" | "pobj" | "nsubj" | "dobj":
                next_states = {"compound", "amod", "nmod", "prep", "relcl", "acl"}
            case "prep":
                next_states = {"pobj"}
            case "amod":
                next_states = {"advmod", "npadvmod"}
            case "relcl" | "acl":
                next_states = {"nsubj", "dobj"}
            case _:
                yield token
                return

        for t in token.children:
            if t in phrase and t.dep_ in next_states and t.lemma_ not in TRIVIAL_WORDS:
                yield from dfs(t, t.dep_)

        yield token

    return sorted(dfs(phrase.root, "compound"))


class RuleBasedPhraseNormalizer:
    """Rule-based phrase normalizer"""

    def __init__(self, phrase_map_rules):
        self.regex_list = {}

        for norm_name, regex_list in phrase_map_rules.items():
            positive_rules = []
            negative_rules = []

            for regex_string in (regex_list or []):
                # Prefix "!" indicates negative rules (must not match)
                if regex_string[0] == "!":
                    regex_string = regex_string[1:]
                    rule_list = negative_rules
                else:
                    rule_list = positive_rules

                # Prefix "=" indicates case-sensitive match (by default do case-insensitive)
                if regex_string[0] == "=":
                    regex_string = regex_string[1:]
                    flag = ""
                else:
                    flag = "i"

                # Prefix/Suffix '\b' (word break) unless the rule explicitly matches the starting/ending position
                if regex_string[0] != "^":
                    regex_string = r'\b' + regex_string

                if regex_string[-1] != "$":
                    regex_string = regex_string + r'\b'

                # Allow space to match any number of white-space characters
                regex_string.replace(' ', r'\s+')

                rule_list.append(f"(?{flag}:{regex_string})")

            # Concat regexes to one for efficiency
            positive_re = re.compile("|".join(positive_rules))
            negative_re = re.compile("|".join(negative_rules) or '$^')  # If no negative rule, '$^' matches nothing
            self.regex_list[norm_name] = (positive_re, negative_re)

    def normalize(self, phrase, fallback_to_stem=True):
        if (phrase.root.lemma_ in TRIVIAL_WORDS
            or (phrase.root.pos_ == "PRON" and phrase.root.lemma_ not in ("I", "we"))):
            yield "UNSPECIFIC"
            return

        original_text = phrase.text.strip(string.punctuation)
        lemma_text = phrase.lemma_.strip(string.punctuation)

        negative_names = set()
        has_match = False

        # First try to match the full phrase or lemmatized full phrase
        for norm_name, (positive_regex, negative_regex) in self.regex_list.items():
            if negative_regex.search(original_text) or negative_regex.search(lemma_text):
                negative_names.add(lemma_text)
            elif positive_regex.search(original_text) or positive_regex.search(lemma_text):
                yield norm_name
                has_match = True

        if not has_match:
            # If no match, aggressively trim the phrase to get its "stem"
            phrase_stem = " ".join(t.lemma_ for t in trim_phrase(phrase))

            for norm_name, (positive_regex, negative_regex) in self.regex_list.items():
                if (norm_name not in negative_names
                    and positive_regex.search(phrase_stem)
                    and not negative_regex.search(phrase_stem)):

                    yield norm_name
                    has_match = True

        # If still no match, yield the stem
        if fallback_to_stem and not has_match:
            yield phrase_stem


class EntityMatcher:
    """Fuzzy entity (company) name matcher"""

    def __init__(self, entity_info_file):
        with open(entity_info_file, "r", encoding="utf-8") as fin:
            entity_info = json.load(fin)

        self.entity_names = {}
        self.ngram_mapping = {}
        self.domain_mapping = {}

        for entity, info in entity_info.items():
            self.entity_names[entity] = info["aliases"]

            for domain in info["domains"]:
                self.domain_mapping[domain] = entity

            for ngram, oov_flag in info["ngrams"].items():
                self.ngram_mapping[ngram] = (entity, oov_flag)

        self.keyword_matching_regex = regex.compile(
            r"\b(?:\L<keywords>)\b",
            keywords=self.ngram_mapping.keys(),
            flags=regex.IGNORECASE
        )

    def match_name(self, name):
        if name.lower() in self.domain_mapping:
            yield self.domain_mapping[name.lower()]
            return

        for m in self.keyword_matching_regex.finditer(name):
            entity, oov_flag = self.ngram_mapping[m[0].lower()]

            if oov_flag:
                yield entity
            else:
                r = regex.compile(r"\b(?:\L<keywords>)\b", keywords=[m[0]])
                for full_name in self.entity_names[entity]:
                    if r.search(full_name):
                        yield entity
                        break
