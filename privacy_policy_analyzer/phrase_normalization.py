import json
import re

import regex
import spacy
import tldextract


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

        # en_core_web_md supports is_oov
        nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer"])

        # Gather all names for NLP processing
        self.entity_names = {e: set(i["aliases"]) for e, i in entity_info.items()}
        name_to_doc = dict()
        all_names = set.union(*[s for s in self.entity_names.values()])

        # Domain to entity mapping. Put domain names in all_names as well.
        self.domain_mapping = dict()

        for entity, info in entity_info.items():
            for full_domain in info["domains"]:
                self.domain_mapping[full_domain] = entity

                domain = tldextract.extract(full_domain).domain
                all_names.add(domain)

        for doc in nlp.pipe(all_names):
            name_to_doc[doc.text] = doc

        # Find all ngrams for fuzzy matching
        all_ngrams = dict()

        for entity, info in entity_info.items():
            ngrams = dict()

            for name in self.entity_names[entity]:
                doc = name_to_doc[name]

                # Find all ngrams that do not start/end with a punct/space
                #   oov_flag: determine whether to do case-sensitive match
                #   like_name_flag: if false, will be dropped later
                for i in range(len(doc)):
                    like_name_flag = False
                    oov_flag = False

                    if doc[i].is_space or doc[i].is_punct:
                        continue

                    for j in range(i, len(doc)):
                        if doc[j].is_space or doc[j].is_punct:
                            continue

                        like_name_flag |= doc[j].pos_ == "PROPN"
                        oov_flag |= doc[j].is_oov
                        ngrams[doc[i:j+1].text.lower()] = (oov_flag, like_name_flag)

            # Also put domain names to ngram list
            for full_domain in info["domains"]:
                domain = tldextract.extract(full_domain).domain

                if domain not in ngrams:
                    doc = name_to_doc[domain]
                    oov_flag = any(t.is_oov for t in doc)
                    ngrams[domain] = (oov_flag, oov_flag)

            # If an ngram can be found in more than one entities, remove it
            for s, (oov_flag, like_name_flag) in ngrams.items():
                if s not in all_ngrams:
                    all_ngrams[s] = (entity, oov_flag, like_name_flag)
                elif all_ngrams[s] is not None:
                    all_ngrams[s] = None

        # Filter out ngrams that has like_name_flag=True and uniquely identifies entities
        self.ngram_mapping = dict()

        for ngram, ngram_info in all_ngrams.items():
            if ngram_info and ngram_info[-1]:
                self.ngram_mapping[ngram] = (ngram_info[0], ngram_info[1])

        # For well-known online trackers, main entity name should always be in the ngram_mapping
        for entity, info in entity_info.items():
            if info["prevalence"] > 2e-5 and entity.lower() not in self.ngram_mapping:
                self.ngram_mapping[entity.lower()] = (entity, False)

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
