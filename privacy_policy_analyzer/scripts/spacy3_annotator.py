#!/usr/bin/env python3

import argparse

import spacy
from privacy_policy_analyzer.document import PolicyDocument
from spacy.matcher import DependencyMatcher


def expand_token(token):
    doc = token.doc
    if token.ent_iob_ != 'O':
        span_start = span_end = token.i

        while doc[span_start].ent_iob_ != 'B':
            span_start -= 1

        while span_end < len(doc) and doc[span_end].ent_iob_ != 'O':
            span_end += 1

        return doc[span_start:span_end]
    else:
        subtoken_pos = {t.i for t in token.subtree}
        left_edge = token.i

        while left_edge - 1 in subtoken_pos:
            prev_token = doc[left_edge - 1]

            if prev_token.is_space or prev_token.pos_ == 'X' or prev_token.ent_iob_ != 'O':
                break

            left_edge -= 1

        return doc[left_edge:token.i + 1]

 
def get_conjuncts(token):
    for child in token.rights:
        if child.dep_ in ["conj", "appos"] and child.pos_ in ["NOUN", "PROPN", "PRON"]:
            yield child


class SubsumAnnotator:
    def __init__(self, nlp):
        self.matcher = DependencyMatcher(nlp.vocab)

        pattern = [
            # anchor token: as
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"ORTH": "as", "DEP": "prep"}
            },
            # "such" as
            {
                "LEFT_ID": "anchor",
                "REL_OP": ";",
                "RIGHT_ID": "r00",
                "RIGHT_ATTRS": {"ORTH": "such"}
            },
            # "SOMETHING" such as
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            },
            # such as "SOMETHING"
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            }
        ]
        self.matcher.add("SUBSUM_SUCH_AS", [pattern])

        pattern = [
            # anchor token: including/like
            {
                "RIGHT_ID": "anchor",
                "RIGHT_ATTRS": {"ORTH": {"IN": ["include", "including", "like"]}}
            },
            # "SOMETHING" including/like
            {
                "LEFT_ID": "anchor",
                "REL_OP": "<",
                "RIGHT_ID": "upper_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            },
            # including "SOMETHING"/like
            {
                "LEFT_ID": "anchor",
                "REL_OP": ">",
                "RIGHT_ID": "lower_token",
                "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}
            }
        ]
        self.matcher.add("SUBSUM_INCLUDING_LIKE", [pattern])

    def test(self, doc):
        matches = self.matcher(doc)

        for match_id, matched_tokens in matches:
            _, (match_spec, ) = self.matcher.get(match_id)
            match_info = {s["RIGHT_ID"]: doc[t] for t, s in zip(matched_tokens, match_spec)}

            lower_tokens = [match_info["lower_token"]]
            lower_tokens.extend(get_conjuncts(lower_tokens[0]))

            if any(t.ent_iob_ != 'O' for t in lower_tokens):
                for lower_token in lower_tokens:
                    print(expand_token(match_info["upper_token"]), expand_token(lower_token), sep=" | ")

        # TODO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdir", help="working directory")
    args = parser.parse_args()

    spacy.prefer_gpu()

    document = PolicyDocument(args.workdir)
    nlp = spacy.load("en_core_web_trf")
    # setup_models(args.ner)

    annotator = SubsumAnnotator(nlp)

    for seg in document.segments:
        doc = document.build_doc(seg, nlp, apply_pipe=True, load_ner=True)


if __name__ == "__main__":
    main()
