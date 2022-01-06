import sys

import yaml
import neuralcoref
import spacy
from spacy.tokens import Token

from pattern_core import (
    PatternMatcher,
    get_chain,
    get_fixed_ent_type,
    iter_entity_dtype_pairs,
)


def main():
    # spacy.prefer_gpu()  # neuralcoref has problems with GPU

    nlp_dir = sys.argv[1]

    with open("pattern.yml") as fin:
        pattern_config = yaml.safe_load(fin)
        pattern_matcher = PatternMatcher(pattern_config)

    nlp = spacy.load("en_core_web_lg")
    nlp_ner = spacy.load(nlp_dir)
    nlp.replace_pipe("ner", nlp_ner.get_pipe("ner"))

    merge_entities = nlp.create_pipe("merge_entities_mod")
    nlp.add_pipe(merge_entities, after="ner")

    merge_noun_chunks = nlp.create_pipe("merge_noun_chunks")
    nlp.add_pipe(merge_noun_chunks)

    Token.set_extension("fixed_ent_type", getter=get_fixed_ent_type)

    # neuralcoref cannot be used together with merge noun chunks/entities
    nlp_coref = spacy.load('en')
    neuralcoref.add_to_pipe(nlp_coref)

    text = input("Text:")
    doc = nlp(text)
    doc_coref = nlp_coref(text)

    for sent, sent_coref in zip(doc.sents, doc_coref.sents):
        if sent.root.norm_ == "include":
            span = None

            for child in sent.root.children:
                if child.dep_ == 'nsubj':
                    start_char_pos = child.idx
                    end_char_pos = child.idx + len(child.text)
                    tok_start = tok_end = None

                    for tok_idx, tok in enumerate(sent_coref):
                        if start_char_pos <= tok.idx < end_char_pos:
                            if tok_start is None:
                                tok_start = tok_idx
                            tok_end = tok_idx

                    span = sent_coref[tok_start:tok_end + 1]
                    break

            dtypes = []
            for tok in sent:
                if tok._.fixed_ent_type == 'DATA':
                    dtypes.append(tok.text)

            coref_cluster = span._.coref_cluster
            if coref_cluster and len(dtypes) > 0:
                for tok in doc:
                    if tok.idx <= coref_cluster.main.root.idx <= tok.idx + len(tok.text):
                        coref_main = tok
                        break
                else:
                    assert False  # unreachable

                doc.user_data[coref_main.i] = dtypes

    for sent in doc.sents:
        for common_parent, ent, dt in iter_entity_dtype_pairs(sent):
            chain = get_chain(common_parent, ent, dt)
            # print(chain)

            if pattern_matcher.test(chain):
                print(
                    "/".join(repr(i.text) for i in chain[0].tokens),
                    repr(common_parent.text),
                    "/".join(repr(i.text) for i in chain[-1].tokens))

                if dt.i in doc.user_data:
                    print(">", repr(dt.text), "INCLUDES", doc.user_data[dt.i])


if __name__ == '__main__':
    main()
