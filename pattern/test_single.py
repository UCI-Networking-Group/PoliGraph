import sys

import yaml

from pattern_core import (
    PatternMatcher,
    get_chain,
    get_fixed_ent_type,
    iter_entity_dtype_pairs,
)
import spacy
from spacy.tokens import Token


def main():
    spacy.prefer_gpu()

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

    text = input("Text:")
    doc = nlp(text)

    for sent in doc.sents:
        for common_parent, ent, dt in iter_entity_dtype_pairs(sent):
            chain = get_chain(common_parent, ent, dt)
            print(chain)

            if pattern_matcher.test(chain):
                print("/".join(i.text for i in chain[0].tokens), common_parent.text, "/".join(i.text for i in chain[-1].tokens))

if __name__ == '__main__':
    main()
