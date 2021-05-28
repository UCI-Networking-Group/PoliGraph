import re
import itertools
import pickle
import sys
import yaml
import spacy
from collections import defaultdict
from pathlib import Path

from spacy.tokens import Token
from pattern_core import PatternMatcher, get_fixed_ent_type, iter_entity_dtype_pairs, get_chain


def main():
    spacy.prefer_gpu()

    nlp_dir = sys.argv[1]
    pl_pickle_dir = Path(sys.argv[2])

    with open("pattern.yml") as fin:
        pattern_config = yaml.safe_load(fin)
        pattern_matcher = PatternMatcher(pattern_config)

    nlp = spacy.load("en_core_web_trf", exclude=["ner"])
    nlp_ner = spacy.load(nlp_dir, nlp.vocab)
    nlp.add_pipe("ner", source=nlp_ner)
    nlp.add_pipe("merge_entities_mod", after="ner")
    nlp.add_pipe("merge_noun_chunks")
    Token.set_extension("fixed_ent_type", getter=get_fixed_ent_type)

    cat_table = defaultdict(list)
    with open("category.yml") as f:
        for cat, pattern_list in yaml.safe_load(f).items():
            for pattern in pattern_list:
                regex = re.compile(r'\b' + pattern + r'\b', flags=re.I)
                cat_table[regex].append(cat)


    for pickle_path in pl_pickle_dir.glob('*.pickle'):
        with pickle_path.open(mode="rb") as fin:
            statements = pickle.load(fin, encoding='utf-8')

        sent_to_pl = dict()
        for entity, collect_flag, dtype, sent, verb in statements:
            if sent not in sent_to_pl:
                sent_to_pl[sent] = set()

            if entity.endswith('implicit'):
                continue

            if entity == 'us':
                entity = 'we'

            for pattern, cat_list in cat_table.items():
                if pattern.search(dtype):
                    for cat in cat_list:
                        sent_to_pl[sent].add((entity, cat))

        for sent, pl_results in sent_to_pl.items():
            doc = nlp(sent)
            our_results = set()

            for sent in doc.sents:
                for common_parent, ent, dt in iter_entity_dtype_pairs(sent):
                    chain = get_chain(common_parent, ent, dt)

                    if pattern_matcher.test(chain):
                        for entity, dtype in itertools.product(chain[0].tokens, chain[-1].tokens):
                            for pattern, cat_list in cat_table.items():
                                if pattern.search(dtype.lemma_):
                                    for cat in cat_list:
                                        our_results.add((entity.lemma_.lower(), cat))

            #if pl_results and not our_results:
            #    print(pl_results)
            if pl_results != our_results:
                print(pl_results, our_results)
            elif pl_results:
                print('GOOD')



if __name__ == '__main__':
    main()
