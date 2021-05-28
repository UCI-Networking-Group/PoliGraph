import itertools
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import spacy
from spacy import displacy
from spacy.language import Language
from spacy.tokens import Token
import yaml


@Language.component(
    "merge_entities_mod",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    retokenizes=True,
)
def merge_entities_mod(doc):
    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            # discard invalid named entities
            if ent.root.pos_ not in ['NOUN', 'PROPN'] or \
               ent.root.dep_ not in ['pobj', 'dobj', 'nsubj', 'nsubjpass', 'dative', 'nmod', 'poss', 'conj', 'appos']:
                for item in ent:
                    item.ent_type_ = 'OTHER'
            else:
                attrs = {
                    "tag": ent.root.tag,
                    "dep": ent.root.dep,
                    "ent_type": ent.label,
                }

                retokenizer.merge(ent, attrs=attrs)

    return doc


def get_fixed_ent_type(token):
    if token.ent_type_ == 'DATA':
        return 'DATA'

    lemma = token.lemma_.lower()

    if token.ent_type_ == 'ORG':
        return 'ENTITY'
    elif re.match(r'^(?:third|3rd)[- ]party\b', lemma):
        return 'ENTITY'
    elif token.lemma_.lower() in ('we', 'us'):
        return 'ENTITY'
    elif token.lemma_.lower().startswith('our '):
        return 'ENTITY'
    elif token.lemma_.lower().startswith('this '):
        return 'ENTITY'

    return ""


def get_chain(root_token, left_token, right_token):
    def get_single_chain(dest):
        chain = []
        chain.append(SimpleNamespace(tokens=[dest], dep=dest.dep_, role=dest._.fixed_ent_type))

        for anc in dest.ancestors:
            if anc == root_token:
                break

            if chain[-1].dep in ['conj', 'appos']:
                chain[-1].tokens.append(anc)
                chain[-1].dep = anc.dep_
                chain[-1].role = chain[-1].role or anc._.fixed_ent_type
            elif anc.dep_ in {'nummod', 'auxpass', 'aux'}:
                continue
            else:
                chain.append(SimpleNamespace(tokens=[anc], dep=anc.dep_, role=anc._.fixed_ent_type))
        else:
            assert False

        while chain and chain[-1].dep in ['conj', 'appos']:
            chain.pop()

        return chain

    left_chain = get_single_chain(left_token)
    right_chain = get_single_chain(right_token)

    root = SimpleNamespace(tokens=[root_token], dep=root_token.dep_, role="ROOT")
    if root.dep in ['xcomp', 'ccomp']:
        root.dep = 'ROOT'

    for tok in root_token.children:
        if tok.dep_ in ['conj', 'appos']:
            root.tokens.append(tok)

    for anc in root_token.ancestors:
        if root.dep in {'conj', 'appos'}:
            root.tokens.append(anc)
            root.dep = anc.dep_
        else:
            break

    final_chain = []
    for item in itertools.chain(left_chain, [root], reversed(right_chain)):
        if item.role != 'ROOT' and item.dep in ['advmod', 'punct']:
            continue

        final_chain.append(item)

    return final_chain


def chain_to_pattern_strings(chain):
    def dfs(idx, ft=()):
        if idx == len(chain):
            yield ",".join(ft)
            return

        if chain[idx].dep in ['pobj', 'dobj', 'nsubj', 'nsubjpass', 'dative', 'nmod', 'poss']:
            yield from dfs(idx + 1, ft + (chain[idx].dep + ":" + chain[idx].role,))
        else:
            for word in {i.lemma_.lower() for i in chain[idx].tokens}:
                yield from dfs(idx + 1, ft + (chain[idx].dep + ":" + word,))

    yield from dfs(0)


def iter_entity_dtype_pairs(sent):
    entities = []
    dtypes = []

    for tok in sent:
        if tok._.fixed_ent_type == 'ENTITY':
            entities.append(tok)
        elif tok._.fixed_ent_type == 'DATA':
            dtypes.append(tok)

    for ent in entities:
        ent_anc = list(ent.ancestors)[::-1]

        for dt in dtypes:
            dt_anc = list(dt.ancestors)[::-1]
            common_parent = None

            for i1, i2 in zip(ent_anc, dt_anc):
                if i1 == i2:
                    common_parent = i1
                else:
                    break

            if common_parent is None:
                continue
            elif common_parent.pos_ != 'VERB':
                continue
            elif common_parent.dep_ in ['advcl', 'acomp', 'advmod', 'punct']:
                continue

            yield common_parent, ent, dt


class PatternMatcher:
    def __init__(self, config):
        self.token_mapping = token_mapping = dict()

        for key, tok_list in config["token_mapping"].items():
            for tok in tok_list:
                if tok not in token_mapping:
                    token_mapping[tok] = []

                token_mapping[tok].append(key)

        self.patterns = patterns = []
        for pattern_str in config["patterns"]:
            pattern = []
            for item in pattern_str.split(","):
                pa, pb = item.split(":", 1)
                pattern.append((pa, pb))

            patterns.append(tuple(pattern))

    def test(self, chain):
        valid_patterns = []
        for p in self.patterns:
            if len(p) == len(chain):
                valid_patterns.append(p)

        for idx, item in enumerate(chain):
            token_set = set()

            for t in item.tokens:
                lem = t.lemma_.lower()
                token_set.add(lem)
                if lem in self.token_mapping:
                    token_set.update(self.token_mapping[lem])

                if t._.fixed_ent_type in ['ENTITY', 'DATA']:
                    token_set.add(t._.fixed_ent_type)

            valid_patterns_next = []
            for p in valid_patterns:
                if p[idx][0] == item.dep:
                    if p[idx][1] == "" or p[idx][1] in token_set:
                        valid_patterns_next.append(p)

            if len(valid_patterns_next) == 0:
                return False
            else:
                valid_patterns = valid_patterns_next

        return True


def main():
    spacy.prefer_gpu()
    nlp_dir = sys.argv[1]
    yml_dir = Path(sys.argv[2])

    with open("pattern.yml") as fin:
        pattern_config = yaml.safe_load(fin)
        pattern_matcher = PatternMatcher(pattern_config)

    nlp = spacy.load("en_core_web_trf", exclude=["ner"])
    nlp_ner = spacy.load(nlp_dir, nlp.vocab)
    nlp.add_pipe("ner", source=nlp_ner)
    nlp.add_pipe("merge_entities_mod", after="ner")
    nlp.add_pipe("merge_noun_chunks")
    Token.set_extension("fixed_ent_type", getter=get_fixed_ent_type)

    for yml_path in yml_dir.glob('*.yml'):
        data = yaml.safe_load(yml_path.open())

        for segment in data['segments']:
            text = segment["segment_text"]
            text = 'We provide customer lists, email addresses and other information identified above to third party companies and contractors'
            doc = nlp(text)
            import pdb
            pdb.set_trace()

            for sent in doc.sents:
                for common_parent, ent, dt in iter_entity_dtype_pairs(sent):
                    chain = get_chain(common_parent, ent, dt)

                    if not pattern_matcher.test(chain):
                        print(sent)
                        print(common_parent, ent, dt, sep=",")
                        for feature in chain_to_pattern_strings(chain):
                            print('-', feature)


if __name__ == "__main__":
    main()
