import importlib.resources as pkg_resources
from types import SimpleNamespace

import yaml

import privacy_policy_analyzer


def discover_entities(token):
    def dfs(token, path):
        if token.dep_ in ["conj", "appos"]:
            for child in token.children:
                yield from dfs(child, path)
        elif token._.ent is None:
            for child in token.children:
                yield from dfs(child, (token,) + path)
        else:
            yield [token, *path]

    yield from dfs(token, ())


def get_first_head(token):
    while token.dep_ in ["conj", "appos"]:
        token = token.head
    return token


def get_conjuncts(token):
    ret = []

    def dfs(current_token):
        ret.append(current_token)

        for child in current_token.children:
            if child.dep_ in ["conj", "appos"]:
                dfs(child)

    dfs(token)
    return ret


class Pattern:
    def __init__(self, pattern, token_mapping):
        self.pattern = []
        for item in pattern.split(","):
            lemma, *properties = item.split(":")
            token_specs = SimpleNamespace(lemma=lemma, dep="*", pos="*")

            for prop in properties:
                if prop.islower():
                    token_specs.dep = prop
                else:
                    token_specs.pos = prop

            self.token_mapping = token_mapping
            self.pattern.append(token_specs)

    def __call__(self, chain):
        def test_lemma(token, lemma_spec):
            if lemma_spec in ["*", "ENTITY", "DATA"]:
                return True

            if lemma_spec in self.token_mapping:
                for lemma in self.token_mapping[lemma_spec]:
                    if test_lemma(token, lemma):
                        return True
            else:
                if token.lemma_ == lemma_spec:
                    return True

            return False

        def test(token, specs):
            p = token
            while p.dep_ in ["conj", "appos"]:
                p = p.head

            if not test_lemma(token, specs.lemma):
                return False

            if specs.dep == "root_like":
                if any(t.pos_ == "VERB" for t in p.ancestors):
                    return False
            elif specs.dep not in ["*", p.dep_.lower()]:
                return False

            if specs.pos not in ["*", token.pos_]:
                return False

            return True

        if len(chain) != len(self.pattern):
            return False

        for conj_tokens, specs in zip(chain, self.pattern):
            for token in conj_tokens:
                if test(token, specs):
                    break
            else:
                return False

        return True


class CollectionAnnotator:
    def __init__(self, nlp):
        with pkg_resources.open_text(privacy_policy_analyzer, "collection_patterns.yml") as fin:
            config = yaml.safe_load(fin)

        self.patterns = []

        for p in config["patterns"]:
            self.patterns.append(Pattern(p, config["token_mapping"]))

    def annotate(self, doc):
        possible_datatypes = set()

        for e in doc.ents:
            if e.label_ == "DATA":
                possible_datatypes.add(get_first_head(e.root))

        for datatype_token in possible_datatypes:
            datatype_chain = [datatype_token]
            current_token = datatype_token

            while current_token.dep_ != "ROOT" and not current_token.is_space:
                parent_token = get_first_head(current_token.head)
                datatype_chain.append(parent_token)

                if parent_token.pos_ == "VERB":
                    for child in parent_token.children:
                        if child == current_token:
                            continue

                        for entity_chain in discover_entities(child):
                            full_chain = [get_conjuncts(t) for t in entity_chain]
                            full_chain.extend(get_conjuncts(t) for t in datatype_chain[::-1])

                            if any(p(full_chain) for p in self.patterns):
                                entity_tokens, *intermediate_chain, datatype_tokens = full_chain

                                entities = [t._.ent for t in entity_tokens]
                                datatypes = [t._.ent for t in datatype_tokens]

                                print("#" * 40)
                                print(current_token.sent, end="\n\n")
                                print(f"> DEP: {entity_tokens[0].dep_}, POS: {entity_tokens[0].pos_}, ENTITIES:", entities)
                                for item in intermediate_chain:
                                    print(f"> DEP: {item[0].dep_}, POS: {item[0].pos_}, TOKEN:", [t.text for t in item])
                                print(f"> DEP: {datatype_tokens[0].dep_}, POS: {datatype_tokens[0].pos_}, DATATYPES:", datatypes)
                                print("#" * 40)

                                doc.user_data["document"].link(datatype_tokens[0], entity_tokens[0], "COLLECTED_BY")

                current_token = parent_token
