import importlib.resources as pkg_resources
from types import SimpleNamespace

import yaml

import privacy_policy_analyzer


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

            if specs.dep not in ["*", p.dep_.lower()]:
                return False

            if specs.pos not in ["*", token.pos_]:
                return False

            return True

        i = 0
        for specs in self.pattern:
            if i == len(chain):
                return False

            current_token = chain[i]
            conj_tokens = [current_token]

            if current_token.dep_ in ["conj", "appos"]:
                while current_token.dep_ in ["conj", "appos"]:
                    current_token = current_token.head
                    conj_tokens.append(current_token)

                    if current_token == chain[i + 1]:
                        i += 1
            else:
                while i + 1 < len(chain) and chain[i + 1].dep_ in ["conj", "appos"]:
                    i += 1
                    current_token = chain[i]
                    conj_tokens.append(current_token)

            for token in conj_tokens:
                if test(token, specs):
                    break
            else:
                return False

            i += 1

        return i == len(chain)


class CollectionAnnotator:
    def __init__(self, nlp):
        with pkg_resources.open_text(privacy_policy_analyzer, "collection_patterns.yml") as fin:
            config = yaml.safe_load(fin)

        self.patterns = []

        for p in config["patterns"]:
            self.patterns.append(Pattern(p, config["token_mapping"]))

    def annotate(self, doc):
        policy_document = doc.user_data["document"]

        def like_data(tok):
            """Check if the phrase started by the token is a data type or subsums a data type"""
            # Use BFS here to avoid a loop.
            bfs_queue = [tok]
            seen = {tok}
            i = 0

            while i < len(bfs_queue):
                tok = bfs_queue[i]
                i += 1

                if tok.ent_type_ == "DATA":
                    return True
                elif tok.ent_type_ == "NN":
                    for linked_token, relationship in policy_document.get_links(tok):
                        if relationship in ["SUBSUM", "COREF"] and linked_token not in seen:
                            bfs_queue.append(linked_token)
                            seen.add(linked_token)

            return False

        dtype_candidates = []
        actor_candidates = []

        for e in doc.ents:
            if like_data(e[0]):
                dtype_candidates.append((e, e.root, list(e.root.ancestors)[::-1]))
            else:
                actor_candidates.append((e, e.root, list(e.root.ancestors)[::-1]))

        for dtype, dtype_root, dtype_ancestors in dtype_candidates:
            for actor, actor_root, actor_ancestors in actor_candidates:
                if dtype_root.is_ancestor(actor_root) or actor_root.is_ancestor(dtype_root):
                    continue

                # Remove common ancestors
                common_ancestor_count = 0
                for a, b in zip(dtype_ancestors, actor_ancestors):
                    if a == b:
                        common_ancestor_count += 1
                    else:
                        break

                if common_ancestor_count == 0:
                    # Skip if not in the same sentence
                    continue

                # Skip if actor and dtype are not connected by a verb
                verb = dtype_ancestors[common_ancestor_count - 1]
                if verb.pos_ != "VERB":
                    continue

                full_chain = [actor_root]
                full_chain.extend(actor_ancestors[-1:common_ancestor_count-1:-1])
                full_chain.append(verb)
                full_chain.extend(dtype_ancestors[common_ancestor_count::])
                full_chain.append(dtype_root)

                if any(p(full_chain) for p in self.patterns):
                    negation = False

                    for token in full_chain:
                        for child in token.children:
                            if child.dep_ == "neg":
                                negation = True
                                break

                    print("#" * 40)
                    print(verb.sent, end="\n\n")

                    if not negation:
                        print(">", actor, verb.lemma_.upper(), dtype)
                        policy_document.link(actor[0], dtype[0], "COLLECT")
                        policy_document.link(dtype[0], actor[0], "COLLECTED_BY")
                    else:
                        print("> (NEG)", actor, verb.lemma_.upper(), dtype)

                    print("#" * 40)
