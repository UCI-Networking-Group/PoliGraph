import argparse
import random
import os
import spacy
import logging
from collections import Counter, defaultdict
import networkx as nx
import csv
import re
from unidecode import unidecode

from privacy_policy_analyzer.document import PolicyDocument, SegmentType

from privacy_policy_analyzer.graph_utils import KGraph, gml_destringizer


def fix_text(text):
    # Tokenizer just doesn't work...
    text = re.sub(r"\be-mail\b", "email", text, flags=re.I)
    text = re.sub(r"\bwi-fi\b", "WiFi", text, flags=re.I)
    text = re.sub(r"\bid\b", "ID", text)
    return unidecode(text)


class KGraphValidationHelper:
    def __init__(self, workdir, nlp):
        gml_path = os.path.join(workdir, "graph.gml")

        self.workdir = workdir
        self.kgraph = kgraph = KGraph(gml_path)

        self.document = PolicyDocument.load(workdir, nlp)
        self.src_to_term = src_to_term = defaultdict(set)

        for u, v, _, edge_data in kgraph.kgraph.edges(keys=True, data=True):
            for u_src, v_src in edge_data["sources"]:
                src_to_term[tuple(u_src)].add(u)
                src_to_term[tuple(v_src)].add(v)

    def validate_edge(self, rel, u, v, text):
        def find_substr(s, sub):
            left = right = 0

            while True:
                left = s.find(sub, right)

                if left == -1:
                    return

                right = left + len(sub)

                yield left, right

        text = fix_text(text)
        u = fix_text(u)
        v = fix_text(v)

        all_sents = [s.strip() for s in text.split("|")]
        all_docs = []

        for segment in self.document.segments:
            if segment.segment_type == SegmentType.LISTITEM:
                continue

            doc = self.document.get_doc_without_context(segment)

            while all_sents and all_sents[0].replace(" ", "") in doc.text.replace(" ", ""):
                all_sents.pop(0)
                all_docs.append(doc)

            if len(all_sents) == 0:
                break

        if len(all_sents) > 0:
            logging.warning("Text %r not found", all_sents[0])

        u_terms = set()
        v_terms = set()

        for doc in all_docs:
            for left, right in find_substr(doc.text.lower(), u.lower()):
                for token in doc.char_span(left, right, alignment_mode="expand"):
                    token_src = token._.src

                    if token_src in self.src_to_term:
                        u_terms.update(self.src_to_term[token_src])

            for left, right in find_substr(doc.text.lower(), v.lower()):
                for token in doc.char_span(left, right, alignment_mode="expand"):
                    token_src = token._.src

                    if token_src in self.src_to_term:
                        v_terms.update(self.src_to_term[token_src])

        logging.info("u=%r, possible terms: %r", u, u_terms)
        logging.info("v=%r, possible terms: %r", v, v_terms)

        if len(u_terms) == 0 or len(v_terms) == 0:
            logging.warning("Empty u / v term list")
            return False

        for u_t in u_terms:
            for v_t in v_terms:
                if rel == "SUBSUM":
                    if self.kgraph.subsum(u_t, v_t):
                        return True
                elif rel == "COLLECT":
                    if any(e == u_t for e in self.kgraph.who_collect(v_t)):
                        return True

        return False


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp", required=True, help="NLP model directory")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("rootdir", help="Root of workdirs")
    args = parser.parse_args()

    nlp = spacy.load(args.nlp)
    total_counter = Counter()
    tp_counter = Counter()

    with open(args.input_csv, encoding="utf-8") as fin:
        policy_id = None

        for row in csv.DictReader(fin):
            if row["policy_id"]:
                policy_id = row["policy_id"]
                current_text = None

                workdir = os.path.join(args.rootdir, policy_id)
                helper = KGraphValidationHelper(workdir, nlp)

            edge_rel = row["rel"]
            edge_from = row["u"]
            edge_to = row["v"]

            if edge_from == edge_to == "":
                continue

            if row["text"] not in ["", "-"]:
                current_text = row["text"].strip()

            total_counter[edge_rel] += 1

            if helper.validate_edge(edge_rel, edge_from, edge_to, current_text):
                tp_counter[edge_rel] += 1

    print(total_counter["COLLECT"], tp_counter["COLLECT"])
    print(total_counter["SUBSUM"], tp_counter["SUBSUM"])


if __name__ == "__main__":
    main()
