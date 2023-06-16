#!/usr/bin/env python3

import argparse
import json
import logging
import os

import networkx as nx

from poligrapher.document import PolicyDocument
from poligrapher.utils import setup_nlp_pipeline


def main():
    logging.basicConfig(format='%(asctime)s [%(levelname)s] <%(name)s> %(message)s', level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_input", help="JSONL dataset")
    parser.add_argument("rootdir", help="Root of workdirs")
    parser.add_argument("--nlp", default="", help="NLP model directory")
    args = parser.parse_args()

    nlp = setup_nlp_pipeline(args.nlp)
    accurate_count = partial_count = incorrect_count = 0

    with open(args.jsonl_input, encoding="utf-8") as fin:
        current_app_id = None
        doc_map = {}

        for line in fin:
            sample = json.loads(line)
            app_id = sample["app_id"]

            if app_id != current_app_id:
                logging.info("Processing %s ...", app_id)

                document = PolicyDocument.load(os.path.join(args.rootdir, app_id), nlp)
                current_app_id = app_id
                doc_map.clear()

                for seg in document.segments:
                    try:
                        doc = document.get_doc_with_context(seg)
                    except KeyError:
                        continue
                    else:
                        doc_map[doc.text] = doc

            text = sample["text"]
            doc = doc_map[text]

            predicted_coref_graph = nx.DiGraph()
            predicted_coref_src_list = set()
            src_list = {t._.src for t in doc}

            for u, v, rel in document.token_relationship.edges(keys=True):
                # Also include subsumptions as conceptually coreferences still "refer" to what they subsume
                if u in src_list and v in src_list:
                    if rel == "SUBSUM":
                        predicted_coref_graph.add_edge(u, v)
                    elif rel == "COREF":
                        predicted_coref_src_list.add(u)
                        predicted_coref_graph.add_edge(u, v)
                        predicted_coref_graph.add_edge(v, u)

            ent_root_list = {}

            for ent_info in sample["entities"]:
                ent = doc.char_span(ent_info["start_offset"], ent_info["end_offset"], alignment_mode="expand")
                ent_root_list[ent_info["id"]] = ent.root

            coref_link_status = {}

            for rel_info in sample["relations"]:
                token1 = ent_root_list[rel_info["from_id"]]
                src1 = token1._.src
                token2 = ent_root_list[rel_info["to_id"]]
                src2 = token2._.src

                coref_link_status.setdefault(src1, [])

                if (predicted_coref_graph.has_node(src1) and predicted_coref_graph.has_node(src2)
                    and nx.has_path(predicted_coref_graph, src1, src2)):
                    coref_link_status[src1].append(True)
                else:
                    coref_link_status[src1].append(False)

            # Any extra labels are count as incorrect
            for src in predicted_coref_src_list:
                if src not in coref_link_status:
                    coref_link_status[src] = [False]

            for flags in coref_link_status.values():
                if all(flags):
                    accurate_count += 1
                elif any(flags):
                    partial_count += 1
                else:
                    incorrect_count += 1

    print("Correct:", accurate_count)
    print("Partially correct:", partial_count)
    print("Incorrect:", incorrect_count)


if __name__ == "__main__":
    main()
