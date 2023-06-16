import argparse
from collections import Counter
import csv
import os

import networkx as nx
from poligrapher.annotators import CollectionAnnotator
from poligrapher.graph_utils import yaml_load_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-o", "--output-path", required=True, help="Output edge stat path")
    args = parser.parse_args()

    non_empty_kgraph_count = 0
    subsum_edge_count = 0
    subsum_sentence_count = 0
    collect_edge_count = 0
    collect_sentence_count = 0
    collect_edge_with_purposes_count = 0
    purpose_phrase_count = 0

    has_we_collect = set()
    has_unspecific = set()

    edge_statistics = Counter()
    edge_purpose_statistics = Counter()

    for d in args.workdirs:
        print(f"Processing {d} ...")

        kgraph_path = os.path.join(d, 'graph-original.yml')
        with open(kgraph_path, encoding="utf-8") as fin:
            graph = yaml_load_graph(fin)

        subsum_sentences = set()
        collect_sentences = set()
        purpose_phrases = set()

        if not nx.is_empty(graph):
            non_empty_kgraph_count += 1

        for u, v, rel, data in graph.edges(keys=True, data=True):
            edge_statistics[(u, v, rel)] += 1

            if rel == "SUBSUM":
                subsum_edge_count += 1
                subsum_sentences.update(data["text"])
            elif rel in CollectionAnnotator.EDGE_TYPES:
                collect_edge_count += 1
                collect_sentences.update(data["text"])

                for phrases in data["purposes"].values():
                    purpose_phrases.update(phrases)

                if data["purposes"]:
                    collect_edge_with_purposes_count += 1
                    edge_purpose_statistics[(u, v, rel)] += 1

                if u == "we":
                    has_we_collect.add(d)

                if u == "UNSPECIFIED_ACTOR" or v == "UNSPECIFIED_ENTITY":
                    has_unspecific.add(d)

        subsum_sentence_count += len(subsum_sentences)
        collect_sentence_count += len(collect_sentences)
        purpose_phrase_count += len(purpose_phrases)

    print("non-empty graphs: ", non_empty_kgraph_count)
    print("SUBSUM:", subsum_edge_count, subsum_sentence_count)
    print("COLLECT:", collect_edge_count, collect_sentence_count)
    print("PURPOSE", collect_edge_with_purposes_count, purpose_phrase_count)
    print("# we COLLECT:", len(has_we_collect))
    print("# UNSPECIFIED:", len(has_unspecific))

    with open(args.output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["u", "v", "rel", "count", "purpose_count"])
        writer.writeheader()

        for (u, v, rel), count in edge_statistics.most_common():
            writer.writerow({
                "u": u,
                "v": v,
                "rel": rel,
                "count": count,
                "purpose_count": edge_purpose_statistics[(u, v, rel)]
            })


if __name__ == "__main__":
    main()
