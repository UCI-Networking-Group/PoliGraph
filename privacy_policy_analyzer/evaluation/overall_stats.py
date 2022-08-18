import argparse
from collections import Counter
import csv
import os

import networkx as nx
from privacy_policy_analyzer.graph_utils import KGraph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-o", "--output-dir", required=True, help="Output dir")
    args = parser.parse_args()

    empty_kgraph_count = 0
    subsum_edge_count = 0
    subsum_sentence_count = 0
    collect_edge_count = 0
    collect_sentence_count = 0
    collect_edge_with_purposes_count = 0
    purpose_sentence_count = 0

    has_we_collect = set()
    has_unspecific = set()

    edge_statistics = Counter()
    edge_purpose_statistics = Counter()

    for d in args.workdirs:
        print(f"Processing {d} ...")

        kgraph_path = os.path.join(d, 'graph_trimmed.gml')
        kgraph = KGraph(kgraph_path)
        subsum_sentences = set()
        collect_sentences = set()
        purpose_sentences = set()

        if nx.is_empty(kgraph.kgraph):
            empty_kgraph_count += 1

        for u, v, rel, data in kgraph.kgraph.edges(keys=True, data=True):
            edge_statistics[(u, v, rel)] += 1

            if rel == "SUBSUM":
                subsum_edge_count += 1
                subsum_sentences.update(data["text"])
            elif rel == "COLLECT":
                collect_edge_count += 1
                collect_sentences.update(data["text"])
                purpose_sentences.update(p[1] for p in data["purposes"])

                if len(data["purposes"]) > 0:
                    collect_edge_with_purposes_count += 1
                    edge_purpose_statistics[(u, v, rel)] += 1

                if u == "we":
                    has_we_collect.add(d)

                if u == "UNSPECIFIC_ACTOR" or v == "UNSPECIFIC_ENTITY":
                    has_unspecific.add(d)

        subsum_sentence_count += len(subsum_sentences)
        collect_sentence_count += len(collect_sentences)
        purpose_sentence_count += len(purpose_sentences)

    print("empty graphs: ", empty_kgraph_count)
    print("SUBSUM:", subsum_edge_count, subsum_sentence_count)
    print("COLLECT:", collect_edge_count, collect_sentence_count)
    print("PURPOSE", collect_edge_with_purposes_count, purpose_sentence_count)
    print("# we COLLECT:", len(has_we_collect))
    print("# UNSPECIFIC:", len(has_unspecific))

    with open(os.path.join(args.output_dir, "edge_stats.csv"), "w", newline="") as fout:
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
