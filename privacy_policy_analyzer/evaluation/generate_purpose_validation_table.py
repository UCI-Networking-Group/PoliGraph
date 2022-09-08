import argparse
import random
import os
import networkx as nx
import csv

from privacy_policy_analyzer.graph_utils import gml_destringizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-o", "--output-file", required=True, help="Output CSV file")
    parser.add_argument("-n", "--num-graph", required=True, type=int,
                        help="Number of graphs")
    parser.add_argument("-m", "--num-phrase-per-graph", required=True, type=int,
                        help="Number of phrases per graph")
    args = parser.parse_args()

    workdirs = sorted(args.workdirs)
    random.seed(0)

    fout = open(args.output_file, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=["policy_id", "text"])
    writer.writeheader()

    for d in random.sample(workdirs, k=args.num_graph):
        kgraph_path = os.path.join(d, 'graph_trimmed.gml')
        graph = nx.read_gml(kgraph_path, destringizer=gml_destringizer)

        purpose_phrases = set()

        for _, _, rel, edge_data in graph.edges(keys=True, data=True):
            if rel == "COLLECT":
                for _, text in edge_data["purposes"]:
                    purpose_phrases.add(text)

        num_edges = min(len(purpose_phrases), args.num_phrase_per_graph)

        for phrase in random.sample(list(purpose_phrases), k=num_edges):
            writer.writerow({
                "policy_id": os.path.basename(os.path.dirname(kgraph_path)),
                "text": phrase,
            })

    fout.close()


if __name__ == "__main__":
    main()
