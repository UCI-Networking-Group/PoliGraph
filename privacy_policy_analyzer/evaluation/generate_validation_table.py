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
    parser.add_argument("-m", "--num-edge-per-graph", required=True, type=int,
                        help="Number of edges per graph")
    args = parser.parse_args()

    workdirs = sorted(args.workdirs)
    random.seed(0)

    fout = open(args.output_file, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=["policy_id", "rel", "u", "v", "text"])
    writer.writeheader()

    for d in random.sample(workdirs, k=args.num_graph):
        kgraph_path = os.path.join(d, 'graph_trimmed.gml')
        graph = nx.read_gml(kgraph_path, destringizer=gml_destringizer)

        all_edges = []

        for u, v, rel in graph.edges(keys=True):
            if rel in ["COLLECT", "SUBSUM"]:
                all_edges.append([u, v, rel])

        num_edges = min(len(all_edges), args.num_edge_per_graph)

        for u, v, rel in random.sample(all_edges, k=num_edges):
            edge_data = graph[u][v][rel]

            writer.writerow({
                "policy_id": os.path.basename(os.path.dirname(kgraph_path)),
                "rel": rel,
                "u": u,
                "v": v,
                "text": "\n".join(edge_data["text"])
            })

    fout.close()


if __name__ == "__main__":
    main()
