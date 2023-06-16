import argparse
import csv
import os
import random
import itertools

from poligrapher.graph_utils import yaml_load_graph


def get_existing_data(graph, existing_data):
    for row in existing_data:
        new_u = u = row["u"]
        new_v = v = row["v"]
        rel = row["rel"]

        if u not in graph or v not in graph:
            candidate_u = []
            candidate_v = []

            for node in graph.nodes:
                if u.lower() in node:
                    candidate_u.append(node)

                if v.lower() in node:
                    candidate_v.append(node)

            for cu, cv in itertools.product(candidate_u, candidate_v):
                if graph.has_edge(cu, cv, rel):
                    new_u, new_v = cu, cv
                    break

        if graph.has_edge(new_u, new_v, rel):
            row["u"] = new_u
            row["v"] = new_v

            edge_data = graph.get_edge_data(new_u, new_v, rel)
            row["text"] = "\n".join(edge_data["text"])

            yield (new_u, new_v, rel), row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-b", "--base", default="", help="Existing data to copy")
    parser.add_argument("-o", "--output-file", required=True, help="Output CSV file")
    parser.add_argument("-m", "--num-edge-per-graph", required=True, type=int,
                        help="Number of edges per graph")
    args = parser.parse_args()

    workdirs = sorted(args.workdirs)
    random.seed(0)

    if args.base:
        with open(args.base, "r", encoding="utf-8", newline="") as fin:
            base_csv_rows = list(csv.DictReader(fin))
    else:
        base_csv_rows = []

    with open(args.output_file, "w", encoding="utf-8", newline="") as fout:
        fieldnames = ["policy_id", "rel", "u", "v", "validation", "text"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for d in workdirs:
            app_id = os.path.basename(os.path.realpath(d))

            with open(os.path.join(d, 'graph-original.yml'), encoding="utf-8") as fin:
                graph = yaml_load_graph(fin)

            selected_data = dict(get_existing_data(graph, filter(lambda t: t["policy_id"] == app_id, base_csv_rows)))
            all_edges = []

            for u, v, rel in graph.edges(keys=True):
                if (u, v, rel) not in selected_data:
                    all_edges.append((u, v, rel))

            num_new_edges = max(0, min(args.num_edge_per_graph - len(selected_data), len(all_edges)))

            for u, v, rel in random.sample(all_edges, k=num_new_edges):
                edge_data = graph.get_edge_data(u, v, rel)

                selected_data[u, v, rel] = {
                    "policy_id": app_id,
                    "rel": rel,
                    "u": u,
                    "v": v,
                    "validation": "",
                    "text": "\n".join(edge_data["text"])
                }

            for row in selected_data.values():
                writer.writerow({n: row[n] for n in fieldnames})


if __name__ == "__main__":
    main()
