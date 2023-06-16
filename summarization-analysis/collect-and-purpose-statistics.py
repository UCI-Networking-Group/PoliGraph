import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path

import networkx as nx
import pandas as pd

from poligrapher.graph_utils import KGraph, load_ontologies

DATATYPE_CATEGORIES = [
    "UNSPECIFIED_DATA",
    #"device identifier",
    "software identifier",
    "hardware identifier",
    "geolocation",
    "government identifier",
    "contact information",
    "protected classification",
    "biometric information",
    "internet activity",
]

ENTITY_CATEGORIES = [
    "we",

    "UNSPECIFIED_ACTOR",

    "advertiser",
    "analytic provider",
    "auth provider",
    "content provider",
    "email service provider",
    "social media",
]

PURPOSE_LABELS = [
    'services',
    'security',
    'legal',
    'advertising',
    'analytics',
]

NON_CORE_PURPOSES = [
    'advertising',
    'analytics',
]

class ParallelHelper:
    def __init__(self, data_ontology, entity_ontology):
        self.data_ontology = data_ontology
        self.entity_ontology = entity_ontology

    def run_on_graph(self, kgraph):
        result = {}

        for data_cat in DATATYPE_CATEGORIES:
            first_party_purposes = set()
            third_party_purposes = set()
            collect_entities = set()

            child_data_types = [data_cat]
            if data_cat != "UNSPECIFIED_DATA":
                child_data_types.extend(nx.descendants(self.data_ontology, data_cat))

            for data_type in child_data_types:
                for entity in kgraph.who_collect(data_type):
                    # First-party
                    if kgraph.subsum("we", entity):
                        first_party_purposes.update(kgraph.purposes(entity, data_type))
                        collect_entities.add("we")
                        continue

                    third_party_purposes.update(kgraph.purposes(entity, data_type))

                    # Third-party category name
                    if entity in ENTITY_CATEGORIES:
                        collect_entities.add(entity)
                        continue

                    # Guess category based on the graph
                    entity_cats = set()

                    for anc in kgraph.ancestors(entity):
                        if anc in ENTITY_CATEGORIES:
                            entity_cats.add(anc)

                    # Guess category based on the ontology
                    if len(entity_cats) == 0 and entity in self.entity_ontology.nodes():
                        parents = {ent_cat for ent_cat, _ in self.entity_ontology.in_edges(entity)}
                        parents.discard("UNKNOWN")

                        if len(parents) == 1:
                            entity_cats.add(parents.pop())

                    collect_entities.update(entity_cats)

            if len(collect_entities) > 0:
                result[data_cat] = {
                    "entities": sorted(collect_entities),
                    "purposes": sorted(first_party_purposes | third_party_purposes),
                    "first_party_purposes": sorted(first_party_purposes),
                    "third_party_purposes": sorted(third_party_purposes),
                }

        return result

    def run(self, privacy_policy_path):
        kgraph_path = os.path.join(privacy_policy_path, 'graph-original.yml')
        kgraph = KGraph(kgraph_path)

        return self.run_on_graph(kgraph)


def worker(input, output):
    data_ontology, entity_ontology = load_ontologies()
    helper = ParallelHelper(data_ontology, entity_ontology)

    for path in iter(input.get, None):
        kgraph_result = helper.run(path)
        output.put((path, kgraph_result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-o", "--output-dir", required=True, help="Output dir")
    args = parser.parse_args()

    nproc = mp.cpu_count()
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    for _ in range(nproc):
        mp.Process(target=worker, args=(input_queue, output_queue)).start()

    for path in args.workdirs:
        input_queue.put(path)

    entity_stats = pd.DataFrame(0, DATATYPE_CATEGORIES,
                                ["total"] + ENTITY_CATEGORIES)
    purpose_stats = pd.DataFrame(0, DATATYPE_CATEGORIES, PURPOSE_LABELS)

    apps_sharing = {d: set() for d in DATATYPE_CATEGORIES}
    non_core_purpose_apps = {d: set() for d in DATATYPE_CATEGORIES}
    set_policy_collect_known_category = set()
    set_policy_sharing_other_data = set()
    set_policy_non_core_purpose = set()
    set_policy_ad_purpose = set()

    for _ in range(len(args.workdirs)):
        path, kgraph_result = output_queue.get()

        for data_cat in kgraph_result:
            if data_cat != "UNSPECIFIED_DATA":
                set_policy_collect_known_category.add(path)

            entity_stats.loc[data_cat, "total"] += 1

            for ent_cat in kgraph_result[data_cat]["entities"]:
                entity_stats.loc[data_cat, ent_cat] += 1

                if ent_cat not in ["we", "UNSPECIFIED_ACTOR"]:
                    apps_sharing[data_cat].add(path)

                    if data_cat not in ["software identifier", "UNSPECIFIED_DATA"]:
                        set_policy_sharing_other_data.add(path)

            for purpose in kgraph_result[data_cat]["purposes"]:
                purpose_stats.loc[data_cat, purpose] += 1

                if purpose in NON_CORE_PURPOSES:
                    non_core_purpose_apps[data_cat].add(path)

                    set_policy_non_core_purpose.add(path)
                    if purpose == "advertising":
                        set_policy_ad_purpose.add(path)

    for _ in range(nproc):
        input_queue.put(None)

    output_dir = Path(args.output_dir)

    entity_stats.to_csv(output_dir / "entity_stats.csv")
    purpose_stats.to_csv(output_dir / "purpose_stats.csv")

    with open(output_dir / "app_num_stats.csv", "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["data_category", "sharing", "non_core_purpose"])
        writer.writeheader()

        for data_cat in DATATYPE_CATEGORIES:
            writer.writerow({
                "data_category": data_cat,
                "sharing": len(apps_sharing[data_cat]),
                "non_core_purpose": len(non_core_purpose_apps[data_cat]),
            })

    print("# of policies that disclose the collection of known categories:", len(set_policy_collect_known_category))
    print("# of policies that disclose to share categories other than software identifiers:",
          len(set_policy_sharing_other_data))
    print("# of policies that disclose non-core usage purposes:", len(set_policy_non_core_purpose))
    print("# of policies that disclose advertising purpose:", len(set_policy_ad_purpose))


if __name__ == "__main__":
    main()
