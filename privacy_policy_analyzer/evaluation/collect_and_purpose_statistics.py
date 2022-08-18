import argparse
import csv
import multiprocessing as mp
import os
from pathlib import Path

import networkx as nx
import pandas as pd

from privacy_policy_analyzer.graph_utils import ExtKGraph, KGraph, load_ontologies

DATATYPE_CATEGORIES = [
    "UNSPECIFIC_DATA",

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

    "UNSPECIFIC_ACTOR",

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
    'personalization',
    'legal',
    'acquisition',
    'advertising',
    'analytics',
    'marketing',
]

NON_CORE_PURPOSES = [
    'analytics',
    'advertising',
    'marketing',
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
            if data_cat != "UNSPECIFIC_DATA":
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
        kgraph_path = os.path.join(privacy_policy_path, 'graph_trimmed.gml')

        kgraph = KGraph(kgraph_path)
        ext_kgraph = ExtKGraph(kgraph_path, self.data_ontology, self.entity_ontology)

        return self.run_on_graph(kgraph), self.run_on_graph(ext_kgraph)


def worker(args, input, output):
    data_ontology, entity_ontology = load_ontologies(args.ontology, args.entity_info)
    helper = ParallelHelper(data_ontology, entity_ontology)

    for path in iter(input.get, None):
        kgraph_result, ext_kgraph_result = helper.run(path)
        output.put((path, kgraph_result, ext_kgraph_result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workdirs", nargs="+", help="Input directories")
    parser.add_argument("-y", "--ontology", required=True, help="Ontology directory")
    parser.add_argument("-e", "--entity-info", required=True, help="Path to entity_info.json")
    parser.add_argument("-o", "--output-dir", required=True, help="Output dir")
    args = parser.parse_args()

    nproc = mp.cpu_count()
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    for _ in range(nproc):
        mp.Process(target=worker, args=(args, input_queue, output_queue)).start()

    for path in args.workdirs:
        input_queue.put(path)

    entity_stats = pd.DataFrame(0, DATATYPE_CATEGORIES,
                                ["total"] + ENTITY_CATEGORIES)
    purpose_stats = pd.DataFrame(0, DATATYPE_CATEGORIES, PURPOSE_LABELS)
    ext_entity_stats = entity_stats.copy()
    ext_purpose_stats = purpose_stats.copy()

    apps_sharing_with_ad_or_analytics = {d: set() for d in DATATYPE_CATEGORIES}
    ext_apps_sharing_with_ad_or_analytics = {d: set() for d in DATATYPE_CATEGORIES}
    non_core_purpose_apps = {d: set() for d in DATATYPE_CATEGORIES}
    ext_non_core_purpose_apps = {d: set() for d in DATATYPE_CATEGORIES}

    for _ in range(len(args.workdirs)):
        path, kgraph_result, ext_kgraph_result = output_queue.get()

        for data_cat in kgraph_result:
            entity_stats.loc[data_cat, "total"] += 1

            for ent_cat in kgraph_result[data_cat]["entities"]:
                entity_stats.loc[data_cat, ent_cat] += 1

                if ent_cat in ["advertiser", "analytic provider"]:
                    apps_sharing_with_ad_or_analytics[data_cat].add(path)

            for purpose in kgraph_result[data_cat]["purposes"]:
                purpose_stats.loc[data_cat, purpose] += 1

                if purpose in NON_CORE_PURPOSES:
                    non_core_purpose_apps[data_cat].add(path)

            for ent_cat in ext_kgraph_result[data_cat]["entities"]:
                ext_entity_stats.loc[data_cat, ent_cat] += 1

                if ent_cat in ["advertiser", "analytic provider"]:
                    ext_apps_sharing_with_ad_or_analytics[data_cat].add(path)

            for purpose in ext_kgraph_result[data_cat]["purposes"]:
                ext_purpose_stats.loc[data_cat, purpose] += 1

                if purpose in NON_CORE_PURPOSES:
                    ext_non_core_purpose_apps[data_cat].add(path)

    for _ in range(nproc):
        input_queue.put(None)

    output_dir = Path(args.output_dir)

    entity_stats.to_csv(output_dir / "entity_stats.csv")
    ext_entity_stats.to_csv(output_dir / "ext_entity_stats.csv")
    purpose_stats.to_csv(output_dir / "purpose_stats.csv")
    ext_purpose_stats.to_csv(output_dir / "ext_purpose_stats.csv")

    with open(output_dir / "app_num_stats.csv", "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["data_category", "bad_sharing", "ext_bad_sharing",
                                                  "non_core_purpose", "ext_non_core_purpose"])
        writer.writeheader()

        for data_cat in DATATYPE_CATEGORIES:
            writer.writerow({
                "data_category": data_cat,
                "bad_sharing": len(apps_sharing_with_ad_or_analytics[data_cat]),
                "ext_bad_sharing": len(ext_apps_sharing_with_ad_or_analytics[data_cat]),
                "non_core_purpose": len(non_core_purpose_apps[data_cat]),
                "ext_non_core_purpose": len(ext_non_core_purpose_apps[data_cat]),
            })


if __name__ == "__main__":
    main()
